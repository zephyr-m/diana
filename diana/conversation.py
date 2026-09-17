"""A persistent Codex conversation and local Piper speech output."""
import asyncio
import contextlib
import os
import re
import time

from .codex import CodexClient, CodexError
from .models import ROOT, MODEL_DIR, VOICE

INSTRUCTIONS = (
    "Ты Диана, голосовой собеседник пользователя. Отвечай по-русски, обычно "
    "двумя-четырьмя короткими предложениями, без Markdown. Это отдельный диалог. "
    "Пока только обсуждай вопросы: не выполняй команды и не используй инструменты. "
    "Если просят действие, объясни, что выполнение действий ещё не подключено."
)


class Conversation:
    def __init__(self, state=None, client=None):
        self.state = state or ROOT / ".local" / "conversation-id"
        self.client = client or CodexClient(timeout=60)
        self.thread_id = None
        self.turn_id = None

    async def start(self):
        await self.client.__aenter__()
        try:
            params = {"cwd": str(ROOT), "sandbox": "read-only",
                      "approvalPolicy": "untrusted", "developerInstructions": INSTRUCTIONS}
            if self.state.exists():
                params["threadId"] = self.state.read_text().strip()
                result = await self.client.request("thread/resume", params)
            else:
                result = await self.client.request("thread/start", params)
            self.thread_id = result["thread"]["id"]
            self.state.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state.with_suffix(".tmp")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as output:
                output.write(self.thread_id + "\n")
            temporary.replace(self.state)
        except BaseException:
            await self.client.close()
            raise

    async def reply(self, text):
        start = asyncio.create_task(self.client.request("turn/start", {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": text, "text_elements": []}],
        }))
        try:
            result = await asyncio.shield(start)
        except asyncio.CancelledError:
            # Stop may arrive before the server returns the turn ID. Resolve
            # that acknowledgement so the remote turn is not left running.
            with contextlib.suppress(Exception):
                result = await start
                await self.client.request("turn/interrupt", {
                    "threadId": self.thread_id, "turnId": result["turn"]["id"]})
            raise
        self.turn_id = result["turn"]["id"]
        answers = []
        completed = False
        try:
            async with asyncio.timeout(180):
                while True:
                    event = await self.client.events.get()
                    params = event.get("params", {})
                    if params.get("threadId") != self.thread_id:
                        continue
                    if event["method"] == "item/completed" and params.get("turnId") == self.turn_id:
                        item = params["item"]
                        if item.get("type") == "agentMessage":
                            answers.append(item["text"])
                    if event["method"] == "turn/completed" and params["turn"]["id"] == self.turn_id:
                        completed = True
                        turn = params["turn"]
                        if turn["status"] != "completed":
                            raise CodexError(f"Turn {turn['status']}: {turn.get('error')}")
                        if not answers:
                            raise CodexError("Codex returned no answer")
                        return "\n".join(answers)
        finally:
            if self.turn_id and not completed:
                with contextlib.suppress(Exception):
                    await self.client.request("turn/interrupt", {
                        "threadId": self.thread_id, "turnId": self.turn_id})
            self.turn_id = None

    async def close(self):
        await self.client.close()


class VoiceDialogue:
    """Keep ASR free while a reply runs. A new verified phrase replaces it."""
    def __init__(self, conversation, voice, report, send_frame, tts_backend="piper"):
        self.conversation, self.voice = conversation, voice
        self.report, self.send_frame = report, send_frame
        self.task = None
        self.trace = None
        from .tts import SpeechSynthesizer
        self.speech = SpeechSynthesizer(voice, tts_backend, report)

    async def stop(self):
        from pipecat.frames.frames import InterruptionFrame
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.send_frame(InterruptionFrame())

    async def submit(self, text, trace=None):
        await self.stop()
        self.trace = trace
        self.task = asyncio.create_task(self._answer(text, trace))

    async def _answer(self, text, trace=None):
        from pipecat.frames.frames import OutputAudioRawFrame
        from loguru import logger
        try:
            if trace:
                await trace.update("codex", "Текст отправлен в отдельный диалог Codex", thread=self.conversation.thread_id)
            started = time.monotonic()
            await self.report("Диана думает…")
            answer = await self.conversation.reply(text)
            if trace:
                trace.duration("codex", started)
                await trace.update("tts", "Ответ получен; готовлю первое предложение", answer=answer)
            await self.report(answer)
            first = True
            # Generate and send sentence-sized chunks; no WAVs on disk.
            for sentence in re.split(r"(?<=[.!?])\s+", answer):
                started = time.monotonic()
                for pcm, rate, channels in await self.speech.synthesize(sentence):
                    if trace and first:
                        trace.duration("tts", started)
                        trace.duration("to_audio", trace.ended)
                        await trace.update("speaking", "Первое аудио передано в выходной тракт WebRTC",
                                           voice=self.speech.last_backend, fallback=self.speech.fallback_reason,
                                           retry_seconds=max(0, round(self.speech.retry_after - time.monotonic())))
                    first = False
                    size = int(rate * channels * 2 * 0.02)
                    for offset in range(0, len(pcm), size):
                        await self.send_frame(OutputAudioRawFrame(
                            audio=pcm[offset:offset + size], sample_rate=rate, num_channels=channels))
                        await asyncio.sleep(0.02)
            if trace:
                await trace.update("done", "Ответ полностью передан в WebRTC; буфер браузера может ещё звучать")
        except asyncio.CancelledError:
            if trace:
                await trace.update("interrupted", "Ответ остановлен новой репликой, кнопкой Стоп или отключением")
            raise
        except Exception:
            if trace:
                await trace.update("error", "Ошибка получения или озвучки ответа; подробности в терминале")
            logger.exception("Ошибка голосового ответа")
            await self.report("Не удалось получить или озвучить ответ. Попробуй ещё раз.")

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.conversation.close()


def load_voice():
    from piper import PiperVoice
    return PiperVoice.load(str(MODEL_DIR / f"{VOICE}.onnx"))
