"""A persistent Codex conversation and local Piper speech output."""
import asyncio
import contextlib
import os
import re

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
        result = await self.client.request("turn/start", {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": text, "text_elements": []}],
        })
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
    def __init__(self, conversation, voice, report, send_frame):
        self.conversation, self.voice = conversation, voice
        self.report, self.send_frame = report, send_frame
        self.task = None
        # Cancellation of to_thread does not stop inference: serialize Piper.
        import threading
        self.voice_lock = threading.Lock()

    async def submit(self, text):
        from pipecat.frames.frames import InterruptionFrame
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.send_frame(InterruptionFrame())
        self.task = asyncio.create_task(self._answer(text))

    def synthesize(self, text):
        with self.voice_lock:
            return [(c.audio_int16_bytes, c.sample_rate, c.sample_channels)
                    for c in self.voice.synthesize(text)]

    async def _answer(self, text):
        from pipecat.frames.frames import OutputAudioRawFrame
        from loguru import logger
        try:
            await self.report("Диана думает…")
            answer = await self.conversation.reply(text)
            await self.report(answer)
            # Generate and send sentence-sized chunks; no WAVs on disk.
            for sentence in re.split(r"(?<=[.!?])\s+", answer):
                for pcm, rate, channels in await asyncio.to_thread(self.synthesize, sentence):
                    size = int(rate * channels * 2 * 0.02)
                    for offset in range(0, len(pcm), size):
                        await self.send_frame(OutputAudioRawFrame(
                            audio=pcm[offset:offset + size], sample_rate=rate, num_channels=channels))
                        await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            raise
        except Exception:
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
