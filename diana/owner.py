"""Register/test an owner using the packaged Pipecat WebRTC page.

This is a diagnostic pipeline. Scores do not authorize Codex commands.
"""
import argparse
import asyncio
import hashlib
import math
import time

import numpy as np
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi import models as RTVI
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from .models import MODEL_DIR, ROOT, SPEAKER_SHA256
from .profile import load_profile, save_profile, unit_vector
from .speaker import SpeakerEncoder
from .speech_buffer import SpeechBuffer

PROFILE = ROOT / ".local" / "owner.json"
WINDOW = 32000  # Two seconds of speech, non-overlapping diagnostic windows.
ACTIVE = False


class OwnerDiagnostic(FrameProcessor):
    def __init__(self, mode, encoder):
        super().__init__()
        self.mode = mode
        self.encoder = encoder
        self.vad = SileroVADAnalyzer(sample_rate=16000)
        self.vad.set_sample_rate(16000)
        self.buffer = SpeechBuffer(window_samples=WINDOW)
        self.vectors = []
        self.reference = load_profile(PROFILE) if mode == "verify" else None
        self.last_hint = 0
        self.complete = False
        self.reporter = None
        self.last_audio = None

    async def report(self, text):
        logger.info(text)
        if self.reporter is not None:
            await self.reporter(text)

    async def cleanup(self):
        await self.vad.cleanup()
        await super().cleanup()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)
            return
        if self.complete:
            return
        if frame.sample_rate != 16000 or frame.num_channels != 1:
            raise ValueError("Ожидалось моно-аудио 16 кГц")
        self.last_audio = time.monotonic()
        state = await self.vad.analyze_audio(frame.audio)
        windows = self.buffer.feed(frame.audio, state == VADState.SPEAKING)
        if self.last_audio - self.last_hint >= 3:
            values = np.frombuffer(frame.audio, dtype="<i2").astype(np.float32) / 32768
            rms = float(np.sqrt(np.mean(values * values))) if len(values) else 0
            db = 20 * math.log10(max(rms, 1e-6))
            status = "речь обнаружена" if state == VADState.SPEAKING else "ожидаю речь"
            progress = (f"регистрация {len(self.vectors)}/6" if self.mode == "enroll"
                        else "проверка голоса; профиль загружен")
            await self.report(f"Уровень микрофона: {db:.0f} dBFS; {status}; "
                              f"собрано речи {len(self.buffer.data) / 32000:.1f}/2 с; {progress}.")
            self.last_hint = self.last_audio
        for pcm in windows:
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            started = time.monotonic()
            vector = await asyncio.to_thread(self.encoder.encode, audio)
            elapsed = time.monotonic() - started
            if self.mode == "enroll":
                # A diagnostic consistency guard, not a calibrated biometric threshold.
                if self.vectors:
                    center = unit_vector(np.mean(self.vectors, axis=0))
                    if float(vector @ center) < 0.45:
                        await self.report("Фрагмент заметно отличается: пропускаю. Говори один, обычным голосом.")
                        continue
                self.vectors.append(vector)
                await self.report(f"Регистрация голоса: {len(self.vectors)}/6.")
                if len(self.vectors) == 6:
                    save_profile(PROFILE, self.vectors)
                    self.complete = True
                    self.buffer.clear()
                    await self.report("Профиль сохранён. Регистрация завершена, можно отключить микрофон.")
                    return
            else:
                score = float(vector @ self.reference)
                await self.report(f"Сходство с владельцем: {score:.3f}; вычисление {elapsed * 1000:.0f} мс; окно речи 2 с.")


async def bot(runner_args):
    global ACTIVE
    if ACTIVE:
        raise RuntimeError("Уже открыта одна микрофонная сессия Diana")
    ACTIVE = True
    dialogue = None
    conversation = None
    try:
        mode = runner_args.cli_args.mode
        if mode == "enroll" and PROFILE.exists():
            raise RuntimeError(f"Профиль уже существует: {PROFILE}. Используй verify.")
        model = MODEL_DIR / "wespeaker-resnet34.onnx"
        with model.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != SPEAKER_SHA256:
                raise ValueError("Контрольная сумма WeSpeaker не совпала")
        encoder = await asyncio.to_thread(SpeakerEncoder, model)
        if mode in ("transcribe", "chat"):
            from .transcribe import OwnerGate, OwnerTranscriber, load_whisper
            whisper = await asyncio.to_thread(load_whisper)
            diagnostic = OwnerTranscriber(OwnerGate(encoder, load_profile(PROFILE), whisper))
        else:
            diagnostic = OwnerDiagnostic(mode, encoder)
        transport = await create_transport(runner_args, {"webrtc": lambda: TransportParams(
            audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=16000,
        )})
        worker = PipelineWorker(Pipeline([transport.input(), diagnostic, transport.output()]),
                                params=PipelineParams(audio_in_sample_rate=16000))
        runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
        await runner.add_workers(worker)

        async def show_status(text):
            await worker.rtvi.push_transport_message(RTVI.BotOutputMessage(
                data=RTVI.BotOutputMessageData(text=text, aggregated_by="sentence",
                                              spoken=False, will_be_spoken=False,
                                              spoken_status="completed")))

        diagnostic.reporter = show_status
        if mode == "chat":
            from .conversation import Conversation, VoiceDialogue, load_voice
            conversation = Conversation()
            voice = await asyncio.to_thread(load_voice)
            await conversation.start()
            dialogue = VoiceDialogue(conversation, voice, show_status, diagnostic.push_frame)
        if mode in ("transcribe", "chat"):
            async def show_transcript(text):
                from datetime import datetime, timezone
                await worker.rtvi.push_transport_message(RTVI.UserTranscriptionMessage(
                    data=RTVI.UserTranscriptionMessageData(
                        text=text, user_id="owner", timestamp=datetime.now(timezone.utc).isoformat(), final=True)))
                if dialogue:
                    await dialogue.submit(text)
            diagnostic.transcript_reporter = show_transcript
        client_ready = asyncio.Event()

        @worker.rtvi.event_handler("on_client_ready")
        async def ready(rtvi):
            client_ready.set()
            welcome = (
                "Регистрация голоса. Говори обычным голосом — нужно собрать шесть фрагментов речи."
                if mode == "enroll" else
                "Проверка голоса. Твой профиль загружен и не изменяется. "
                "Говори — здесь появятся значения сходства с твоим профилем."
            )
            if mode == "transcribe":
                welcome = ("Распознавание включено. Скажи фразу и сделай паузу. "
                           "После проверки твоего голоса здесь появится текст. "
                           "Порог 0.35 предварительный. В Codex текст пока не отправляется.")
            elif mode == "chat":
                welcome = ("Диана готова к разговору. Скажи фразу и сделай паузу — "
                           "я отвечу голосом. Контекст разговора сохраняется. "
                           "Новая подтверждённая реплика прерывает предыдущий ответ.")
            await diagnostic.report(welcome)

        async def watch_audio():
            await client_ready.wait()
            while not diagnostic.complete:
                await asyncio.sleep(5)
                if diagnostic.last_audio is None or time.monotonic() - diagnostic.last_audio > 5:
                    await diagnostic.report("Звук не поступает. Проверь выбранный микрофон и кнопку Mute на странице.")

        @transport.event_handler("on_client_disconnected")
        async def disconnected(transport, client):
            await runner.cancel()

        logger.info("Режим {}. Подключи микрофон на странице Pipecat. Результаты — в этом терминале.", mode)
        watchdog = asyncio.create_task(watch_audio())
        try:
            await runner.run()
        finally:
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)
    finally:
        if dialogue:
            await dialogue.close()
        elif conversation:
            await conversation.close()
        ACTIVE = False


if __name__ == "__main__":
    from pipecat.runner.run import main
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["enroll", "verify", "transcribe", "chat"], required=True)
    main(parser)
