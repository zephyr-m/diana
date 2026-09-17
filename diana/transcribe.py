"""Owner-gated, local utterance transcription. No Codex calls or audio storage."""
import asyncio
from dataclasses import dataclass

import numpy as np
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor
import time

from .models import MODEL_DIR
from .telemetry import Trace

THRESHOLD = 0.35  # Provisional, based on the owner's initial small calibration set.
RATE_BYTES = 32000


@dataclass
class Utterance:
    audio: bytes
    voiced: bytes
    truncated: bool = False


class UtteranceBuffer:
    """Keep pauses and pre-roll for ASR; separate voiced samples for verification.

    Long utterances are capped; ignore their remaining audio until a pause so
    the user sees one explicit truncation rather than an unbounded queue.
    """
    def __init__(self):
        self.audio = bytearray()
        self.voiced = bytearray()
        self.preroll = bytearray()
        self.gap = 0
        self.discarding = False

    def feed(self, pcm, speaking):
        if self.discarding:
            self.gap = 0 if speaking else self.gap + len(pcm)
            if self.gap >= int(RATE_BYTES * 0.8):
                self.discarding = False
                self.gap = 0
            return None
        if not self.audio and not speaking:
            self.preroll.extend(pcm)
            del self.preroll[:-int(RATE_BYTES * 0.3)]
            return None
        if not self.audio:
            self.audio.extend(self.preroll)
            self.preroll.clear()
        self.audio.extend(pcm)
        if speaking:
            self.voiced.extend(pcm)
            self.gap = 0
        else:
            self.gap += len(pcm)
        truncated = len(self.audio) >= RATE_BYTES * 20
        if self.gap < int(RATE_BYTES * 0.8) and not truncated:
            return None
        result = Utterance(bytes(self.audio[:RATE_BYTES * 20]),
                           bytes(self.voiced[:RATE_BYTES * 20]), truncated)
        self.audio.clear()
        self.voiced.clear()
        self.gap = 0
        self.discarding = truncated
        return result


class OwnerGate:
    def __init__(self, encoder, reference, whisper, threshold=THRESHOLD):
        self.encoder, self.reference, self.whisper = encoder, reference, whisper
        self.threshold = threshold

    def recognize(self, utterance, metrics=None, progress=None, verify_owner=True):
        metrics = metrics if metrics is not None else {}
        started = time.monotonic()
        voice = np.frombuffer(utterance.voiced, dtype="<i2").astype(np.float32) / 32768
        score = None
        if verify_owner:
            if len(voice) < 16000:
                return "short", None, ""
            # Check overlapping windows, including the tail: an accepted beginning
            # must not automatically admit a later, different speaker.
            width = min(32000, len(voice))
            starts = list(range(0, len(voice) - width + 1, 16000))
            if starts[-1] != len(voice) - width:
                starts.append(len(voice) - width)
            scores = [float(self.encoder.encode(voice[i:i + width]) @ self.reference) for i in starts]
            metrics["verify"] = round((time.monotonic() - started) * 1000, 1)
            score = min(scores)
            if not np.isfinite(scores).all() or score < self.threshold:
                return "rejected", score, ""
        if progress:
            progress(score)
        audio = np.frombuffer(utterance.audio, dtype="<i2").astype(np.float32) / 32768
        started = time.monotonic()
        segments, _ = self.whisper.transcribe(audio, language="ru", beam_size=1,
                                             condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        metrics["whisper"] = round((time.monotonic() - started) * 1000, 1)
        return "accepted" if text else "empty", score, text


class OwnerTranscriber(FrameProcessor):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate
        self.vad = SileroVADAnalyzer(sample_rate=16000)
        self.vad.set_sample_rate(16000)
        self.buffer = UtteranceBuffer()
        self.queue = asyncio.Queue(maxsize=2)
        self.consumer = None
        self.reporter = None
        self.transcript_reporter = None
        self.last_audio = None
        self.complete = False
        self.was_speaking = False
        self.verify_owner = False
        self.telemetry = None
        self.trace = None
        self.last_voiced = None
        self.last_meter = 0

    async def report(self, text):
        logger.info(text)
        if self.reporter:
            await self.reporter(text)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)
            return
        if frame.sample_rate != 16000 or frame.num_channels != 1:
            raise ValueError("Ожидалось моно-аудио 16 кГц")
        self.last_audio = time.monotonic()
        if self.consumer is None:
            self.consumer = asyncio.create_task(self._consume())
            await self.report("Микрофон подключён. Скажи фразу и сделай небольшую паузу.")
        state = await self.vad.analyze_audio(frame.audio)
        speaking = state == VADState.SPEAKING
        if self.telemetry and self.last_audio - self.last_meter > 0.15:
            samples = np.frombuffer(frame.audio, dtype="<i2").astype(np.float32) / 32768
            db = 20 * np.log10(max(float(np.sqrt(np.mean(samples * samples))), 1e-6))
            await self.telemetry({"kind": "meter", "db": round(float(db), 1), "speaking": speaking})
            self.last_meter = self.last_audio
        if speaking and not self.was_speaking and not self.buffer.audio and not self.buffer.discarding:
            self.trace = Trace(self.telemetry)
            await self.trace.update("listening", "Микрофон: обнаружена речь")
            await self.report("Слушаю…")
        if speaking:
            self.last_voiced = self.last_audio
        self.was_speaking = speaking
        utterance = self.buffer.feed(frame.audio, speaking)
        if utterance is not None:
            trace = self.trace or Trace(self.telemetry)
            trace.ended = self.last_voiced or time.monotonic()
            trace.duration("pause", trace.ended)
            trace.data["queued_at"] = time.monotonic()
            await trace.update("queued", "Реплика завершена; ожидает обработки", truncated=utterance.truncated)
            self.trace = None
            try:
                self.queue.put_nowait((utterance, trace))
            except asyncio.QueueFull:
                await trace.update("rejected", "Очередь обработки заполнена")
                await self.report("Не успеваю обработать речь. Эта реплика пропущена; подожди результат и повтори.")

    async def _consume(self):
        while True:
            utterance, trace = await self.queue.get()
            try:
                trace.duration("queue", trace.data.pop("queued_at"))
                verify_owner = self.verify_owner  # Snapshot: changes apply to the next processed phrase.
                await trace.update("verify" if verify_owner else "whisper",
                                   "Проверяю все окна голоса" if verify_owner else "Проверка голоса выключена; принимается любая речь",
                                   verify_owner=verify_owner)
                loop = asyncio.get_running_loop()
                def progress(score):
                    asyncio.run_coroutine_threadsafe(trace.update(
                        "whisper", "Голос подтверждён; распознавание локальным Whisper" if verify_owner else "Whisper: без проверки владельца",
                        score=score, threshold=self.gate.threshold), loop).result(timeout=5)
                result, score, text = await asyncio.to_thread(self.gate.recognize, utterance, trace.data["metrics"], progress, verify_owner)
                reasons = {"short": "Меньше секунды речи для проверки владельца", "rejected": "Голос не прошёл проверку; Whisper и Codex не вызваны",
                           "empty": "Голос подтверждён, Whisper не разобрал слова", "accepted": "Все окна голоса прошли порог; текст распознан"}
                if not verify_owner:
                    reasons.update(accepted="Текст распознан без проверки владельца", empty="Whisper не разобрал слова; проверка владельца выключена")
                await trace.update("recognized" if result == "accepted" else "rejected", reasons[result],
                                   score=score if score is not None and np.isfinite(score) else None,
                                   threshold=self.gate.threshold, text=text)
                if utterance.truncated:
                    await self.report("Реплика длиннее 20 секунд: обработано начало. Остальное повтори после паузы.")
                if result == "short":
                    await self.report("Слишком коротко для проверки голоса. Пока скажи фразу подлиннее.")
                elif result == "rejected":
                    await self.report(f"Голос не подтверждён ({score:.3f}, порог {THRESHOLD}). Реплика отклонена.")
                elif result == "empty":
                    await self.report("Слова разобрать не удалось. Повтори фразу.")
                elif self.transcript_reporter:
                    await self.transcript_reporter(text, trace)
                    await self.report(f"Голос подтверждён ({score:.3f}). Текст распознан локально." if verify_owner else "Текст распознан без проверки голоса.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await trace.update("error", f"Ошибка обработки: {type(exc).__name__}")
                logger.error("Ошибка распознавания: {}", type(exc).__name__)
                await self.report("Ошибка обработки реплики. Повтори её; если ошибка повторится, сообщи об этом.")
            finally:
                self.queue.task_done()

    async def cleanup(self):
        if self.consumer:
            self.consumer.cancel()
            await asyncio.gather(self.consumer, return_exceptions=True)
        await self.vad.cleanup()
        await super().cleanup()


def load_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(str(MODEL_DIR / "whisper-small"), device="cpu", compute_type="int8",
                        cpu_threads=4, local_files_only=True)
