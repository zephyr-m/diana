"""Edge Svetlana with local Piper fallback, returning mono PCM for WebRTC."""
import asyncio
import io
import threading
import time

SVETLANA = "ru-RU-SvetlanaNeural"


def decode_mp3(data):
    import av
    resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
    parts = []
    with av.open(io.BytesIO(data), format="mp3") as source:
        for frame in source.decode(audio=0):
            for pcm in resampler.resample(frame):
                parts.append(pcm.to_ndarray().tobytes())
        for pcm in resampler.resample(None):
            parts.append(pcm.to_ndarray().tobytes())
    if not parts:
        raise ValueError("Empty Edge audio")
    return [(b"".join(parts), 24000, 1)]


async def edge_synthesize(text):
    import edge_tts
    data = bytearray()
    async with asyncio.timeout(15):
        async for chunk in edge_tts.Communicate(text, SVETLANA).stream():
            if chunk["type"] == "audio":
                data.extend(chunk["data"])
    return await asyncio.to_thread(decode_mp3, bytes(data))


class SpeechSynthesizer:
    def __init__(self, voice, backend="piper", report=None):
        self.voice, self.backend, self.report = voice, backend, report
        self.lock = threading.Lock()
        self.retry_after = 0
        self.last_backend = backend
        self.fallback_reason = None

    def piper_synthesize(self, text):
        # Cancelling to_thread doesn't stop inference; serialize access.
        with self.lock:
            return [(c.audio_int16_bytes, c.sample_rate, c.sample_channels)
                    for c in self.voice.synthesize(text)]

    async def synthesize(self, text):
        if self.backend == "edge" and time.monotonic() >= self.retry_after:
            try:
                result = await edge_synthesize(text)
                self.last_backend, self.fallback_reason = "edge", None
                return result
            except Exception as exc:
                from loguru import logger
                logger.warning("Edge TTS unavailable: {}", type(exc).__name__)
                self.retry_after = time.monotonic() + 60
                self.fallback_reason = f"Edge: {type(exc).__name__}; повторная попытка через минуту"
                if self.report:
                    await self.report("Светлана недоступна. Временно озвучиваю локальным голосом Piper.")
        self.last_backend = "piper"
        return await asyncio.to_thread(self.piper_synthesize, text)
