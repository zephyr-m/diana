"""Offline diagnostic of VAD and speaker inference; never enrolls a profile."""
import asyncio
import numpy as np
import soundfile as sf
import soxr
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState
from diana.models import ROOT, MODEL_DIR
from diana.speaker import SpeakerEncoder


async def main():
    audio, rate = sf.read(ROOT / ".local/voice-sample.wav", dtype="float32")
    audio = soxr.resample(audio, rate, 16000)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    vad = SileroVADAnalyzer(sample_rate=16000)
    vad.set_sample_rate(16000)
    voiced = 0
    for start in range(0, len(pcm) - 640, 640):
        if await vad.analyze_audio(pcm[start:start + 640]) == VADState.SPEAKING:
            voiced += 1
    assert voiced > 0, "VAD did not find speech"
    vector = SpeakerEncoder(MODEL_DIR / "wespeaker-resnet34.onnx").encode(audio)
    assert vector.shape == (256,)
    print(f"VAD: {voiced} speech frames; WeSpeaker: {len(vector)} dimensions; no profile saved")


asyncio.run(main())
