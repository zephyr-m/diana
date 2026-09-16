"""Real-model smoke check with a synthetic reference kept only in memory.

Never modifies owner.json. Checks model plumbing, not owner discrimination.
"""
import numpy as np
import soundfile as sf
import soxr
from diana.models import ROOT, MODEL_DIR
from diana.speaker import SpeakerEncoder
from diana.transcribe import OwnerGate, Utterance, load_whisper

samples, rate = sf.read(ROOT / ".local/voice-sample.wav", dtype="float32")
samples = soxr.resample(samples, rate, 16000)
pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
encoder = SpeakerEncoder(MODEL_DIR / "wespeaker-resnet34.onnx")
reference = encoder.encode(samples)
gate = OwnerGate(encoder, reference, load_whisper())
result, score, text = gate.recognize(Utterance(pcm, pcm))
assert result == "accepted" and "диана" in text.lower(), (result, score, text)
print(f"Real-model smoke test: {result}, minimum similarity {score:.3f}; {text}")
print("Owner profile untouched. This test uses only the synthetic sample.")
