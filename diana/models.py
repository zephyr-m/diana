"""Download and smoke-test the local speech and speaker models (user-run)."""
import argparse
import gc
import json
import time
import wave
import hashlib
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / ".local" / "models"
VOICE = "ru_RU-irina-medium"
SPEAKER_REVISION = "ff1ac5bca8ef11e90662b879aa923979e0bd277b"
SPEAKER_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/"
    f"{SPEAKER_REVISION}/voxceleb_resnet34.onnx"
)
SPEAKER_SHA256 = "9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1"


def prepare_speaker(download=True):
    import numpy as np
    import soundfile as sf
    import soxr
    from .speaker import SpeakerEncoder

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sample = ROOT / ".local" / "voice-sample.wav"
    if not sample.is_file():
        raise RuntimeError("Сначала запусти полную проверку: python -m diana.models")
    path = MODEL_DIR / "wespeaker-resnet34.onnx"
    if download and not path.exists():
        print("Скачиваю WeSpeaker ResNet34 ONNX…", flush=True)
        temporary = path.with_suffix(".partial")
        with urllib.request.urlopen(SPEAKER_URL, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        with temporary.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != SPEAKER_SHA256:
                raise RuntimeError("Контрольная сумма WeSpeaker не совпала. Повтори скачивание.")
        # Validate the graph before accepting the download as complete.
        SpeakerEncoder(temporary)
        temporary.replace(path)
    encoder = SpeakerEncoder(path)
    audio, rate = sf.read(ROOT / ".local" / "voice-sample.wav", dtype="float32")
    if rate != 16000:
        audio = soxr.resample(audio, rate, 16000)
    embedding = encoder.encode(audio)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    report = {"model": str(path), "url": SPEAKER_URL, "sha256": digest,
              "revision": SPEAKER_REVISION,
              "dimensions": len(embedding), "norm": float(np.linalg.norm(embedding)),
              "note": "Model inference only. Synthetic voice is NOT enrolled as owner."}
    (ROOT / ".local" / "speaker-model-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def prepare(download=True):
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    from piper import PiperVoice
    from piper.download_voices import download_voice

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    whisper_dir = MODEL_DIR / "whisper-small"
    if download:
        print("Скачиваю Whisper small (многоязычный)…", flush=True)
        download_model("small", output_dir=str(whisper_dir))
        print("Скачиваю русский голос Piper…", flush=True)
        download_voice(VOICE, MODEL_DIR)

    print("Проверяю синтез русского голоса…", flush=True)
    voice = PiperVoice.load(str(MODEL_DIR / f"{VOICE}.onnx"))
    sample = ROOT / ".local" / "voice-sample.wav"
    with wave.open(str(sample), "wb") as output:
        voice.synthesize_wav("Привет. Это Диана. Проверка голосовой связи.", output)
    del voice
    gc.collect()
    print("Проверяю распознавание на CPU…", flush=True)
    model = WhisperModel(str(whisper_dir), device="cpu", compute_type="int8",
                         cpu_threads=4, local_files_only=True)
    started = time.monotonic()
    segments, info = model.transcribe(str(sample), language="ru", beam_size=1)
    text = " ".join(segment.text.strip() for segment in segments)
    if not text.strip():
        raise RuntimeError("Whisper вернул пустой текст для тестовой записи")
    report = {"whisper": "small", "voice": VOICE, "recognized": text,
              "transcription_seconds": round(time.monotonic() - started, 2),
              "audio_seconds": info.duration,
              "note": "Synthetic speech smoke test, not microphone accuracy or latency."}
    (ROOT / ".local" / "models-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Голос для прослушивания: {sample}")
    del model
    gc.collect()
    prepare_speaker(download)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Do not download models")
    parser.add_argument("--speaker-only", action="store_true", help="Resume at WeSpeaker using the existing test WAV")
    args = parser.parse_args()
    if args.speaker_only:
        prepare_speaker(download=not args.check_only)
    else:
        prepare(download=not args.check_only)
