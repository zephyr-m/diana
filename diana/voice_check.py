"""Check imports and record dependency versions; does not open the microphone."""
import importlib
import importlib.metadata
import json
from pathlib import Path


def main():
    for name in ("pipecat", "aiortc", "onnxruntime", "faster_whisper", "piper", "torch", "torchaudio", "soundfile", "soxr"):
        importlib.import_module(name)
    packages = ("pipecat-ai", "aiortc", "onnxruntime", "faster-whisper", "piper-tts", "torch", "torchaudio", "soundfile", "soxr")
    report = {name: importlib.metadata.version(name) for name in packages}
    output = Path(".local")
    output.mkdir(mode=0o700, exist_ok=True)
    (output / "voice-dependencies.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
