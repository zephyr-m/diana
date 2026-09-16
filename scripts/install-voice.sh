#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if ! command -v uv >/dev/null; then
  echo 'Запусти из папки Diana: nix-shell --run "bash scripts/install-voice.sh"' >&2
  exit 1
fi
# CPU wheels avoid downloading the CUDA runtime on this laptop.
uv sync --extra voice
uv run --no-sync python -m diana.voice_check
echo 'Зависимости проверены. Модели устанавливаются отдельной командой.'
