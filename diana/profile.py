"""Local owner profile: embeddings only, no recorded audio."""
import json
import os
from pathlib import Path

import numpy as np

from .models import SPEAKER_SHA256


def unit_vector(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (256,) or not np.isfinite(vector).all():
        raise ValueError("Некорректный голосовой профиль")
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        raise ValueError("Пустой голосовой профиль")
    return vector / norm


def save_profile(path, vectors):
    if len(vectors) < 6:
        raise ValueError("Нужно минимум шесть фрагментов речи")
    centroid = unit_vector(np.mean([unit_vector(v) for v in vectors], axis=0))
    data = {"version": 1, "model_sha256": SPEAKER_SHA256,
            "segments": len(vectors), "embedding": centroid.tolist()}
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    # Exclusive creation: enrollment must never silently overwrite an owner.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(data, output)
    return centroid


def load_profile(path):
    data = json.loads(Path(path).read_text())
    if data.get("version") != 1 or data.get("model_sha256") != SPEAKER_SHA256:
        raise ValueError("Профиль создан другой версией модели; требуется новая регистрация")
    return unit_vector(data["embedding"])
