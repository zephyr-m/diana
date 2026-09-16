import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from diana.profile import load_profile, save_profile, unit_vector


class ProfileTests(unittest.TestCase):
    def test_round_trip_private_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "owner.json"
            vector = np.ones(256, dtype=np.float32)
            expected = save_profile(path, [vector] * 6)
            np.testing.assert_allclose(load_profile(path), expected)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_profile(path, [-vector] * 6)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_vectors_rejected(self):
        for value in (np.zeros(256), np.ones(255), np.full(256, np.nan)):
            with self.assertRaises(ValueError):
                unit_vector(value)

    def test_model_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "owner.json"
            path.write_text(json.dumps({"version": 1, "model_sha256": "other",
                                        "embedding": [1.0] * 256}))
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_partial_enrollment_not_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "owner.json"
            with self.assertRaises(ValueError):
                save_profile(path, [np.ones(256)] * 5)
            self.assertFalse(path.exists())
