import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from diana.transcribe import OwnerGate, Utterance, UtteranceBuffer, RATE_BYTES


class UtteranceTests(unittest.TestCase):
    def test_keeps_initial_audio_and_internal_pause(self):
        buffer = UtteranceBuffer()
        buffer.feed(b"a" * 640, False)
        buffer.feed(b"b" * RATE_BYTES, True)
        buffer.feed(b"c" * 3200, False)
        buffer.feed(b"d" * RATE_BYTES, True)
        phrase = buffer.feed(b"e" * 25600, False)
        self.assertTrue(phrase.audio.startswith(b"a" * 640 + b"b" * RATE_BYTES))
        self.assertIn(b"c" * 3200, phrase.audio)
        self.assertEqual(phrase.voiced, b"b" * RATE_BYTES + b"d" * RATE_BYTES)
        self.assertFalse(buffer.audio)

    def test_silence_has_bounded_memory_and_no_utterance(self):
        buffer = UtteranceBuffer()
        for _ in range(1000):
            self.assertIsNone(buffer.feed(b"0" * 640, False))
        self.assertLessEqual(len(buffer.preroll), 9600)
        self.assertFalse(buffer.audio)

    def test_long_speech_is_capped_then_waits_for_pause(self):
        buffer = UtteranceBuffer()
        phrase = buffer.feed(b"a" * RATE_BYTES * 20, True)
        self.assertTrue(phrase.truncated)
        self.assertIsNone(buffer.feed(b"b" * RATE_BYTES, True))
        buffer.feed(b"0" * RATE_BYTES, False)
        buffer.feed(b"c" * RATE_BYTES, True)
        phrase = buffer.feed(b"0" * RATE_BYTES, False)
        self.assertEqual(phrase.voiced, b"c" * RATE_BYTES)


class GateTests(unittest.TestCase):
    def setup_gate(self, scores):
        encoder = Mock()
        encoder.encode.side_effect = [np.array([score]) for score in scores]
        whisper = Mock()
        whisper.transcribe.return_value = ([SimpleNamespace(text=" Тестовая фраза. ")], None)
        return OwnerGate(encoder, np.array([1.0]), whisper), whisper

    def phrase(self, seconds):
        pcm = np.full(int(16000 * seconds), 2000, dtype="<i2").tobytes()
        return Utterance(pcm, pcm)

    def test_foreign_speech_never_reaches_whisper(self):
        gate, whisper = self.setup_gate([0.12])
        self.assertEqual(gate.recognize(self.phrase(2))[0], "rejected")
        whisper.transcribe.assert_not_called()

    def test_disabled_verification_skips_encoder_and_accepts_short_speech(self):
        gate, whisper = self.setup_gate([])
        metrics = {}
        result, score, text = gate.recognize(self.phrase(0.5), metrics, verify_owner=False)
        self.assertEqual((result, score, text), ("accepted", None, "Тестовая фраза."))
        gate.encoder.encode.assert_not_called()
        self.assertNotIn("verify", metrics)
        whisper.transcribe.assert_called_once()

    def test_reenabled_verification_rejects_foreign_voice(self):
        gate, whisper = self.setup_gate([0.12])
        gate.recognize(self.phrase(2), verify_owner=False)
        whisper.reset_mock()
        self.assertEqual(gate.recognize(self.phrase(2), verify_owner=True)[0], "rejected")
        whisper.transcribe.assert_not_called()

    def test_change_of_speaker_blocks_whole_phrase(self):
        gate, whisper = self.setup_gate([0.65, 0.61, 0.10])
        self.assertEqual(gate.recognize(self.phrase(4))[0], "rejected")
        whisper.transcribe.assert_not_called()

    def test_unchecked_tail_cannot_bypass_gate(self):
        gate, whisper = self.setup_gate([0.65, 0.11])
        self.assertEqual(gate.recognize(self.phrase(2.5))[0], "rejected")
        whisper.transcribe.assert_not_called()

    def test_owner_transcribed_once(self):
        gate, whisper = self.setup_gate([0.65, 0.60])
        result, score, text = gate.recognize(self.phrase(3))
        self.assertEqual((result, score, text), ("accepted", 0.60, "Тестовая фраза."))
        whisper.transcribe.assert_called_once()

    def test_short_or_invalid_identity_never_reaches_whisper(self):
        gate, whisper = self.setup_gate([])
        self.assertEqual(gate.recognize(self.phrase(0.5))[0], "short")
        whisper.transcribe.assert_not_called()
        gate, whisper = self.setup_gate([0.65, float("nan")])
        self.assertEqual(gate.recognize(self.phrase(3))[0], "rejected")
        whisper.transcribe.assert_not_called()
