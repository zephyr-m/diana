import unittest
from diana.speech_buffer import SpeechBuffer


class SpeechBufferTests(unittest.TestCase):
    def test_short_gap_keeps_speech_but_excludes_silence(self):
        buffer = SpeechBuffer(window_samples=4, max_gap_samples=3)
        self.assertEqual(buffer.feed(b"aaaa", True), [])
        self.assertEqual(buffer.feed(b"00", False), [])
        self.assertEqual(buffer.feed(b"bbbb", True), [b"aaaabbbb"])

    def test_long_gap_discards_previous_speaker_fragment(self):
        buffer = SpeechBuffer(window_samples=4, max_gap_samples=3)
        buffer.feed(b"aaaa", True)
        buffer.feed(b"000000", False)
        self.assertEqual(buffer.feed(b"bbbb", True), [])
        self.assertEqual(buffer.feed(b"cccc", True), [b"bbbbcccc"])

    def test_partial_frames_and_multiple_windows(self):
        buffer = SpeechBuffer(window_samples=2)
        self.assertEqual(buffer.feed(b"aa", True), [])
        self.assertEqual(buffer.feed(b"bbbbcccc", True), [b"aabb", b"bbcc"])
        self.assertEqual(bytes(buffer.data), b"cc")

    def test_silence_cannot_produce_window(self):
        buffer = SpeechBuffer(window_samples=2)
        self.assertEqual(buffer.feed(b"0" * 100000, False), [])
        self.assertEqual(len(buffer.data), 0)
