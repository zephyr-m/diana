"""Collect voiced PCM while tolerating short gaps between syllables."""


class SpeechBuffer:
    def __init__(self, window_samples=32000, max_gap_samples=12800):
        self.window_bytes = window_samples * 2
        self.max_gap_bytes = max_gap_samples * 2
        self.data = bytearray()
        self.gap_bytes = 0

    def feed(self, pcm, speaking):
        if not speaking:
            self.gap_bytes += len(pcm)
            if self.gap_bytes >= self.max_gap_bytes:
                self.clear()
            return []
        self.gap_bytes = 0
        self.data.extend(pcm)
        windows = []
        while len(self.data) >= self.window_bytes:
            windows.append(bytes(self.data[:self.window_bytes]))
            del self.data[:self.window_bytes]
        return windows

    def clear(self):
        self.data.clear()
        self.gap_bytes = 0
