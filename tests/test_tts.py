import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from diana.tts import SpeechSynthesizer


class Voice:
    def synthesize(self, text):
        return [SimpleNamespace(audio_int16_bytes=b"\x00\x00", sample_rate=22050, sample_channels=1)]


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_edge_failure_falls_back_and_cools_down(self):
        report = AsyncMock()
        speech = SpeechSynthesizer(Voice(), "edge", report)
        with patch("diana.tts.edge_synthesize", new_callable=AsyncMock) as edge:
            edge.side_effect = TimeoutError()
            self.assertEqual((await speech.synthesize("hello"))[0][1], 22050)
            await speech.synthesize("again")
            edge.assert_awaited_once()
            report.assert_awaited_once()

    async def test_cancellation_never_starts_fallback(self):
        voice = Voice()
        with patch.object(voice, "synthesize") as piper:
            with patch("diana.tts.edge_synthesize", new_callable=AsyncMock) as edge:
                edge.side_effect = asyncio.CancelledError()
                with self.assertRaises(asyncio.CancelledError):
                    await SpeechSynthesizer(voice, "edge").synthesize("hello")
                piper.assert_not_called()

    async def test_piper_mode_never_contacts_edge(self):
        with patch("diana.tts.edge_synthesize", new_callable=AsyncMock) as edge:
            await SpeechSynthesizer(Voice(), "piper").synthesize("hello")
            edge.assert_not_called()
