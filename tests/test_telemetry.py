import asyncio
import time
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np

from diana.telemetry import Trace
from diana.transcribe import OwnerGate, Utterance


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_has_scoped_id_and_measured_duration(self):
        sender = AsyncMock()
        trace = Trace(sender)
        trace.duration("queue", time.monotonic() - .1)
        await trace.update("rejected", "too short")
        self.assertNotEqual(trace.data["id"], Trace().data["id"])
        self.assertGreaterEqual(trace.data["metrics"]["queue"], 99)
        self.assertEqual(sender.call_args.args[0]["events"][0]["reason"], "too short")

    async def test_rejected_voice_has_no_asr_metric_or_progress(self):
        encoder, whisper, progress = Mock(), Mock(), Mock()
        encoder.encode.return_value = np.array([.1])
        gate = OwnerGate(encoder, np.array([1.]), whisper)
        pcm = np.full(32000, 1000, dtype="<i2").tobytes()
        metrics = {}
        result = gate.recognize(Utterance(pcm, pcm), metrics, progress)
        self.assertEqual(result[0], "rejected")
        self.assertIn("verify", metrics)
        self.assertNotIn("whisper", metrics)
        progress.assert_not_called()
        whisper.transcribe.assert_not_called()

    async def test_console_does_not_replace_diagnostic_route(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from diana.web import install
        app = FastAPI()
        install(app)
        @app.get("/client/")
        async def legacy():
            return {"legacy": True}
        @app.get("/")
        async def legacy_redirect():
            return {"wrong": True}
        with TestClient(app) as client:
            self.assertIn("Голосовой пульт", client.get("/").text)
            self.assertEqual(client.get("/client/").json(), {"legacy": True})
            self.assertEqual(client.get("/console/../diana/owner.py").status_code, 404)
