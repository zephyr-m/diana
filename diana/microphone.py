"""Browser microphone diagnostic using Pipecat's packaged WebRTC client.

Based on Pipecat's voice-agent transport/pipeline pattern:
https://github.com/pipecat-ai/pipecat/blob/main/examples/getting-started/06-voice-agent.py
No audio files, transcription, or Codex requests are produced.
"""
import math
import time

import numpy as np
from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner


class MicrophoneMeter(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.last_report = 0.0
        self.samples = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            values = np.frombuffer(frame.audio, dtype="<i2").astype(np.float32)
            self.samples += len(values)
            now = time.monotonic()
            if len(values) and now - self.last_report >= 1:
                rms = float(np.sqrt(np.mean(values * values))) / 32768
                db = 20 * math.log10(max(rms, 1e-6))
                logger.info("Микрофон: {} Гц, каналов {}, уровень {:.0f} dBFS, сэмплов {}",
                            frame.sample_rate, frame.num_channels, db, self.samples)
                self.last_report = now
            return
        await self.push_frame(frame, direction)


async def bot(runner_args):
    transport = await create_transport(runner_args, {"webrtc": lambda: TransportParams(
        audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=16000,
    )})
    pipeline = Pipeline([transport.input(), MicrophoneMeter(), transport.output()])
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=16000))
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def connected(transport, client):
        logger.info("Микрофон подключён. Говори; уровень будет показан в терминале.")

    @transport.event_handler("on_client_disconnected")
    async def disconnected(transport, client):
        await runner.cancel()

    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
