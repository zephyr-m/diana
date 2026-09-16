"""One live Codex exchange, resume check, and actual Piper inference."""
import asyncio
from pathlib import Path
import tempfile

from diana.conversation import Conversation, load_voice


async def main():
    with tempfile.TemporaryDirectory(prefix="diana-check-") as directory:
        state = Path(directory) / "conversation-id"
        conversation = Conversation(state=state)
        try:
            await conversation.start()
            answer = await conversation.reply("Ответь одним коротким предложением: голосовая связь работает.")
            print("Codex:", answer, flush=True)
            thread_id = conversation.thread_id
        finally:
            await conversation.close()
        resumed = Conversation(state=state)
        try:
            await resumed.start()
            assert resumed.thread_id == thread_id
            print("Resume: OK", flush=True)
        finally:
            await resumed.close()
        voice = await asyncio.to_thread(load_voice)
        chunks = await asyncio.to_thread(lambda: list(voice.synthesize(answer)))
        duration = sum(len(c.audio_int16_bytes) / (c.sample_rate * c.sample_channels * 2) for c in chunks)
        assert duration > 0
        print(f"Piper: {duration:.2f} seconds of audio; owner profile untouched", flush=True)


asyncio.run(main())
