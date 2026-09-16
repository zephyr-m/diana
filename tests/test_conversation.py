import asyncio
from pathlib import Path
import tempfile
import unittest

from diana.conversation import Conversation, VoiceDialogue
from diana.codex import CodexError


class FakeClient:
    def __init__(self):
        self.events = asyncio.Queue()
        self.calls = []

    async def __aenter__(self):
        return self

    async def close(self):
        pass

    async def request(self, method, params):
        self.calls.append((method, params))
        if method.startswith("thread/"):
            return {"thread": {"id": "saved-thread"}}
        return {"turn": {"id": "current-turn"}}


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_saved_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "thread"
            client = FakeClient()
            await Conversation(state, client).start()
            self.assertEqual(state.stat().st_mode & 0o777, 0o600)
            await Conversation(state, client).start()
            self.assertEqual(client.calls[-1][0], "thread/resume")
            self.assertEqual(client.calls[-1][1]["threadId"], "saved-thread")

    async def test_ignore_other_turns_and_threads(self):
        client = FakeClient()
        conversation = Conversation(client=client)
        conversation.thread_id = "thread"
        for thread, turn, text in [("other", "current-turn", "wrong"),
                                   ("thread", "old-turn", "stale"),
                                   ("thread", "current-turn", "answer")]:
            await client.events.put({"method": "item/completed", "params": {
                "threadId": thread, "turnId": turn,
                "item": {"type": "agentMessage", "text": text}}})
        await client.events.put({"method": "turn/completed", "params": {
            "threadId": "thread", "turn": {"id": "current-turn", "status": "completed"}}})
        self.assertEqual(await conversation.reply("hello"), "answer")
        self.assertEqual(len(client.calls), 1)

    async def test_cancel_interrupts_remote_turn(self):
        client = FakeClient()
        conversation = Conversation(client=client)
        conversation.thread_id = "thread"
        task = asyncio.create_task(conversation.reply("hello"))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(client.calls[-1], ("turn/interrupt", {
            "threadId": "thread", "turnId": "current-turn"}))

    async def test_failed_turn_does_not_return_partial_answer(self):
        client = FakeClient()
        conversation = Conversation(client=client)
        conversation.thread_id = "thread"
        await client.events.put({"method": "turn/completed", "params": {
            "threadId": "thread", "turn": {"id": "current-turn", "status": "failed"}}})
        with self.assertRaises(CodexError):
            await conversation.reply("hello")

    async def test_new_phrase_cancels_answer_and_flushes_audio(self):
        from pipecat.frames.frames import InterruptionFrame
        calls = []
        class SlowConversation:
            async def reply(self, text):
                calls.append(text)
                await asyncio.Event().wait()
            async def close(self):
                pass
        async def sink(value):
            calls.append(value)
        dialogue = VoiceDialogue(SlowConversation(), None, sink, sink)
        await dialogue.submit("first")
        await asyncio.sleep(0)
        first = dialogue.task
        await dialogue.submit("second")
        await asyncio.sleep(0)
        self.assertTrue(first.cancelled())
        self.assertEqual(sum(isinstance(c, InterruptionFrame) for c in calls), 2)
        self.assertIn("second", calls)
        await dialogue.close()
