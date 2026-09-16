import asyncio
import sys
import unittest

from diana.codex import CodexClient, CodexError


# A separate process exercises actual JSONL pipes, not mocked request methods.
SERVER = r'''
import json, sys
held = None
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if "id" not in msg:
        continue
    if method == "initialize":
        out = {"id": msg["id"], "result": {"userAgent": "test"}}
    elif method == "wait":
        continue
    elif method == "die":
        break
    elif method == "fail":
        out = {"id": msg["id"], "error": {"code": -1, "message": "failed"}}
    elif method == "first":
        held = msg["id"]
        continue
    elif method == "second":
        print(json.dumps({"method": "item/agentMessage/delta", "params": {"delta": "hello"}}), flush=True)
        print(json.dumps({"id": msg["id"], "result": "second"}), flush=True)
        out = {"id": held, "result": "first"}
    elif method == "ask":
        print(json.dumps({"id": "approval", "method": "item/commandExecution/requestApproval", "params": {}}), flush=True)
        reply = json.loads(sys.stdin.readline())
        out = {"id": msg["id"], "result": reply}
    else:
        out = {"id": msg["id"], "result": {"data": []}}
    print(json.dumps(out), flush=True)
'''


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        return CodexClient([sys.executable, "-u", "-c", SERVER], timeout=2)

    async def test_out_of_order_responses_and_notification(self):
        async with self.client() as client:
            answers = await asyncio.gather(client.request("first"), client.request("second"))
            self.assertEqual(answers, ["first", "second"])
            event = await asyncio.wait_for(client.events.get(), 1)
            self.assertEqual(event["params"]["delta"], "hello")
            self.assertEqual(client.pending, {})

    async def test_rpc_error(self):
        async with self.client() as client:
            with self.assertRaisesRegex(CodexError, "failed"):
                await client.request("fail")

    async def test_eof_fails_pending_request(self):
        async with self.client() as client:
            with self.assertRaisesRegex(CodexError, "closed"):
                await client.request("die")

    async def test_timeout_does_not_poison_next_request(self):
        async with self.client() as client:
            client.timeout = 0.05
            with self.assertRaises(TimeoutError):
                await client.request("wait")
            client.timeout = 2
            self.assertEqual(await client.list_threads(), {"data": []})
            self.assertEqual(client.pending, {})

    async def test_approval_is_never_accepted_implicitly(self):
        async with self.client() as client:
            reply = await client.request("ask")
            self.assertEqual(reply["error"]["code"], -32601)
            self.assertNotIn("result", reply)


if __name__ == "__main__":
    unittest.main()
