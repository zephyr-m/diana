"""Small stdio adapter for the installed Codex JSON-RPC protocol.

No credentials are copied; authentication stays in Codex. This first-stage
client deliberately rejects server requests (including approvals) until a UI
for them is implemented.
"""
import asyncio
import contextlib
import itertools
import json


class CodexError(RuntimeError):
    pass


class CodexClient:
    def __init__(self, command=None, timeout=20):
        self.command = command or ["codex", "app-server", "--stdio"]
        self.timeout = timeout
        self.events = asyncio.Queue()
        self.pending = {}
        self.ids = itertools.count(1)
        self.process = None
        self.reader = None
        self.server_info = None

    async def __aenter__(self):
        self.process = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Keep protocol stdout separate from diagnostic stderr.
            limit=16 * 1024 * 1024,
        )
        self.reader = asyncio.create_task(self._read())
        try:
            self.server_info = await self.request("initialize", {
                "clientInfo": {"name": "diana", "version": "0.1.0"},
            })
            await self._send({"method": "initialized"})
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _send(self, message):
        if self.process is None or self.process.returncode is not None:
            raise CodexError("Codex process is not running")
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def request(self, method, params=None):
        request_id = next(self.ids)
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method,
                              "params": params or {}})
            return await asyncio.wait_for(future, self.timeout)
        finally:
            self.pending.pop(request_id, None)

    async def _read(self):
        failure = CodexError("Codex closed the protocol connection")
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if "method" in message:
                    if "id" in message:
                        await self._send({"id": message["id"], "error": {
                            "code": -32601,
                            "message": "Diana approval/tool UI is not implemented",
                        }})
                    await self.events.put(message)
                    continue
                future = self.pending.get(message.get("id"))
                if future is None or future.done():
                    continue
                if "error" in message:
                    future.set_exception(CodexError(str(message["error"])))
                else:
                    future.set_result(message.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = CodexError(f"Protocol reader failed: {exc}")
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(failure)

    async def list_threads(self, limit=10):
        return await self.request("thread/list", {
            "limit": limit, "sortKey": "updated_at",
            "useStateDbOnly": True,
            "sourceKinds": ["cli", "vscode", "appServer", "unknown"],
        })

    async def close(self):
        if self.process is not None and self.process.returncode is None:
            self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 2)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 2)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        self.process.kill()
                    await self.process.wait()
        if self.reader is not None:
            self.reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader
