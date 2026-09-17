"""Per-utterance facts and monotonic durations; never model reasoning."""
import time
import uuid


class Trace:
    def __init__(self, sender=None, source="voice", ended=None):
        self.sender = sender
        self.ended = ended or time.monotonic()
        self.data = {"id": uuid.uuid4().hex[:12], "source": source,
                     "metrics": {}, "events": [], "stage": "listening"}

    async def update(self, stage=None, reason=None, **fields):
        if stage:
            self.data["stage"] = stage
            self.data["events"].append({"stage": stage, "at": time.time(), "reason": reason})
        if reason:
            self.data["reason"] = reason
        self.data.update(fields)
        if self.sender:
            await self.sender({"kind": "trace", **self.data})

    def duration(self, name, started):
        self.data["metrics"][name] = round((time.monotonic() - started) * 1000, 1)
