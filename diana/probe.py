"""Metadata-only connectivity check: python3 -m diana.probe."""
import argparse
import asyncio
import json
import sys

from .codex import CodexClient


async def probe(args):
    command = ["codex", "app-server", "proxy"] if args.proxy else None
    async with CodexClient(command=command) as client:
        result = await client.list_threads()
        threads = result.get("data", [])
        report = {
            "connected": True,
            "transport": "proxy" if args.proxy else "separate-stdio-server",
            "server": client.server_info,
            "threads": [{"id": t["id"], "cwd": t.get("cwd"),
                         "status": t.get("status")} for t in threads],
            "more_threads": bool(result.get("nextCursor")),
            "note": "Metadata access does not prove control of a task in another app.",
        }
        if args.thread:
            item = await client.request("thread/read", {
                "threadId": args.thread, "includeTurns": False,
            })
            report["requested_thread"] = {"id": item["thread"]["id"],
                                          "readable": True}
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--thread", help="Read metadata for a known thread ID")
    args = parser.parse_args()
    try:
        asyncio.run(probe(args))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Diana probe failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
