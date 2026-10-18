"""CLI: `python -m agentic_rag.cli "question"` (interactive approval for SQL)."""
from __future__ import annotations

import argparse
import json

from agentic_rag.service import AgentService


def main() -> None:
    p = argparse.ArgumentParser(prog="agentic-rag")
    p.add_argument("question")
    p.add_argument("--user", default="demo")
    args = p.parse_args()

    svc = AgentService()
    out = svc.ask(args.question, user_id=args.user)
    while out["status"] == "awaiting_approval":
        print(f"\n[approval needed] The agent wants to run:\n  {out['request']['sql']}")
        ok = input("Approve? [y/N] ").strip().lower() == "y"
        out = svc.resume(out["thread_id"], ok, user_id=args.user)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
