#!/usr/bin/env python3
# build-id: oa-048f6ab0
"""Why-Agent CLI.

    python run.py "Why did EU revenue drop in week 41?" [--run-id ID] [--json] [--out PATH]
    python run.py --serve [--port N]

Exit codes: 0 success or partial, 1 structured error result, 2 usage error.
With --json, stdout is a single JSON object and log lines go to stderr.
Without --json, stdout carries the log lines (the last one is "Completed in ...")
and the human-readable answer is printed to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.errors import InvalidRequest, error_body  # noqa: E402
from agent.orchestrator import run_question  # noqa: E402

GENERATED_BY = "run.py"


class UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        raise UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="run.py", description="Why-Agent: explain a business metric change with cited evidence.")
    p.add_argument("question", nargs="?", help="the business question, e.g. 'Why did EU revenue drop in week 41?'")
    p.add_argument("--run-id", dest="run_id", default=None, help="run id ([A-Za-z0-9._-]{1,64})")
    p.add_argument("--json", dest="as_json", action="store_true", help="print only the result JSON to stdout")
    p.add_argument("--out", default=None, help="also write the result JSON to this path")
    p.add_argument("--serve", action="store_true", help="start the HTTP server")
    p.add_argument("--port", type=int, default=None, help="HTTP port (default $PORT or 8000)")
    p.add_argument("--sample", action="store_true", help="shorthand for SAMPLE_MODE=true")
    return p


def _usage_error(message: str, as_json: bool) -> int:
    if as_json:
        body = error_body(InvalidRequest(message), run_id=None, trace_id=None, log_file=None,
                          execution_time_seconds=0.0)
        print(json.dumps(body, indent=2, ensure_ascii=False))
    else:
        print(f"usage error: {message}", file=sys.stderr)
        print("usage: run.py QUESTION [--run-id ID] [--json] [--out PATH] | run.py --serve [--port N]",
              file=sys.stderr)
    return 2


def _print_human(result: dict) -> None:
    out = sys.stderr
    if result.get("status") == "error":
        err = result.get("error", {})
        print(f"\nERROR {err.get('type')}: {err.get('message')}", file=out)
        return
    o = result.get("output", {})
    ans = o.get("answer", {})
    print("", file=out)
    print(f"Status: {result.get('status')}", file=out)
    print(ans.get("text") or o.get("summary", ""), file=out)
    if o.get("recommendations"):
        print("\nRecommendations:", file=out)
        for r in o["recommendations"]:
            print(f"- {r}", file=out)
    if ans.get("caveats"):
        print("\nCaveats:", file=out)
        for c in ans["caveats"]:
            print(f"- {c}", file=out)
    print(f"\nTrace: {result.get('log_file')}", file=out)


def main(argv: Optional[list] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except UsageError as e:
        return _usage_error(str(e), "--json" in argv)

    if args.sample:
        os.environ["SAMPLE_MODE"] = "true"

    if args.serve:
        from agent.server import serve
        serve(args.port)
        return 0

    if args.question is None or not args.question.strip():
        return _usage_error("a question is required (or use --serve)", args.as_json)

    options = {"log_to_stderr": True} if args.as_json else None
    result = run_question(args.question, run_id=args.run_id, options=options)
    result["generated_by"] = GENERATED_BY
    text = json.dumps(result, indent=2, ensure_ascii=False)

    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")

    if args.as_json:
        print(text)
    else:
        _print_human(result)

    return 0 if result.get("status") in ("success", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
