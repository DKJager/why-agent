#!/usr/bin/env python3
"""Generate samples/outputs/* by running the three sample questions through run.py.

Outputs are only ever produced by this script; never write or edit them by hand.

    python scripts/run_samples.py [--out-dir DIR] [--hmem PATH]

For every question in samples/questions.json it deletes the H-MEM file first
(once, before the first question), then runs, in order:

    python run.py "<q>" --run-id <id> --json --out <out-dir>/<id>.json

with SAMPLE_MODE=true and HMEM_PATH=<hmem>, and copies each trace to
<out-dir>/<id>.trace.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "samples", "outputs"))
    ap.add_argument("--hmem", default=os.path.join(ROOT, ".hmem", "samples.json"))
    ap.add_argument("--questions", default=os.path.join(ROOT, "samples", "questions.json"))
    args = ap.parse_args(argv)

    with open(args.questions, "r", encoding="utf-8") as fh:
        questions = json.load(fh)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    hmem = os.path.abspath(args.hmem)
    os.makedirs(os.path.dirname(hmem), exist_ok=True)
    if os.path.exists(hmem):
        os.remove(hmem)

    env = dict(os.environ)
    env["SAMPLE_MODE"] = "true"
    env["HMEM_PATH"] = hmem
    env["ENGINE"] = "stub"
    env.setdefault("LOG_DIR", "logs")
    log_dir = env["LOG_DIR"]
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(ROOT, log_dir)

    for item in questions:
        qid, question = item["id"], item["question"]
        trace_src = os.path.join(log_dir, f"trace-{qid}.jsonl")
        if os.path.exists(trace_src):
            os.remove(trace_src)  # the trace file is opened in append mode
        out_path = os.path.join(out_dir, f"{qid}.json")
        cmd = [sys.executable, os.path.join(ROOT, "run.py"), question,
               "--run-id", qid, "--json", "--out", out_path]
        print(f"[run_samples] {qid}: {question}", file=sys.stderr)
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            print(f"[run_samples] {qid} failed with exit code {proc.returncode}", file=sys.stderr)
            return 1
        if not os.path.exists(trace_src):
            print(f"[run_samples] trace not found: {trace_src}", file=sys.stderr)
            return 1
        shutil.copyfile(trace_src, os.path.join(out_dir, f"{qid}.trace.jsonl"))
        print(f"[run_samples] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
