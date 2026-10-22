import json
import re
import subprocess
import sys
from pathlib import Path

from _helpers import Q1, REPO, out_of, read_trace, run, trace_path

ACTIONS = {
    "start_run", "parse_question", "recall_memory", "delegate_task", "source_query",
    "provider_call", "compute_delta", "propose_claims", "review_claims", "accept_claim",
    "reject_claim", "request_refinement", "refine", "store_memory", "synthesize_answer",
    "fallback", "complete_run", "error",
}
STATUSES = {"success", "error", "rejected", "retry", "fallback", "skipped"}
AGENTS = {"Planner", "Investigator", "Critic", "Explainer", "Orchestrator"}
TS = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")


def _check_records(records, run_id):
    assert records
    last_step = 0
    for r in records:
        assert isinstance(r["timestamp"], str) and TS.match(r["timestamp"]), r["timestamp"]
        assert r["run_id"] == run_id
        assert isinstance(r["agent_name"], str) and r["agent_name"] in AGENTS
        assert isinstance(r["action"], str) and r["action"] in ACTIONS
        assert isinstance(r["input_summary"], str) and len(r["input_summary"]) <= 300
        assert isinstance(r["output_summary"], str) and len(r["output_summary"]) <= 300
        assert r["target_agent"] is None or isinstance(r["target_agent"], str)
        c = r["confidence"]
        assert c is None or (isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= c <= 1)
        assert isinstance(r["retry_count"], int) and not isinstance(r["retry_count"], bool)
        assert r["status"] in STATUSES
        assert isinstance(r["step"], int)
        assert r["step"] > last_step
        last_step = r["step"]


def test_trace_file_in_process(sample_env):
    run_id = "t06-inproc"
    res = run(Q1, run_id)
    out_of(res)
    path = trace_path(sample_env.logs, run_id)
    assert path.exists()
    records = read_trace(path)
    _check_records(records, run_id)

    names = {r["agent_name"] for r in records}
    assert {"Planner", "Investigator", "Critic", "Explainer"} <= names
    assert any(r.get("kind") == "source_call" for r in records)
    assert any(r.get("kind") == "provider_call" for r in records)
    ops = {(r.get("details") or {}).get("op") for r in records if r.get("kind") == "engine_call"}
    assert "recall" in ops and "store" in ops
    assert any(r["action"] == "delegate_task" and r["target_agent"] == "Investigator"
               for r in records)

    assert res["trace_id"] == f"trace-{run_id}"
    assert Path(res["log_file"]).name == f"trace-{run_id}.jsonl"
    assert Path(res["log_file"]).exists() or (REPO / res["log_file"]).exists()


def test_stdout_format_and_json_mode(sample_env, subprocess_env):
    run_id = "t06-cli"
    proc = subprocess.run(
        [sys.executable, "run.py", Q1, "--run-id", run_id],
        cwd=str(REPO), env=subprocess_env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    pat = re.compile(r"^\[(INFO|WARNING|ERROR)\] run_id=" + re.escape(run_id) + r" ")
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    matching = [l for l in lines if pat.match(l)]
    records = read_trace(trace_path(sample_env.logs, run_id))
    _check_records(records, run_id)
    assert len(matching) >= len(records)
    assert re.search(r"Completed in", lines[-1]) and pat.match(lines[-1])

    run_id2 = "t06-json"
    proc2 = subprocess.run(
        [sys.executable, "run.py", Q1, "--run-id", run_id2, "--json"],
        cwd=str(REPO), env=subprocess_env, capture_output=True, text=True, timeout=120,
    )
    assert proc2.returncode == 0, proc2.stderr
    body = json.loads(proc2.stdout)
    assert isinstance(body, dict)
    assert body["run_id"] == run_id2
    assert body["trace_id"] == f"trace-{run_id2}"
    assert Path(body["log_file"]).name == f"trace-{run_id2}.jsonl"
    assert trace_path(sample_env.logs, run_id2).exists()
    _check_records(read_trace(trace_path(sample_env.logs, run_id2)), run_id2)
