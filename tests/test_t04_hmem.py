# build-id: oa-b85bde9e
import json
import re
import subprocess
import sys

from agent.agents.investigator import Investigator
from agent.engine import HMemStub

from _helpers import Q1, Q2, REPO, out_of, read_trace, run, trace_path


def _records(path):
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _capture_plan(monkeypatch):
    holder = {}
    orig = Investigator.run

    def wrapper(self, ctx, plan):
        holder["plan"] = plan
        return orig(self, ctx, plan)

    monkeypatch.setattr(Investigator, "run", wrapper)
    return holder


def _plan_shape(plan):
    return sorted((sq.source, sq.priority) for sq in plan.subquestions)


def test_recall_across_sessions(sample_env, monkeypatch, tmp_path, subprocess_env):
    # Session 1
    res1 = run(Q1, "t04-s1")
    out1 = out_of(res1)
    assert out1["memory"]["stored"], "nothing stored"
    recs = _records(sample_env.hmem)
    entities = {r["entity"] for r in recs}
    assert {"carrier:NordExpress", "region:EU", "week:2026-W41"} <= entities
    stored_ids = {r["memory_id"] for r in recs}

    # Session 2 in a brand new process
    env = dict(subprocess_env)
    proc = subprocess.run(
        [sys.executable, "run.py", Q2, "--run-id", "t04-s2", "--json"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    res2 = json.loads(proc.stdout)
    out2 = out_of(res2)
    recalled_ids = set(re.findall(r"m-[0-9a-f]{12}", json.dumps(out2["memory"]["recalled"])))
    assert recalled_ids & stored_ids, "session-1 memory not recalled"
    prior_ids = set(re.findall(r"m-[0-9a-f]{12}", json.dumps(out2["answer"]["prior_knowledge"])))
    assert prior_ids & stored_ids

    trace = read_trace(trace_path(sample_env.logs, "t04-s2"))
    seen = set()
    for r in trace:
        d = r.get("details") or {}
        if r.get("kind") == "engine_call" and d.get("op") == "recall":
            seen |= set(d.get("memory_ids") or [])
    assert seen & stored_ids

    # Plan differs from a run with an empty store
    holder = _capture_plan(monkeypatch)
    run(Q2, "t04-s2b")
    plan_with = holder.pop("plan")
    shipments = [sq for sq in plan_with.subquestions if sq.source == "shipments"]
    assert shipments and min(sq.priority for sq in shipments) == 1

    monkeypatch.setenv("HMEM_PATH", str(tmp_path / "empty" / "store.json"))
    res_ctrl = run(Q2, "t04-ctrl")
    plan_empty = holder["plan"]
    assert out_of(res_ctrl)["memory"]["recalled"] == []
    assert _plan_shape(plan_with) != _plan_shape(plan_empty)


def test_stub_confirm_and_supersede(tmp_path):
    path = tmp_path / "m" / "store.json"
    stub = HMemStub(str(path))
    r1 = stub.store("carrier:X", "driver:carrier_delay@2026-W41", "X delayed orders",
                    evidence_refs=["E1@run:a"], confidence=0.9, run_id="a")
    r2 = stub.store("carrier:X", "driver:carrier_delay@2026-W41", "X delayed orders",
                    evidence_refs=["E1@run:b"], confidence=0.9, run_id="b")
    assert r2.memory_id == r1.memory_id
    assert r2.confirmations == 2
    assert len(stub.recall(entities=["carrier:X"])) == 1

    r3 = stub.store("carrier:X", "driver:carrier_delay@2026-W41", "Y is different now",
                    evidence_refs=["E2@run:c"], confidence=0.9, run_id="c")
    assert r3.memory_id != r1.memory_id
    old = stub.get(r1.memory_id)
    assert old.status == "superseded"
    assert old.superseded_by == r3.memory_id
    recalled = stub.recall(entities=["carrier:X"])
    assert [r.memory_id for r in recalled] == [r3.memory_id]

    # persistence across instances
    again = HMemStub(str(path))
    assert [r.memory_id for r in again.recall(entities=["carrier:X"])] == [r3.memory_id]
    assert stub.forget(r3.memory_id) is True
