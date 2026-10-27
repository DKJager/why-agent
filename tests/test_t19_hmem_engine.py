"""T19: the Meterless H-MEM engine (agent.hmem) and its use as Why-Agent memory.

1. The official H-MEM conformance vectors (v1, verbatim) all pass.
2. Mutation controls: deliberately broken engines FAIL the same runner, so a
   green run means something.
3. The HMemMeterless adapter: confirm, supersede, recall, ledger, persistence,
   migration of a stub store.
4. End to end: ENGINE=meterless stores findings in run 1 and a brand-new
   process recalls them in run 2, changing the plan.
"""
import hashlib
import json
import re
import subprocess
import sys

import pytest

from agent.engine import HMemMeterless, HMemStub
from agent.hmem import HMEM
from agent.hmem.conformance import runner
from agent.agents.investigator import Investigator

from _helpers import Q1, Q2, REPO, out_of, read_trace, run, trace_path

OFFICIAL_VECTORS_SHA256 = "10afb5a07233466576ba6feb0d88408488909ba2721ee9157fdb932132917029"
T0 = 1_750_000_000_000


# ---------------------------------------------------------------- conformance
def test_vectors_are_the_official_file():
    assert hashlib.sha256(runner.VECTORS.read_bytes()).hexdigest() == OFFICIAL_VECTORS_SHA256


def test_conformance_all_green():
    results = runner.run_checks(HMEM)
    vectors = json.loads(runner.VECTORS.read_text(encoding="utf-8"))
    expected = sum(len(v) for v in vectors["sections"].values())
    assert len(results) == expected == 13
    failed = [r for r in results if r["status"] != "PASS"]
    assert not failed, failed


def test_conformance_cli_exit_code():
    proc = subprocess.run([sys.executable, "-m", "agent.hmem.conformance.runner"],
                          cwd=str(REPO), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "13/13 checks passed" in proc.stdout


class _AcceptsNoSource(HMEM):
    def add(self, content, source, **kw):
        return super().add(content, source or "anonymous", **kw)


class _WrongDelta(HMEM):
    def feedback(self, memory_id, kind):
        m = super().feedback(memory_id, kind)
        m.confidence = round(m.confidence + 0.01, 10)
        return m


class _NoSupersedePenalty(HMEM):
    def score(self, m, q):
        s = super().score(m, q)
        if m.superseded_by:
            s["score"] = min(1.0, s["raw"] + 0.2) * m.confidence + 0.5
        return s


class _DreamMaterialises(HMEM):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        orig = self.dreaming.dream

        def dream():
            ps = orig()
            for p in ps:
                self.add(content=p["content"], source="dreaming", derived_from=p["derivedFrom"])
            return ps
        self.dreaming.dream = dream


@pytest.mark.parametrize("impl,case", [
    (_AcceptsNoSource, "rv-001"),
    (_WrongDelta, "fd-001"),
    (_NoSupersedePenalty, "ro-001"),
    (_DreamMaterialises, "db-001"),
])
def test_mutation_controls_fail(impl, case):
    results = {r["id"]: r for r in runner.run_checks(impl)}
    assert results[case]["status"] == "FAIL"


# ---------------------------------------------------------------- engine unit
def test_feedback_moves_ranking():
    h = HMEM(clock=lambda: T0)
    a = h.add(content="NordExpress delayed EU orders in week 41", source="t", confidence=0.7)
    b = h.add(content="NordExpress delayed EU orders in week 41", source="t", confidence=0.7)
    h.feedback(a.id, "helpful")
    h.feedback(b.id, "not_helpful")
    order = [x["memory"].id for x in h.query("NordExpress delayed orders", threshold=0)["memories"]]
    assert order == [a.id, b.id]
    trace = h.query("NordExpress delayed orders", threshold=0)["trace"]
    assert trace["rankerVersion"] and trace["weightProfile"] == "weights@2026-10"


def test_review_tag_is_not_reinjected():
    h = HMEM(clock=lambda: T0)
    m = h.add(content="carrier NordExpress caused the delay", source="t")
    h.feedback(m.id, "wrong")
    assert h.query("NordExpress delay", threshold=0)["memories"] == []
    assert h.query("NordExpress delay", threshold=0, strategy="audit")["memories"]


def test_goal_run_scope_does_not_leak():
    h = HMEM(clock=lambda: T0)
    h.add(content="scratch note for run A about carriers", source="t", goal_run_id="A")
    assert h.query("carriers", threshold=0, goal_run_id="B", include_unscoped=False)["memories"] == []
    assert h.query("carriers", threshold=0, goal_run_id="A")["memories"]


def test_every_mutation_is_ledgered(tmp_path):
    h = HMEM(clock=lambda: T0, path=str(tmp_path / "h.json"))
    a = h.add(content="always retry NordExpress shipments", source="t", confidence=0.9)
    b = h.add(content="never retry NordExpress shipments", source="t", confidence=0.3)
    h.feedback(a.id, "helpful")
    h.conflicts.auto_resolve(h.conflicts.scan()[0])
    h.delete(b.id)
    actions = [e["action"] for e in h.ledger.all()]
    assert actions[:3] == ["create", "create", "feedback"]
    assert "resolve" in actions and actions[-1] == "delete"
    again = HMEM(clock=lambda: T0, path=str(tmp_path / "h.json"))
    assert [e["action"] for e in again.ledger.all()] == actions


# ---------------------------------------------------------------- adapter
KEY = "driver:carrier_delay@2026-W41"


def test_adapter_confirm_supersede_recall(tmp_path):
    path = tmp_path / "m" / "store.json"
    eng = HMemMeterless(str(path))
    r1 = eng.store("carrier:X", KEY, "X delayed orders", evidence_refs=["E1@run:a"],
                   confidence=0.8, run_id="a")
    assert re.fullmatch(r"m-[0-9a-f]{12}", r1.memory_id)
    r2 = eng.store("carrier:X", KEY, "X delayed orders", evidence_refs=["E1@run:b"],
                   confidence=0.8, run_id="b")
    assert r2.memory_id == r1.memory_id and r2.confirmations == 2
    assert r2.confidence == 0.85  # re-confirmation is H-MEM "helpful" feedback (+0.05)
    assert set(r2.evidence_refs) == {"E1@run:a", "E1@run:b"}

    r3 = eng.store("carrier:X", KEY, "Y is different now", evidence_refs=["E2@run:c"],
                   confidence=0.9, run_id="c")
    old = eng.get(r1.memory_id)
    assert old.status == "superseded" and old.superseded_by == r3.memory_id
    assert [r.memory_id for r in eng.recall(entities=["carrier:X"])] == [r3.memory_id]
    assert eng.last_trace["suppressedSuperseded"] == 1
    assert eng.recall(entities=["carrier:Z"]) == []

    again = HMemMeterless(str(path))
    assert [r.memory_id for r in again.recall(entities=["carrier:X"])] == [r3.memory_id]
    hist = [e["action"] for e in again.hmem.ledger.history(r1.memory_id)]
    assert hist == ["create", "feedback", "supersede"]
    assert again.forget(r3.memory_id) is True
    assert again.hmem.ledger.history(r3.memory_id)[-1]["action"] == "delete"
    assert again.health()["state"] == "online"


def test_adapter_migrates_a_stub_store(tmp_path):
    path = tmp_path / "store.json"
    stub = HMemStub(str(path))
    a = stub.store("carrier:X", KEY, "X delayed orders", evidence_refs=["E1@run:a"],
                   confidence=0.9, run_id="a")
    b = stub.store("carrier:X", KEY, "Y now", evidence_refs=["E2@run:b"], confidence=0.9, run_id="b")
    eng = HMemMeterless(str(path))
    assert json.loads(path.read_text())["engine"] == "hmem"
    assert eng.get(a.memory_id).status == "superseded"
    assert [r.memory_id for r in eng.recall(entities=["carrier:X"])] == [b.memory_id]
    assert eng.hmem.store.get(b.memory_id).provenance["origin"] == "import"


def test_adapter_rejects_corrupt_file(tmp_path):
    from agent.errors import ConfigError
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ConfigError):
        HMemMeterless(str(p))


# ---------------------------------------------------------------- end to end
def test_end_to_end_recall_across_processes(sample_env, monkeypatch, subprocess_env):
    monkeypatch.setenv("ENGINE", "meterless")
    out1 = out_of(run(Q1, "t19-s1"))
    assert out1["memory"]["stored"]
    store = json.loads(sample_env.hmem.read_text(encoding="utf-8"))
    assert store["engine"] == "hmem"
    stored_ids = {r["id"] for r in store["records"]}
    ents = {e["name"] for r in store["records"] for e in r["entities"]}
    assert {"carrier:NordExpress", "region:EU", "week:2026-W41"} <= ents
    assert all(r["provenance"]["origin"] == "goal_run" for r in store["records"])
    assert {e["action"] for e in store["ledger"]} == {"create"}

    env = dict(subprocess_env, ENGINE="meterless")
    proc = subprocess.run([sys.executable, "run.py", Q2, "--run-id", "t19-s2", "--json"],
                          cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out2 = out_of(json.loads(proc.stdout))
    recalled = set(re.findall(r"m-[0-9a-f]{12}", json.dumps(out2["memory"]["recalled"])))
    assert recalled & stored_ids

    trace = read_trace(trace_path(sample_env.logs, "t19-s2"))
    seen = set()
    for r in trace:
        d = r.get("details") or {}
        if r.get("kind") == "engine_call" and d.get("op") == "recall":
            seen |= set(d.get("memory_ids") or [])
    assert seen & stored_ids

    # The same finding confirmed again is H-MEM feedback, written to the ledger.
    out_of(run(Q1, "t19-s3"))
    store = json.loads(sample_env.hmem.read_text(encoding="utf-8"))
    assert "feedback" in {e["action"] for e in store["ledger"]}
    assert any(r["accessCount"] >= 1 for r in store["records"])


def test_recall_changes_the_plan(sample_env, monkeypatch, tmp_path):
    monkeypatch.setenv("ENGINE", "meterless")
    run(Q1, "t19-p1")
    holder = {}
    orig = Investigator.run

    def wrapper(self, ctx, plan):
        holder["plan"] = plan
        return orig(self, ctx, plan)

    monkeypatch.setattr(Investigator, "run", wrapper)
    run(Q2, "t19-p2")
    with_mem = sorted((sq.source, sq.priority) for sq in holder["plan"].subquestions)
    monkeypatch.setenv("HMEM_PATH", str(tmp_path / "empty" / "store.json"))
    res = run(Q2, "t19-p3")
    assert out_of(res)["memory"]["recalled"] == []
    without = sorted((sq.source, sq.priority) for sq in holder["plan"].subquestions)
    assert with_mem != without
