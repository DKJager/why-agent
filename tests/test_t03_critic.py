import copy
import json

from agent.agents.critic import Critic
from agent.agents.investigator import Investigator
from agent.models import Claim, NumberRef

from _helpers import Q1, read_trace, run, trace_path

UNCITED = "Customers disliked a price increase"


def _codes(review, claim_id):
    return {r.code for r in review.rejected if r.claim_id == claim_id}


def _accepted_ids(review):
    return {c.claim_id for c in review.accepted}


def _craft(orig, critic, ctx, evidence):
    evidence = list(evidence)
    ev0 = evidence[0]
    ev_orders = next(e for e in evidence if e.source == "orders")
    out = {}

    # (a) no citation
    c = Claim(claim_id="T3a", kind="observation", text="Orders fell", numbers=[], evidence_ids=[])
    out["a"] = orig(critic, ctx, [c], evidence)

    # (b) unknown evidence
    c = Claim(claim_id="T3b", kind="observation", text="Orders fell", numbers=[],
              evidence_ids=["E99"])
    out["b"] = orig(critic, ctx, [c], evidence)

    # (c) altered number
    ev_num = next(e for e in evidence if any(abs(float(v)) > 0 for v in e.values.values()))
    name = next(k for k, v in ev_num.values.items() if abs(float(v)) > 0)
    val = float(ev_num.values[name]) * 1.10
    c = Claim(claim_id="T3c", kind="observation", text="The value is shown here",
              numbers=[NumberRef(ev_num.evidence_id, name, val, "int")],
              evidence_ids=[ev_num.evidence_id])
    out["c"] = orig(critic, ctx, [c], evidence)

    # (d) ungrounded number in text
    c = Claim(claim_id="T3d", kind="observation",
              text="Revenue changed by 987654 units", numbers=[],
              evidence_ids=[ev0.evidence_id])
    out["d"] = orig(critic, ctx, [c], evidence)

    # (e) driver with a single source
    c = Claim(claim_id="T3e", kind="driver", text="A single sourced driver", numbers=[],
              evidence_ids=[ev_orders.evidence_id])
    out["e"] = orig(critic, ctx, [c], evidence)

    # (f) non-existent row_ref
    ev_mod = copy.deepcopy(evidence)
    for e in ev_mod:
        if e.evidence_id == ev_orders.evidence_id:
            e.row_refs = ["O-999999"]
            e.row_ref_total = 1
    c = Claim(claim_id="T3f", kind="observation", text="Rows were cited", numbers=[],
              evidence_ids=[ev_orders.evidence_id])
    out["f"] = orig(critic, ctx, [c], ev_mod)

    # (g) clean claim
    c = Claim(claim_id="T3g", kind="observation", text="Evidence was retrieved", numbers=[],
              evidence_ids=[ev_orders.evidence_id])
    out["g"] = orig(critic, ctx, [c], evidence)
    return out


def test_critic_rules_unit(sample_env, monkeypatch):
    orig = Critic.review
    state = {}

    def wrapper(self, ctx, claims, evidence):
        if "done" not in state:
            state["done"] = True
            try:
                state["results"] = _craft(orig, self, ctx, evidence)
            except Exception as e:  # noqa: BLE001
                state["exc"] = e
        return orig(self, ctx, claims, evidence)

    monkeypatch.setattr(Critic, "review", wrapper)
    run(Q1, "t03-unit")
    assert "exc" not in state, repr(state.get("exc"))
    r = state["results"]

    assert "NO_CITATION" in _codes(r["a"], "T3a")
    assert "T3a" not in _accepted_ids(r["a"])
    assert "UNKNOWN_EVIDENCE" in _codes(r["b"], "T3b")
    assert "NUMBER_MISMATCH" in _codes(r["c"], "T3c")
    assert "UNGROUNDED_NUMBER" in _codes(r["d"], "T3d")
    assert "DRIVER_SINGLE_SOURCE" in _codes(r["e"], "T3e")
    assert "ROWREF_MISSING" in _codes(r["f"], "T3f")
    assert "T3g" in _accepted_ids(r["g"])
    assert not _codes(r["g"], "T3g")


def test_uncited_claim_never_reaches_answer(sample_env, monkeypatch):
    monkeypatch.setenv("MAX_REFINEMENTS", "2")
    orig_run = Investigator.run

    def run_with_uncited(self, ctx, plan):
        evidence, claims = orig_run(self, ctx, plan)
        claims = list(claims) + [Claim(claim_id="C-uncited", kind="observation",
                                       text=UNCITED, numbers=[], evidence_ids=[])]
        return evidence, claims

    monkeypatch.setattr(Investigator, "run", run_with_uncited)
    res = run(Q1, "t03-int")
    assert res["status"] in ("success", "partial"), res
    out = res["output"]

    assert "price increase" not in out["answer"]["text"].lower()
    assert "price increase" not in json.dumps(out["findings"]).lower()
    assert "price increase" not in json.dumps(out["answer"]["drivers"]).lower()

    assert "NO_CITATION" in json.dumps(out["unverified"])

    records = read_trace(trace_path(sample_env.logs, "t03-int"))
    actions = [r["action"] for r in records]
    assert "reject_claim" in actions
    assert actions.count("request_refinement") == 2
