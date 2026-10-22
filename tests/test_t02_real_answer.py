# build-id: oa-d6bb9d14
import json

from _helpers import Q1, drivers, out_of, run, truth


def _find_wow(output):
    for claim in output["findings"]:
        if claim.get("kind") == "delta":
            for n in claim.get("numbers", []):
                if "wow" in n["name"].lower():
                    return n["value"]
    for ev in output["evidence"]:
        for k, v in (ev.get("values") or {}).items():
            if "wow" in k.lower():
                return v
    return None


def test_q1_planted_cause_found(sample_env):
    res = run(Q1, "t02-q1")
    out = out_of(res)
    t = truth(sample_env.data_dir, "EU", 41)

    # sanity on the data itself (independent of the agent)
    assert t["wow_pct"] <= -20.0

    drs = drivers(res)
    assert drs, "no driver reported"
    top = drs[0]
    assert "NordExpress" in top["text"]
    # the name must be present in the evidence rows (found, not hardcoded)
    assert "NordExpress" in json.dumps(out["evidence"])

    assert top["explained_share"] >= 0.6
    assert t["top_segment"][0] == "NordExpress"
    assert abs(top["explained_share"] - min(1.0, t["top_share"])) <= 0.01  # corrected 22 Oct per committee #1631: shares above 100% are capped and labelled

    wow = _find_wow(out)
    assert wow is not None, "no WoW number on the delta claim"
    cands = [t["wow_pct"], t["wow_pct"] / 100.0]
    assert any(abs(wow - c) <= (0.1 if c is cands[0] else 0.001) for c in cands), (wow, cands)

    deltas = [c for c in out["findings"] if c.get("kind") == "delta"]
    assert deltas

    sources = {e["source"] for e in out["evidence"]}
    assert {"orders", "shipments", "tickets"} <= sources

    ruled = [c for c in out["findings"] if c.get("kind") == "ruled_out"]
    ruled += out["answer"].get("ruled_out", [])
    assert any("demand" in json.dumps(r).lower() for r in ruled), "demand not ruled out"

    for d in drs:
        assert "home" not in d["text"].lower().split(), d
        assert "category:home" not in json.dumps(d).lower()
