"""T16: explained-share cap (SHARE_OVER_100) and region-code case."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agents.critic import Critic
from agent.models import Claim, Evidence

ROOT = Path(__file__).resolve().parent.parent
_RE_EXPL = re.compile(r"explain\w*[^%]{0,60}?(\d[\d,]*(?:\.\d+)?)\s*%", re.I)


class _Trace:
    def emit(self, *a, **k):
        return None


def _fake_ctx():
    ctx = SimpleNamespace()
    ctx.counters = {}
    ctx.trace = _Trace()
    ctx.emit = lambda *a, **k: None
    ctx.engine = SimpleNamespace(get=lambda mid: None)

    def _no_query(*a, **k):  # pragma: no cover - must not be needed for a rejected claim
        raise AssertionError("source must not be queried for a rejected claim")

    ctx.query_source = _no_query
    return ctx


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="E1", source="orders", description="x",
        query={}, query_text="", values={}, rows=[], row_refs=[], row_ref_total=0,
        retrieved_at="2026-10-18T00:00:00Z",
    )


def test_critic_rejects_share_over_100():
    claim = Claim(claim_id="C1", kind="driver", text="A driver moved revenue [E1].",
                  numbers=[], evidence_ids=["E1"], explained_share=1.06, confidence=0.9)
    review = Critic().review(_fake_ctx(), [claim], [_evidence()])
    assert claim not in review.accepted
    assert any(r.claim_id == "C1" and r.code == "SHARE_OVER_100" for r in review.rejected)


def test_critic_rejects_negative_share():
    claim = Claim(claim_id="C1", kind="driver", text="A driver moved revenue [E1].",
                  numbers=[], evidence_ids=["E1"], explained_share=-0.1, confidence=0.9)
    review = Critic().review(_fake_ctx(), [claim], [_evidence()])
    assert any(r.code == "SHARE_OVER_100" for r in review.rejected)


@pytest.fixture
def sample_run_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMPLE_MODE", "true")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("HMEM_PATH", str(tmp_path / "hmem.json"))
    monkeypatch.setenv("DATA_DIR", str(ROOT / "data" / "sample"))
    for k in ("METERLESS_API_KEY", "MODEL_API_KEY", "ZETARIS_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def _all_text(result: dict) -> str:
    out = result["output"]
    ans = out["answer"]
    parts = [ans.get("headline", ""), ans.get("text", ""), out.get("summary", "")]
    parts += [d.get("text", "") for d in ans.get("drivers", [])]
    parts += [d.get("text", "") for d in ans.get("ruled_out", [])]
    parts += [json.dumps(f) for f in out.get("findings", [])]
    return "\n".join(str(p) for p in parts)


def test_eu_run_has_no_share_above_100(sample_run_env):
    from agent.orchestrator import run_question
    result = run_question("Why did EU revenue drop in week 41?", run_id="t16-eu")
    assert result["status"] in ("success", "partial")
    text = _all_text(result)
    for m in _RE_EXPL.finditer(text):
        assert float(m.group(1).replace(",", "")) <= 100.0, m.group(0)
    for d in result["output"]["answer"]["drivers"]:
        if d.get("explained_share") is not None:
            assert 0.0 <= d["explained_share"] <= 1.0


def test_us_run_keeps_region_code_case(sample_run_env):
    from agent.orchestrator import run_question
    result = run_question("Why did US revenue drop in week 41?", run_id="t16-us")
    assert result["status"] in ("success", "partial")
    ans = result["output"]["answer"]
    for text in (ans["text"], ans["headline"]):
        assert "US" in text
        assert " us " not in text
