"""T11: run bounds (deadline, step cap, per-call provider timeout)."""
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.orchestrator import run_question  # noqa: E402
from agent.sources import CsvSource  # noqa: E402
from agent.provider import SampleProvider  # noqa: E402

Q1 = "Why did EU revenue drop in week 41?"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMPLE_MODE", "true")
    monkeypatch.setenv("DATA_DIR", str(ROOT / "data" / "sample"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("HMEM_PATH", str(tmp_path / "hmem" / "store.json"))
    monkeypatch.setenv("ENGINE", "stub")
    for k in ("RUN_TIMEOUT_S", "MAX_STEPS", "MAX_MODEL_CALLS", "PROVIDER_TIMEOUT_S",
              "MAX_REFINEMENTS", "METERLESS_API_KEY", "MODEL_API_KEY", "ZETARIS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _guarded(fn, limit=60):
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:  # pragma: no cover
            box["exc"] = e

    t = threading.Thread(target=target, daemon=True)
    started = time.monotonic()
    t.start()
    t.join(limit)
    assert not t.is_alive(), "run_question hung past the guard"
    assert "exc" not in box, f"run_question raised: {box.get('exc')!r}"
    return box["result"], time.monotonic() - started


def test_slow_source_hits_run_timeout(env):
    env.setenv("RUN_TIMEOUT_S", "1")
    original = CsvSource.query

    def slow(self, spec, **kw):
        time.sleep(1.5)
        return original(self, spec, **kw)

    env.setattr(CsvSource, "query", slow)
    result, elapsed = _guarded(lambda: run_question(Q1, run_id="t11-timeout"))
    assert result["status"] in ("partial", "error")
    if result["status"] == "error":
        assert result["error"]["type"] == "RunTimeout"
    assert elapsed < 10, f"run took {elapsed:.1f}s despite RUN_TIMEOUT_S=1"


def test_max_steps_cap(env):
    env.setenv("MAX_STEPS", "5")
    result, _ = _guarded(lambda: run_question(Q1, run_id="t11-steps"))
    assert result["status"] in ("partial", "error")
    if result["status"] == "error":
        assert result["error"]["type"] == "IterationLimit"
    else:
        assert result["warnings"], "a partial result must carry a warning"


def _record_timeouts(env):
    seen = []
    original = SampleProvider.chat

    def wrapper(self, messages, *a, **kw):
        seen.append(kw.get("timeout_s"))
        return original(self, messages, *a, **kw)

    env.setattr(SampleProvider, "chat", wrapper)
    return seen


def test_provider_timeout_capped_by_remaining_budget(env):
    env.setenv("RUN_TIMEOUT_S", "60")
    seen = _record_timeouts(env)
    result, _ = _guarded(lambda: run_question(Q1, run_id="t11-cap"))
    assert result["status"] in ("success", "partial")
    assert seen, "provider.chat was never called"
    assert all(t is not None for t in seen), "each call must pass timeout_s"
    assert all(0 < t <= 60 + 1e-6 for t in seen), seen
    assert seen[0] >= 50, "first call should get nearly the whole remaining budget"


def test_provider_timeout_uses_config_when_smaller(env):
    env.setenv("RUN_TIMEOUT_S", "300")
    env.setenv("PROVIDER_TIMEOUT_S", "10")
    seen = _record_timeouts(env)
    result, _ = _guarded(lambda: run_question(Q1, run_id="t11-cfg"))
    assert result["status"] in ("success", "partial")
    assert seen and all(t == pytest.approx(10) for t in seen), seen


def test_provider_timeout_default_is_180(env):
    env.setenv("RUN_TIMEOUT_S", "300")
    seen = _record_timeouts(env)
    result, _ = _guarded(lambda: run_question(Q1, run_id="t11-default"))
    assert result["status"] in ("success", "partial")
    assert seen and all(t is not None and t <= 180 + 1e-6 for t in seen), seen
    assert seen[0] == pytest.approx(180, abs=1)
