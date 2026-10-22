import json

import pytest

from agent.agents.planner import Planner
from agent.server import handle_run

from _helpers import Q1, free_port, read_trace, trace_path


def _assert_shape(body):
    assert body["status"] == "error"
    for k in ("run_id", "error", "trace_id", "execution_time_seconds"):
        assert k in body, k
    for k in ("type", "message", "recoverable"):
        assert k in body["error"], k
    assert isinstance(body["error"]["recoverable"], bool)
    assert isinstance(body["error"]["message"], str)


def _assert_terminal_error_trace(logs, body, required=True):
    p = trace_path(logs, body["run_id"])
    if not p.exists():
        assert not required, "trace file missing"
        return
    records = read_trace(p)
    assert records
    assert records[-1]["action"] == "error"
    assert any(r["action"] == "error" for r in records)


def _payload(query=Q1, run_id=None, options=None):
    p = {"input": {"query": query}}
    if run_id:
        p["run_id"] = run_id
    if options is not None:
        p["options"] = options
    return p


@pytest.mark.parametrize("bad", ["{not valid json", None, [], 42])
def test_invalid_payload_400(sample_env, bad):
    status, body = handle_run(bad)
    assert status == 400
    _assert_shape(body)
    assert body["error"]["type"] == "InvalidRequest"


def test_empty_query_400(sample_env):
    status, body = handle_run(_payload(query=""))
    assert status == 400
    _assert_shape(body)
    assert body["error"]["type"] == "InvalidRequest"
    assert body["error"]["recoverable"] is True
    _assert_terminal_error_trace(sample_env.logs, body, required=False)


def test_bad_run_id_400(sample_env):
    status, body = handle_run(_payload(run_id="bad id/../x"))
    assert status == 400
    _assert_shape(body)
    assert body["error"]["type"] == "InvalidRequest"


def test_unparseable_question_422(sample_env):
    status, body = handle_run(_payload(query="hello", run_id="t08-hello"))
    assert status == 422
    _assert_shape(body)
    assert body["error"]["type"] == "QuestionNotSupported"
    _assert_terminal_error_trace(sample_env.logs, body)


def test_live_mode_without_credentials_is_config_error(sample_env, monkeypatch):
    monkeypatch.setenv("SAMPLE_MODE", "false")
    status, body = handle_run(_payload(run_id="t08-noconf"))
    assert status == 500
    _assert_shape(body)
    assert body["error"]["type"] == "ConfigError"
    _assert_terminal_error_trace(sample_env.logs, body)


def test_rt_local_token_is_unauthorized_not_unavailable(sample_env, monkeypatch):
    dead = free_port()
    monkeypatch.setenv("SAMPLE_MODE", "false")
    monkeypatch.setenv("METERLESS_API_KEY", "rt-local")
    monkeypatch.setenv("METERLESS_BASE_URL", f"http://127.0.0.1:{dead}/v1")
    monkeypatch.setenv("ZETARIS_BASE_URL", f"http://127.0.0.1:{dead}")
    monkeypatch.setenv("ZETARIS_API_KEY", "zk-test-key")
    status, body = handle_run(_payload(run_id="t08-rtlocal"))
    _assert_shape(body)
    assert body["error"]["type"] == "ProviderUnauthorized"
    assert body["error"]["type"] != "ProviderUnavailable"
    assert status == 502
    _assert_terminal_error_trace(sample_env.logs, body)


def test_engine_meterless_not_configured(sample_env, monkeypatch):
    monkeypatch.setenv("ENGINE", "meterless")
    status, body = handle_run(_payload(run_id="t08-engine"))
    assert status == 503
    _assert_shape(body)
    assert body["error"]["type"] == "EngineNotConfigured"
    _assert_terminal_error_trace(sample_env.logs, body)


def test_no_rows_is_no_data(sample_env):
    status, body = handle_run(_payload(query="Why did EU revenue drop in week 12?",
                                       run_id="t08-nodata"))
    _assert_shape(body)
    assert body["error"]["type"] == "NoData"
    _assert_terminal_error_trace(sample_env.logs, body)


def test_forced_exception_is_internal_error(sample_env, monkeypatch):
    def boom(self, ctx, question):
        raise RuntimeError("boom")

    monkeypatch.setattr(Planner, "run", boom)
    status, body = handle_run(_payload(run_id="t08-internal"))
    assert status == 500
    _assert_shape(body)
    assert body["error"]["type"] == "InternalError"
    assert body["error"]["recoverable"] is False
    dumped = json.dumps(body)
    assert "Traceback" not in dumped
    assert 'File "' not in dumped
    _assert_terminal_error_trace(sample_env.logs, body)


def test_fallback_to_sample_degrades(sample_env, monkeypatch):
    dead = free_port()
    monkeypatch.setenv("SAMPLE_MODE", "false")
    monkeypatch.setenv("METERLESS_API_KEY", "mk-real-looking-token-123")
    monkeypatch.setenv("METERLESS_BASE_URL", f"http://127.0.0.1:{dead}/v1")
    monkeypatch.setenv("ZETARIS_BASE_URL", f"http://127.0.0.1:{dead}")
    monkeypatch.setenv("ZETARIS_API_KEY", "zk-test-key")
    status, body = handle_run(_payload(run_id="t08-fallback",
                                       options={"fallback_to_sample": True}))
    assert status == 200, body
    assert body["status"] in ("success", "partial")
    assert body["degraded"] is True
    assert body["warnings"]
    records = read_trace(trace_path(sample_env.logs, "t08-fallback"))
    assert any(r["action"] == "fallback" for r in records)
