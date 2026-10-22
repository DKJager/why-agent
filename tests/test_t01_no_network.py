import socket

import requests
import requests.sessions

from _helpers import SAMPLE_QUESTIONS, run


def _install_guards(monkeypatch):
    calls = []

    def make(name):
        def f(*a, **k):
            calls.append(name)
            raise AssertionError("network access attempted via " + name)
        return f

    monkeypatch.setattr(socket.socket, "connect", make("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", make("socket.create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", make("socket.getaddrinfo"))
    monkeypatch.setattr(requests.sessions.Session, "request", make("Session.request"))
    return calls


def _run_all(prefix):
    results = []
    for rid, q in SAMPLE_QUESTIONS:
        results.append(run(q, f"{prefix}-{rid}"))
    return results


def _check(results):
    for res in results:
        assert res["status"] in ("success", "partial"), res
        mode = res["mode"]
        assert mode["source"] == "csv"
        assert mode["provider"] == "sample"
        assert mode["engine"] == "hmem-stub"
        assert mode["sample_mode"] is True


def test_sample_mode_makes_no_network_calls(sample_env, monkeypatch):
    calls = _install_guards(monkeypatch)
    results = _run_all("t01a")
    _check(results)
    assert calls == []


def test_sample_mode_ignores_garbage_live_settings(sample_env, monkeypatch, tmp_path):
    monkeypatch.setenv("ZETARIS_BASE_URL", "https://zetaris.invalid")
    monkeypatch.setenv("ZETARIS_API_KEY", "garbage-zk-key")
    monkeypatch.setenv("METERLESS_BASE_URL", "http://203.0.113.9:8088/v1")
    monkeypatch.setenv("METERLESS_API_KEY", "garbage-mk-key")
    monkeypatch.setenv("MODEL_BASE_URL", "http://203.0.113.10:8088")
    monkeypatch.setenv("MODEL_API_KEY", "garbage-model-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "garbage-nv-key")
    monkeypatch.setenv("HMEM_PATH", str(tmp_path / "other" / "store.json"))
    calls = _install_guards(monkeypatch)
    results = _run_all("t01b")
    _check(results)
    assert calls == []
