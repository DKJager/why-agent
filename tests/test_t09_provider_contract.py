"""T09: Meterless provider contract, exercised against a fake local OpenAI-compatible server."""
import inspect
import json
import re
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import provider as provider_mod  # noqa: E402
from agent.provider import MeterlessProvider, normalize_base_url  # noqa: E402
from agent.errors import ProviderUnauthorized  # noqa: E402

TOKEN = "tok-T09-abcdef123456"


class FakeServer:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a, **k):
                pass

            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw.decode("utf-8", "replace")
                path = self.path.split("?")[0]
                req = {
                    "method": self.command,
                    "path": path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
                outer.requests.append(req)
                handler = outer.routes.get((self.command, path))
                if handler is None:
                    status, payload = 404, {"error": "not found"}
                elif callable(handler):
                    status, payload = handler(req)
                else:
                    status, payload = handler
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = _handle
            do_POST = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def by_path(self, method, path):
        return [r for r in self.requests if r["method"] == method and r["path"] == path]


MODELS_OK = (200, {"object": "list", "data": [
    {"id": "meterless/auto", "object": "model"},
    {"id": "meterless/fast", "object": "model"},
    {"id": "meterless/deep", "object": "model"},
]})


def chat_ok(req):
    return 200, {
        "id": "c1", "model": "meterless/auto",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SAMPLE_MODE", raising=False)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost,::1")


@pytest.fixture
def make_server():
    servers = []

    def _make(routes):
        s = FakeServer(routes)
        servers.append(s)
        return s

    yield _make
    for s in servers:
        s.stop()


def _provider(srv, **kw):
    return MeterlessProvider(base_url=srv.url + "/v1", token=TOKEN, **kw)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


MESSAGES = [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "TASK:ping\n{}"},
]


# ---------------------------------------------------------------- headers / body

def test_headers_and_string_content(make_server):
    srv = make_server({
        ("GET", "/v1/models"): MODELS_OK,
        ("GET", "/healthz"): (200, {"status": "ok"}),
        ("POST", "/v1/chat/completions"): chat_ok,
    })
    p = _provider(srv)
    p.list_models()
    p.health()
    res = p.chat(MESSAGES)
    assert isinstance(res.text, str) and res.text == "pong"

    assert srv.requests, "no request reached the fake server"
    for r in srv.requests:
        if r["path"] == "/healthz":
            continue  # informational probe; auth not mandated there
        assert r["headers"].get("authorization") == f"Bearer {TOKEN}", r["path"]
        assert r["headers"].get("meterless-app") == "why-agent", r["path"]

    chats = srv.by_path("POST", "/v1/chat/completions")
    assert len(chats) == 1
    c = chats[0]
    assert c["headers"].get("meterless-scope") == "user"
    body = c["body"]
    assert isinstance(body, dict)
    assert body["messages"], "messages missing"
    for m in body["messages"]:
        assert isinstance(m["content"], str), "content must be a plain string, never a parts array"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_default_model_is_meterless_auto(make_server):
    srv = make_server({("POST", "/v1/chat/completions"): chat_ok})
    _provider(srv).chat(MESSAGES)
    assert srv.requests[0]["body"]["model"] == "meterless/auto"


def test_explicit_model_id_is_used(make_server):
    srv = make_server({("POST", "/v1/chat/completions"): chat_ok})
    _provider(srv, model_id="meterless/deep").chat(MESSAGES)
    assert srv.requests[0]["body"]["model"] == "meterless/deep"


def test_chat_returns_model_and_text(make_server):
    srv = make_server({("POST", "/v1/chat/completions"): chat_ok})
    res = _provider(srv).chat(MESSAGES)
    assert isinstance(res.text, str)
    assert res.model == "meterless/auto"
    assert res.http_status == 200


def test_json_mode_sets_response_format_and_retries_without_on_400(make_server):
    def chat(req):
        if isinstance(req["body"], dict) and "response_format" in req["body"]:
            return 400, {"error": "response_format unsupported"}
        return 200, {"model": "meterless/auto",
                     "choices": [{"message": {"role": "assistant", "content": "{\"a\": 1}"}}],
                     "usage": {}}

    srv = make_server({("POST", "/v1/chat/completions"): chat})
    res = _provider(srv).chat(MESSAGES, json_mode=True)
    chats = srv.by_path("POST", "/v1/chat/completions")
    assert len(chats) == 2
    assert chats[0]["body"]["response_format"] == {"type": "json_object"}
    assert "response_format" not in chats[1]["body"]
    assert json.loads(res.text) == {"a": 1}


@pytest.mark.parametrize("status", [401, 403])
def test_chat_auth_failure_raises_unauthorized_without_retry(make_server, status):
    srv = make_server({("POST", "/v1/chat/completions"): (status, {"error": "nope"})})
    with pytest.raises(ProviderUnauthorized):
        _provider(srv).chat(MESSAGES)
    assert len(srv.by_path("POST", "/v1/chat/completions")) == 1


# ---------------------------------------------------------------- health

@pytest.mark.parametrize("models_resp,expected", [
    ((401, {"error": "unauthorized"}), "unauthorized"),
    ((403, {"error": "forbidden"}), "unauthorized"),
    (MODELS_OK, "online"),
    ((200, {"object": "list", "data": []}), "offline"),
    ((200, {"object": "list", "data": [{"id": "gpt-4o"}, {"id": "llama3"}]}), "offline"),
    ((500, {"error": "boom"}), "offline"),
])
def test_health_decided_by_models(make_server, models_resp, expected):
    srv = make_server({
        ("GET", "/v1/models"): models_resp,
        ("GET", "/healthz"): (200, {"status": "ok"}),
    })
    h = _provider(srv).health()
    assert h.state == expected
    assert isinstance(h.detail, str)
    if expected == "online":
        assert "meterless/auto" in h.models


def test_health_connection_refused_is_offline():
    port = _free_port()
    p = MeterlessProvider(base_url=f"http://127.0.0.1:{port}/v1", token=TOKEN)
    h = p.health()
    assert h.state == "offline"


def test_healthz_alone_never_online(make_server):
    srv = make_server({("GET", "/healthz"): (200, {"status": "ok"})})  # /v1/models -> 404
    h = _provider(srv).health()
    assert h.state != "online"
    assert h.state == "offline"


# ---------------------------------------------------------------- models

def test_list_models_keeps_unknown_meterless_ids(make_server):
    srv = make_server({("GET", "/v1/models"): (200, {"object": "list", "data": [
        {"id": "meterless/auto"},
        {"id": "meterless/foo.bar"},
        {"id": "meterless/fusion"},
        {"id": "gpt-4o"},
        {"id": "other/thing"},
    ]})})
    ids = _provider(srv).list_models()
    assert all(isinstance(i, str) for i in ids)
    assert set(ids) == {"meterless/auto", "meterless/foo.bar", "meterless/fusion"}


# ---------------------------------------------------------------- timeout

def test_default_timeout_is_180():
    default = inspect.signature(MeterlessProvider.__init__).parameters["timeout_s"].default
    assert default == 180
    p = MeterlessProvider(base_url="http://127.0.0.1:1/v1", token=TOKEN)
    for attr in ("timeout_s", "timeout", "_timeout_s", "_timeout"):
        if hasattr(p, attr):
            assert getattr(p, attr) == 180
            break


# ---------------------------------------------------------------- autodetect

class _Probe(str):
    """A probe result that works as a plain state string or as a health-like object."""

    @property
    def state(self):
        return str(self)

    @property
    def models(self):
        return ["meterless/auto"] if str(self) == "online" else []

    @property
    def detail(self):
        return "fake"

    def __getitem__(self, k):
        if k == "state":
            return str(self)
        return str.__getitem__(self, k)


def _host_port(args, kwargs):
    vals = list(args) + list(kwargs.values())
    if len(vals) >= 2 and isinstance(vals[1], int):
        return str(vals[0]).strip("[]"), vals[1]
    for v in vals:
        if isinstance(v, str):
            m = re.match(r"^(?:\w+://)?(\[[^\]]+\]|[^/:\[]+)(?::(\d+))?", v)
            if m and m.group(2):
                return m.group(1).strip("[]"), int(m.group(2))
    raise AssertionError(f"cannot interpret probe args {args} {kwargs}")


PORTS = [8088, 8090, 8787, 11434]
HOSTS = ["127.0.0.1", "localhost", "::1"]
ORDER = [(h, p) for p in PORTS for h in HOSTS]


def _reset_autodetect_cache(monkeypatch):
    for name in dir(provider_mod):
        low = name.lower()
        if ("cache" in low or "detected" in low) and not callable(getattr(provider_mod, name)):
            monkeypatch.setattr(provider_mod, name, None, raising=False)


def _run_autodetect(responses):
    calls = []

    def probe(*args, **kwargs):
        hp = _host_port(args, kwargs)
        calls.append(hp)
        return _Probe(responses.get(hp, "offline"))

    sig = inspect.signature(provider_mod.autodetect)
    params = list(sig.parameters.values())
    assert params, "autodetect must accept an injectable probe function"
    named = [p for p in params if "probe" in p.name.lower()]
    if named:
        result = provider_mod.autodetect(**{named[0].name: probe})
    else:
        result = provider_mod.autodetect(probe)
    return result, calls


def test_autodetect_port_major_order_and_first_online(monkeypatch):
    _reset_autodetect_cache(monkeypatch)
    hit = ("localhost", 8090)
    result, calls = _run_autodetect({hit: "online", ("::1", 8787): "online"})
    assert result is not None
    assert normalize_base_url(result) == "http://localhost:8090/v1"
    idx = ORDER.index(hit)
    assert calls == ORDER[: idx + 1]


def test_autodetect_unauthorized_counts_as_found(monkeypatch):
    _reset_autodetect_cache(monkeypatch)
    hit = ("::1", 8787)
    result, calls = _run_autodetect({hit: "unauthorized"})
    assert result is not None
    assert normalize_base_url(result) == "http://[::1]:8787/v1"
    assert calls == ORDER[: ORDER.index(hit) + 1]


def test_autodetect_none_when_all_offline(monkeypatch):
    _reset_autodetect_cache(monkeypatch)
    result, calls = _run_autodetect({})
    assert result is None
    assert calls == ORDER
