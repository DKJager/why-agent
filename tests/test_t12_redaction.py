# build-id: oa-0b4bd7db
"""T12: secrets never reach trace, stdout, stderr or result JSON."""
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

TOKEN = "sk-test-SECRET123456"
ZKEY = "zk-SECRET"
NVKEY = "nvapi-SECRETNVIDIA123456"
SECRETS = [TOKEN, ZKEY, NVKEY, "SECRET123456"]
Q1 = "Why did EU revenue drop in week 41?"


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
                    body = None
                req = {"method": self.command, "path": self.path.split("?")[0],
                       "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                outer.requests.append(req)
                handler = outer.routes.get((self.command, req["path"]))
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
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _echo401(req):
    # A hostile upstream that echoes our credentials back in its error body.
    return 401, {"error": f"unauthorized: {req['headers'].get('authorization')} "
                          f"token={TOKEN} key={ZKEY}"}


MODELS_OK = (200, {"object": "list", "data": [{"id": "meterless/auto"}, {"id": "meterless/fast"}]})
CHAT_OK = (200, {"model": "meterless/auto",
                 "choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}})


def _clean_env():
    drop = ("METERLESS", "MODEL_", "ZETARIS", "NVIDIA", "SAMPLE_MODE", "HMEM", "LOG_", "DATA_DIR",
            "ENGINE", "RUN_TIMEOUT", "MAX_", "PROVIDER_TIMEOUT", "SOURCE_TIMEOUT", "PORT",
            "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_cli(tmp_path, env, json_mode):
    out = tmp_path / "result.json"
    cmd = [sys.executable, "run.py", Q1, "--run-id", "t12-run", "--out", str(out)]
    if json_mode:
        cmd.append("--json")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    return proc, out


def _collect_texts(tmp_path, proc, out):
    texts = {"stdout": proc.stdout, "stderr": proc.stderr}
    for p in tmp_path.rglob("*"):
        if p.is_file():
            texts[str(p)] = p.read_text(encoding="utf-8", errors="ignore")
    return texts


def _assert_clean(texts):
    for name, text in texts.items():
        for s in SECRETS:
            assert s not in text, f"secret {s!r} leaked in {name}"
        for m in re.finditer(r"Bearer\s+(\S+)", text):
            assert m.group(1).startswith("***redacted***"), f"unmasked bearer in {name}: {m.group(0)}"
        assert not re.search(r"sk-[A-Za-z0-9_-]{6,}", text), f"sk- style secret in {name}"


def _live_env(tmp_path, meterless, zetaris):
    env = _clean_env()
    env.update({
        "SAMPLE_MODE": "false",
        "METERLESS_BASE_URL": meterless.url + "/v1",
        "METERLESS_API_KEY": TOKEN,
        "ZETARIS_REST_URL": zetaris.url,
        "ZETARIS_ORG_ID": "1",
        "ZETARIS_API_KEY": ZKEY,
        "NVIDIA_API_KEY": NVKEY,
        "METERLESS_AUTODETECT": "false",
        "ENGINE": "stub",
        "LOG_DIR": str(tmp_path / "logs"),
        "HMEM_PATH": str(tmp_path / "hmem" / "store.json"),
        "SOURCE_TIMEOUT_S": "5",
        "PROVIDER_TIMEOUT_S": "10",
        "RUN_TIMEOUT_S": "60",
    })
    return env


@pytest.fixture
def servers():
    made = []

    def _make(routes):
        s = FakeServer(routes)
        made.append(s)
        return s

    yield _make
    for s in made:
        s.stop()


@pytest.mark.parametrize("json_mode", [True, False])
@pytest.mark.parametrize("scenario", ["provider_rejects", "source_rejects"])
def test_live_flow_never_leaks_secrets(tmp_path, servers, scenario, json_mode):
    if scenario == "provider_rejects":
        meterless = servers({("GET", "/v1/models"): _echo401,
                             ("POST", "/v1/chat/completions"): _echo401,
                             ("GET", "/healthz"): (200, {"status": "ok"})})
    else:
        meterless = servers({("GET", "/v1/models"): MODELS_OK,
                             ("POST", "/v1/chat/completions"): CHAT_OK,
                             ("GET", "/healthz"): (200, {"status": "ok"})})
    zetaris = servers({("POST", "/sql-editor/sqls/run"): _echo401})

    env = _live_env(tmp_path, meterless, zetaris)
    proc, out = _run_cli(tmp_path, env, json_mode)

    # the credentials were genuinely in play (so the grep below is meaningful)
    sent = [r["headers"].get("authorization", "") for r in meterless.requests + zetaris.requests]
    assert any(TOKEN in a or ZKEY in a for a in sent), "fake servers never saw a credential"

    assert proc.returncode in (0, 1)
    if json_mode:
        body = json.loads(proc.stdout)
        assert isinstance(body, dict)
    else:
        body = json.loads(out.read_text(encoding="utf-8"))
    assert body["status"] == "error"
    assert out.exists()

    trace = tmp_path / "logs" / "trace-t12-run.jsonl"
    assert trace.exists()
    _assert_clean(_collect_texts(tmp_path, proc, out))


@pytest.mark.parametrize("json_mode", [True, False])
def test_sample_mode_with_keys_set_never_leaks(tmp_path, json_mode):
    env = _clean_env()
    env.update({
        "SAMPLE_MODE": "true",
        "METERLESS_API_KEY": TOKEN,
        "MODEL_API_KEY": TOKEN,
        "ZETARIS_API_KEY": ZKEY,
        "NVIDIA_API_KEY": NVKEY,
        "LOG_DIR": str(tmp_path / "logs"),
        "HMEM_PATH": str(tmp_path / "hmem" / "store.json"),
        "DATA_DIR": str(ROOT / "data" / "sample"),
    })
    proc, out = _run_cli(tmp_path, env, json_mode)
    assert proc.returncode == 0, proc.stderr
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["status"] in ("success", "partial")
    assert (tmp_path / "logs" / "trace-t12-run.jsonl").exists()
    _assert_clean(_collect_texts(tmp_path, proc, out))
