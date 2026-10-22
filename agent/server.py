# build-id: oa-23ea9159
"""HTTP interface (DESIGN section 12.2).

``handle_run(payload) -> (status, body)`` is a pure function; the HTTP
handler is a thin wrapper around it.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Tuple

from .config import Config
from .errors import (
    InvalidRequest,
    WhyAgentError,
    error_body,
    http_status_for,
    wrap_unexpected,
)
from .orchestrator import run_question
from .trace import Redactor, TraceWriter

BIND_HOST = "0.0.0.0"
MAX_BODY_BYTES = 64 * 1024


def _new_run_id() -> str:
    return "run-" + time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:4]


def _safe_config() -> Optional[Config]:
    try:
        return Config.from_env()
    except WhyAgentError:
        return None


def _request_error(exc: BaseException, run_id: Optional[str] = None) -> Tuple[int, dict]:
    """Structured error for problems found before a run starts, with a minimal trace."""
    started = time.monotonic()
    err = wrap_unexpected(exc)
    rid = run_id or _new_run_id()
    cfg = _safe_config() or Config()
    redactor = Redactor.from_env(extra=cfg.secret_values())
    trace_id: Optional[str] = None
    log_file: Optional[str] = None
    try:
        tw = TraceWriter(rid, cfg.log_dir, redactor=redactor, level=cfg.log_level, echo=False)
        try:
            tw.emit("Orchestrator", "start_run", "", "request rejected before the run started")
            tw.emit("Orchestrator", "error", "", redactor.redact(f"{err.type}: {err.message}"),
                    status="error", details={"error_type": err.type})
            trace_id, log_file = tw.trace_id, tw.path
        finally:
            tw.close()
    except Exception:
        pass
    body = error_body(err, run_id=rid, trace_id=trace_id, log_file=log_file,
                      execution_time_seconds=time.monotonic() - started, redact=redactor.redact)
    return http_status_for(err.type), body


def handle_run(payload: Any) -> Tuple[int, dict]:
    """Run one question described by a request payload. Never raises."""
    try:
        if isinstance(payload, (bytes, bytearray)):
            if len(payload) > MAX_BODY_BYTES:
                return _request_error(InvalidRequest("Request body is too large (max 64 KB)."))
            try:
                payload = payload.decode("utf-8")
            except UnicodeDecodeError:
                return _request_error(InvalidRequest("Request body is not valid UTF-8."))
        if isinstance(payload, str):
            if len(payload.encode("utf-8")) > MAX_BODY_BYTES:
                return _request_error(InvalidRequest("Request body is too large (max 64 KB)."))
            try:
                payload = json.loads(payload)
            except ValueError:
                return _request_error(InvalidRequest("Request body is not valid JSON."))
        if not isinstance(payload, dict):
            return _request_error(InvalidRequest("Request body must be a JSON object."))

        inp = payload.get("input")
        query = inp.get("query") if isinstance(inp, dict) else None
        run_id = payload.get("run_id")
        options = payload.get("options")
        body = run_question(query, run_id=run_id, options=options)
        if body.get("status") == "error":
            return http_status_for(body.get("error", {}).get("type", "InternalError")), body
        return 200, body
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return _request_error(e)


def _sample_mode() -> bool:
    cfg = _safe_config()
    return bool(cfg.sample_mode) if cfg else False


class _Handler(BaseHTTPRequestHandler):
    server_version = "WhyAgent/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # silence default access log
        return

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _route_error(self, status: int, message: str) -> None:
        etype = "InvalidRequest"
        body = {
            "run_id": None, "status": "error",
            "error": {"type": etype, "message": message, "recoverable": True,
                      "details": {"http_status": status}},
            "trace_id": None, "log_file": None, "execution_time_seconds": 0.0,
        }
        self._send(status, body)

    def _path(self) -> str:
        return self.path.split("?", 1)[0].split("#", 1)[0]

    def do_GET(self) -> None:
        path = self._path()
        if path == "/healthz":
            self._send(200, {"status": "ok", "sample_mode": _sample_mode()})
        elif path == "/run":
            self._route_error(405, "Use POST for /run.")
        else:
            self._route_error(404, "Not found.")

    def do_POST(self) -> None:
        path = self._path()
        if path == "/healthz":
            return self._route_error(405, "Use GET for /healthz.")
        if path != "/run":
            return self._route_error(404, "Not found.")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            status, body = _request_error(InvalidRequest("Invalid Content-Length header."))
            self.close_connection = True
            return self._send(status, body)
        if length > MAX_BODY_BYTES:
            status, body = _request_error(InvalidRequest("Request body is too large (max 64 KB)."))
            self.close_connection = True
            return self._send(status, body)
        raw = self.rfile.read(length) if length else b""
        status, body = handle_run(raw)
        self._send(status, body)

    def _method_not_allowed(self) -> None:
        if self._path() in ("/run", "/healthz"):
            self._route_error(405, "Method not allowed.")
        else:
            self._route_error(404, "Not found.")

    do_PUT = do_DELETE = do_PATCH = _method_not_allowed


def make_server(port: Optional[int] = None, host: str = BIND_HOST) -> ThreadingHTTPServer:
    """Server factory; binds ``host`` (default 0.0.0.0). Port 0 picks a free port."""
    if port is None:
        cfg = _safe_config()
        port = cfg.port if cfg else 8000
    ThreadingHTTPServer.daemon_threads = True
    return ThreadingHTTPServer((host, int(port)), _Handler)


def serve(port: Optional[int] = None) -> None:
    server = make_server(port)
    addr, actual = server.server_address[:2]
    print(f"[INFO] Why-Agent listening on {addr}:{actual} (POST /run, GET /healthz)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["handle_run", "make_server", "serve", "BIND_HOST", "MAX_BODY_BYTES"]
