"""Model provider: Meterless Runtime client, URL helpers, SampleProvider (DESIGN section 7)."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Protocol
from urllib.parse import parse_qs, urlsplit

import requests

from . import rules
from .errors import (
    ConfigError,
    InvalidRequest,
    ProviderUnauthorized,
    ProviderUnavailable,
    UpstreamTimeout,
)
from .models import ParsedQuestion

Message = dict

DEFAULT_BASE_URL = "http://127.0.0.1:8088/v1"
PLACEHOLDER_TOKENS = {"", "rt-local", "your-key-here", "changeme"}
APP_ID = "why-agent"
_STRIP_SUFFIXES = ("/healthz", "/v1/models", "/v1/chat/completions", "/v1")
_RE_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
AUTODETECT_PORTS = (8088, 8090, 8787, 11434)
AUTODETECT_HOSTS = ("127.0.0.1", "localhost", "[::1]")


@dataclass
class ChatResult:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    http_status: Optional[int] = None


@dataclass
class ProviderHealth:
    state: Literal["online", "unauthorized", "offline"]
    base_url: str
    models: list = field(default_factory=list)
    detail: str = ""


class ModelProvider(Protocol):
    name: str

    def chat(self, messages: list, *, model: Optional[str] = None, temperature: float = 0.0,
             max_tokens: int = 1024, json_mode: bool = False,
             timeout_s: Optional[float] = None) -> ChatResult: ...

    def list_models(self) -> list: ...

    def health(self) -> ProviderHealth: ...


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------
def _normalize(raw: Optional[str], extra_suffixes: tuple = ()) -> str:
    s = (raw or "").strip()
    if not s:
        return DEFAULT_BASE_URL
    if not _RE_SCHEME.match(s):
        s = "http://" + s
    try:
        parts = urlsplit(s)
        netloc = parts.netloc
        scheme = parts.scheme
        path = parts.path
    except ValueError:
        raise ConfigError("Invalid base URL") from None
    if not netloc:
        raise ConfigError("Invalid base URL")
    path = path.rstrip("/")
    suffixes = tuple(extra_suffixes) + _STRIP_SUFFIXES
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if path.endswith(suf):
                path = path[: -len(suf)].rstrip("/")
                changed = True
                break
    return f"{scheme}://{netloc}{path}/v1"


def normalize_base_url(raw: Optional[str]) -> str:
    return _normalize(raw)


def parse_connect_link(raw: str) -> tuple:
    """Return (base_url, key). NEVER log the input or the key."""
    if not isinstance(raw, str):
        return (normalize_base_url(None), None)
    s = raw.strip()
    key: Optional[str] = None
    try:
        parts = urlsplit(s if _RE_SCHEME.match(s) else "http://" + s)
        for chunk in (parts.fragment, parts.query):
            vals = parse_qs(chunk).get("k")
            if vals and vals[0]:
                key = vals[0]
                break
    except ValueError:
        return (normalize_base_url(s), None)
    if key is None:
        return (normalize_base_url(s), None)
    return (_normalize(s, extra_suffixes=("/connect",)), key)


def healthz_url(base_url: str) -> str:
    b = (base_url or "").rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b.rstrip("/") + "/healthz"


def validate_token(token: Optional[str]) -> str:
    t = (token or "").strip()
    if t.lower() in PLACEHOLDER_TOKENS:
        raise ProviderUnauthorized("placeholder")
    return t


# ---------------------------------------------------------------------------
# MeterlessProvider
# ---------------------------------------------------------------------------
def _is_meterless_id(i: Any) -> bool:
    return isinstance(i, str) and i.startswith("meterless/")


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S | re.I)
    return m.group(1) if m else text


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                out.append(p["text"])
        return "".join(out)
    return "" if content is None else str(content)


class MeterlessProvider:
    name = "meterless"

    def __init__(self, base_url: Optional[str], token: Optional[str], app_id: str = APP_ID,
                 model_id: str = "meterless/auto", timeout_s: float = 180.0) -> None:
        self.base_url = normalize_base_url(base_url)
        self._token = validate_token(token)
        self.app_id = app_id or APP_ID
        self.model_id = model_id or "meterless/auto"
        self.timeout_s = float(timeout_s)

    def __repr__(self) -> str:
        return f"MeterlessProvider(base_url={self.base_url!r}, token='***', model_id={self.model_id!r})"

    # ---- transport ----------------------------------------------------
    def _headers(self, scope: bool = False) -> dict:
        h = {
            "Authorization": f"Bearer {self._token}",
            "meterless-app": self.app_id,
        }
        if scope:
            h["meterless-scope"] = "user"
        return h

    def _send(self, method: str, url: str, *, body: Optional[dict] = None, scope: bool = False,
              timeout: float) -> Any:
        headers = self._headers(scope)
        if body is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(2):
            try:
                if method == "GET":
                    resp = requests.get(url, headers=headers, timeout=timeout)
                else:
                    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            except requests.Timeout:
                raise UpstreamTimeout("Meterless Runtime request timed out") from None
            except requests.ConnectionError:
                if attempt == 0:
                    continue
                raise ProviderUnavailable("Meterless Runtime is unreachable") from None
            except requests.RequestException:
                raise ProviderUnavailable("Meterless Runtime request failed") from None
            if resp.status_code in (502, 503) and attempt == 0:
                continue
            return resp
        raise ProviderUnavailable("Meterless Runtime is unavailable")

    @staticmethod
    def _check_auth(resp: Any) -> None:
        if resp.status_code in (401, 403):
            raise ProviderUnauthorized(
                f"Meterless Runtime rejected the token (HTTP {resp.status_code}).",
                details={"http_status": resp.status_code})

    # ---- API ----------------------------------------------------------
    def chat(self, messages: list, *, model: Optional[str] = None, temperature: float = 0.0,
             max_tokens: int = 1024, json_mode: bool = False,
             timeout_s: Optional[float] = None) -> ChatResult:
        timeout = self.timeout_s if timeout_s is None else max(0.001, float(timeout_s))
        used_model = model or self.model_id
        msgs = [{"role": str(m["role"]), "content": _content_text(m.get("content"))} for m in messages]
        body: dict = {
            "model": used_model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        url = self.base_url + "/chat/completions"
        resp = self._send("POST", url, body=body, scope=True, timeout=timeout)
        self._check_auth(resp)
        if resp.status_code == 400 and json_mode:
            body.pop("response_format", None)
            resp = self._send("POST", url, body=body, scope=True, timeout=timeout)
            self._check_auth(resp)
        if resp.status_code != 200:
            raise ProviderUnavailable(
                f"Meterless Runtime returned HTTP {resp.status_code}.",
                details={"http_status": resp.status_code})
        try:
            obj = resp.json()
            choice = obj["choices"][0]
            text = _content_text((choice.get("message") or {}).get("content"))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ProviderUnavailable("Meterless Runtime returned an unexpected response",
                                      details={"http_status": resp.status_code}) from None
        if json_mode:
            text = _strip_fences(text)
        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
        return ChatResult(text=text, model=str(obj.get("model") or used_model), usage=usage,
                          http_status=resp.status_code)

    def list_models(self) -> list:
        resp = self._send("GET", self.base_url + "/models", timeout=min(self.timeout_s, 30.0))
        self._check_auth(resp)
        if resp.status_code != 200:
            raise ProviderUnavailable(f"Meterless Runtime returned HTTP {resp.status_code}.",
                                      details={"http_status": resp.status_code})
        try:
            data = resp.json().get("data", [])
            ids = [m.get("id") if isinstance(m, dict) else m for m in data]
        except (ValueError, AttributeError, TypeError):
            raise ProviderUnavailable("Meterless Runtime returned an unexpected models response") from None
        return [i for i in ids if _is_meterless_id(i)]

    def health(self) -> ProviderHealth:
        headers = self._headers()
        notes: list[str] = []
        try:
            hz = requests.get(healthz_url(self.base_url), headers=headers, timeout=3.0)
            notes.append(f"healthz={hz.status_code}")
        except requests.RequestException:
            notes.append("healthz=unreachable")
        try:
            resp = requests.get(self.base_url + "/models", headers=headers, timeout=5.0)
        except requests.RequestException:
            return ProviderHealth("offline", self.base_url, [], "; ".join(notes + ["models=unreachable"]))
        notes.append(f"models={resp.status_code}")
        if resp.status_code in (401, 403):
            return ProviderHealth("unauthorized", self.base_url, [], "; ".join(notes))
        if resp.status_code == 200:
            try:
                data = resp.json().get("data", [])
                ids = [m.get("id") if isinstance(m, dict) else m for m in data]
                ids = [i for i in ids if _is_meterless_id(i)]
            except (ValueError, AttributeError, TypeError):
                ids = []
            if ids:
                return ProviderHealth("online", self.base_url, ids, "; ".join(notes))
            notes.append("no meterless/* models")
        return ProviderHealth("offline", self.base_url, [], "; ".join(notes))


# ---------------------------------------------------------------------------
# autodetect
# ---------------------------------------------------------------------------
_DETECTED_BASE_URL: Optional[str] = None


def detected_base_url() -> Optional[str]:
    return _DETECTED_BASE_URL


def reset_autodetect() -> None:
    global _DETECTED_BASE_URL
    _DETECTED_BASE_URL = None


def _default_probe(base_url: str, timeout_s: float = 1.5) -> str:
    try:
        resp = requests.get(base_url + "/models", headers={"meterless-app": APP_ID}, timeout=timeout_s)
    except requests.RequestException:
        return "offline"
    if resp.status_code in (401, 403):
        return "unauthorized"
    if resp.status_code == 200:
        try:
            data = resp.json().get("data", [])
            if any(_is_meterless_id(m.get("id") if isinstance(m, dict) else m) for m in data):
                return "online"
        except (ValueError, AttributeError, TypeError):
            pass
    return "offline"


def _state_of(res: Any) -> str:
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        return str(res.get("state", "offline"))
    return str(getattr(res, "state", "offline"))


def autodetect(probe: Optional[Callable] = None, timeout_s: float = 1.5) -> Optional[str]:
    """Probe local ports (port-major). Returns the first base URL whose state is
    online or unauthorized; remembers the base URL (never the token) in memory."""
    global _DETECTED_BASE_URL
    if probe is None and _DETECTED_BASE_URL:
        return _DETECTED_BASE_URL
    fn = probe or _default_probe
    two_args = True
    try:
        params = list(inspect.signature(fn).parameters.values())
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        has_var = any(p.kind == p.VAR_POSITIONAL for p in params)
        two_args = has_var or len(positional) >= 2
    except (TypeError, ValueError):
        pass
    for port in AUTODETECT_PORTS:
        for host in AUTODETECT_HOSTS:
            base = f"http://{host}:{port}/v1"
            try:
                res = fn(base, timeout_s) if two_args else fn(base)
            except Exception:
                continue
            if _state_of(res) in ("online", "unauthorized"):
                _DETECTED_BASE_URL = base
                return base
    return None


# ---------------------------------------------------------------------------
# SampleProvider
# ---------------------------------------------------------------------------
class SampleProvider:
    """Deterministic, offline. Delegates TASK:<name> messages to agent.rules."""

    name = "sample"
    model = "sample"

    def chat(self, messages: list, *, model: Optional[str] = None, temperature: float = 0.0,
             max_tokens: int = 1024, json_mode: bool = False,
             timeout_s: Optional[float] = None) -> ChatResult:
        content = ""
        for m in reversed(messages or []):
            if m.get("role") == "user":
                content = _content_text(m.get("content"))
                break
        if not content.startswith("TASK:"):
            raise InvalidRequest("SampleProvider expects a message of the form 'TASK:<name>\\n<json>'")
        head, _, rest = content.partition("\n")
        task = head[len("TASK:"):].strip()
        try:
            payload = json.loads(rest) if rest.strip() else {}
        except ValueError:
            raise InvalidRequest("SampleProvider payload is not valid JSON") from None
        if not isinstance(payload, dict):
            raise InvalidRequest("SampleProvider payload must be a JSON object")

        if task == "parse_question":
            parsed = rules.parse_question(str(payload.get("question", "")), payload.get("default_year"))
            text = json.dumps(parsed.to_dict(), sort_keys=True)
        elif task == "plan":
            parsed = ParsedQuestion.from_dict(payload["parsed"])
            text = json.dumps(rules.plan(parsed, payload.get("recalled") or []).to_dict(), sort_keys=True)
        elif task == "narrate":
            text = rules.narrate(payload)
        else:
            raise InvalidRequest(f"Unknown task: {task}")
        return ChatResult(text=text, model=self.model, usage={}, http_status=None)

    def list_models(self) -> list:
        return ["sample"]

    def health(self) -> ProviderHealth:
        return ProviderHealth("online", "sample://local", ["sample"], "offline deterministic provider")
