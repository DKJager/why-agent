# build-id: oa-d02013e3
"""Trace writer, redactor and stdout logger (DESIGN §11)."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, TextIO
from urllib.parse import parse_qs, urlsplit

MASK = "***redacted***"

REQUIRED_FIELDS = (
    "timestamp",
    "run_id",
    "agent_name",
    "action",
    "input_summary",
    "output_summary",
    "target_agent",
    "confidence",
    "retry_count",
    "status",
)
STATUSES = ("success", "error", "rejected", "retry", "fallback", "skipped")
KINDS = ("agent_step", "source_call", "provider_call", "engine_call")
AGENT_NAMES = ("Planner", "Investigator", "Critic", "Explainer", "Orchestrator")
ACTIONS = (
    "start_run", "parse_question", "recall_memory", "delegate_task", "source_query",
    "provider_call", "compute_delta", "propose_claims", "review_claims", "accept_claim",
    "reject_claim", "request_refinement", "refine", "store_memory", "synthesize_answer",
    "fallback", "complete_run", "error",
)
SECRET_ENV_NAMES = (
    "METERLESS_API_KEY", "MODEL_API_KEY", "ZETARIS_API_KEY", "NVIDIA_API_KEY",
)

SUMMARY_MAX = 300
_LEVEL_NUM = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

_RE_BEARER = re.compile(r"Bearer\s+\S+")
_RE_CONNECT = re.compile(r"#k=[^\s&]+")
_RE_SK = re.compile(r"sk-[A-Za-z0-9_-]{6,}")


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Redactor:
    """Replaces secrets in strings (and nested JSON-like objects)."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: set[str] = set()
        self.add_secrets(secrets)

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None, extra: Iterable[str] = ()
    ) -> "Redactor":
        env = os.environ if env is None else env
        r = cls()
        for name in SECRET_ENV_NAMES:
            r.add_secret(env.get(name))
        r.add_secrets(extra)
        return r

    def add_secret(self, value: Optional[str]) -> None:
        if not isinstance(value, str):
            return
        value = value.strip()
        if len(value) >= 4:
            self._secrets.add(value)

    def add_secrets(self, values: Iterable[Optional[str]]) -> None:
        for v in values:
            self.add_secret(v)

    def add_connect_link(self, raw: Optional[str]) -> None:
        """Register the ``k=`` value of a connect link. The link is never stored."""
        if not isinstance(raw, str) or "k=" not in raw:
            return
        try:
            parts = urlsplit(raw if "://" in raw else "http://" + raw)
            for chunk in (parts.fragment, parts.query):
                for v in parse_qs(chunk).get("k", []):
                    self.add_secret(v)
        except ValueError:
            return

    def redact(self, text: Any) -> str:
        s = text if isinstance(text, str) else str(text)
        for secret in sorted(self._secrets, key=len, reverse=True):
            if secret in s:
                s = s.replace(secret, MASK)
        s = _RE_BEARER.sub("Bearer " + MASK, s)
        s = _RE_CONNECT.sub(MASK, s)
        s = _RE_SK.sub(MASK, s)
        return s

    def redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, Mapping):
            return {self.redact(str(k)): self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [self.redact_obj(v) for v in obj]
        return obj


def summarize(text: Any, redactor: Optional[Redactor] = None, limit: int = SUMMARY_MAX) -> str:
    s = "" if text is None else (text if isinstance(text, str) else str(text))
    if redactor is not None:
        s = redactor.redact(s)
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return s


class TraceWriter:
    """Append-only JSONL trace plus one stdout line per record."""

    def __init__(
        self,
        run_id: str,
        log_dir: str = "logs",
        *,
        redactor: Optional[Redactor] = None,
        level: str = "INFO",
        stream: Optional[TextIO] = None,
        to_stderr: bool = False,
        echo: bool = True,
    ) -> None:
        self.run_id = run_id
        self.trace_id = f"trace-{run_id}"
        self.redactor = redactor if redactor is not None else Redactor.from_env()
        self.level = _LEVEL_NUM.get(str(level).upper(), 20)
        self._stream = stream
        self.to_stderr = to_stderr
        self.echo = echo
        self.log_dir = log_dir
        self._path = Path(log_dir) / f"trace-{run_id}.jsonl"
        self._lock = threading.Lock()
        self._step = 0
        self._fh: Optional[TextIO] = None
        self._open()

    # ---- file handling -------------------------------------------------
    @property
    def path(self) -> str:
        return self._path.as_posix()

    log_file = path

    @property
    def step(self) -> int:
        return self._step

    def _open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                finally:
                    self._fh.close()
                    self._fh = None

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- emitting ------------------------------------------------------
    def _out(self) -> TextIO:
        if self._stream is not None:
            return self._stream
        return sys.stderr if self.to_stderr else sys.stdout

    def _echo(self, level_name: str, line: str) -> None:
        if not self.echo or _LEVEL_NUM[level_name] < self.level:
            return
        out = self._out()
        out.write(line + "\n")
        out.flush()

    def emit(
        self,
        agent_name: str,
        action: str,
        input_summary: Any = "",
        output_summary: Any = "",
        *,
        target_agent: Optional[str] = None,
        confidence: Optional[float] = None,
        retry_count: int = 0,
        status: str = "success",
        kind: str = "agent_step",
        duration_ms: int = 0,
        details: Optional[dict] = None,
    ) -> dict:
        if status not in STATUSES:
            raise ValueError(f"invalid trace status: {status!r}")
        if kind not in KINDS:
            raise ValueError(f"invalid trace kind: {kind!r}")
        if confidence is not None:
            confidence = max(0.0, min(1.0, float(confidence)))
        red = self.redactor
        out_sum = summarize(output_summary, red)
        with self._lock:
            self._step += 1
            record = {
                "timestamp": utc_now_iso(),
                "run_id": self.run_id,
                "agent_name": agent_name,
                "action": action,
                "input_summary": summarize(input_summary, red),
                "output_summary": out_sum,
                "target_agent": None if target_agent is None else red.redact(target_agent),
                "confidence": confidence,
                "retry_count": int(retry_count),
                "status": status,
                "step": self._step,
                "kind": kind,
                "duration_ms": int(duration_ms),
                "details": red.redact_obj(details or {}),
            }
            line = json.dumps(record, ensure_ascii=False, default=str)
            if self._fh is None:
                self._open()
            assert self._fh is not None
            self._fh.write(line + "\n")
            self._fh.flush()

            if status == "error":
                lvl = "ERROR"
            elif status in ("rejected", "fallback"):
                lvl = "WARNING"
            else:
                lvl = "INFO"
            self._echo(lvl, f"[{lvl}] run_id={self.run_id} {agent_name} {action}: {out_sum}")
        return record

    def note(self, level: str, message: str) -> None:
        """A plain log line (not a trace record)."""
        lvl = str(level).upper() if str(level).upper() in _LEVEL_NUM else "INFO"
        self._echo(lvl, f"[{lvl}] run_id={self.run_id} {self.redactor.redact(message)}")

    def log_completed(self, seconds: float) -> None:
        """Final stdout line of a run."""
        with self._lock:
            self._echo("INFO", f"[INFO] run_id={self.run_id} Completed in {seconds:.2f} seconds")


def read_trace(path: str) -> list[dict]:
    """Parse a JSONL trace file into a list of records."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
