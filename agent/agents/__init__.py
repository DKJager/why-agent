"""Agents of the Why-Agent (DESIGN section 9) and their shared RunContext."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..errors import IterationLimit, RunTimeout
from ..provider import ChatResult
from ..sources import QueryResult, QuerySpec

AGENT_ROLES = [
    {"name": "Planner",
     "role": "Parses the question, recalls memory, splits it into per-source sub-questions"},
    {"name": "Investigator",
     "role": "Queries orders/tickets/shipments, computes deltas, joins sources, proposes cited claims"},
    {"name": "Critic",
     "role": "Rejects any claim not backed by a re-verified cited aggregate or row; max 2 refinement loops"},
    {"name": "Explainer",
     "role": "Writes the final answer using only accepted claims, with citations"},
]

SYSTEM_PROMPT = (
    "You are a component of a business-analysis agent. You only help with language: "
    "parsing questions, wording sub-questions and writing narrative. You never invent "
    "numbers. Reply with a single JSON object when asked for JSON."
)


def make_messages(task: str, payload: dict) -> list:
    """Message list in the ``TASK:<name>\\n<json>`` format used by the providers."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"TASK:{task}\n" + json.dumps(payload, sort_keys=True, default=str)},
    ]


def parse_json_object(text: str) -> Optional[dict]:
    """Tolerantly parse a JSON object out of model text. None on failure."""
    t = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S | re.I)
    if m:
        t = m.group(1)
    try:
        obj = json.loads(t)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


@dataclass
class RunContext:
    """Shared state of one run. ``deadline`` is a ``time.monotonic()`` value."""

    config: Any
    run_id: str
    source: Any
    provider: Any
    engine: Any
    trace: Any
    deadline: Optional[float] = None
    counters: dict = field(default_factory=lambda: {"steps": 0, "model_calls": 0})

    # ---- budget -----------------------------------------------------------
    def remaining(self) -> float:
        if self.deadline is None:
            return float("inf")
        return self.deadline - time.monotonic()

    def check_deadline(self) -> None:
        if self.remaining() <= 0:
            raise RunTimeout("The run deadline was exceeded.",
                             details={"run_timeout_s": getattr(self.config, "run_timeout_s", None)})

    def count_step(self) -> None:
        self.counters["steps"] = self.counters.get("steps", 0) + 1
        if self.counters["steps"] > getattr(self.config, "max_steps", 60):
            raise IterationLimit("MAX_STEPS was reached.",
                                 details={"max_steps": getattr(self.config, "max_steps", 60)})

    def count_model_call(self) -> None:
        if self.counters.get("model_calls", 0) >= getattr(self.config, "max_model_calls", 12):
            raise IterationLimit("MAX_MODEL_CALLS was reached.",
                                 details={"max_model_calls": getattr(self.config, "max_model_calls", 12)})
        self.counters["model_calls"] = self.counters.get("model_calls", 0) + 1

    def provider_timeout(self) -> float:
        return max(0.001, min(float(getattr(self.config, "provider_timeout_s", 180.0)),
                              self.remaining()))

    def source_timeout(self) -> float:
        return max(0.001, min(float(getattr(self.config, "source_timeout_s", 20.0)),
                              self.remaining()))

    # ---- tracing helpers ------------------------------------------------------
    def emit(self, agent_name: str, action: str, input_summary: Any = "", output_summary: Any = "",
             **kw: Any) -> dict:
        """Emit an agent-step trace record; counts against MAX_STEPS."""
        self.check_deadline()
        self.count_step()
        return self.trace.emit(agent_name, action, input_summary, output_summary, **kw)

    def call_provider(self, agent_name: str, task: str, payload: dict, *, json_mode: bool = True,
                      max_tokens: int = 1024, retry_count: int = 0) -> ChatResult:
        """One traced model call with budget and timeout handling."""
        self.check_deadline()
        self.count_model_call()
        t0 = time.monotonic()
        try:
            res = self.provider.chat(
                make_messages(task, payload),
                temperature=0.0, max_tokens=max_tokens, json_mode=json_mode,
                timeout_s=self.provider_timeout(),
            )
        except Exception as e:
            self.trace.emit(
                agent_name, "provider_call", f"task={task}", f"{type(e).__name__}",
                target_agent=f"provider:{getattr(self.provider, 'name', 'provider')}",
                status="error", kind="provider_call", retry_count=retry_count,
                duration_ms=int((time.monotonic() - t0) * 1000),
                details={"provider": getattr(self.provider, "name", "provider"), "task": task,
                         "http_status": getattr(e, "details", {}).get("http_status")
                         if isinstance(getattr(e, "details", None), dict) else None},
            )
            raise
        self.trace.emit(
            agent_name, "provider_call", f"task={task}", f"model={res.model} chars={len(res.text or '')}",
            target_agent=f"provider:{getattr(self.provider, 'name', 'provider')}",
            kind="provider_call", retry_count=retry_count,
            duration_ms=int((time.monotonic() - t0) * 1000),
            details={"provider": getattr(self.provider, "name", "provider"), "model": res.model,
                     "task": task, "http_status": res.http_status, "usage": res.usage},
        )
        return res

    def query_source(self, agent_name: str, spec: QuerySpec, *, purpose: str = "",
                     retry_count: int = 0) -> QueryResult:
        """One traced source query."""
        self.check_deadline()
        t0 = time.monotonic()
        src = getattr(self.source, "name", "source")
        try:
            res = self.source.query(spec, timeout_s=self.source_timeout())
        except Exception as e:
            self.trace.emit(
                agent_name, "source_query", purpose or f"table={spec.table}", type(e).__name__,
                target_agent=f"source:{src}", status="error", kind="source_call",
                retry_count=retry_count, duration_ms=int((time.monotonic() - t0) * 1000),
                details={"source": src, "table": spec.table, "query_text": "", "row_count": 0,
                         "http_status": getattr(self.source, "last_http_status", None)},
            )
            raise
        self.trace.emit(
            agent_name, "source_query", purpose or f"table={spec.table}",
            f"{res.row_count} row(s)", target_agent=f"source:{src}", kind="source_call",
            retry_count=retry_count, duration_ms=int((time.monotonic() - t0) * 1000),
            details={"source": src, "table": spec.table, "query_text": res.query_text,
                     "row_count": res.row_count,
                     "http_status": getattr(self.source, "last_http_status", None)},
        )
        return res

    def emit_engine(self, agent_name: str, action: str, op: str, entity: str, memory_ids: list,
                    input_summary: Any = "", output_summary: Any = "", *, retry_count: int = 0,
                    status: str = "success") -> dict:
        return self.trace.emit(
            agent_name, action, input_summary, output_summary,
            target_agent=f"engine:{getattr(self.engine, 'name', 'engine')}",
            kind="engine_call", status=status, retry_count=retry_count,
            details={"engine": getattr(self.engine, "name", "engine"), "op": op,
                     "entity": entity, "memory_ids": list(memory_ids)},
        )


__all__ = ["RunContext", "AGENT_ROLES", "make_messages", "parse_json_object", "SYSTEM_PROMPT"]
