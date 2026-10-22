"""Orchestrator: the investigation loop, limits and error mapping (DESIGN section 10).

``run_question`` never raises: it returns either the success JSON (section 12.3)
or the structured error JSON (section 13).
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Optional

from . import rules
from .agents import AGENT_ROLES, RunContext
from .agents.critic import Critic
from .agents.explainer import Explainer
from .agents.investigator import Investigator
from .agents.planner import Planner
from .config import Config
from .engine import HMemMeterless, HMemStub, build_engine
from .errors import (
    ConfigError,
    InternalError,
    InvalidRequest,
    IterationLimit,
    ProviderUnauthorized,
    ProviderUnavailable,
    RunTimeout,
    SourceUnauthorized,
    SourceUnavailable,
    UpstreamTimeout,
    WhyAgentError,
    error_body,
    wrap_unexpected,
)
from .models import Answer, MemoryRecord
from .provider import (
    MeterlessProvider,
    SampleProvider,
    autodetect,
    parse_connect_link,
    validate_token,
)
from .sources import CsvSource, ZetarisSource
from .trace import Redactor, TraceWriter

ORCH = "Orchestrator"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MAX_QUESTION_CHARS = 1000
FALLBACK_ERRORS = (ProviderUnavailable, ProviderUnauthorized, SourceUnavailable,
                   SourceUnauthorized, UpstreamTimeout)
_RE_MARKER = re.compile(r"\s*\[E\d+\]")
STORE_MIN_CONFIDENCE = 0.7


def _default_run_id() -> str:
    return "run-" + time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:4]


# ---------------------------------------------------------------------------
# component construction
# ---------------------------------------------------------------------------
def _apply_connect_link(config: Config) -> Config:
    raw = config.meterless_base_url
    if raw and "k=" in raw:
        base, key = parse_connect_link(raw)
        changes: dict = {"meterless_base_url": base}
        if key and not config.meterless_runtime_token:
            changes["meterless_runtime_token"] = key
        return config.replace(**changes)
    return config


def _build_components(config: Config) -> dict:
    if config.sample_mode:
        source: Any = CsvSource(config.data_dir)
        provider: Any = SampleProvider()
        if config.engine == "meterless":
            HMemMeterless()  # raises NotConfigured
        engine: Any = HMemStub(config.hmem_path) if config.engine == "stub" else build_engine(config)
        return {"source": source, "provider": provider, "engine": engine, "sample": True,
                "config": config}

    config = _apply_connect_link(config)
    token = config.meterless_runtime_token
    if token:
        try:
            validate_token(token)
        except ProviderUnauthorized:
            raise ProviderUnauthorized(
                "The Meterless token is a placeholder; set a real METERLESS_API_KEY.",
                details={"reason": "placeholder"}) from None
    config.validate_live()
    base = config.meterless_base_url
    if base is None:
        base = autodetect()
        if base is None:
            raise ProviderUnavailable("No Meterless Runtime was found on the local ports.")
    provider = MeterlessProvider(base, token, model_id=config.model_id,
                                 timeout_s=config.provider_timeout_s)
    source = ZetarisSource(config.zetaris_base_url, config.zetaris_api_key, config.zetaris_tables,
                           config.source_timeout_s)
    if config.engine == "meterless":
        HMemMeterless()  # raises NotConfigured
    engine = build_engine(config)
    return {"source": source, "provider": provider, "engine": engine, "sample": False,
            "config": config}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def _store_findings(ctx: RunContext, accepted: list, evidence: list) -> list:
    stored: list = []
    if not getattr(ctx.config, "memory_enabled", True):
        return stored
    ev_by_id = {e.evidence_id: e for e in evidence}
    for c in accepted:
        if c.kind not in ("driver", "delta") or c.confidence < STORE_MIN_CONFIDENCE:
            continue
        fact = _RE_MARKER.sub("", c.text).strip()
        fact_key = c.fact_key or f"{c.kind}:{c.claim_id}"
        refs = [f"{i}@run:{ctx.run_id}" for i in c.evidence_ids]
        for i in c.evidence_ids:
            ev = ev_by_id.get(i)
            if ev is not None:
                refs.extend(str(r) for r in ev.row_refs[:5])
        for entity in c.entities:
            try:
                rec = ctx.engine.store(entity, fact_key, fact, evidence_refs=refs,
                                       confidence=c.confidence, run_id=ctx.run_id)
            except WhyAgentError as e:
                ctx.emit_engine(ORCH, "store_memory", "store", entity, [], f"{fact_key}",
                                f"store failed: {type(e).__name__}", status="skipped")
                continue
            stored.append(rec)
            ctx.emit_engine(ORCH, "store_memory", "store", entity, [rec.memory_id],
                            f"{fact_key}", f"stored {rec.memory_id} (confirmations={rec.confirmations})",
                            status="success")
    return stored


def _log_rejections(state: dict, claims: list, review: Any) -> None:
    """Remember every rejection (with its claim) across refinement rounds."""
    log = state.setdefault("unverified_log", [])
    by_id = {c.claim_id: c for c in claims}
    seen = {(r.code, r.detail, getattr(c, "text", None)) for r, c in log}
    for r in review.rejected:
        c = by_id.get(r.claim_id)
        key = (r.code, r.detail, getattr(c, "text", None))
        if key in seen:
            continue
        seen.add(key)
        log.append((r, c))


def _still_unverified(state: dict) -> list:
    """Logged (rejection, claim) pairs whose claim was never accepted."""
    review = state["review"]
    acc_text = {c.text for c in review.accepted}
    acc_key = {(c.kind, c.fact_key) for c in review.accepted if c.fact_key}
    out = []
    for r, c in state.get("unverified_log", []):
        if c is not None:
            if c.text in acc_text:
                continue
            if c.fact_key and (c.kind, c.fact_key) in acc_key:
                continue
        out.append((r, c))
    return out


def _pipeline(ctx: RunContext, question: str, state: dict) -> None:
    planner, investigator, critic = Planner(), Investigator(), Critic()
    explainer = Explainer(critic)
    plan = planner.run(ctx, question)
    state["plan"] = plan
    state["recalled"] = list(getattr(ctx, "recalled_memory", []) or [])

    evidence, claims = investigator.run(ctx, plan)
    max_ref = int(getattr(ctx.config, "max_refinements", 2))
    review = None
    for attempt in range(max_ref + 1):
        review = critic.review(ctx, claims, evidence)
        state.update(review=review, evidence=evidence, claims=claims)
        _log_rejections(state, claims, review)
        if not review.rejected:
            break
        if attempt >= max_ref:
            break
        ctx.emit("Critic", "request_refinement",
                 f"{len(review.rejected)} rejection(s)",
                 ",".join(sorted({r.code for r in review.rejected})),
                 target_agent="Investigator", retry_count=attempt + 1, status="retry")
        # the claims actually reviewed (including any added after Investigator.run)
        ctx.last_claims = list(claims)
        evidence, claims = investigator.refine(ctx, plan, evidence, review.rejected)

    state["stored"] = _store_findings(ctx, review.accepted, evidence)
    unverified = [r for r, _ in _still_unverified(state)]
    answer = explainer.run(ctx, plan, review, evidence, unverified, state["recalled"])
    state["answer"] = answer


# ---------------------------------------------------------------------------
# output assembly
# ---------------------------------------------------------------------------
def _unverified(state: dict) -> list:
    out = []
    for r, c in _still_unverified(state):
        d = r.to_dict()
        if c is not None:
            d["kind"] = c.kind
            d["text"] = c.text
        out.append(d)
    return out


def _mode(comps: dict) -> dict:
    prov = comps["provider"]
    model = getattr(prov, "model", None) or getattr(prov, "model_id", None) or "unknown"
    return {"sample_mode": bool(comps["sample"]), "source": comps["source"].name,
            "provider": prov.name, "engine": comps["engine"].name, "model": model}


def _partial_answer(state: dict, comps: dict, exc: WhyAgentError) -> Answer:
    plan, review = state["plan"], state["review"]
    payload = {"parsed": plan.parsed.to_dict(),
               "accepted": [c.to_dict() for c in review.accepted],
               "recalled": [r.to_dict() for r in state.get("recalled", [])]}
    parts = rules.answer_parts(payload)
    caveats = [f"The run stopped early ({exc.type}); this answer uses only claims verified before the stop.",
               f"{len({r.claim_id for r, _ in _still_unverified(state)})} claim(s) could not be verified and are excluded."]
    if comps["sample"]:
        caveats.append("Sample mode: all data is fictional synthetic data; no real company, "
                       "customer or carrier is represented.")
    return Answer(headline=parts["headline"], drivers=parts["drivers"], ruled_out=parts["ruled_out"],
                  prior_knowledge=parts["prior_knowledge"], caveats=caveats, text=parts["text"])


def _output(question: str, run_id: str, status: str, state: dict, answer: Answer, comps: dict,
            trace: TraceWriter, degraded: bool, warnings: list, started: float) -> dict:
    plan, review = state["plan"], state["review"]
    accepted = [c for c in review.accepted]
    findings = [c.to_dict() for c in accepted if c.kind != "memory"]
    payload = {"parsed": plan.parsed, "accepted": [c.to_dict() for c in accepted]}
    body = {
        "run_id": run_id,
        "status": status,
        "output": {
            "question": question,
            "parsed": plan.parsed.to_output(),
            "summary": answer.headline,
            "answer": answer.to_dict(),
            "findings": findings,
            "unverified": _unverified(state),
            "evidence": [e.to_dict() for e in state.get("evidence", [])],
            "recommendations": rules.recommendations(payload),
            "memory": {"recalled": [r.to_dict() for r in state.get("recalled", [])],
                       "stored": [r.to_dict() for r in state.get("stored", [])]},
            "artifacts": [{"type": "trace", "path": trace.path}],
        },
        "agents": [dict(a) for a in AGENT_ROLES],
        "mode": _mode(comps),
        "degraded": bool(degraded),
        "warnings": list(warnings),
        "trace_id": trace.trace_id,
        "log_file": trace.path,
        "execution_time_seconds": round(time.monotonic() - started, 3),
    }
    if comps["source"].name == "csv":
        try:
            body["data_fingerprint"] = "sha256:" + comps["source"].fingerprint()
        except WhyAgentError:
            pass
    return body


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------
def run_question(question: str, *, run_id: Optional[str] = None, options: Optional[dict] = None,
                 config: Optional[Config] = None) -> dict:
    started = time.monotonic()
    pending: Optional[BaseException] = None

    if run_id is None:
        run_id = _default_run_id()
    elif not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        pending = InvalidRequest("run_id must match [A-Za-z0-9._-]{1,64}")
        run_id = _default_run_id()

    base_cfg: Optional[Config] = config
    try:
        if base_cfg is None:
            base_cfg = Config.from_env()
        cfg = base_cfg.with_options(options)
    except WhyAgentError as e:
        if pending is None:
            pending = e
        cfg = base_cfg if base_cfg is not None else Config()

    opts = options if isinstance(options, dict) else {}
    to_stderr = bool(opts.get("log_to_stderr")) or os.environ.get("WHY_AGENT_LOG_STDERR") == "1"
    redactor = Redactor.from_env(extra=cfg.secret_values())
    redactor.add_connect_link(cfg.meterless_base_url)
    redactor.add_connect_link(os.environ.get("METERLESS_BASE_URL"))
    redactor.add_connect_link(os.environ.get("MODEL_BASE_URL"))

    try:
        trace = TraceWriter(run_id, cfg.log_dir, redactor=redactor, level=cfg.log_level,
                            to_stderr=to_stderr)
    except OSError:
        return error_body(InternalError("Could not open the trace file.", cause_type="OSError"),
                          run_id=run_id, trace_id=None, log_file=None,
                          execution_time_seconds=time.monotonic() - started, redact=redactor.redact)

    state: dict = {}
    comps: Optional[dict] = None
    degraded = False
    warnings: list = []
    try:
        try:
            q_summary = question if isinstance(question, str) else ""
            trace.emit(ORCH, "start_run", q_summary, f"run started (sample_mode={cfg.sample_mode})",
                       target_agent="Planner")
            if pending is not None:
                raise pending
            if not isinstance(question, str) or not question.strip():
                raise InvalidRequest("The question must be a non-empty string.")
            if len(question) > MAX_QUESTION_CHARS:
                raise InvalidRequest(f"The question must be at most {MAX_QUESTION_CHARS} characters.")
            fb = opts.get("fallback_to_sample", False)
            if fb is not None and not isinstance(fb, bool):
                raise InvalidRequest("options.fallback_to_sample must be a boolean")
            fallback = bool(fb)

            deadline = time.monotonic() + cfg.run_timeout_s
            active_cfg = cfg
            while True:
                try:
                    comps = _build_components(active_cfg)
                    active_cfg = comps["config"]
                    ctx = RunContext(config=active_cfg, run_id=run_id, source=comps["source"],
                                     provider=comps["provider"], engine=comps["engine"],
                                     trace=trace, deadline=deadline)
                    _pipeline(ctx, question, state)
                    break
                except FALLBACK_ERRORS as e:
                    if active_cfg.sample_mode or not fallback:
                        raise
                    degraded = True
                    msg = f"{e.type}: live mode unavailable, continued with sample source and provider."
                    warnings.append(msg)
                    trace.emit(ORCH, "fallback", "live components", redactor.redact(msg),
                               status="fallback", details={"error_type": e.type})
                    state.clear()
                    comps = None
                    active_cfg = cfg.replace(sample_mode=True)

            status = "success"
            answer = state["answer"]
            result = _output(question, run_id, status, state, answer, comps, trace, degraded,
                             warnings, started)
            trace.emit(ORCH, "complete_run", question, f"status={status}", status="success",
                       details={"claims_accepted": len(state["review"].accepted),
                                "claims_unverified": len(_still_unverified(state))})
        except (IterationLimit, RunTimeout) as e:
            if state.get("review") is None or comps is None or state.get("plan") is None:
                raise
            warnings.append(f"Run stopped early ({e.type}): {redactor.redact(e.message)}")
            trace.emit(ORCH, "fallback", "run limit", f"{e.type}: partial answer from accepted claims",
                       status="fallback", details={"error_type": e.type})
            answer = _partial_answer(state, comps, e)
            result = _output(question, run_id, "partial", state, answer, comps, trace, degraded,
                             warnings, started)
            trace.emit(ORCH, "complete_run", question, "status=partial", status="success")
    except BaseException as exc:  # noqa: BLE001 - converted to the error contract
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            trace.close()
            raise
        err = wrap_unexpected(exc)
        details = {"error_type": err.type, "exception": type(exc).__name__,
                   "recoverable": bool(err.recoverable)}
        if isinstance(err, InternalError) and err.cause_type:
            details["exception"] = err.cause_type
        try:
            trace.emit(ORCH, "error", question if isinstance(question, str) else "",
                       redactor.redact(f"{err.type}: {err.message}"), status="error",
                       details=details)
        except Exception:
            pass
        result = error_body(err, run_id=run_id, trace_id=trace.trace_id, log_file=trace.path,
                            execution_time_seconds=time.monotonic() - started,
                            redact=redactor.redact)
    finally:
        try:
            trace.log_completed(time.monotonic() - started)
        finally:
            trace.close()
    return result


__all__ = ["run_question"]
