"""Planner agent (DESIGN section 9): parse -> recall memory -> plan -> delegate."""

from __future__ import annotations

from typing import Any, Optional

from .. import rules
from ..errors import NoData, QuestionNotSupported
from ..models import MemoryRecord, ParsedQuestion, Plan
from ..sources import Agg, QuerySpec
from . import RunContext, parse_json_object

AGENT = "Planner"
_METRICS = ("net_revenue", "ticket_count")
_REGIONS = ("EU", "UK", "US", "ALL")
_DIRECTIONS = ("drop", "rise", "change")
_SOURCES = ("orders", "tickets", "shipments", "memory")
_DATA_SOURCES = ("orders", "tickets", "shipments")


def _entities(metric: str, region: str, year: int, week: int) -> tuple:
    ents = []
    if region != "ALL":
        ents.append(f"region:{region}")
    ents.append(f"week:{year}-W{week:02d}")
    ents.append(f"metric:{metric}")
    return tuple(ents)


def _validate_parsed(obj: Optional[dict], default_year: int) -> Optional[ParsedQuestion]:
    if not isinstance(obj, dict):
        return None
    try:
        metric = obj["metric"]
        region = obj["region"]
        direction = obj.get("direction", "change")
        year = obj.get("iso_year", default_year)
        week = obj["iso_week"]
        if isinstance(year, bool) or isinstance(week, bool):
            return None
        year, week = int(year), int(week)
    except (KeyError, TypeError, ValueError):
        return None
    if metric not in _METRICS or region not in _REGIONS or direction not in _DIRECTIONS:
        return None
    if not (1 <= week <= 53) or not (2000 <= year <= 2100):
        return None
    return ParsedQuestion(metric=metric, region=region, iso_year=year, iso_week=week,
                          direction=direction, entities=_entities(metric, region, year, week))


def _carrier_memory(recalled: list) -> bool:
    return any(rules._mentions_carrier(r) for r in recalled)


def _validate_plan(obj: Optional[dict], parsed: ParsedQuestion, recalled: list) -> Optional[Plan]:
    if not isinstance(obj, dict):
        return None
    try:
        raw = dict(obj)
        raw["parsed"] = parsed.to_dict()
        raw["recalled_memory_ids"] = [r.memory_id for r in recalled]
        raw.setdefault("rationale", "")
        plan = Plan.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None
    subs = plan.subquestions
    if not subs:
        return None
    ids = set()
    for s in subs:
        if s.source not in _SOURCES or not s.text.strip() or s.id in ids:
            return None
        ids.add(s.id)
    srcs = {s.source for s in subs}
    if not all(d in srcs for d in _DATA_SOURCES):
        return None
    if recalled and "memory" not in srcs:
        return None
    if _carrier_memory(recalled):
        first = min(subs, key=lambda s: s.priority)
        if first.source != "shipments":
            return None
    subs.sort(key=lambda s: s.priority)
    return plan


class Planner:
    name = AGENT

    # ---- step 0: default year -------------------------------------------
    def _default_year(self, ctx: RunContext) -> int:
        spec = QuerySpec(table="orders", aggs=(Agg("max", "iso_year", "max_iso_year"),), limit=1)
        res = ctx.query_source(AGENT, spec, purpose="max iso_year present in orders")
        val = res.rows[0].get("max_iso_year") if res.rows else None
        if val is None:
            raise NoData("The orders source contains no rows.")
        return int(val)

    # ---- step 1: parse --------------------------------------------------------
    def _parse(self, ctx: RunContext, question: str) -> ParsedQuestion:
        default_year = self._default_year(ctx)
        parsed: Optional[ParsedQuestion] = None
        reason = ""
        res = ctx.call_provider(AGENT, "parse_question",
                                {"question": question, "default_year": default_year},
                                json_mode=True, max_tokens=400)
        parsed = _validate_parsed(parse_json_object(res.text), default_year)
        if parsed is None:
            reason = "model output failed validation"
        else:
            # Cross-check against the deterministic rules; they win on disagreement.
            try:
                ref = rules.parse_question(question, default_year)
            except QuestionNotSupported:
                raise
            if (ref.metric, ref.region, ref.iso_week) != (parsed.metric, parsed.region, parsed.iso_week):
                parsed = None
                reason = "model output disagreed with rule-based parse"
        if parsed is None:
            ctx.trace.emit(AGENT, "fallback", "parse_question", f"using rules.parse_question ({reason})",
                           status="fallback")
            parsed = rules.parse_question(question, default_year)
        ctx.emit(AGENT, "parse_question", question,
                 f"metric={parsed.metric} region={parsed.region} week={parsed.iso_year}-W{parsed.iso_week:02d} "
                 f"direction={parsed.direction}")
        return parsed

    # ---- step 2: recall ---------------------------------------------------------
    def _recall(self, ctx: RunContext, parsed: ParsedQuestion) -> list:
        if not getattr(ctx.config, "memory_enabled", True):
            ctx.emit(AGENT, "recall_memory", "memory disabled", "skipped", status="skipped")
            return []
        records: list[MemoryRecord] = ctx.engine.recall(entities=list(parsed.entities))
        if not records:
            ctx.emit_engine(AGENT, "recall_memory", "recall", ",".join(parsed.entities), [],
                            f"entities={list(parsed.entities)}", "no memory recalled")
        for r in records:
            ctx.emit_engine(AGENT, "recall_memory", "recall", r.entity, [r.memory_id],
                            f"entities={list(parsed.entities)}",
                            f"{r.entity}: {r.fact}")
        return records

    # ---- step 3: plan -------------------------------------------------------------
    def _plan(self, ctx: RunContext, parsed: ParsedQuestion, recalled: list) -> Plan:
        payload = {"parsed": parsed.to_dict(), "recalled": [r.to_dict() for r in recalled]}
        res = ctx.call_provider(AGENT, "plan", payload, json_mode=True, max_tokens=1200)
        plan = _validate_plan(parse_json_object(res.text), parsed, recalled)
        if plan is None:
            ctx.trace.emit(AGENT, "fallback", "plan", "using rules.plan (model output failed validation)",
                           status="fallback")
            plan = rules.plan(parsed, recalled)
        return plan

    # ---- public ------------------------------------------------------------------------
    def run(self, ctx: RunContext, question: str) -> Plan:
        parsed = self._parse(ctx, question)
        recalled = self._recall(ctx, parsed)
        plan = self._plan(ctx, parsed, recalled)
        first = min(plan.subquestions, key=lambda s: s.priority)
        ctx.emit(AGENT, "delegate_task", f"{len(plan.subquestions)} sub-question(s)",
                 f"first priority: {first.source}; recalled memory: {len(recalled)}",
                 target_agent="Investigator")
        # Remember the recalled records for later stages (Explainer / output).
        setattr(ctx, "recalled_memory", recalled)
        return plan
