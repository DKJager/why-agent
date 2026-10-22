"""Deterministic parse / plan / narrate rules (DESIGN sections 7.4, 9.4).

Used by ``SampleProvider`` and as the fallback when a model misbehaves.
Nothing here invents numbers: every number in the rendered text comes
verbatim from the accepted claims (which the Critic has verified), and no
carrier or category name is hardcoded; names only appear if the claims
(built from the data) contain them.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from .errors import QuestionNotSupported
from .models import ParsedQuestion, Plan, SubQuestion

DEFAULT_YEAR = 2026

_RE_TICKET = re.compile(r"\b(tickets?|support|complaints?)\b", re.I)
_RE_REVENUE = re.compile(r"\b(revenues?|sales)\b", re.I)
_RE_EU = re.compile(r"\bEU\b|\bEurope(?:an)?\b", re.I)
_RE_UK = re.compile(r"\bUK\b|\bUnited Kingdom\b|\bBritain\b|\bGreat Britain\b")
_RE_US = re.compile(r"\bUS\b|\bUSA\b|\bUnited States\b")
_RE_WEEK = re.compile(r"\bweek\s*(?:no\.?\s*)?(\d{1,2})\b", re.I)
_RE_WEEK_W = re.compile(r"\bW(\d{1,2})\b")
_RE_YEAR_W = re.compile(r"\b(20\d{2})\s*-?\s*W(\d{1,2})\b", re.I)
_RE_YEAR = re.compile(r"\b(20\d{2})\b")
_RE_DROP = re.compile(r"\b(drop(?:ped|s)?|fell|fall(?:s|en)?|down|declin\w*|decreas\w*|dip(?:ped)?|plunge\w*|slump\w*)\b", re.I)
_RE_RISE = re.compile(r"\b(ris(?:e|es|en)|rose|spik\w*|up|increas\w*|jump\w*|surg\w*|grew|grow\w*)\b", re.I)


# ---------------------------------------------------------------------------
# small helpers (dataclass-or-dict access)
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_list(v: Any) -> list:
    if v is None:
        return []
    return list(v)


def _strip_period(s: str) -> str:
    return str(s).strip().rstrip(".").strip()


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def parse_question(question: str, default_year: Optional[int] = None) -> ParsedQuestion:
    """Regex rules. Raises QuestionNotSupported if no metric or no week is found."""
    q = question if isinstance(question, str) else ""
    if _RE_TICKET.search(q):
        metric = "ticket_count"
    elif _RE_REVENUE.search(q):
        metric = "net_revenue"
    else:
        raise QuestionNotSupported(
            "Could not find a supported metric (revenue/sales or support tickets) in the question.")

    year = int(default_year) if default_year else DEFAULT_YEAR
    week: Optional[int] = None
    m = _RE_YEAR_W.search(q)
    if m:
        year, week = int(m.group(1)), int(m.group(2))
    else:
        m = _RE_WEEK.search(q) or _RE_WEEK_W.search(q)
        if m:
            week = int(m.group(1))
        my = _RE_YEAR.search(q)
        if my:
            year = int(my.group(1))
    if week is None or not (1 <= week <= 53):
        raise QuestionNotSupported("Could not find a valid ISO week number (1-53) in the question.")

    if _RE_EU.search(q):
        region = "EU"
    elif _RE_UK.search(q):
        region = "UK"
    elif _RE_US.search(q):
        region = "US"
    else:
        region = "ALL"

    d_drop = _RE_DROP.search(q)
    d_rise = _RE_RISE.search(q)
    if d_drop and d_rise:
        direction = "drop" if d_drop.start() <= d_rise.start() else "rise"
    elif d_drop:
        direction = "drop"
    elif d_rise:
        direction = "rise"
    else:
        direction = "change"

    ents = []
    if region != "ALL":
        ents.append(f"region:{region}")
    ents.append(f"week:{year}-W{week:02d}")
    ents.append(f"metric:{metric}")
    return ParsedQuestion(metric=metric, region=region, iso_year=year, iso_week=week,
                          direction=direction, entities=tuple(ents))


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def _region_label(parsed: Any) -> str:
    r = _get(parsed, "region", "ALL")
    return "all regions" if r == "ALL" else str(r)


def _metric_label(parsed: Any) -> str:
    return "support tickets" if _get(parsed, "metric") == "ticket_count" else "net revenue"


def _subject(parsed: Any) -> str:
    if _get(parsed, "region", "ALL") == "ALL":
        return f"Overall {_metric_label(parsed)}"
    return f"{_region_label(parsed)} {_metric_label(parsed)}"


def _verb(parsed: Any) -> str:
    return {"drop": "dropped", "rise": "rose"}.get(_get(parsed, "direction"), "changed")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def _mentions_carrier(rec: Any) -> bool:
    entity = str(_get(rec, "entity", "") or "")
    fact = str(_get(rec, "fact", "") or "")
    return entity.startswith("carrier:") or "carrier" in fact.lower()


def plan(parsed: ParsedQuestion, recalled: Iterable[Any] = ()) -> Plan:
    """Build per-source sub-questions. Recalled memory mentioning a carrier
    promotes shipments to priority 1 and adds a ``memory`` sub-question."""
    recalled = list(recalled or ())
    subject = _subject(parsed)
    week_label = f"week {parsed.iso_week} of {parsed.iso_year}"
    region = _region_label(parsed)
    ticket_q = parsed.metric == "ticket_count"

    texts = {
        "orders": (
            f"How did {subject.lower()} for {region} in {week_label} compare with the prior week "
            f"and the four-week baseline, and which country, category, carrier and cancellation "
            f"reason explain the change?",
            "Quantify the change and decompose it into segments; separate demand from fulfilment.",
        ),
        "shipments": (
            f"Which carriers show late deliveries for {region} shipments around {week_label}, "
            f"and do they match the cancelled orders?",
            "Corroborate a candidate carrier with delivery delay data.",
        ),
        "tickets": (
            f"Which support ticket categories changed for {region} in {week_label}, and are they "
            f"linked to the orders of the candidate segment?",
            "Check customer-side signals and join them to the candidate orders.",
        ),
    }
    order = ["tickets", "orders", "shipments"] if ticket_q else ["orders", "shipments", "tickets"]
    carrier_memory = any(_mentions_carrier(r) for r in recalled)
    if carrier_memory:
        order.remove("shipments")
        order.insert(0, "shipments")

    subs: list[SubQuestion] = []
    for i, src in enumerate(order, start=1):
        text, purpose = texts[src]
        subs.append(SubQuestion(id=f"Q{i}", source=src, text=text, purpose=purpose, priority=i))
    if recalled:
        facts = "; ".join(str(_get(r, "entity", "")) for r in recalled[:5])
        subs.append(SubQuestion(
            id=f"Q{len(subs) + 1}", source="memory",
            text=f"Which prior findings (H-MEM) relate to {facts}? Treat them as hypotheses to test.",
            purpose="Recalled memory is a hypothesis, never evidence.",
            priority=len(subs) + 1,
        ))
    rationale = f"Investigate {subject.lower()} for {region} in {week_label} across orders, shipments and tickets."
    if carrier_memory:
        rationale += " Recalled memory points at a carrier, so shipments are checked first."
    elif recalled:
        rationale += " Recalled memory is used as a hypothesis only."
    return Plan(parsed=parsed, subquestions=subs,
                recalled_memory_ids=[str(_get(r, "memory_id", "")) for r in recalled],
                rationale=rationale)


# ---------------------------------------------------------------------------
# narrate
# ---------------------------------------------------------------------------
def _sorted_drivers(drivers: list) -> list:
    def key(c: Any):
        s = _get(c, "explained_share")
        return (0 if s is not None else 1, -(s if s is not None else 0.0), str(_get(c, "claim_id", "")))
    return sorted(drivers, key=key)


def _split(payload: dict) -> dict:
    parsed = payload.get("parsed")
    claims = _as_list(payload.get("accepted", payload.get("claims")))
    by_kind: dict[str, list] = {}
    for c in claims:
        by_kind.setdefault(str(_get(c, "kind")), []).append(c)
    return {
        "parsed": parsed,
        "delta": by_kind.get("delta", []),
        "driver": _sorted_drivers(by_kind.get("driver", [])),
        "ruled_out": by_kind.get("ruled_out", []),
        "observation": by_kind.get("observation", []),
        "memory": by_kind.get("memory", []),
        "recalled": _as_list(payload.get("recalled")),
    }


def headline(payload: dict) -> str:
    s = _split(payload)
    parsed = s["parsed"]
    subject = _subject(parsed)
    wk = _get(parsed, "iso_week")
    if s["driver"]:
        return f"{subject} {_verb(parsed)} in week {wk}: {_strip_period(_get(s['driver'][0], 'text'))}"
    if s["observation"]:
        return f"No significant change in {subject.lower()} in week {wk}; there is no cause to explain."
    if s["ruled_out"] or s["delta"]:
        return (f"{subject} {_verb(parsed)} in week {wk}, but no verified driver could be established "
                f"from the available evidence.")
    return f"No verified finding could be established for {subject.lower()} in week {wk}."


def narrate(payload: dict) -> str:
    """Render the narrative from accepted claims only (claim texts verbatim)."""
    s = _split(payload)
    lines: list[str] = [headline(payload), ""]
    for c in s["delta"]:
        lines.append(_get(c, "text"))
    for c in s["observation"]:
        lines.append(_get(c, "text"))
    if s["driver"]:
        lines += ["", "Main driver(s):"]
        for c in s["driver"]:
            lines.append(f"- {_get(c, 'text')}")
    if s["ruled_out"]:
        lines += ["", "Ruled out:"]
        for c in s["ruled_out"]:
            lines.append(f"- {_get(c, 'text')}")
    if s["memory"]:
        lines += ["", "Prior knowledge (H-MEM, a hypothesis and not evidence):"]
        for c in s["memory"]:
            lines.append(f"- {_get(c, 'text')}")
    return "\n".join(lines).strip()


render_answer = narrate


def answer_parts(payload: dict) -> dict:
    """Structured pieces for the Explainer: headline, drivers, ruled_out,
    prior_knowledge and text (all derived from accepted claims / recalled memory)."""
    s = _split(payload)
    drivers = []
    for i, c in enumerate(s["driver"], start=1):
        drivers.append({
            "rank": i,
            "claim_id": _get(c, "claim_id"),
            "text": _get(c, "text"),
            "explained_share": _get(c, "explained_share"),
            "confidence": _get(c, "confidence"),
            "evidence_ids": list(_get(c, "evidence_ids", []) or []),
        })
    ruled = [{
        "claim_id": _get(c, "claim_id"),
        "text": _get(c, "text"),
        "evidence_ids": list(_get(c, "evidence_ids", []) or []),
    } for c in s["ruled_out"]]
    prior = [{
        "memory_id": _get(r, "memory_id"),
        "entity": _get(r, "entity"),
        "fact_key": _get(r, "fact_key"),
        "fact": _get(r, "fact"),
        "confidence": _get(r, "confidence"),
        "last_confirmed_at": _get(r, "last_confirmed_at"),
        "label": "prior knowledge (hypothesis, not evidence)",
    } for r in s["recalled"]]
    return {
        "headline": headline(payload),
        "drivers": drivers,
        "ruled_out": ruled,
        "prior_knowledge": prior,
        "text": narrate(payload),
    }


def recommendations(payload: dict) -> list[str]:
    """Rule-based recommendations from confirmed drivers. Names come from claim entities."""
    s = _split(payload)
    out: list[str] = []
    parsed = s["parsed"]
    region = _get(parsed, "region", "ALL")
    where = "" if region == "ALL" else f" {region}"
    for c in s["driver"]:
        carriers = [e.split(":", 1)[1] for e in (_get(c, "entities", ()) or ())
                    if isinstance(e, str) and e.startswith("carrier:")]
        for name in carriers:
            out.append(f"Review the carrier SLA and reroute{where} orders away from {name} "
                       f"until delivery times recover.")
            out.append(f"Proactively contact customers with delayed {name} orders to reduce cancellations.")
        if not carriers:
            out.append("Investigate the dominant segment identified above and monitor it weekly.")
    seen: set[str] = set()
    uniq = []
    for r in out:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq
