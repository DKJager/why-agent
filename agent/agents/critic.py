"""Critic agent (DESIGN section 9.2).

Rejects every claim that is not backed by a cited, re-verified aggregate or
row. The Critic re-executes ``Evidence.query`` through the DataSource and
re-derives the named values with the Investigator's pure derivers.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..errors import SourceQueryError
from ..models import Claim, Evidence, Rejection, Review
from ..sources import PRIMARY_KEYS, Filter, QuerySpec
from . import RunContext
from .investigator import fmt_number, recompute_values

AGENT = "Critic"

REL_TOL = 0.005
ABS_TOL = 0.01
ROWREF_SAMPLE = 5
ALL_SOURCES = ("orders", "shipments", "tickets")

_RE_MARKER = re.compile(r"\[E(\d+)\]")
_RE_WEEKS = re.compile(r"\b[Ww]eeks?\s*\d+(?:\s*[-\u2013]\s*\d+)?")
_RE_WLABEL = re.compile(r"(?:\b\d{4}\s*-\s*)?\bW\d{1,2}\b")
_RE_NUM = re.compile(r"\u20ac?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_RE_YEAR = re.compile(r"^(?:19|20)\d{2}$")


def _close(a: float, b: float) -> bool:
    d = abs(float(a) - float(b))
    return d <= ABS_TOL or d <= REL_TOL * max(abs(float(a)), abs(float(b)))


def _numeric_tokens(text: str) -> list:
    t = _RE_MARKER.sub(" ", text or "")
    t = _RE_WLABEL.sub(" ", t)
    t = _RE_WEEKS.sub(" ", t)
    return _RE_NUM.findall(t)


def _markers(text: str) -> list:
    return [f"E{m}" for m in _RE_MARKER.findall(text or "")]


def _forms(nr: Any) -> set:
    """Accepted textual forms of a NumberRef: as formatted, plus its magnitude.

    Signed values (e.g. a WoW drop of -30.1) are written in prose as "dropped 30.1%";
    the sign is carried by the NumberRef, the direction by the wording."""
    out = {fmt_number(nr.value, nr.fmt)}
    out.add(fmt_number(abs(nr.value), nr.fmt))
    return out


def _sample(items: list, k: int) -> list:
    n = len(items)
    if n <= k:
        return list(items)
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


class Critic:
    name = AGENT

    # ------------------------------------------------------------------ checks
    def _check_recompute(self, ctx: RunContext, ev: Evidence, retry: int) -> Optional[tuple]:
        """Return None if OK, else (code, detail, needs)."""
        try:
            spec = QuerySpec.from_dict(ev.query)
            res = ctx.query_source(AGENT, spec, purpose=f"re-verify {ev.evidence_id}",
                                   retry_count=retry)
            new = recompute_values(ev.query, res.rows)
        except SourceQueryError as e:
            return ("RECOMPUTE_FAILED", f"{ev.evidence_id}: query failed ({e.message})",
                    f"re-run query for {ev.evidence_id}")
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            return ("RECOMPUTE_FAILED", f"{ev.evidence_id}: cannot recompute ({type(e).__name__})",
                    f"re-run query for {ev.evidence_id}")
        if new is None:
            return ("RECOMPUTE_FAILED", f"{ev.evidence_id}: no derivation recorded",
                    f"re-run query for {ev.evidence_id}")
        for k, stored in ev.values.items():
            got = new.get(k)
            if got is None or not _close(stored, got):
                return ("RECOMPUTE_FAILED",
                        f"{ev.evidence_id}.{k}: stored {stored} vs recomputed {got}",
                        f"re-run query for {ev.evidence_id}")
        return None

    def _check_rowrefs(self, ctx: RunContext, ev: Evidence, retry: int) -> Optional[tuple]:
        refs = [str(r) for r in (ev.row_refs or [])]
        if not refs:
            return None
        pk = PRIMARY_KEYS.get(ev.source)
        if pk is None:
            return ("ROWREF_MISSING", f"{ev.evidence_id}: unknown source {ev.source}", None)
        sample = _sample(refs, ROWREF_SAMPLE)
        try:
            spec = QuerySpec(ev.source, filters=(Filter(pk, "in", tuple(sample)),),
                             select=(pk,), limit=5000)
            res = ctx.query_source(AGENT, spec, purpose=f"verify row refs of {ev.evidence_id}",
                                   retry_count=retry)
        except SourceQueryError as e:
            return ("ROWREF_MISSING", f"{ev.evidence_id}: row lookup failed ({e.message})",
                    f"re-run query for {ev.evidence_id}")
        found = {str(r.get(pk)) for r in res.rows}
        missing = [r for r in sample if r not in found]
        if missing:
            return ("ROWREF_MISSING",
                    f"{ev.evidence_id}: row(s) not found in source: {', '.join(missing[:3])}",
                    f"re-run query for {ev.evidence_id}")
        return None

    # ------------------------------------------------------------------ review
    def review(self, ctx: RunContext, claims: list, evidence: list) -> Review:
        retry = int(ctx.counters.get("refinements", 0))
        ev_by_id = {e.evidence_id: e for e in evidence}
        recompute_cache: dict = {}
        rowref_cache: dict = {}
        accepted: list = []
        rejected: list = []

        for c in claims:
            rej: list = []

            def add(code: str, detail: str, needs: Optional[str] = None) -> None:
                rej.append(Rejection(c.claim_id, code, detail, needs))

            if c.kind == "memory":
                rec = None
                if c.memory_id:
                    try:
                        rec = ctx.engine.get(c.memory_id)
                    except Exception:
                        rec = None
                if not c.memory_id or rec is None:
                    add("MEMORY_NOT_FOUND", "memory claim has no resolvable memory_id")
            else:
                if not c.evidence_ids:
                    add("NO_CITATION", "claim cites no evidence",
                        "evidence from orders, shipments or tickets")
                unknown = [i for i in c.evidence_ids if i not in ev_by_id]
                for m in _markers(c.text):
                    if m not in ev_by_id and m not in unknown:
                        unknown.append(m)
                for nr in c.numbers:
                    if nr.evidence_id not in ev_by_id and nr.evidence_id not in unknown:
                        unknown.append(nr.evidence_id)
                if unknown:
                    add("UNKNOWN_EVIDENCE", "unknown evidence id(s): " + ", ".join(unknown),
                        "valid evidence ids")

                # C3
                for nr in c.numbers:
                    ev = ev_by_id.get(nr.evidence_id)
                    if ev is None:
                        continue
                    actual = ev.values.get(nr.name)
                    if actual is None:
                        add("NUMBER_MISMATCH", f"{nr.evidence_id}.{nr.name} does not exist",
                            f"re-run query for {nr.evidence_id}")
                    elif not _close(nr.value, actual):
                        add("NUMBER_MISMATCH",
                            f"{nr.evidence_id}.{nr.name}: claimed {nr.value} vs evidence {actual}",
                            f"re-run query for {nr.evidence_id}")

                # C4
                allowed: set = set()
                for nr in c.numbers:
                    allowed |= _forms(nr)
                bad = []
                for tok in _numeric_tokens(c.text):
                    if tok in allowed:
                        continue
                    if _RE_YEAR.match(tok):
                        continue
                    bad.append(tok)
                if bad:
                    add("UNGROUNDED_NUMBER",
                        "number(s) in text without a matching reference: " + ", ".join(bad[:5]),
                        "numeric references for every number in the text")

                # C6b
                if c.kind == "driver":
                    srcs = {ev_by_id[i].source for i in c.evidence_ids if i in ev_by_id}
                    if len(srcs) < 2:
                        missing = [s for s in ALL_SOURCES if s not in srcs]
                        add("DRIVER_SINGLE_SOURCE",
                            f"driver is backed by {len(srcs)} distinct source(s), needs at least 2",
                            "evidence from " + " or ".join(missing))

                # C5 / C6 only for otherwise clean claims (saves source calls)
                if not rej:
                    for i in c.evidence_ids:
                        ev = ev_by_id[i]
                        if i not in recompute_cache:
                            recompute_cache[i] = self._check_recompute(ctx, ev, retry)
                        r = recompute_cache[i]
                        if r is not None:
                            add(r[0], r[1], r[2])
                            break
                if not rej:
                    for i in c.evidence_ids:
                        ev = ev_by_id[i]
                        if i not in rowref_cache:
                            rowref_cache[i] = self._check_rowrefs(ctx, ev, retry)
                        r = rowref_cache[i]
                        if r is not None:
                            add(r[0], r[1], r[2])
                            break

            if rej:
                rejected.extend(rej)
                for r in rej:
                    ctx.trace.emit(AGENT, "reject_claim", f"{c.claim_id} ({c.kind})",
                                   f"{r.code}: {r.detail}", target_agent="Investigator",
                                   confidence=c.confidence, retry_count=retry, status="rejected")
            else:
                accepted.append(c)
                ctx.trace.emit(AGENT, "accept_claim", f"{c.claim_id} ({c.kind})",
                               f"accepted, cites {','.join(c.evidence_ids) or 'memory'}",
                               target_agent="Explainer", confidence=c.confidence,
                               retry_count=retry)

        ctx.emit(AGENT, "review_claims", f"{len(claims)} claim(s), {len(evidence)} evidence item(s)",
                 f"accepted {len(accepted)}, rejected {len({r.claim_id for r in rejected})} claim(s)",
                 target_agent="Explainer" if not rejected else "Investigator",
                 retry_count=retry, status="success" if not rejected else "rejected")
        return Review(accepted=accepted, rejected=rejected)

    # --------------------------------------------------------------- narrative
    def review_narrative(self, ctx: RunContext, text: str, accepted: list,
                         evidence: list) -> list:
        ev_ids = {e.evidence_id for e in evidence}
        cited: set = set()
        allowed: set = set()
        for c in accepted:
            cited.update(c.evidence_ids)
            for nr in c.numbers:
                allowed |= _forms(nr)
            if c.kind == "memory":
                allowed.update(_numeric_tokens(c.text))
        out: list = []
        bad_markers = [m for m in dict.fromkeys(_markers(text))
                       if m not in ev_ids or (m not in cited)]
        if bad_markers:
            out.append(Rejection("narrative", "UNKNOWN_EVIDENCE",
                                 "marker(s) not backed by an accepted claim: " + ", ".join(bad_markers),
                                 None))
        bad = []
        for tok in _numeric_tokens(text):
            if tok in allowed or _RE_YEAR.match(tok):
                continue
            if tok not in bad:
                bad.append(tok)
        if bad:
            out.append(Rejection("narrative", "UNGROUNDED_NUMBER",
                                 "number(s) not mapped to an accepted claim: " + ", ".join(bad[:5]),
                                 None))
        ctx.trace.emit(AGENT, "review_claims", "narrative",
                       "narrative verified" if not out else f"narrative rejected: {out[0].code}",
                       target_agent="Explainer", status="success" if not out else "rejected",
                       retry_count=int(ctx.counters.get("refinements", 0)))
        return out


__all__ = ["Critic"]
