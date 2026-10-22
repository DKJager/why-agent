"""Investigator agent (DESIGN section 9.1).

Deterministic in both modes: it builds ``QuerySpec`` objects, runs them through
the DataSource, derives named aggregate values *in code* and proposes claims
whose numbers are references into those values. The model never produces a
number here.

Every Evidence item records, next to the re-runnable ``query``, the name of the
pure function that derives its ``values`` from the result rows
(``query["derive"]``) and its parameters (``query["params"]``). A reviewer can
therefore re-execute the query and call :func:`recompute_values`.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from ..errors import NoData
from ..models import Claim, Evidence, NumberRef, Plan
from ..sources import Agg, Filter, QuerySpec
from ..trace import utc_now_iso
from . import RunContext

AGENT = "Investigator"

SIGNIFICANT_DROP = 5.0   # percent, on both comparisons
DOMINANCE = 0.40         # share of the delta
DEMAND_TOL = 8.0         # percent: gross within this band => demand unchanged
LATE_DAYS = 3

REBUILD_CODES = {
    "RECOMPUTE_FAILED", "NUMBER_MISMATCH", "UNGROUNDED_NUMBER",
    "ROWREF_MISSING", "DRIVER_SINGLE_SOURCE",
}


# ---------------------------------------------------------------------------
# formatting / small helpers
# ---------------------------------------------------------------------------
def fmt_number(v: float, fmt: str) -> str:
    """Render a number exactly as it must appear in claim text."""
    if fmt == "pct1":
        return f"{v:.1f}%"
    if fmt == "eur0":
        return f"€{v:,.0f}"
    if fmt == "int":
        return f"{int(round(v))}"
    return f"{v:.1f}"  # days1


def _f(x: Any) -> float:
    try:
        return float(x) if x is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _r(x: Any, n: int = 2) -> float:
    return round(float(x), n)


def _pct(new: float, old: float) -> float:
    if old:
        return (new - old) / old * 100.0
    return 100.0 if new else 0.0


def _san(s: Any) -> str:
    return re.sub(r"\W", "_", str(s))


def _wl(year: int, week: int) -> str:
    return f"{year}-W{week:02d}"


def _filters(table: str, p: Any, lo: int, hi: int, extra: tuple = ()) -> tuple:
    ycol, wcol = (("ship_iso_year", "ship_iso_week") if table == "shipments"
                  else ("iso_year", "iso_week"))
    f = [Filter(ycol, "=", p.iso_year), Filter(wcol, ">=", lo), Filter(wcol, "<=", hi)]
    if p.region != "ALL":
        f.append(Filter("region", "=", p.region))
    f.extend(extra)
    return tuple(f)


def _mean(d: dict, keys: list) -> float:
    return sum(d.get(k, 0.0) for k in keys) / len(keys) if keys else 0.0


# ---------------------------------------------------------------------------
# derivers: pure functions rows -> (values, info)
# ---------------------------------------------------------------------------
def _d_weekly_revenue(rows: list, p: dict) -> tuple:
    w = int(p["week"])
    gross: dict = {}
    canc: dict = {}
    cnt: dict = {}
    for r in rows:
        k = int(r["iso_week"])
        g = _f(r.get("gross"))
        gross[k] = gross.get(k, 0.0) + g
        cnt[k] = cnt.get(k, 0) + int(_f(r.get("n")))
        if r.get("status") == "cancelled":
            canc[k] = canc.get(k, 0.0) + g
    net = {k: gross[k] - canc.get(k, 0.0) for k in gross}
    bw = [k for k in range(w - 4, w) if k in gross]
    nw, npv, nb = net.get(w, 0.0), net.get(w - 1, 0.0), _mean(net, bw)
    gw, gp, gb = gross.get(w, 0.0), gross.get(w - 1, 0.0), _mean(gross, bw)
    wow, base = _pct(nw, npv), _pct(nw, nb)
    gwow, gbase = _pct(gw, gp), _pct(gw, gb)
    vals = {
        "net_w": _r(nw), "net_prev": _r(npv), "net_base_mean": _r(nb),
        "net_wow_pct": _r(wow, 1), "net_wow_abs_pct": _r(abs(wow), 1),
        "net_base_pct": _r(base, 1), "net_base_abs_pct": _r(abs(base), 1),
        "net_delta_eur": _r(nw - npv), "net_change_abs_eur": _r(abs(nw - npv)),
        "gross_w": _r(gw), "gross_prev": _r(gp),
        "gross_wow_pct": _r(gwow, 1), "gross_wow_abs_pct": _r(abs(gwow), 1),
        "gross_base_abs_pct": _r(abs(gbase), 1),
        "cancelled_w": _r(canc.get(w, 0.0)), "cancelled_prev": _r(canc.get(w - 1, 0.0)),
        "orders_w": float(cnt.get(w, 0)),
    }
    return vals, {}


def _d_weekly_tickets(rows: list, p: dict) -> tuple:
    w = int(p["week"])
    cnt: dict = {}
    for r in rows:
        k = int(r["iso_week"])
        cnt[k] = cnt.get(k, 0) + int(_f(r.get("n")))
    bw = [k for k in range(w - 4, w) if k in cnt]
    tw, tp = float(cnt.get(w, 0)), float(cnt.get(w - 1, 0))
    base = _mean({k: float(v) for k, v in cnt.items()}, bw)
    wow, bpct = _pct(tw, tp), _pct(tw, base)
    vals = {
        "tickets_w": tw, "tickets_prev": tp, "tickets_base_mean": float(round(base)),
        "tickets_wow_pct": _r(wow, 1), "tickets_wow_abs_pct": _r(abs(wow), 1),
        "tickets_base_pct": _r(bpct, 1), "tickets_base_abs_pct": _r(abs(bpct), 1),
        "tickets_change_abs": abs(tw - tp), "tickets_total": float(sum(cnt.values())),
    }
    return vals, {}


def _d_segment_delta(rows: list, p: dict) -> tuple:
    w, dim = int(p["week"]), p["dim"]
    net: dict = {}
    for r in rows:
        k = (int(r["iso_week"]), r[dim])
        if r.get("status") != "cancelled":
            net[k] = net.get(k, 0.0) + _f(r.get("gross"))
        else:
            net.setdefault(k, 0.0)
    segs = sorted({s for (_, s) in net})
    total = _f(p.get("net_delta"))
    deltas = {s: net.get((w, s), 0.0) - net.get((w - 1, s), 0.0) for s in segs}
    shares = {s: (max(0.0, deltas[s] / total) if total else 0.0) for s in segs}
    vals: dict = {}
    for s in segs:
        vals[f"net_abs_delta_{_san(s)}"] = _r(abs(deltas[s]))
        vals[f"share_pct_{_san(s)}"] = _r(shares[s] * 100, 1)
    top = max(segs, key=lambda s: (shares[s], s)) if segs else None
    vals["top_seg_net_abs_delta_eur"] = _r(abs(deltas[top])) if top is not None else 0.0
    vals["top_seg_share_pct"] = _r(shares[top] * 100, 1) if top is not None else 0.0
    return vals, {"top": top, "share": shares.get(top, 0.0) if top is not None else 0.0}


def _d_cancel_segments(rows: list, p: dict) -> tuple:
    w = int(p["week"])
    total = _f(p.get("net_delta"))
    seg: dict = {}
    for r in rows:
        key = (r.get("carrier"), r.get("cancel_reason"))
        s = seg.setdefault(key, {"w": 0.0, "prev": 0.0, "n": 0, "ids": []})
        wk = int(r["iso_week"])
        if wk == w:
            s["w"] += _f(r.get("order_value_eur"))
            s["n"] += 1
            s["ids"].append(r["order_id"])
        elif wk == w - 1:
            s["prev"] += _f(r.get("order_value_eur"))
    cancel_total_w = sum(s["w"] for s in seg.values())
    best, best_share = None, -1.0
    for key in sorted(seg, key=lambda k: (str(k[0]), str(k[1]))):
        s = seg[key]
        share = (-(s["w"] - s["prev"]) / total) if total else 0.0
        if share > best_share:
            best, best_share = key, share
    vals = {"cancel_total_w_eur": _r(cancel_total_w), "cand_cancel_w_eur": 0.0,
            "cand_cancel_prev_eur": 0.0, "cand_incr_abs_eur": 0.0, "cand_orders_w": 0.0,
            "cand_share_pct": 0.0}
    info: dict = {"carrier": None, "reason": None, "share": 0.0, "incr": 0.0, "ids": []}
    if best is not None:
        s = seg[best]
        share = max(0.0, best_share)
        ids = sorted(s["ids"])
        vals.update({
            "cand_cancel_w_eur": _r(s["w"]), "cand_cancel_prev_eur": _r(s["prev"]),
            "cand_incr_abs_eur": _r(abs(s["w"] - s["prev"])), "cand_orders_w": float(s["n"]),
            "cand_share_pct": _r(share * 100, 1),
        })
        info = {"carrier": best[0], "reason": best[1], "share": share,
                "incr": s["w"] - s["prev"], "ids": ids,
                "refs": ids[:20], "ref_total": len(ids)}
    return vals, info


def _d_shipments(rows: list, p: dict) -> tuple:
    c, w = p["carrier"], int(p["week"])
    ids = set(p.get("order_ids") or [])

    def late(r: dict) -> bool:
        return _f(r.get("delay_days")) >= LATE_DAYS

    def wk(r: dict) -> int:
        return int(r["ship_iso_week"])

    cw = [r for r in rows if r["carrier"] == c and wk(r) == w]
    cb = [r for r in rows if r["carrier"] == c and w - 4 <= wk(r) < w]
    ot = [r for r in rows if r["carrier"] != c and wk(r) == w]

    def share(l: list) -> float:
        return 100.0 * sum(1 for r in l if late(r)) / len(l) if l else 0.0

    def avg(l: list) -> float:
        return sum(_f(r.get("delay_days")) for r in l) / len(l) if l else 0.0

    late_ids = {r["order_id"] for r in rows if r["carrier"] == c and late(r)}
    refs = sorted(r["shipment_id"] for r in cw if late(r))
    vals = {
        "ship_total_w": float(len(cw)),
        "ship_late_w": float(sum(1 for r in cw if late(r))),
        "ship_late_share_w_pct": _r(share(cw), 1),
        "ship_avg_delay_w": _r(avg(cw), 1),
        "ship_late_share_base_pct": _r(share(cb), 1),
        "ship_avg_delay_base": _r(avg(cb), 1),
        "ship_late_share_other_pct": _r(share(ot), 1),
        "join_total_n": float(len(ids)),
        "join_late_n": float(len(ids & late_ids)),
    }
    return vals, {"refs": refs[:20], "ref_total": len(refs)}


def _d_tickets(rows: list, p: dict) -> tuple:
    w = int(p["week"])
    ids = set(p.get("order_ids") or [])
    cnt: dict = {}
    for r in rows:
        k = int(r["iso_week"])
        cnt[k] = cnt.get(k, 0) + 1
    bw = [k for k in range(w - 4, w) if k in cnt]
    base = _mean({k: float(v) for k, v in cnt.items()}, bw)
    tw = float(cnt.get(w, 0))
    linked = sorted(r["ticket_id"] for r in rows
                    if int(r["iso_week"]) == w and r.get("order_id") in ids)
    vals = {
        "tickets_w": tw, "tickets_base_mean": float(round(base)),
        "tickets_change_abs_pct": _r(abs(_pct(tw, base)), 1),
        "linked_w": float(len(linked)),
        "linked_share_pct": _r(100.0 * len(linked) / tw, 1) if tw else 0.0,
        "tickets_up": 1.0 if (tw >= 1.25 * base and tw - base >= 3) else 0.0,
    }
    return vals, {"refs": linked[:20], "ref_total": len(linked)}


def _d_category_tickets(rows: list, p: dict) -> tuple:
    w = int(p["week"])
    cnt: dict = {}
    weeks: set = set()
    for r in rows:
        k = (int(r["iso_week"]), r["category"])
        cnt[k] = cnt.get(k, 0) + int(_f(r.get("n")))
        weeks.add(int(r["iso_week"]))
    bw = [k for k in range(w - 4, w) if k in weeks]
    cats = sorted({c for (_, c) in cnt})
    wv = {c: float(cnt.get((w, c), 0)) for c in cats}
    bv = {c: (sum(cnt.get((k, c), 0) for k in bw) / len(bw) if bw else 0.0) for c in cats}
    total_incr = sum(wv.values()) - sum(bv.values())
    sign = 1 if total_incr >= 0 else -1
    inc = {c: (wv[c] - bv[c]) * sign for c in cats}
    spike = ([c for c in cats if inc[c] > 0 and inc[c] >= 0.25 * abs(total_incr)]
             if total_incr else [])
    spike_incr = sum(inc[c] for c in spike)
    share = min(1.0, spike_incr / abs(total_incr)) if total_incr else 0.0
    others = [c for c in cats if c not in spike and inc[c] > 0]
    other = max(others, key=lambda c: (inc[c], c)) if others else None
    vals = {
        "total_incr_abs": float(round(abs(total_incr))),
        "spike_incr_abs": float(round(spike_incr)),
        "spike_share_pct": _r(share * 100, 1),
        "spike_w": float(sum(wv[c] for c in spike)),
        "spike_base": float(round(sum(bv[c] for c in spike))),
        "other_incr_abs": float(round(inc[other])) if other else 0.0,
    }
    return vals, {"spike": spike, "share": share, "sign": sign, "other": other}


def _d_ticket_ids(rows: list, p: dict) -> tuple:
    ids = sorted({r["order_id"] for r in rows if r.get("order_id")})
    refs = sorted(r["ticket_id"] for r in rows)
    return ({"tickets_with_orders_n": float(len(rows)), "linked_order_ids_n": float(len(ids))},
            {"order_ids": ids, "refs": refs[:20], "ref_total": len(refs)})


def _d_linked_orders(rows: list, p: dict) -> tuple:
    by: dict = {}
    for r in rows:
        by.setdefault(r.get("carrier"), []).append(r)
    total = len(rows)
    top = max(sorted(by, key=str), key=lambda k: len(by[k])) if by else None
    n_top = len(by[top]) if top is not None else 0
    canc = sum(1 for r in by.get(top, []) if r.get("status") == "cancelled")
    share = (n_top / total) if total else 0.0
    ids = sorted(r["order_id"] for r in by.get(top, []))
    vals = {
        "linked_orders_n": float(total), "top_carrier_orders_n": float(n_top),
        "top_carrier_share_pct": _r(share * 100, 1), "top_carrier_cancelled_n": float(canc),
    }
    return vals, {"top": top, "share": share, "ids": ids, "refs": ids[:20], "ref_total": len(ids)}


DERIVERS: dict[str, Callable[[list, dict], tuple]] = {
    "weekly_revenue": _d_weekly_revenue,
    "weekly_tickets": _d_weekly_tickets,
    "segment_delta": _d_segment_delta,
    "cancel_segments": _d_cancel_segments,
    "shipments": _d_shipments,
    "tickets": _d_tickets,
    "category_tickets": _d_category_tickets,
    "ticket_ids": _d_ticket_ids,
    "linked_orders": _d_linked_orders,
}


def recompute_values(query: dict, rows: list) -> Optional[dict]:
    """Re-derive ``Evidence.values`` from freshly executed rows of ``Evidence.query``."""
    fn = DERIVERS.get((query or {}).get("derive"))
    if fn is None:
        return None
    return fn(rows, dict(query.get("params") or {}))[0]


def _confidence(share: float, n_sources: int, first_pass: bool) -> float:
    return round(0.4 * min(1.0, max(0.0, share)) + 0.3 * (min(n_sources, 3) / 3.0)
                 + 0.3 * (1.0 if first_pass else 0.5), 2)


# ---------------------------------------------------------------------------
# one investigation
# ---------------------------------------------------------------------------
class _Run:
    def __init__(self, ctx: RunContext, plan: Plan, retry: int) -> None:
        self.ctx = ctx
        self.plan = plan
        self.p = plan.parsed
        self.retry = retry
        self.evs: list[Evidence] = []
        self.claims: list[Claim] = []
        self.note = ""

    # ---- helpers -----------------------------------------------------------
    def rlabel(self) -> str:
        return "Overall" if self.p.region == "ALL" else self.p.region

    def ents(self, *extra: str) -> tuple:
        out = list(extra)
        if self.p.region != "ALL":
            out.append(f"region:{self.p.region}")
        out.append(f"week:{_wl(self.p.iso_year, self.p.iso_week)}")
        return tuple(out)

    def ev(self, source: str, description: str, spec: QuerySpec, derive: str,
           params: dict) -> tuple:
        res = self.ctx.query_source(AGENT, spec, purpose=description, retry_count=self.retry)
        values, info = DERIVERS[derive](res.rows, params)
        q = spec.as_dict()
        q["derive"] = derive
        q["params"] = params
        refs = list(info.get("refs", []))[:20]
        e = Evidence(
            evidence_id=f"E{len(self.evs) + 1}", source=source, description=description,
            query=q, query_text=res.query_text, values={k: float(v) for k, v in values.items()},
            rows=[dict(r) for r in res.rows[:50]], row_refs=refs,
            row_ref_total=int(info.get("ref_total", len(refs))), retrieved_at=utc_now_iso(),
        )
        self.evs.append(e)
        return e, info

    @staticmethod
    def N(refs: list, ev: Evidence, name: str, fmt: str) -> str:
        refs.append(NumberRef(ev.evidence_id, name, float(ev.values[name]), fmt))
        return fmt_number(ev.values[name], fmt)

    @staticmethod
    def NS(refs: list, ev: Evidence, name: str, fmt: str) -> str:
        """Cite a SIGNED value; the text shows its magnitude (direction is in the wording)."""
        refs.append(NumberRef(ev.evidence_id, name, float(ev.values[name]), fmt))
        return fmt_number(abs(ev.values[name]), fmt)

    @staticmethod
    def mk(ev: Evidence) -> str:
        return f"[{ev.evidence_id}]"

    def add_claim(self, kind: str, text: str, refs: list, **kw: Any) -> Claim:
        ids: list[str] = []
        for r in refs:
            if r.evidence_id not in ids:
                ids.append(r.evidence_id)
        c = Claim(claim_id=f"C{len(self.claims) + 1}", kind=kind, text=text,
                  numbers=list(refs), evidence_ids=ids, **kw)
        self.claims.append(c)
        return c

    def n_sources(self, ids: list) -> int:
        by = {e.evidence_id: e.source for e in self.evs}
        return len({by[i] for i in ids if i in by})

    # ---- shared claim builders ---------------------------------------------------
    def _observation(self, e1: Evidence, wow: str, base: str, subject: str) -> None:
        p = self.p
        refs: list = []
        text = (f"{subject} changed by only {self.N(refs, e1, wow, 'pct1')} versus the prior week and "
                f"{self.N(refs, e1, base, 'pct1')} versus the four-week mean, below the significance "
                f"threshold, so there is no significant change to explain {self.mk(e1)}.")
        self.add_claim("observation", text, refs,
                       entities=self.ents(f"metric:{p.metric}"), confidence=0.9)

    def _demand_ruled_out(self, e1: Evidence) -> bool:
        v = e1.values
        if v["gross_wow_abs_pct"] <= DEMAND_TOL and v["gross_base_abs_pct"] <= DEMAND_TOL:
            refs: list = []
            text = (f"Demand is not the cause: gross order value before cancellations changed by only "
                    f"{self.N(refs, e1, 'gross_wow_abs_pct', 'pct1')} versus the prior week and "
                    f"{self.N(refs, e1, 'gross_base_abs_pct', 'pct1')} versus the four-week mean "
                    f"{self.mk(e1)}.")
            self.add_claim("ruled_out", text, refs, entities=self.ents("metric:gross_order_value"),
                           confidence=0.9)
            return True
        return False

    # ---- revenue flow ---------------------------------------------------------------
    def _revenue(self) -> None:
        p, ctx = self.p, self.ctx
        w, y = p.iso_week, p.iso_year
        lo = max(1, w - 4)
        subject = f"{self.rlabel()} net revenue"
        spec1 = QuerySpec(
            "orders", filters=_filters("orders", p, lo, w), group_by=("iso_week", "status"),
            aggs=(Agg("sum", "order_value_eur", "gross"), Agg("count", "order_id", "n")),
            order_by=(("iso_week", "asc"), ("status", "asc")))
        e1, _ = self.ev("orders", f"Weekly net revenue, gross order value and cancellations "
                        f"({self.rlabel()}, weeks {lo}-{w})", spec1, "weekly_revenue", {"week": w})
        v = e1.values
        if v["orders_w"] <= 0:
            raise NoData(f"No orders found for region {p.region} in week {w} of {y}.")
        ctx.emit(AGENT, "compute_delta", f"{subject}, week {w} vs prior week and four-week mean",
                 f"week-over-week {v['net_wow_pct']}%, vs four-week mean {v['net_base_pct']}%",
                 target_agent="Critic")
        if not (v["net_wow_abs_pct"] >= SIGNIFICANT_DROP and v["net_base_abs_pct"] >= SIGNIFICANT_DROP):
            self._observation(e1, "net_wow_abs_pct", "net_base_abs_pct", subject)
            self._demand_ruled_out(e1)
            return

        neg = v["net_wow_pct"] < 0
        base_neg = v["net_base_pct"] < 0
        refs: list = []
        text = (f"{subject} was {self.N(refs, e1, 'net_w', 'eur0')} in week {w}, "
                f"{'down' if neg else 'up'} {self.NS(refs, e1, 'net_wow_pct', 'pct1')} from "
                f"{self.N(refs, e1, 'net_prev', 'eur0')} in week {w - 1} and "
                f"{self.N(refs, e1, 'net_base_abs_pct', 'pct1')} "
                f"{'below' if base_neg else 'above'} the four-week mean of "
                f"{self.N(refs, e1, 'net_base_mean', 'eur0')} {self.mk(e1)}.")
        self.add_claim("delta", text, refs, entities=self.ents(f"metric:{p.metric}"),
                       fact_key=f"delta:{p.metric}@{_wl(y, w)}", confidence=0.95)

        if not self._demand_ruled_out(e1) and v["gross_wow_abs_pct"] > DEMAND_TOL:
            refs = []
            text = (f"Gross order value before cancellations went "
                    f"{'down' if v['gross_wow_pct'] < 0 else 'up'} by "
                    f"{self.NS(refs, e1, 'gross_wow_pct', 'pct1')} versus the prior week, so part of "
                    f"the change is demand-side {self.mk(e1)}.")
            self.add_claim("delta", text, refs, entities=self.ents("metric:gross_order_value"),
                           fact_key=f"delta:demand@{_wl(y, w)}", confidence=0.9)

        # I3: decomposition by segment (decoys are ruled out here)
        for dim in ("category", "country"):
            spec = QuerySpec(
                "orders", filters=_filters("orders", p, max(1, w - 1), w),
                group_by=("iso_week", dim, "status"), aggs=(Agg("sum", "order_value_eur", "gross"),),
                limit=5000)
            e, info = self.ev("orders", f"Net revenue change by {dim} (week {w} vs prior week)", spec,
                              "segment_delta", {"week": w, "dim": dim, "net_delta": v["net_delta_eur"]})
            if info["top"] is not None and info["share"] < DOMINANCE:
                refs = []
                text = (f"{dim.capitalize()} '{info['top']}' is ruled out as the cause: it accounts for "
                        f"only {self.N(refs, e, 'top_seg_share_pct', 'pct1')} of the "
                        f"{self.N(refs, e1, 'net_change_abs_eur', 'eur0')} change in net revenue, below "
                        f"the dominance threshold {self.mk(e)}.")
                extra = (f"category:{info['top']}",) if dim == "category" else ()
                self.add_claim("ruled_out", text, refs, entities=self.ents(*extra), confidence=0.8)

        # I5: cancelled orders -> candidate segment
        spec5 = QuerySpec(
            "orders",
            filters=_filters("orders", p, max(1, w - 1), w, (Filter("status", "=", "cancelled"),)),
            select=("order_id", "iso_week", "carrier", "cancel_reason", "order_value_eur"),
            order_by=(("order_value_eur", "desc"),), limit=5000)
        e5, i5 = self.ev("orders", f"Cancelled orders by carrier and reason (week {w} vs prior week)",
                         spec5, "cancel_segments", {"week": w, "net_delta": v["net_delta_eur"]})
        if i5["carrier"] is None or i5["share"] < DOMINANCE:
            if i5["carrier"] is not None:
                refs = []
                text = (f"The largest cancellation segment (reason '{i5['reason']}' on carrier "
                        f"{i5['carrier']}) explains only {self.N(refs, e5, 'cand_share_pct', 'pct1')} of "
                        f"the {self.N(refs, e1, 'net_change_abs_eur', 'eur0')} change, below the "
                        f"dominance threshold {self.mk(e5)}.")
                self.add_claim("ruled_out", text, refs, entities=self.ents(), confidence=0.7)
            return

        cand, reason, share = i5["carrier"], i5["reason"], i5["share"]

        # carrier's orders (for the ticket join)
        spec_c = QuerySpec(
            "orders",
            filters=_filters("orders", p, max(1, w - 1), w, (Filter("carrier", "=", cand),)),
            select=("order_id",), limit=5000)
        res_c = ctx.query_source(AGENT, spec_c, purpose=f"order ids of carrier {cand}",
                                 retry_count=self.retry)
        carrier_ids = sorted(r["order_id"] for r in res_c.rows)

        # I6: shipments, in-app join with the cancelled order ids
        spec6 = QuerySpec(
            "shipments", filters=_filters("shipments", p, max(1, w - 4), w + 1),
            select=("shipment_id", "order_id", "carrier", "ship_iso_week", "delay_days"), limit=5000)
        e6, _ = self.ev("shipments", f"Shipment delays by carrier (weeks {max(1, w - 4)}-{w + 1}) joined to "
                        f"cancelled orders", spec6, "shipments",
                        {"carrier": cand, "week": w, "order_ids": i5["ids"]})

        # I7: tickets, in-app join with the carrier's orders
        spec7 = QuerySpec(
            "tickets", filters=_filters("tickets", p, lo, w),
            select=("ticket_id", "iso_week", "category", "order_id"), limit=5000)
        e7, _ = self.ev("tickets", f"Support tickets (weeks {lo}-{w}) joined to {cand} orders", spec7,
                        "tickets", {"week": w, "order_ids": carrier_ids})

        # I8: driver claim
        up = i5["incr"] >= 0
        refs = []
        t1 = (f"Cancelled orders with reason '{reason}' on carrier {cand} "
              f"{'rose' if up else 'fell'} from {self.N(refs, e5, 'cand_cancel_prev_eur', 'eur0')} in week "
              f"{w - 1} to {self.N(refs, e5, 'cand_cancel_w_eur', 'eur0')} in week {w}, which explains "
              f"{self.N(refs, e5, 'cand_share_pct', 'pct1')} of the "
              f"{self.N(refs, e1, 'net_change_abs_eur', 'eur0')} {'drop' if neg else 'rise'} in net "
              f"revenue {self.mk(e5)}.")
        t2 = (f"{cand} shipments were late (three or more days) in "
              f"{self.N(refs, e6, 'ship_late_share_w_pct', 'pct1')} of cases in week {w} versus "
              f"{self.N(refs, e6, 'ship_late_share_base_pct', 'pct1')} in the four prior weeks, and "
              f"{self.N(refs, e6, 'join_late_n', 'int')} of {self.N(refs, e6, 'join_total_n', 'int')} "
              f"affected orders had a late shipment {self.mk(e6)}.")
        parts = [t1, t2]
        v7 = e7.values
        if v7["tickets_up"] >= 1.0:
            parts.append(
                f"Support tickets rose to {self.N(refs, e7, 'tickets_w', 'int')} in week {w} (four-week "
                f"mean {self.N(refs, e7, 'tickets_base_mean', 'int')}), and "
                f"{self.N(refs, e7, 'linked_w', 'int')} of them "
                f"({self.N(refs, e7, 'linked_share_pct', 'pct1')}) are tied to {cand} orders "
                f"{self.mk(e7)}.")
        ids: list[str] = []
        for r in refs:
            if r.evidence_id not in ids:
                ids.append(r.evidence_id)
        conf = _confidence(share, self.n_sources(ids), self.retry == 0)
        self.add_claim("driver", " ".join(parts), refs,
                       entities=(f"carrier:{cand}",) + self.ents(),
                       fact_key=f"driver:carrier_{_san(reason)}@{_wl(y, w)}",
                       explained_share=round(share, 4), confidence=conf)
        self._hypothesis_note(cand)

    # ---- ticket flow ---------------------------------------------------------------
    def _tickets(self) -> None:
        p, ctx = self.p, self.ctx
        w, y = p.iso_week, p.iso_year
        lo = max(1, w - 4)
        subject = f"{self.rlabel()} support tickets"
        spec1 = QuerySpec("tickets", filters=_filters("tickets", p, lo, w), group_by=("iso_week",),
                          aggs=(Agg("count", "ticket_id", "n"),), order_by=(("iso_week", "asc"),))
        e1, _ = self.ev("tickets", f"Weekly support tickets ({self.rlabel()}, weeks {lo}-{w})", spec1,
                        "weekly_tickets", {"week": w})
        v = e1.values
        if v["tickets_total"] <= 0:
            raise NoData(f"No support tickets found for region {p.region} around week {w} of {y}.")
        ctx.emit(AGENT, "compute_delta", f"{subject}, week {w} vs prior week and four-week mean",
                 f"week-over-week {v['tickets_wow_pct']}%, vs four-week mean {v['tickets_base_pct']}%",
                 target_agent="Critic")
        if not (v["tickets_wow_abs_pct"] >= SIGNIFICANT_DROP
                and v["tickets_base_abs_pct"] >= SIGNIFICANT_DROP):
            self._observation(e1, "tickets_wow_abs_pct", "tickets_base_abs_pct", subject)
            return

        refs: list = []
        text = (f"{subject} were {self.N(refs, e1, 'tickets_w', 'int')} in week {w}, "
                f"{'down' if v['tickets_wow_pct'] < 0 else 'up'} "
                f"{self.NS(refs, e1, 'tickets_wow_pct', 'pct1')} from "
                f"{self.N(refs, e1, 'tickets_prev', 'int')} in week {w - 1} and "
                f"{self.N(refs, e1, 'tickets_base_abs_pct', 'pct1')} "
                f"{'below' if v['tickets_base_pct'] < 0 else 'above'} the four-week mean of "
                f"{self.N(refs, e1, 'tickets_base_mean', 'int')} {self.mk(e1)}.")
        self.add_claim("delta", text, refs, entities=self.ents(f"metric:{p.metric}"),
                       fact_key=f"delta:{p.metric}@{_wl(y, w)}", confidence=0.95)

        spec2 = QuerySpec("tickets", filters=_filters("tickets", p, lo, w),
                          group_by=("iso_week", "category"), aggs=(Agg("count", "ticket_id", "n"),),
                          limit=5000)
        e2, i2 = self.ev("tickets", f"Tickets by category (weeks {lo}-{w})", spec2,
                         "category_tickets", {"week": w})
        direction = "above" if i2["sign"] >= 0 else "below"
        if i2["other"] is not None:
            refs = []
            text = (f"Category '{i2['other']}' changed by only "
                    f"{self.N(refs, e2, 'other_incr_abs', 'int')} tickets versus its four-week mean, so it "
                    f"is not a main driver {self.mk(e2)}.")
            self.add_claim("ruled_out", text, refs, entities=self.ents(f"category:{i2['other']}"),
                           confidence=0.8)
        if not i2["spike"]:
            return

        spike = tuple(i2["spike"])
        spec3 = QuerySpec(
            "tickets",
            filters=_filters("tickets", p, w, w, (Filter("category", "in", spike),
                                                  Filter("order_id", "not_null"))),
            select=("ticket_id", "order_id", "category", "iso_week"), limit=5000)
        e3, i3 = self.ev("tickets", f"Week {w} tickets in the changed categories with an order id",
                         spec3, "ticket_ids", {"week": w})
        order_ids = tuple(i3["order_ids"])
        if not order_ids:
            return
        spec4 = QuerySpec("orders", filters=(Filter("order_id", "in", order_ids),),
                          select=("order_id", "carrier", "status", "cancel_reason", "order_value_eur"),
                          limit=5000)
        e4, i4 = self.ev("orders", "Orders behind the changed tickets, by carrier and status", spec4,
                         "linked_orders", {})
        if i4["top"] is None:
            return
        top, cshare = i4["top"], i4["share"]
        if cshare < DOMINANCE:
            refs = []
            text = (f"No single carrier dominates the orders behind these tickets: the largest, {top}, "
                    f"holds only {self.N(refs, e4, 'top_carrier_share_pct', 'pct1')} of "
                    f"{self.N(refs, e4, 'linked_orders_n', 'int')} linked orders {self.mk(e4)}.")
            self.add_claim("ruled_out", text, refs, entities=self.ents(), confidence=0.7)
            return

        spec5 = QuerySpec(
            "shipments", filters=_filters("shipments", p, max(1, w - 4), w + 1),
            select=("shipment_id", "order_id", "carrier", "ship_iso_week", "delay_days"), limit=5000)
        e5, _ = self.ev("shipments", f"Shipment delays for carrier {top} (weeks {max(1, w - 4)}-{w + 1})",
                        spec5, "shipments", {"carrier": top, "week": w, "order_ids": i4["ids"]})

        cats = " and ".join(f"'{c}'" for c in spike)
        refs = []
        t1 = (f"The change is concentrated in the {cats} ticket categories: "
              f"{self.N(refs, e2, 'spike_incr_abs', 'int')} tickets {direction} the four-week mean, "
              f"{self.N(refs, e2, 'spike_share_pct', 'pct1')} of the total change {self.mk(e2)}.")
        t2 = (f"{self.N(refs, e4, 'top_carrier_orders_n', 'int')} of the "
              f"{self.N(refs, e4, 'linked_orders_n', 'int')} orders behind these tickets "
              f"({self.N(refs, e4, 'top_carrier_share_pct', 'pct1')}) belong to carrier {top}, and "
              f"{self.N(refs, e4, 'top_carrier_cancelled_n', 'int')} of those orders were cancelled "
              f"{self.mk(e4)}.")
        t3 = (f"{top} shipments were late (three or more days) in "
              f"{self.N(refs, e5, 'ship_late_share_w_pct', 'pct1')} of cases in week {w} versus "
              f"{self.N(refs, e5, 'ship_late_share_base_pct', 'pct1')} in the four prior weeks "
              f"{self.mk(e5)}.")
        ids: list[str] = []
        for r in refs:
            if r.evidence_id not in ids:
                ids.append(r.evidence_id)
        explained = i2["share"] * cshare
        conf = _confidence(explained, self.n_sources(ids), self.retry == 0)
        self.add_claim("driver", " ".join((t1, t2, t3)), refs,
                       entities=(f"carrier:{top}",) + self.ents(),
                       fact_key=f"driver:ticket_carrier@{_wl(y, w)}",
                       explained_share=round(explained, 4), confidence=conf)
        self._hypothesis_note(top)

    # ---- memory hypothesis (trace only; memory is never evidence) -------------------------
    def _hypothesis_note(self, carrier: str) -> None:
        recalled = getattr(self.ctx, "recalled_memory", None) or []
        hyp = [str(getattr(r, "entity", "")).split(":", 1)[1] for r in recalled
               if str(getattr(r, "entity", "")).startswith("carrier:")]
        if hyp:
            self.note = (f"; memory hypothesis carrier {hyp[0]}: "
                         + ("supported by current data" if hyp[0] == carrier
                            else f"not supported (data points at {carrier})"))

    # ---- entry -----------------------------------------------------------------------------
    def go(self) -> tuple:
        if self.p.metric == "ticket_count":
            self._tickets()
        else:
            self._revenue()
        kinds = ",".join(c.kind for c in self.claims)
        drivers = [c.confidence for c in self.claims if c.kind == "driver"]
        self.ctx.emit(AGENT, "propose_claims", f"{len(self.evs)} evidence item(s)",
                      f"{len(self.claims)} claim(s): {kinds}{self.note}",
                      target_agent="Critic", retry_count=self.retry,
                      confidence=max(drivers) if drivers else None)
        return self.evs, self.claims


class Investigator:
    name = AGENT

    def run(self, ctx: RunContext, plan: Plan) -> tuple:
        evidence, claims = _Run(ctx, plan, 0).go()
        # remembered so refine() can keep claims it cannot repair
        ctx.last_claims = claims
        return evidence, claims

    def refine(self, ctx: RunContext, plan: Plan, evidence: list, rejections: list) -> tuple:
        rnd = ctx.counters.get("refinements", 0) + 1
        ctx.counters["refinements"] = rnd
        old: list = list(getattr(ctx, "last_claims", None) or [])
        by_id = {c.claim_id: c for c in old}
        rebuild = [r for r in rejections if r.code in REBUILD_CODES]
        needs = "; ".join(sorted({r.needs for r in rejections if r.needs}))[:200]
        ctx.emit(AGENT, "refine", f"{len(rejections)} rejection(s): "
                 + ",".join(sorted({r.code for r in rejections})),
                 f"{'re-running queries' if (rebuild or not old) else 'no repair possible'}"
                 f"{'; needs: ' + needs if needs else ''}",
                 target_agent="Critic", retry_count=rnd)
        if not rebuild and old:
            ctx.last_claims = old
            return list(evidence), old
        new_ev, new_claims = _Run(ctx, plan, rnd).go()
        new_texts = {c.text for c in new_claims}
        new_keys = {(c.kind, c.fact_key) for c in new_claims if c.fact_key}
        # every rejected claim that was not re-grounded is carried forward unchanged
        kept: list = []
        seen_ids: set = set()
        for r in rejections:
            c = by_id.get(r.claim_id)
            if c is None or c.claim_id in seen_ids:
                continue
            seen_ids.add(c.claim_id)
            if c.text in new_texts:
                continue
            if c.fact_key and (c.kind, c.fact_key) in new_keys:
                continue
            kept.append(c)
        taken = {c.claim_id for c in new_claims}
        n = len(new_claims)
        for c in kept:
            if c.claim_id in taken:
                n += 1
                while f"C{n}" in taken:
                    n += 1
                c.claim_id = f"C{n}"
            taken.add(c.claim_id)
        claims = list(new_claims) + kept
        ctx.last_claims = claims
        return new_ev, claims


__all__ = ["Investigator", "DERIVERS", "recompute_values", "fmt_number",
           "SIGNIFICANT_DROP", "DOMINANCE", "DEMAND_TOL"]
