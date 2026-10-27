"""H-MEM conformance runner, ported check-for-check from Meterless/H-MEM
``conformance/runner.ts`` and driven by its ``vectors.json`` (version 1, verbatim).

    python -m agent.hmem.conformance.runner

Exit codes match the original: 0 all checks pass, 1 a check failed,
2 the implementation could not be loaded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
VECTORS = HERE / "vectors.json"
DAY = 24 * 60 * 60 * 1000


def _approx(a, b, eps=1e-9) -> bool:
    return a is not None and abs(a - b) < eps


def run_checks(HMEM) -> list[dict]:
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    T0 = vectors["clock"]
    S = vectors["sections"]
    results: list[dict] = []
    ids: set[str] = set()

    def record(section: str, cid: str, desc: str, fn: Callable[[], None]) -> None:
        if cid in ids:
            raise RuntimeError(f"duplicate case id {cid}")
        ids.add(cid)
        try:
            fn()
            results.append({"id": cid, "section": section, "description": desc, "status": "PASS"})
        except Exception as err:  # noqa: BLE001 - a failed check, not a crash
            results.append({"id": cid, "section": section, "description": desc, "status": "FAIL",
                            "detail": f"{type(err).__name__}: {err}"})

    def check(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    for c in S["record-validation"]:
        def f(c=c):
            h = HMEM(clock=lambda: T0)
            rejected = False
            try:
                h.add(content=c["input"]["content"], source=c["input"]["source"])
            except Exception:  # noqa: BLE001
                rejected = True
            check(rejected == (c["expect"] == "reject"),
                  f"expected {c['expect']}, got {'reject' if rejected else 'accept'}")
        record("record-validation", c["id"], c["description"], f)

    for c in S["feedback-deltas"]:
        def f(c=c):
            h = HMEM(clock=lambda: T0)
            m = h.add(content="feedback target record", confidence=c["start"], source="conformance")
            before = m.access_count
            h.feedback(m.id, c["kind"])
            after = h.store.get(m.id)
            check(_approx(after.confidence, c["expectConfidence"]),
                  f"confidence {after.confidence}, expected {c['expectConfidence']}")
            if "expectAccessDelta" in c:
                check(after.access_count - before == c["expectAccessDelta"],
                      f"accessCount delta {after.access_count - before}, expected {c['expectAccessDelta']}")
            if c.get("expectTag"):
                check(c["expectTag"] in after.tags, f"missing tag {c['expectTag']}")
        record("feedback-deltas", c["id"], c["description"], f)

    for c in S["ranking-order"]:
        def f(c=c):
            h = HMEM(clock=lambda: T0)
            by = {}
            for r in c["records"]:
                by[r["key"]] = h.add(content=r["content"], confidence=r["confidence"], layer=r["layer"],
                                     source="conformance")
            for r in c["records"]:
                if r.get("supersededByKey"):
                    by[r["key"]].superseded_by = by[r["supersededByKey"]].id
            res = h.query(c["query"], threshold=0, topN=10)
            order = [x["memory"].id for x in res["memories"]]
            exp = [by[k].id for k in c["expectOrder"]]
            for i in range(len(exp) - 1):
                check(exp[i] in order and exp[i + 1] in order
                      and order.index(exp[i]) < order.index(exp[i + 1]),
                      f"expected {c['expectOrder'][i]} before {c['expectOrder'][i + 1]}, got {order}")
        record("ranking-order", c["id"], c["description"], f)

    for c in S["sleep-guardrails"]:
        def f(c=c):
            now = {"t": T0}
            h = HMEM(clock=lambda: now["t"])
            by = {}
            for r in c["records"]:
                by[r["key"]] = h.add(content=r["content"], source=r["source"],
                                     derived_from=r.get("derivedFrom"))
            now["t"] = T0 + c["ageDays"] * DAY
            before = sorted(m.id for m in h.store.all())
            preview = h.sleep.preview()
            check(sorted(m.id for m in h.store.all()) == before, "preview mutated the store")
            for k in c["expectArchived"]:
                check(by[k].id in preview["toArchive"], f"{k} should be archivable")
            for k in c["expectProtected"]:
                check(by[k].id not in preview["toArchive"], f"{k} is guardrail-protected and must not archive")
            report = h.sleep.execute(preview)
            for k in c["expectProtected"]:
                check(h.store.get(by[k].id) is not None, f"{k} was deleted despite guardrails")
            check(isinstance(report.get("backupId"), str) and len(report["backupId"]) > 0,
                  "execute must return a backupId")
        record("sleep-guardrails", c["id"], c["description"], f)

    for c in S["conflict-gate"]:
        def f(c=c):
            h = HMEM(clock=lambda: T0)
            a = h.add(content=c["a"]["content"], confidence=c["a"]["confidence"], source="conformance")
            b = h.add(content=c["b"]["content"], confidence=c["b"]["confidence"], source="conformance")
            found = h.conflicts.scan()
            check(len(found) > 0, "opposing always/never pair on similar content must be detected")
            outcome = h.conflicts.autoResolve(found[0])
            check(outcome in ("resolved", "queued"), f"outcome must be resolved or queued, got {outcome}")
            if outcome == "resolved":
                loser = [m for m in (h.store.get(a.id), h.store.get(b.id)) if m and m.superseded_by]
                check(len(loser) > 0, "resolved conflicts must mark the loser supersededBy")
            check(h.store.get(a.id) is not None and h.store.get(b.id) is not None,
                  "conflict resolution must never delete records")
        record("conflict-gate", c["id"], c["description"], f)

    for c in S["ledger-audit"]:
        def f(c=c):
            h = HMEM(clock=lambda: T0)
            m = h.add(content="audited record", confidence=c["start"], source="conformance")
            h.feedback(m.id, c["kind"])
            hist = h.ledger.history(m.id)
            actions = [e["action"] for e in hist]
            check(actions == c["expectActions"], f"actions {actions}, expected {c['expectActions']}")
            fb = hist[-1]
            check(_approx((fb.get("previousState") or {}).get("confidence"), c["expectPrev"]),
                  f"previousState.confidence {(fb.get('previousState') or {}).get('confidence')}, expected {c['expectPrev']}")
            check(_approx((fb.get("newState") or {}).get("confidence"), c["expectNew"]),
                  f"newState.confidence {(fb.get('newState') or {}).get('confidence')}, expected {c['expectNew']}")
        record("ledger-audit", c["id"], c["description"], f)

    for c in S["dream-boundary"]:
        def f(c=c):
            h = HMEM(clock=lambda: T0)
            for r in c["records"]:
                h.add(content=r["content"], type=r["type"], source="conformance")
            before = len(h.store.all())
            proposals = h.dreaming.dream()
            check(len(proposals) > 0, "two related preferences must produce at least one proposal")
            check(len(h.store.all()) == before, "dreaming must not materialize anything before approval")
            created = h.dreaming.approve(proposals[0]["id"])
            if created:
                check(created.layer == "long_term", "approved proposals materialize as long_term")
                check(len(created.derived_from or []) > 0, "approved proposals carry derivedFrom lineage")
        record("dream-boundary", c["id"], c["description"], f)

    return results


def main() -> int:
    try:
        from agent.hmem import HMEM
    except Exception as err:  # noqa: BLE001
        print(f"conformance: could not load implementation: {err}", file=sys.stderr)
        return 2
    results = run_checks(HMEM)
    passed = sum(r["status"] == "PASS" for r in results)
    print("\nH-MEM conformance: implementation = agent.hmem (Python)")
    section = ""
    for r in results:
        if r["section"] != section:
            section = r["section"]
            print(f"\n  {section}")
        print(f"    {r['status']} {r['id']} {r['description']}"
              + (f"\n         {r['detail']}" if r.get("detail") else ""))
    print(f"\n  {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    sys.exit(main())
