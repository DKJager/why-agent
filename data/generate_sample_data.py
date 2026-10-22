#!/usr/bin/env python3
"""Generate the FICTIONAL synthetic dataset used by the Why-Agent.

Everything produced here is synthetic: "Northwind-Lite", all customers, orders,
carriers (NordExpress, RheinFracht, ...) and tickets are invented. No real
company, person or carrier is represented. The output is CC0.

Usage:
    python data/generate_sample_data.py [OUT_DIR]

The generator is seeded and uses only the stdlib ``random`` module, so it
reproduces the committed CSVs byte-for-byte.

Planted story (see DESIGN.md section 5.2):
  * A fictional NordExpress hub outage (Rotterdam) from 2026-10-01 to 2026-10-10.
  * NordExpress EU orders placed in ISO week 41 are mostly cancelled with
    cancel_reason=delivery_delay; support tickets about delivery spike.
  * EU demand (gross order value) is unchanged: the drop is fulfilment.
  * Decoys: EU 'home' category about -4% in week 41; UK payment_issue tickets +3.
  * Control: UK and US stay flat.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from datetime import date, datetime, timedelta

SEED = 20261022
FIRST_MONDAY = date(2026, 8, 24)  # Monday of ISO week 35, 2026
WEEKS = list(range(35, 43))
DATA_AS_OF = date(2026, 10, 18)
AS_OF_END = datetime(2026, 10, 18, 23, 59, 59)

CULPRIT = "NordExpress"
OUTAGE_START = date(2026, 10, 1)
OUTAGE_END = date(2026, 10, 10)
WEEK40_OUTAGE_END = date(2026, 10, 4)

DOW_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.1, 0.8, 0.7]

CATEGORIES = [
    ("home", 0.25, 45.0),
    ("electronics", 0.15, 120.0),
    ("apparel", 0.25, 40.0),
    ("accessories", 0.20, 25.0),
    ("outdoor", 0.15, 70.0),
]
CAT_NAMES = [c[0] for c in CATEGORIES]
CAT_W = [c[1] for c in CATEGORIES]
CAT_PRICE = {c[0]: c[2] for c in CATEGORIES}

REGIONS = {
    "EU": {
        "base": 150,
        "gross": 10500.0,
        "countries": ["DE", "FR", "NL", "ES", "IT"],
        "country_w": [0.30, 0.25, 0.15, 0.15, 0.15],
        "carriers": ["NordExpress", "RheinFracht", "AlpenPost"],
        "carrier_w": [0.45, 0.30, 0.25],
        "transit": 3,
    },
    "UK": {
        "base": 45,
        "gross": 3400.0,
        "countries": ["GB"],
        "country_w": [1.0],
        "carriers": ["BritParcel", "IslePost"],
        "carrier_w": [0.70, 0.30],
        "transit": 2,
    },
    "US": {
        "base": 55,
        "gross": 4400.0,
        "countries": ["US"],
        "country_w": [1.0],
        "carriers": ["StarFreight", "PrairieExpress"],
        "carrier_w": [0.60, 0.40],
        "transit": 4,
    },
}

CARRIER_HUB = {
    "NordExpress": "Rotterdam",
    "RheinFracht": "Leipzig",
    "AlpenPost": "Munich",
    "BritParcel": "Birmingham",
    "IslePost": "Leeds",
    "StarFreight": "Memphis",
    "PrairieExpress": "Dallas",
}

BASE_CANCEL_RATE = 0.03
CANCEL_REASONS = ["customer_request", "payment_failed", "out_of_stock", "delivery_delay"]
CANCEL_REASON_W = [0.40, 0.30, 0.20, 0.10]

ORDER_COLS = [
    "order_id", "order_ts", "order_date", "iso_year", "iso_week", "region", "country",
    "customer_id", "category", "sku", "qty", "unit_price_eur", "order_value_eur",
    "carrier", "status", "cancel_reason", "status_updated_ts",
]
SHIPMENT_COLS = [
    "shipment_id", "order_id", "carrier", "region", "origin_hub", "ship_date",
    "ship_iso_year", "ship_iso_week", "promised_date", "delivered_date", "delay_days", "status",
]
TICKET_COLS = [
    "ticket_id", "created_ts", "created_date", "iso_year", "iso_week", "region", "country",
    "order_id", "category", "priority", "status", "sentiment",
]


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _week_monday(week: int) -> date:
    return FIRST_MONDAY + timedelta(days=7 * (week - 35))


# --------------------------------------------------------------------------
# orders
# --------------------------------------------------------------------------
def gen_orders(rng: random.Random) -> list[dict]:
    orders: list[dict] = []
    for week in WEEKS:
        monday = _week_monday(week)
        for region, cfg in REGIONS.items():
            n = cfg["base"] + rng.randint(-3, 3)
            target = cfg["gross"] * (1 + rng.uniform(-0.012, 0.012))
            batch: list[dict] = []
            for _ in range(n):
                day = rng.choices(range(7), weights=DOW_WEIGHTS)[0]
                d = monday + timedelta(days=day)
                dt = datetime(d.year, d.month, d.day) + timedelta(seconds=rng.randint(0, 86399))
                country = rng.choices(cfg["countries"], weights=cfg["country_w"])[0]
                category = rng.choices(CAT_NAMES, weights=CAT_W)[0]
                sku = f"{category[:3].upper()}-{rng.randint(1, 40):03d}"
                qty = rng.choices([1, 2, 3], weights=[0.70, 0.22, 0.08])[0]
                price = CAT_PRICE[category] * rng.uniform(0.85, 1.15)
                carrier = rng.choices(cfg["carriers"], weights=cfg["carrier_w"])[0]
                customer = rng.randint(1, 600)
                iso = d.isocalendar()
                batch.append({
                    "ts": dt,
                    "date": d,
                    "order_ts": _ts(dt),
                    "order_date": d.isoformat(),
                    "iso_year": iso[0],
                    "iso_week": iso[1],
                    "region": region,
                    "country": country,
                    "customer_id": f"C-{customer:05d}",
                    "category": category,
                    "sku": sku,
                    "qty": qty,
                    "price": price,
                    "carrier": carrier,
                    "status": "delivered",
                    "cancel_reason": "",
                })
            raw = sum(o["qty"] * o["price"] for o in batch)
            scale = target / raw
            for o in batch:
                p = round(o["price"] * scale, 2)
                if region == "EU" and week == 41 and o["category"] == "home":
                    p = round(p * 0.96, 2)  # decoy 1: organic noise
                o["unit_price_eur"] = p
                o["order_value_eur"] = round(o["qty"] * p, 2)
            orders.extend(batch)

    orders.sort(key=lambda o: o["order_ts"])
    for i, o in enumerate(orders, start=1):
        o["num"] = i
        o["order_id"] = f"O-{i:06d}"
    return orders


def _cancel(o: dict, reason: str) -> None:
    o["status"] = "cancelled"
    o["cancel_reason"] = reason


def apply_cancellations(rng: random.Random, orders: list[dict]) -> None:
    groups: dict[tuple[str, int], list[dict]] = {}
    for o in orders:
        groups.setdefault((o["region"], o["iso_week"]), []).append(o)
    for key in sorted(groups):
        region, week = key
        grp = groups[key]
        excluded: set[str] = set()
        if region == "EU" and week == 41:
            ne = [o for o in grp if o["carrier"] == CULPRIT]
            k = round(rng.uniform(0.57, 0.63) * len(ne))
            for o in rng.sample(ne, k):
                _cancel(o, "delivery_delay")
            excluded = {o["order_id"] for o in ne}
        elif region == "EU" and week == 40:
            ne = [o for o in grp if o["carrier"] == CULPRIT
                  and OUTAGE_START <= o["date"] <= WEEK40_OUTAGE_END]
            k = round(rng.uniform(0.10, 0.15) * len(ne))
            for o in rng.sample(ne, k):
                _cancel(o, "delivery_delay")
            excluded = {o["order_id"] for o in ne}
        eligible = [o for o in grp if o["order_id"] not in excluded]
        c = round(BASE_CANCEL_RATE * len(eligible))
        for o in rng.sample(eligible, c):
            _cancel(o, rng.choices(CANCEL_REASONS, weights=CANCEL_REASON_W)[0])


# --------------------------------------------------------------------------
# shipments (+ final order status)
# --------------------------------------------------------------------------
def build_shipments(rng: random.Random, orders: list[dict]) -> list[dict]:
    shipments: list[dict] = []
    for o in orders:
        region = o["region"]
        od = o["date"]
        ship = min(od + timedelta(days=rng.randint(0, 1)), DATA_AS_OF)
        promised = ship + timedelta(days=REGIONS[region]["transit"])
        outage = (o["carrier"] == CULPRIT and region == "EU"
                  and OUTAGE_START <= ship <= OUTAGE_END)
        planned = rng.randint(4, 9) if outage else rng.randint(0, 1)
        delivered = promised + timedelta(days=planned)
        undelivered = delivered > DATA_AS_OF
        # DESIGN 5.1: delay = (delivered_date or data_as_of) - promised_date, floored at 0
        end = DATA_AS_OF if undelivered else delivered
        delay = max(0, (end - promised).days)
        if undelivered:
            delivered_str = ""
            status = "held" if outage else "in_transit"
        else:
            delivered_str = delivered.isoformat()
            status = "delivered"
        iso = ship.isocalendar()
        shipments.append({
            "shipment_id": f"S-{o['num']:06d}",
            "order_id": o["order_id"],
            "carrier": o["carrier"],
            "region": region,
            "origin_hub": CARRIER_HUB[o["carrier"]],
            "ship_date": ship.isoformat(),
            "ship_iso_year": iso[0],
            "ship_iso_week": iso[1],
            "promised_date": promised.isoformat(),
            "delivered_date": delivered_str,
            "delay_days": delay,
            "status": status,
        })
        # final order status + last update timestamp
        if o["status"] == "cancelled":
            upd = o["ts"] + timedelta(hours=rng.randint(12, 96))
        elif delivered_str:
            o["status"] = "delivered"
            upd = datetime(delivered.year, delivered.month, delivered.day, 9) + timedelta(
                minutes=rng.randint(0, 540))
        else:
            o["status"] = "shipped"
            upd = o["ts"] + timedelta(hours=rng.randint(2, 30))
        if upd > AS_OF_END:
            upd = AS_OF_END
        o["status_updated_ts"] = _ts(upd)
    return shipments


# --------------------------------------------------------------------------
# tickets
# --------------------------------------------------------------------------
DELAY_CATS = ("delivery_delay", "where_is_my_order")


def gen_tickets(rng: random.Random, orders: list[dict]) -> list[dict]:
    by_rw: dict[tuple[str, int], list[dict]] = {}
    for o in orders:
        by_rw.setdefault((o["region"], o["iso_week"]), []).append(o)
    spike_pool = [o for o in orders if o["region"] == "EU" and o["carrier"] == CULPRIT
                  and date(2026, 9, 30) <= o["date"] <= date(2026, 10, 11)]
    tickets: list[dict] = []

    def add(region: str, d: date, category: str, order: dict | None) -> None:
        secs = rng.randint(0, 86399)
        dt = datetime(d.year, d.month, d.day) + timedelta(seconds=secs)
        if category in DELAY_CATS:
            priority = rng.choices(["low", "normal", "high"], weights=[0.2, 0.5, 0.3])[0]
            sentiment = rng.choices(["negative", "neutral", "positive"], weights=[0.70, 0.25, 0.05])[0]
        elif category == "payment_issue":
            priority = rng.choices(["low", "normal", "high"], weights=[0.2, 0.5, 0.3])[0]
            sentiment = rng.choices(["negative", "neutral", "positive"], weights=[0.5, 0.4, 0.1])[0]
        else:
            priority = rng.choices(["low", "normal", "high"], weights=[0.5, 0.4, 0.1])[0]
            sentiment = rng.choices(["negative", "neutral", "positive"], weights=[0.25, 0.5, 0.25])[0]
        recent = d >= date(2026, 10, 12)
        status = "open" if rng.random() < (0.7 if recent else 0.1) else "closed"
        if order is not None:
            country = order["country"]
        else:
            cfg = REGIONS[region]
            country = rng.choices(cfg["countries"], weights=cfg["country_w"])[0]
        iso = d.isocalendar()
        tickets.append({
            "ts": dt,
            "created_ts": _ts(dt),
            "created_date": d.isoformat(),
            "iso_year": iso[0],
            "iso_week": iso[1],
            "region": region,
            "country": country,
            "order_id": order["order_id"] if order is not None else "",
            "category": category,
            "priority": priority,
            "status": status,
            "sentiment": sentiment,
        })

    def pick_date(o: dict, monday: date, sunday: date) -> date:
        start = max(o["date"], monday)
        return start + timedelta(days=rng.randint(0, (sunday - start).days))

    for week in WEEKS:
        monday = _week_monday(week)
        sunday = monday + timedelta(days=6)
        # ---- EU
        pool = by_rw[("EU", week)]
        n_delay = rng.randint(43, 47) if week == 41 else rng.randint(6, 10)
        for _ in range(n_delay):
            cat = rng.choice(DELAY_CATS)
            if week == 41:
                o = rng.choice(spike_pool)
            else:
                o = rng.choice(pool)
            add("EU", pick_date(o, monday, sunday), cat, o)
        for _ in range(rng.randint(4, 7)):
            cat = rng.choices(["refund_request", "product_question", "payment_issue"],
                              weights=[0.35, 0.40, 0.25])[0]
            o = rng.choice(pool) if rng.random() < 0.5 else None
            d = pick_date(o, monday, sunday) if o is not None else monday + timedelta(days=rng.randint(0, 6))
            add("EU", d, cat, o)
        # ---- UK / US
        for region, lo, hi in (("UK", 4, 7), ("US", 5, 8)):
            pool = by_rw[(region, week)]
            for _ in range(rng.randint(lo, hi)):
                cat = rng.choices(
                    ["delivery_delay", "where_is_my_order", "refund_request", "product_question", "payment_issue"],
                    weights=[0.15, 0.15, 0.25, 0.30, 0.15])[0]
                if cat in DELAY_CATS or rng.random() < 0.5:
                    o = rng.choice(pool)
                    d = pick_date(o, monday, sunday)
                else:
                    o = None
                    d = monday + timedelta(days=rng.randint(0, 6))
                add(region, d, cat, o)
            if region == "UK" and week == 41:  # decoy 2
                for _ in range(3):
                    add("UK", monday + timedelta(days=rng.randint(0, 6)), "payment_issue", None)

    tickets.sort(key=lambda t: t["created_ts"])
    for i, t in enumerate(tickets, start=1):
        t["ticket_id"] = f"T-{i:06d}"
    return tickets


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _write_csv(path: str, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in cols])


README_TEXT = """# Sample data

FICTIONAL SYNTHETIC DATA, no licence restrictions (CC0).

"Northwind-Lite", every customer, order, carrier and ticket in these files is
invented. No real company, person or carrier is represented.

Contents: `orders.csv`, `shipments.csv`, `tickets.csv` covering ISO weeks 35-42
of 2026 (2026-08-24 to 2026-10-18, data_as_of 2026-10-18).

Regenerate (byte-identical) with:

    python data/generate_sample_data.py
"""


def generate(out_dir: str) -> dict[str, str]:
    rng = random.Random(SEED)
    orders = gen_orders(rng)
    apply_cancellations(rng, orders)
    shipments = build_shipments(rng, orders)
    tickets = gen_tickets(rng, orders)
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "orders": os.path.join(out_dir, "orders.csv"),
        "shipments": os.path.join(out_dir, "shipments.csv"),
        "tickets": os.path.join(out_dir, "tickets.csv"),
    }
    _write_csv(paths["orders"], ORDER_COLS, orders)
    _write_csv(paths["shipments"], SHIPMENT_COLS, shipments)
    _write_csv(paths["tickets"], TICKET_COLS, tickets)
    with open(os.path.join(out_dir, "README.md"), "w", newline="", encoding="utf-8") as fh:
        fh.write(README_TEXT)
    return paths


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = argv[0] if argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")
    paths = generate(out_dir)
    for name, p in paths.items():
        print(f"wrote {name}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
