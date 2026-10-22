# build-id: oa-fbc7a467
"""T15: the generator is byte-reproducible and the planted-data invariants hold.

All invariants are recomputed with the plain csv module from the shipped CSVs.
"""
import ast
import csv
import os
import shutil
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample"
NAMES = ("orders.csv", "tickets.csv", "shipments.csv")
AS_OF = date(2026, 10, 18)
DELAY_CATS = {"delivery_delay", "where_is_my_order"}


def _rows(name, d=DATA):
    with open(Path(d) / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def orders():
    return _rows("orders.csv")


@pytest.fixture(scope="module")
def shipments():
    return _rows("shipments.csv")


@pytest.fixture(scope="module")
def tickets():
    return _rows("tickets.csv")


def _net(orders, region, week, gross=False):
    return sum(float(r["order_value_eur"]) for r in orders
               if r["region"] == region and int(r["iso_year"]) == 2026 and int(r["iso_week"]) == week
               and (gross or r["status"] != "cancelled"))


def _baseline(orders, region, gross=False):
    return statistics.mean(_net(orders, region, w, gross) for w in (37, 38, 39, 40))


# ------------------------------------------------------------ reproducibility

def test_generator_regenerates_byte_identically(tmp_path):
    work = tmp_path / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "logs", ".hmem", ".pytest_cache", "node_modules", ".venv", "venv", "*.pyc"))
    for n in NAMES:
        (work / "data" / "sample" / n).unlink()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SAMPLE_MODE", "DATA_DIR"))}
    proc = subprocess.run([sys.executable, "data/generate_sample_data.py"], cwd=work, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for n in NAMES:
        regenerated = work / "data" / "sample" / n
        assert regenerated.exists(), f"generator did not write {n}"
        assert regenerated.read_bytes() == (DATA / n).read_bytes(), f"{n} differs after regeneration"


def test_generator_is_stdlib_only():
    tree = ast.parse((ROOT / "data" / "generate_sample_data.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    assert "random" in mods
    assert not (mods - set(sys.stdlib_module_names)), mods - set(sys.stdlib_module_names)


def test_data_readme_labels_fictional():
    text = (DATA / "README.md").read_text(encoding="utf-8")
    assert "FICTIONAL" in text
    assert "CC0" in text
    assert "generate_sample_data.py" in text


# ------------------------------------------------------------ schema

def test_columns_present(orders, shipments, tickets):
    assert {"order_id", "order_ts", "order_date", "iso_year", "iso_week", "region", "country",
            "customer_id", "category", "sku", "qty", "unit_price_eur", "order_value_eur", "carrier",
            "status", "cancel_reason", "status_updated_ts"} <= set(orders[0])
    assert {"shipment_id", "order_id", "carrier", "region", "origin_hub", "ship_date", "ship_iso_year",
            "ship_iso_week", "promised_date", "delivered_date", "delay_days", "status"} <= set(shipments[0])
    assert {"ticket_id", "created_ts", "created_date", "iso_year", "iso_week", "region", "country",
            "order_id", "category", "priority", "status", "sentiment"} <= set(tickets[0])


def test_row_counts(orders, shipments, tickets):
    assert 1500 <= len(orders) <= 2500
    assert len(shipments) == len(orders)
    assert 150 <= len(tickets) <= 350
    assert {r["order_id"] for r in shipments} == {r["order_id"] for r in orders}
    assert len({r["order_id"] for r in orders}) == len(orders)


def test_window_and_week_consistency(orders, shipments, tickets):
    for r in orders:
        d = date.fromisoformat(r["order_date"])
        y, w, _ = d.isocalendar()
        assert (int(r["iso_year"]), int(r["iso_week"])) == (y, w)
        assert date(2026, 8, 24) <= d <= AS_OF
        assert 35 <= w <= 42
    for r in shipments:
        y, w, _ = date.fromisoformat(r["ship_date"]).isocalendar()
        assert (int(r["ship_iso_year"]), int(r["ship_iso_week"])) == (y, w)
    for r in tickets:
        y, w, _ = date.fromisoformat(r["created_date"]).isocalendar()
        assert (int(r["iso_year"]), int(r["iso_week"])) == (y, w)


def test_value_and_domain_rules(orders, shipments, tickets):
    for r in orders:
        assert int(r["qty"]) >= 1
        assert abs(int(r["qty"]) * float(r["unit_price_eur"]) - float(r["order_value_eur"])) < 0.011
        assert r["region"] in {"EU", "UK", "US"}
        assert r["status"] in {"delivered", "shipped", "cancelled"}
        if r["status"] == "cancelled":
            assert r["cancel_reason"] in {"delivery_delay", "customer_request", "payment_failed",
                                          "out_of_stock"}
        else:
            assert r["cancel_reason"] == ""
    ids = {r["order_id"] for r in orders}
    for t in tickets:
        assert t["order_id"] == "" or t["order_id"] in ids
        assert t["category"] in {"delivery_delay", "where_is_my_order", "refund_request",
                                 "product_question", "payment_issue"}


def test_delay_days_formula(shipments):
    for r in shipments:
        end = date.fromisoformat(r["delivered_date"]) if r["delivered_date"] else AS_OF
        expected = max(0, (end - date.fromisoformat(r["promised_date"])).days)
        assert int(r["delay_days"]) == expected, r["shipment_id"]


# ------------------------------------------------------------ §5.2 invariants

def test_eu_net_revenue_drop_at_least_20pct(orders):
    n40, n41 = _net(orders, "EU", 40), _net(orders, "EU", 41)
    assert n40 > 0
    assert n41 <= 0.8 * n40, f"EU net w41={n41:.0f} vs w40={n40:.0f}"


def _nord_cancelled(orders, week):
    return sum(float(r["order_value_eur"]) for r in orders
               if r["region"] == "EU" and int(r["iso_week"]) == week and r["status"] == "cancelled"
               and r["carrier"] == "NordExpress" and r["cancel_reason"] == "delivery_delay")


def test_nordexpress_explains_at_least_60pct_of_delta(orders):
    delta = _net(orders, "EU", 40) - _net(orders, "EU", 41)
    assert delta > 0
    share = (_nord_cancelled(orders, 41) - _nord_cancelled(orders, 40)) / delta
    assert share >= 0.6, f"share={share:.2f}"


def test_eu_demand_unchanged(orders):
    gross41 = _net(orders, "EU", 41, gross=True)
    base = _baseline(orders, "EU", gross=True)
    assert abs(gross41 - base) / base <= 0.10, f"{gross41:.0f} vs {base:.0f}"


@pytest.mark.parametrize("region", ["US", "UK"])
def test_controls_stable(orders, region):
    n41 = _net(orders, region, 41)
    base = _baseline(orders, region)
    assert abs(n41 - base) / base <= 0.08, f"{region}: {n41:.0f} vs {base:.0f}"


def test_nordexpress_share_of_eu_orders(orders):
    eu = [r for r in orders if r["region"] == "EU"]
    share = sum(1 for r in eu if r["carrier"] == "NordExpress") / len(eu)
    assert 0.35 <= share <= 0.55, share


def test_nordexpress_cancel_rates(orders):
    w41 = [r for r in orders if r["region"] == "EU" and int(r["iso_week"]) == 41
           and r["carrier"] == "NordExpress"]
    assert len(w41) >= 20
    rate = sum(1 for r in w41 if r["status"] == "cancelled" and r["cancel_reason"] == "delivery_delay") / len(w41)
    assert 0.45 <= rate <= 0.75, rate

    others = [r for r in orders if r["region"] == "EU" and int(r["iso_week"]) == 41
              and r["carrier"] != "NordExpress"]
    assert sum(1 for r in others if r["status"] == "cancelled") / len(others) <= 0.10

    early = [r for r in orders if r["region"] == "EU" and r["carrier"] == "NordExpress"
             and "2026-10-01" <= r["order_date"] <= "2026-10-04"]
    if len(early) >= 5:
        e_rate = sum(1 for r in early if r["status"] == "cancelled"
                     and r["cancel_reason"] == "delivery_delay") / len(early)
        assert e_rate <= 0.35, e_rate
        assert e_rate < rate


def test_planted_shipment_delays(shipments):
    eu = [r for r in shipments if r["region"] == "EU"]
    planted = [r for r in eu if r["carrier"] == "NordExpress" and "2026-10-01" <= r["ship_date"] <= "2026-10-10"]
    rest = [r for r in eu if r not in planted]
    assert len(planted) >= 20
    assert sum(1 for r in planted if 4 <= int(r["delay_days"]) <= 9) / len(planted) >= 0.95
    assert sum(1 for r in rest if int(r["delay_days"]) <= 1) / len(rest) >= 0.95


def test_ticket_spike_tied_to_nordexpress(orders, tickets):
    def count(week):
        return sum(1 for t in tickets if t["region"] == "EU" and int(t["iso_week"]) == week
                   and t["category"] in DELAY_CATS)

    baseline = statistics.mean(count(w) for w in range(35, 41))
    assert 3 <= baseline <= 15, baseline
    assert count(41) >= 30, count(41)

    carrier = {r["order_id"]: r["carrier"] for r in orders}
    spike = [t for t in tickets if t["region"] == "EU" and int(t["iso_week"]) == 41
             and t["category"] in DELAY_CATS and t["order_id"]]
    assert spike
    tied = sum(1 for t in spike if carrier.get(t["order_id"]) == "NordExpress") / len(spike)
    assert tied >= 0.6, tied
