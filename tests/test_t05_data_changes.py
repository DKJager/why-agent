import json
import re

from _helpers import (Q1, Q3, copy_data, drivers, out_of, REPO, rewrite_csv, run)


def _fresh_run(question, run_id, monkeypatch, tmp_path, data_dir):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HMEM_PATH", str(tmp_path / f"hm-{run_id}" / "store.json"))
    return run(question, run_id)


def test_answer_follows_the_data(sample_env, monkeypatch, tmp_path):
    base = _fresh_run(Q1, "t05-base", monkeypatch, tmp_path, sample_env.data_dir)
    base_out = out_of(base)
    assert "NordExpress" in json.dumps(drivers(base))

    # (a) unplant
    unplanted = copy_data(REPO / "data" / "sample", tmp_path / "unplanted")

    def fix_orders(i, r):
        if (r["region"] == "EU" and r["carrier"] == "NordExpress"
                and r["iso_week"] in ("40", "41") and r["status"] == "cancelled"):
            r["status"] = "delivered"
            r["cancel_reason"] = ""

    def fix_ship(i, r):
        if (r["region"] == "EU" and r["carrier"] == "NordExpress"
                and r["ship_iso_week"] in ("40", "41")):
            r["delay_days"] = str(i % 2)

    rewrite_csv(unplanted / "orders.csv", fix_orders)
    rewrite_csv(unplanted / "shipments.csv", fix_ship)

    res_a = _fresh_run(Q1, "t05-a", monkeypatch, tmp_path, unplanted)
    out_a = out_of(res_a)
    headline = out_a["answer"]["headline"].lower()
    assert re.search(r"no (significant|material|notable)|not significant|insignificant", headline), headline
    assert "NordExpress" not in json.dumps(out_a["answer"]["drivers"])
    assert not [c for c in out_a["findings"] if c.get("kind") == "driver"]
    assert out_a["answer"]["text"] != base_out["answer"]["text"]
    assert res_a["data_fingerprint"] != base["data_fingerprint"]

    # (b) relabel
    relabel = copy_data(REPO / "data" / "sample", tmp_path / "relabel")
    for name in ("orders.csv", "shipments.csv", "tickets.csv"):
        p = relabel / name
        p.write_bytes(p.read_bytes().replace(b"NordExpress", b"RheinFracht"))
    res_b = _fresh_run(Q1, "t05-b", monkeypatch, tmp_path, relabel)
    drs_b = drivers(res_b)
    assert drs_b and "RheinFracht" in drs_b[0]["text"]
    assert "NordExpress" not in json.dumps(res_b)
    assert res_b["data_fingerprint"] != base["data_fingerprint"]

    # (c) US control never names a driver
    res_c = _fresh_run(Q3, "t05-c", monkeypatch, tmp_path, sample_env.data_dir)
    out_c = out_of(res_c)
    assert out_c["answer"]["drivers"] == []
    assert not [c for c in out_c["findings"] if c.get("kind") == "driver"]
