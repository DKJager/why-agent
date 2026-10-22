# build-id: oa-b47de00e
import csv
import json
import shutil
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

Q1 = "Why did EU revenue drop in week 41?"
Q2 = "Why did EU support tickets spike in week 41?"
Q3 = "Why did US revenue drop in week 41?"
SAMPLE_QUESTIONS = [("t-sample-01", Q1), ("t-sample-02", Q2), ("t-sample-03", Q3)]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rewrite_csv(path, fn):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for i, row in enumerate(rows):
        fn(i, row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def truth(data_dir, region, week, year=None):
    """Recompute metrics with the plain csv module."""
    rows = read_csv(Path(data_dir) / "orders.csv")
    if year is None:
        year = max(int(r["iso_year"]) for r in rows)

    def agg(w):
        net = gross = canc = 0.0
        for r in rows:
            if r["region"] != region or int(r["iso_year"]) != year or int(r["iso_week"]) != w:
                continue
            v = float(r["order_value_eur"])
            gross += v
            if r["status"] == "cancelled":
                canc += v
            else:
                net += v
        return {"net": net, "gross": gross, "cancelled": canc}

    cur, prev = agg(week), agg(week - 1)
    base = [agg(week - k) for k in range(1, 5)]
    seg = {}
    for w, sign in ((week, 1), (week - 1, -1)):
        for r in rows:
            if (r["region"] == region and int(r["iso_year"]) == year
                    and int(r["iso_week"]) == w and r["status"] == "cancelled"):
                key = (r["carrier"], r["cancel_reason"])
                seg[key] = seg.get(key, 0.0) + sign * float(r["order_value_eur"])
    delta = prev["net"] - cur["net"]
    top = max(seg, key=lambda k: seg[k]) if seg else None
    share = (seg[top] / delta) if (top and delta > 0) else None
    return {
        "cur": cur,
        "prev": prev,
        "wow_pct": (cur["net"] / prev["net"] - 1) * 100 if prev["net"] else None,
        "vs_mean_pct": (cur["net"] / (sum(b["net"] for b in base) / 4) - 1) * 100,
        "gross_vs_mean_pct": (cur["gross"] / (sum(b["gross"] for b in base) / 4) - 1) * 100,
        "top_segment": top,
        "top_share": share,
    }


def read_trace(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def trace_path(logs, run_id):
    return Path(logs) / f"trace-{run_id}.jsonl"


def run(question, run_id, *, options=None):
    from agent.orchestrator import run_question
    kwargs = {"run_id": run_id}
    if options is not None:
        kwargs["options"] = options
    return run_question(question, **kwargs)


def out_of(result):
    assert result["status"] in ("success", "partial"), result
    return result["output"]


def drivers(result):
    return out_of(result)["answer"]["drivers"]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def copy_data(src, dst):
    shutil.copytree(src, dst)
    return Path(dst)
