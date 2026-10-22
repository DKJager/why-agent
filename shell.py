"""Track 02 local shell: ask why, query several sources, return evidence-backed answer.
Model hook is optional (open-weight Bedrock seat only); default is deterministic."""
import csv, sys
def load(p): return list(csv.DictReader(open(p)))
def weekly(rows, key=None):
    out = {}
    for r in rows:
        k = (r["region"], int(r["week"])); out[k] = out.get(k, 0) + float(r[key]) if key else out.get(k, 0) + 1
    return out
def explain(region, d="data"):
    o = weekly(load(d + "/orders.csv"), "revenue"); t = load(d + "/tickets.csv")
    w0, w1 = o.get((region, 40), 0), o.get((region, 41), 0); ev = []
    ev.append(("orders.csv", "revenue %s wk40=%s wk41=%s" % (region, w0, w1)))
    top = {}
    for r in t:
        if r["region"] == region and r["week"] == "41": top[r["topic"]] = top.get(r["topic"], 0) + 1
    ev.append(("tickets.csv", "wk41 topics %s" % top))
    why = "drop" if w1 < w0 else "no drop"
    return {"question": "why did %s revenue change" % region, "finding": why, "evidence": ev}
if __name__ == "__main__": print(explain(sys.argv[1] if len(sys.argv) > 1 else "EU"))
