"""Builds dashboard.html from reports/*.json. Self-contained (data embedded, no server, works offline).
Called automatically at the end of every run_daily; can also be run by hand:  python -m pipeline.dashboard
"""
from __future__ import annotations
import datetime as dt, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = pathlib.Path(__file__).with_name("dashboard_template.html")
MAX_RUNS_PER_MARKET = 90                                  # about a month of open/midday/close runs
MODE_ORDER = {"open": 0, "midday": 1, "close": 2}
MAX_SCANS_PER_MARKET = 5                                  # scans are big (whole index); keep the page light

def load_runs() -> list:
    runs = []
    for p in (ROOT / "reports").glob("*.json"):
        try:
            date, market, mode = p.stem.split("_")
            doc = json.loads(p.read_text())
        except Exception:
            continue                                      # skip anything that is not a pipeline report
        runs.append({"id": p.stem, "date": date, "market": market, "mode": mode,
                     "generated": doc.get("generated"), "results": doc.get("results", []),
                     "unresolved": doc.get("unresolved", []), "changes": doc.get("changes"),
                     "dividends": doc.get("upcoming_dividends")})
    runs.sort(key=lambda r: (r["date"], MODE_ORDER.get(r["mode"], 9)), reverse=True)
    out, seen = [], {}
    for r in runs:
        seen[r["market"]] = seen.get(r["market"], 0) + 1
        if seen[r["market"]] <= MAX_RUNS_PER_MARKET:
            out.append(r)
    return out

def load_scans() -> list:
    out = []
    for mkt in ("US", "IN"):
        for p in sorted((ROOT / "reports" / "scan").glob(f"*_{mkt}_scan.json"), reverse=True)[:MAX_SCANS_PER_MARKET]:
            try:
                doc = json.loads(p.read_text())
            except Exception:
                continue
            doc["id"] = p.stem
            out.append(doc)
    return out

def build() -> pathlib.Path:
    payload = {"built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "runs": load_runs(), "scans": load_scans()}
    blob = json.dumps(payload, default=str).replace("</", "<\\/")   # keep embedded JSON from closing the script tag
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", blob)
    out = ROOT / "dashboard.html"
    out.write_text(html)
    return out

if __name__ == "__main__":
    print(build())
