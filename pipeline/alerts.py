"""macOS notifications for upcoming ex-dividend dates. Information only, not a recommendation to buy.
Each alert fires once (state in data/alerts_sent.json):
  - "upcoming": first time an ex-date within WINDOW_DAYS is seen
  - "last day": the trading day before the ex-date (US and India both settle T+1, so you must own the
    shares by that day's close to receive the dividend). Exchange holidays are not accounted for.
"""
from __future__ import annotations
import datetime as dt, json, os, pathlib, subprocess, sys, urllib.request
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / "alerts_sent.json"
LOG = ROOT / "logs" / "alerts.log"
WINDOW_DAYS = 14
CUR = {"US": "$", "IN": "₹"}

def notify(title: str, subtitle: str, message: str) -> None:
    """Phone push via ntfy.sh when NTFY_TOPIC is set (cloud runs); macOS Notification Center when on a Mac."""
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        body = json.dumps({"topic": topic, "title": title, "message": f"{message}\n{subtitle}", "tags": ["moneybag"],
                           "click": os.environ.get("DASHBOARD_URL", "")}).encode()
        req = urllib.request.Request("https://ntfy.sh/", data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=20).read()
    if sys.platform == "darwin":
        # text goes in as argv, so quotes in names cannot break the script
        script = ('on run argv\n display notification (item 3 of argv) with title (item 1 of argv) '
                  'subtitle (item 2 of argv) sound name "Glass"\nend run')
        subprocess.run(["osascript", "-e", script, title, subtitle, message], capture_output=True, timeout=20)

def _load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}

def upcoming_dividends(results: list, today: dt.date) -> list:
    out = []
    for r in results:
        dv = r.get("dividends") or {}
        if r.get("status") != "ok" or not dv.get("next_ex_date"):
            continue
        ex = dt.date.fromisoformat(dv["next_ex_date"][:10])
        if not (today <= ex <= today + dt.timedelta(days=WINDOW_DAYS)):
            continue
        last_buy = np.busday_offset(np.datetime64(ex), -1, roll="backward").astype(dt.date)
        ttm, close = dv.get("trailing_12m"), r.get("close")
        out.append({"symbol": r["symbol"], "name": r.get("name"), "ex_date": str(ex), "last_buy_day": str(last_buy),
                    "pay_date": dv.get("pay_date"), "last_amount": dv.get("last_amount"),
                    "yield_ttm": round(ttm / close, 4) if ttm and close else None})
    return sorted(out, key=lambda d: d["ex_date"])

def dividend_alerts(market: str, results: list, today: dt.date, send: bool = True) -> list:
    sent = _load()
    fired = []
    for d in upcoming_dividends(results, today):
        ex, last_buy = dt.date.fromisoformat(d["ex_date"]), dt.date.fromisoformat(d["last_buy_day"])
        if today > last_buy:
            continue                                   # too late to qualify; nothing useful to say
        kind = "last day" if today == last_buy else "upcoming"
        key = f"{market}:{d['symbol']}:{d['ex_date']}:{kind}"
        if key in sent:
            continue
        sym = d["symbol"].replace(".NS", "")
        amt = f"last paid {CUR[market]}{d['last_amount']:g}" if d["last_amount"] else "amount not published"
        yld = f", {d['yield_ttm']:.2%} trailing yield" if d["yield_ttm"] else ""
        pay = f" Pays {d['pay_date']}." if d["pay_date"] else ""
        if kind == "last day":
            title, msg = f"{sym}: last day to own for dividend", f"Ex-date tomorrow ({ex:%a %b %d}). Shares must be held by today's close. {amt}{yld}.{pay}"
        else:
            title, msg = f"{sym}: ex-dividend {ex:%a %b %d}", f"Own by close {last_buy:%a %b %d} to qualify. {amt}{yld}.{pay}"
        subtitle = (d.get("name") or sym) + " · info only, not advice"
        if send:
            notify(title, subtitle, msg)
        sent[key] = str(today)
        fired.append({"key": key, "title": title, "message": msg})
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a") as f:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M} {title} | {msg}\n")
    cutoff = str(today - dt.timedelta(days=60))
    sent = {k: v for k, v in sent.items() if v >= cutoff}
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(sent, indent=1))
    return fired
