"""Once-a-day wide market scan (S&P 500 / Nifty 100 from config/scan_universe.yaml). Screens only, no model:
with hundreds of names a model WILL look predictive on some by luck. Output is a list of things to look into.
Usage: python -m pipeline.scan --market US|IN
Writes reports/scan/YYYY-MM-DD_<MKT>_scan.json, sends one digest notification for new high-yield ex-dividend dates,
and rebuilds the dashboard.
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, pathlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import yaml
from . import alerts
from .data import now_utc
from .run_daily import market_today

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "scan"
MIN_ALERT_YIELD = 0.03               # scan dividend alerts only above this trailing yield
SUSPECT_YIELD = 0.15                 # above this, trailing yield probably includes a special dividend
FLAG_SIGMA = 2.5
KEEP_SCANS = 60

def load_scan_universe(market: str) -> tuple[list, dict]:
    cfg = yaml.safe_load((ROOT / "config" / "scan_universe.yaml").read_text())
    return cfg[market], {"retrieved": cfg.get("retrieved"), "source": cfg["sources"][market]}

def watchlist_symbols(market: str) -> set:
    cfg = yaml.safe_load((ROOT / "config" / "universe.yaml").read_text())
    return {r["symbol"] for r in cfg["us" if market == "US" else "india"] if r.get("symbol")}

def screen(sym: str, df: pd.DataFrame, latest: pd.Timestamp) -> dict:
    """Price/volume/dividend screens for one ticker from the bulk download."""
    df = df.dropna(subset=["Close"])
    if len(df) < 30:
        return {"symbol": sym, "status": "failed", "reason": "insufficient price history"}
    c, v = df["Close"], df["Volume"]
    r = c.pct_change()
    vol20 = float(r.iloc[-21:-1].std())                   # typical daily move, excluding today
    ret1 = float(r.iloc[-1])
    avgv = float(v.iloc[-21:-1].mean())
    hi, lo = float(c.max()), float(c.min())
    row = {"symbol": sym, "status": "ok", "asof": str(df.index[-1].date()), "close": round(float(c.iloc[-1]), 4),
           "ret_1d": round(ret1, 4), "ret_20d": round(float(c.iloc[-1] / c.iloc[-21] - 1), 4) if len(c) > 21 else None,
           "move_vs_vol": round(ret1 / vol20, 2) if vol20 else None,
           "volume_ratio_20d": round(float(v.iloc[-1]) / avgv, 2) if avgv else None,
           "dist_52w_high": round(float(c.iloc[-1]) / hi - 1, 3), "dist_52w_low": round(float(c.iloc[-1]) / lo - 1, 3), "flags": []}
    d = df["Dividends"] if "Dividends" in df else pd.Series(dtype=float)
    d = d[d > 0]
    ttm = float(d[d.index > df.index[-1] - pd.Timedelta(days=365)].sum()) if len(d) else 0.0
    row["dividends"] = {"trailing_12m": ttm or None, "last_amount": float(d.iloc[-1]) if len(d) else None,
                        "last_ex_date": str(d.index[-1].date()) if len(d) else None, "next_ex_date": None, "pay_date": None}
    row["yield_ttm"] = round(ttm / row["close"], 4) if ttm and row["close"] else None
    f = row["flags"]
    if (latest - df.index[-1]).days > 4: f.append(f"STALE data (last {row['asof']})")
    if row["move_vs_vol"] is not None and abs(row["move_vs_vol"]) >= FLAG_SIGMA: f.append(f"large 1-day move ({row['move_vs_vol']} x 20d vol)")
    if row["volume_ratio_20d"] and row["volume_ratio_20d"] >= 2: f.append(f"volume {row['volume_ratio_20d']}x 20-day average")
    if row["dist_52w_high"] > -0.01: f.append("at/near 52-week high")
    if row["dist_52w_low"] < 0.01: f.append("at/near 52-week low")
    if row["yield_ttm"] and row["yield_ttm"] > SUSPECT_YIELD: f.append("yield looks inflated (special dividend?) - check")
    return row

def next_ex_dates(syms: list, today: dt.date) -> dict:
    """Ex-dividend calendar lookups, only for dividend payers. Missing stays missing."""
    import yfinance as yf
    def one(s):
        try:
            cal = yf.Ticker(s).calendar
            ex, pay = (cal.get("Ex-Dividend Date"), cal.get("Dividend Date")) if isinstance(cal, dict) else (None, None)
            if ex and pd.Timestamp(ex).date() >= today:
                return s, str(ex), str(pay) if pay else None
        except Exception:
            pass
        return s, None, None
    with ThreadPoolExecutor(8) as ex:
        return {s: (e, p) for s, e, p in ex.map(one, syms)}

def attention(r: dict) -> float:
    s = min(abs(r["move_vs_vol"]), 5) if r.get("move_vs_vol") is not None else 0.0
    s += min(max((r.get("volume_ratio_20d") or 1) - 1, 0), 3)
    s += 0.5 * len([x for x in r["flags"] if "52-week" in x])
    return round(s, 2)

def run(market: str, send_alerts: bool = True) -> pathlib.Path:
    import yfinance as yf
    names, meta = load_scan_universe(market)
    today = market_today(market)
    syms = [n["symbol"] for n in names]
    raw = yf.download(syms, period="1y", auto_adjust=True, actions=True, group_by="ticker", threads=True, progress=False)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    closes = raw.xs("Close", axis=1, level=1)
    latest = closes.dropna(how="all").index.max()          # last day with real prices (Yahoo can add an empty row for today)
    info = {n["symbol"]: n for n in names}
    rows = []
    for s in syms:
        if s not in raw.columns.get_level_values(0):
            rows.append({"symbol": s, "status": "failed", "reason": "no data returned"}); continue
        try:
            r = screen(s, raw[s], latest)
        except Exception as e:
            r = {"symbol": s, "status": "failed", "reason": str(e)[:200]}
        r.update(name=info[s]["name"], sector=info[s].get("sector"))
        rows.append(r)
    ok = [r for r in rows if r["status"] == "ok"]
    payers = [r["symbol"] for r in ok if r["dividends"]["trailing_12m"]]
    for s, (ex, pay) in next_ex_dates(payers, today).items():
        r = next(x for x in ok if x["symbol"] == s)
        r["dividends"].update(next_ex_date=ex, pay_date=pay)
    for r in ok:
        r["attention"] = attention(r)
    ok.sort(key=lambda r: r["attention"], reverse=True)

    # sector view: equal-weighted average 1-day move per sector
    sectors = {}
    for r in ok:
        sectors.setdefault(r.get("sector") or "Unknown", []).append(r["ret_1d"])
    sector_moves = sorted(({"sector": k, "n": len(v), "avg_ret_1d": round(float(np.mean(v)), 4),
                            "up": sum(x > 0 for x in v)} for k, v in sectors.items()), key=lambda x: x["avg_ret_1d"], reverse=True)

    divs = [d for d in alerts.upcoming_dividends(ok, today)]
    n = len(ok)
    summary = {"scanned": len(rows), "ok": n, "failed": len(rows) - n,
               "big_moves": sum(1 for r in ok if any(f.startswith("large 1-day") for f in r["flags"])),
               # two-sided normal tail beyond 2.5 sigma is ~1.2%; real returns have fatter tails, so this is a floor
               "big_moves_expected_by_chance": round(n * 2 * (1 - 0.5 * (1 + math.erf(FLAG_SIGMA / math.sqrt(2)))), 1),
               "unusual_volume": sum(1 for r in ok if any(f.startswith("volume") for f in r["flags"])),
               "near_high": sum(1 for r in ok if "at/near 52-week high" in r["flags"]),
               "near_low": sum(1 for r in ok if "at/near 52-week low" in r["flags"]),
               "breadth_up": sum(1 for r in ok if r["ret_1d"] > 0)}
    doc = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "retrieved": now_utc(),
           "market": market, "date": str(today), "asof": str(latest.date()), "universe": meta,
           "source": "yfinance (delayed/EOD)", "summary": summary, "sector_moves": sector_moves,
           "upcoming_dividends": divs, "results": ok + [r for r in rows if r["status"] != "ok"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{today}_{market}_scan.json"
    out.write_text(json.dumps(doc, default=str))
    for old in sorted(OUT_DIR.glob(f"*_{market}_scan.json"))[:-KEEP_SCANS]:
        old.unlink()                                       # keep disk use bounded (~60 scans per market)
    if send_alerts:
        try:
            dividend_digest(market, divs, today)
        except Exception as e:
            print(f"scan dividend alert failed: {e}")
    try:
        from . import dashboard
        dashboard.build()
    except Exception as e:
        print(f"dashboard build failed: {e}")
    return out

def dividend_digest(market: str, divs: list, today: dt.date) -> None:
    """ONE notification per scan listing newly seen high-yield ex-dates (watchlist names are alerted separately)."""
    skip = watchlist_symbols(market)
    sent = alerts._load()
    new = [d for d in divs if d["symbol"] not in skip and d["yield_ttm"] and MIN_ALERT_YIELD <= d["yield_ttm"] <= SUSPECT_YIELD
           and dt.date.fromisoformat(d["last_buy_day"]) >= today and f"scan:{market}:{d['symbol']}:{d['ex_date']}" not in sent]
    if not new:
        return
    new.sort(key=lambda d: d["yield_ttm"], reverse=True)
    parts = [f"{d['symbol'].replace('.NS', '')} {d['yield_ttm']:.1%} (ex {dt.date.fromisoformat(d['ex_date']):%b %d})" for d in new]
    title = f"{len(new)} {'US' if market == 'US' else 'India'} stock{'s' if len(new) > 1 else ''} ≥{MIN_ALERT_YIELD:.0%} yield going ex-dividend"
    alerts.notify(title, "Market scan · info only, not advice · see dashboard", ", ".join(parts[:6]) + (f" +{len(parts) - 6} more" if len(parts) > 6 else ""))
    for d in new:
        sent[f"scan:{market}:{d['symbol']}:{d['ex_date']}"] = str(today)
    alerts.STATE.parent.mkdir(exist_ok=True); alerts.LOG.parent.mkdir(exist_ok=True)
    alerts.STATE.write_text(json.dumps(sent, indent=1))
    with alerts.LOG.open("a") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M} {title} | {', '.join(parts)}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["US", "IN"], required=True)
    ap.add_argument("--no-alerts", action="store_true")
    a = ap.parse_args()
    out = run(a.market, send_alerts=not a.no_alerts)
    s = json.loads(out.read_text())["summary"]
    print(f"{out.name}: {s}")

if __name__ == "__main__":
    main()
