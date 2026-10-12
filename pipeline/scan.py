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
from .data import _retry, now_utc
from .run_daily import market_today

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "scan"
MIN_ALERT_YIELD = 0.03               # scan dividend alerts only above this trailing yield
SUSPECT_YIELD = 0.15                 # above this, trailing yield probably includes a special dividend
FLAG_SIGMA = 2.5
KEEP_SCANS = 60
CACHE_DAYS = 7                       # refresh each name's per-share fundamentals about weekly
INFO_PER_RUN = 120                   # Yahoo quote-summary calls per scan (rate-limit budget)

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

def _num(v):
    try:
        v = float(v); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def refresh_fundamentals(market: str, syms: list) -> dict:
    """Per-share fundamentals change quarterly, so they are cached (data/fundamentals_<MKT>.json) and refreshed in a
    rolling window: at most INFO_PER_RUN names per scan, stalest first. Valuation ratios are then computed from TODAY's
    price. This keeps Yahoo requests low enough to avoid rate limits."""
    import yfinance as yf
    path = ROOT / "data" / f"fundamentals_{market}.json"
    try:
        cache = json.loads(path.read_text())
    except Exception:
        cache = {}
    cutoff = str(dt.date.today() - dt.timedelta(days=CACHE_DAYS))
    stale = sorted((s for s in syms if cache.get(s, {}).get("fetched", "") < cutoff), key=lambda s: cache.get(s, {}).get("fetched", ""))
    def one(s):
        try:
            i = _retry(lambda: yf.Ticker(s).info, tries=3, wait=3.0) or {}
        except Exception:
            return s, None
        if not i.get("symbol") and not i.get("quoteType"):
            return s, None
        return s, {"fetched": str(dt.date.today()), "eps_trailing": _num(i.get("trailingEps")), "eps_forward": _num(i.get("forwardEps")),
                   "book_value": _num(i.get("bookValue")), "shares": _num(i.get("sharesOutstanding")), "payout_ratio": _num(i.get("payoutRatio"))}
    with ThreadPoolExecutor(4) as ex:                    # 8 workers triggered Yahoo rate limits (HTTP 429)
        for s, v in ex.map(one, stale[:INFO_PER_RUN]):
            if v: cache[s] = v
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return cache

def valuation(close: float, f: dict | None) -> dict:
    if not f:
        return {}
    ratio = lambda d: round(close / d, 2) if d and d > 0 else None   # negative earnings: P/E not meaningful
    return {"pe_trailing": ratio(f.get("eps_trailing")), "pe_forward": ratio(f.get("eps_forward")),
            "price_to_book": ratio(f.get("book_value")), "market_cap": close * f["shares"] if f.get("shares") else None,
            "payout_ratio": f.get("payout_ratio"), "fundamentals_as_of": f.get("fetched")}

def ex_dividend_dates(payers: list, today: dt.date) -> dict:
    """Ex-dividend calendar lookups for dividend payers only. Missing stays missing."""
    import yfinance as yf
    def one(s):
        try:
            cal = _retry(lambda: yf.Ticker(s).calendar, tries=2)
            ex, pay = (cal.get("Ex-Dividend Date"), cal.get("Dividend Date")) if isinstance(cal, dict) else (None, None)
            if ex and pd.Timestamp(ex).date() >= today:
                return s, {"next_ex_date": str(ex), "pay_date": str(pay) if pay else None}
        except Exception:
            pass
        return s, {}
    with ThreadPoolExecutor(4) as ex:
        return dict(ex.map(one, payers))

def attention(r: dict) -> float:
    s = min(abs(r["move_vs_vol"]), 5) if r.get("move_vs_vol") is not None else 0.0
    s += min(max((r.get("volume_ratio_20d") or 1) - 1, 0), 3)
    s += 0.5 * len([x for x in r["flags"] if "52-week" in x])
    return round(s, 2)

def download(syms: list) -> pd.DataFrame:
    """Bulk daily history. Names Yahoo drops (rate limit, timeout) are retried once after a pause."""
    import time
    import yfinance as yf
    get = lambda ss: yf.download(ss, period="1y", auto_adjust=True, actions=True, group_by="ticker", threads=True, progress=False)
    raw = get(syms)
    missing = [s for s in syms if s not in raw.columns.get_level_values(0) or raw[s]["Close"].dropna().empty]
    if len(missing) > 0.05 * len(syms):
        time.sleep(45)
        again = get(missing if len(missing) > 1 else missing * 2)
        got = [s for s in missing if s in again.columns.get_level_values(0) and not again[s]["Close"].dropna().empty]
        if got:
            raw = pd.concat([raw.drop(columns=got, level=0, errors="ignore"), again[got]], axis=1)
    return raw

def run(market: str, send_alerts: bool = True) -> pathlib.Path:
    import yfinance as yf
    names, meta = load_scan_universe(market)
    today = market_today(market)
    syms = [n["symbol"] for n in names]
    raw = download(syms)
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
    if len(ok) < 0.5 * len(rows):
        # a mostly-failed download (rate limit, outage) must not overwrite the last good scan
        raise RuntimeError(f"scan aborted: only {len(ok)} of {len(rows)} names returned prices (Yahoo rate limit or outage?)")
    fcache = refresh_fundamentals(market, [r["symbol"] for r in ok])
    exd = ex_dividend_dates([r["symbol"] for r in ok if r["dividends"]["trailing_12m"]], today)
    for r in ok:
        r["dividends"].update(exd.get(r["symbol"], {}))
        r.update(valuation(r["close"], fcache.get(r["symbol"])))
        # REITs pay out more than accounting earnings by design (judge them on FFO), so they are not flagged
        if (r.get("payout_ratio") or 0) > 1 and r.get("yield_ttm") and r.get("sector") not in ("Real Estate", "Realty"):
            r["flags"].append("payout ratio above 100% of earnings (dividend may be at risk)")
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
