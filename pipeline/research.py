"""Decision-support context for watchlist names: valuation (vs own sector), business quality, dividend track record,
trend and risk vs the market, upcoming events, and recent headlines. Every block is EVIDENCE with its source;
missing fields stay None (unknown), never 0. Nothing here is a buy/sell rule.
"""
from __future__ import annotations
import datetime as dt, json, pathlib, urllib.parse
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARK = {"US": ("^GSPC", "S&P 500"), "IN": ("^NSEI", "Nifty 50")}

def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def fundamentals(info: dict) -> dict:
    """Snapshot from Yahoo's quote summary (secondary source; check filings for anything that matters)."""
    g = lambda k: _num(info.get(k))
    de = g("debtToEquity")
    return {"source": "Yahoo Finance quote summary (secondary)",
            "market_cap": g("marketCap"), "pe_trailing": g("trailingPE"), "pe_forward": g("forwardPE"),
            "price_to_book": g("priceToBook"), "ev_to_ebitda": g("enterpriseToEbitda"),
            "profit_margin": g("profitMargins"), "operating_margin": g("operatingMargins"), "roe": g("returnOnEquity"),
            "revenue_growth": g("revenueGrowth"), "earnings_growth": g("earningsGrowth"),
            "debt_to_equity": de / 100 if de is not None else None,          # Yahoo reports this as a percent
            "current_ratio": g("currentRatio"), "free_cashflow": g("freeCashflow"),
            "payout_ratio": g("payoutRatio"), "beta_yahoo": g("beta"),
            "yahoo_sector": info.get("sector"), "industry": info.get("industry")}

def dividend_record(divs: pd.Series, today: dt.date) -> dict:
    """Track record from the payment history: consecutive years of growth, 5-year growth rate, any cut."""
    out = {"years_paid": 0, "consecutive_increases": 0, "cagr_5y": None, "cut_in_5y": None}
    if divs is None or not len(divs):
        return out
    s = divs.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    annual = s.groupby(s.index.year).sum()
    annual = annual[annual.index < today.year]                 # current year is incomplete
    if not len(annual):
        return out
    out["years_paid"] = int(len(annual))
    streak = 0
    for a, b in zip(annual.values[::-1][1:], annual.values[::-1][:-1]):   # walk back from the latest full year
        if b > a * 1.001: streak += 1
        else: break
    out["consecutive_increases"] = streak
    last5 = annual.iloc[-6:]
    if len(last5) == 6 and last5.iloc[0] > 0:
        out["cagr_5y"] = round(float((last5.iloc[-1] / last5.iloc[0]) ** (1 / 5) - 1), 4)
        out["cut_in_5y"] = bool((last5.diff().dropna() < -0.001 * last5.iloc[:-1].values).any())
    return out

def trend_risk(px: pd.DataFrame, bench: pd.Series | None) -> dict:
    """Where the price sits vs its own history and vs the market. Context, not a timing signal."""
    c = px["close"]
    r = c.pct_change().dropna()
    ret = lambda n: round(float(c.iloc[-1] / c.iloc[-1 - n] - 1), 4) if len(c) > n else None
    out = {"ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_12m": ret(252),
           "above_50d": bool(c.iloc[-1] > c.iloc[-50:].mean()) if len(c) >= 50 else None,
           "above_200d": bool(c.iloc[-1] > c.iloc[-200:].mean()) if len(c) >= 200 else None,
           "vol_annual": round(float(r.iloc[-252:].std() * np.sqrt(252)), 4) if len(r) > 60 else None}
    last = c.iloc[-252:]
    out["max_drawdown_1y"] = round(float((last / last.cummax() - 1).min()), 4)
    if bench is not None and len(bench) > 60:
        b = bench.reindex(c.index).ffill()
        br = b.pct_change()
        both = pd.concat([r, br], axis=1, keys=["s", "b"], sort=True).dropna().iloc[-252:]
        if len(both) > 60 and both["b"].var() > 0:
            out["beta_1y"] = round(float(both["s"].cov(both["b"]) / both["b"].var()), 2)
        bb = b.dropna()
        for n, k in ((63, "3m"), (252, "12m")):          # stock return minus market return over the same window
            if out.get(f"ret_{k}") is not None and len(bb) > n:
                out[f"rel_{k}"] = round(out[f"ret_{k}"] - float(bb.iloc[-1] / bb.iloc[-1 - n] - 1), 4)
    return out

def next_earnings(calendar, today: dt.date):
    try:
        ds = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
        ds = [d for d in (ds or []) if pd.Timestamp(d).date() >= today]
        return str(min(ds)) if ds else None
    except Exception:
        return None

def news(name: str, market: str, max_items: int = 6) -> list:
    """Recent headlines (Google News RSS, last 7 days). Secondary sources: read the article before relying on it."""
    import feedparser
    hl, gl = ("en-US", "US") if market == "US" else ("en-IN", "IN")
    # bias toward market-relevant coverage (company blogs and product reviews crowd out results/analyst news otherwise)
    q = urllib.parse.quote(f'"{name}" (stock OR shares OR earnings OR results OR dividend OR analyst) when:7d')
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={gl}:en")
    items, seen = [], set()
    for e in feed.entries:
        title = e.get("title", "")
        src = (e.get("source") or {}).get("title", "")
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        key = title.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        ts = e.get("published_parsed")
        items.append({"title": title, "source": src, "link": e.get("link", ""),
                      "published": dt.datetime(*ts[:6], tzinfo=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else None})
    items.sort(key=lambda x: x["published"] or "", reverse=True)
    return items[:max_items]

def sector_context(symbol: str, market: str, f: dict) -> dict | None:
    """Where this name's valuation sits within its own sector, using the latest market scan (same-day snapshot)."""
    scans = sorted((ROOT / "reports" / "scan").glob(f"*_{market}_scan.json"))
    if not scans:
        return None
    doc = json.loads(scans[-1].read_text())
    rows = [r for r in doc["results"] if r.get("status") == "ok"]
    me = next((r for r in rows if r["symbol"] == symbol), None)
    sector = me.get("sector") if me else None
    if not sector:
        return None
    peers = [r for r in rows if r.get("sector") == sector]
    out = {"sector": sector, "n_peers": len(peers), "scan_date": doc.get("date"),
           "universe": "S&P 500" if market == "US" else "Nifty 100"}
    for key, mine in (("pe_trailing", f.get("pe_trailing")), ("pe_forward", f.get("pe_forward")), ("price_to_book", f.get("price_to_book"))):
        vals = [v for v in (_num(r.get(key)) for r in peers) if v is not None and v > 0]
        if mine is None or mine <= 0 or len(vals) < 4:
            continue
        out[key] = {"median": round(float(np.median(vals)), 2), "pct_rank": round(float(np.mean([v < mine for v in vals])), 2), "n": len(vals)}
    return out

def fetch_benchmark(market: str, synthetic: bool = False):
    if synthetic:
        return None
    from . import data
    try:
        return data.fetch_prices(BENCHMARK[market][0])["close"]
    except Exception:
        return None
