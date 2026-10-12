"""Data layer. Every record carries source, retrieval time and data-as-of date."""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _retry(fn, tries: int = 3, wait: float = 2.0):
    """Yahoo occasionally rate-limits or times out; retry with backoff before reporting a failure."""
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait * (2 ** i))

def fetch_prices(symbol: str, years: int = 5, synthetic: bool = False) -> pd.DataFrame:
    """Daily OHLCV indexed by date. attrs: source, retrieved, asof. Delayed/EOD data, not live."""
    if synthetic:
        df = _synthetic(symbol, years)
        src = "synthetic (test only)"
    else:
        import yfinance as yf
        def get():
            raw = yf.Ticker(symbol).history(period=f"{years}y", auto_adjust=True)
            if raw.empty:
                raise ValueError(f"no price data for {symbol}")
            return raw
        raw = _retry(get)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df.dropna(subset=["close"])                    # Yahoo sometimes emits a price-less row for today (seen on NSE)
        src = "yfinance (delayed/EOD)"
    df.attrs.update(source=src, retrieved=now_utc(), asof=str(df.index[-1].date()))
    return df

def fetch_profile(symbol: str, synthetic: bool = False) -> dict:
    """One Yahoo round-trip set per name: quote-summary info, event calendar, dividend history. Failures stay empty."""
    if synthetic:
        return {"info": {}, "calendar": {}, "dividends": pd.Series(dtype=float)}
    import yfinance as yf
    t = yf.Ticker(symbol)
    out = {}
    for key, get in (("info", lambda: t.info), ("calendar", lambda: t.calendar), ("dividends", lambda: t.dividends)):
        try:
            out[key] = _retry(get, tries=2)
        except Exception:
            out[key] = None
    return out

def fetch_dividends(symbol: str, synthetic: bool = False, profile: dict | None = None, today: dt.date | None = None) -> dict:
    """Trailing dividends and next ex-date if published. Missing stays None, never 0."""
    out = {"symbol": symbol, "trailing_12m": None, "last_amount": None, "last_ex_date": None, "next_ex_date": None, "pay_date": None,
           "source": "synthetic (test only)" if synthetic else "yfinance", "retrieved": now_utc()}
    if synthetic:
        return out
    today = today or dt.date.today()
    profile = profile or fetch_profile(symbol)
    d = profile.get("dividends")
    if d is not None and len(d):
        d = d.copy()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        # trailing 12 months from TODAY: a company that stopped paying must show no trailing dividend
        out["trailing_12m"] = float(d[d.index > pd.Timestamp(today) - pd.Timedelta(days=365)].sum())
        out["last_ex_date"] = str(d.index[-1].date())
        out["last_amount"] = float(d.iloc[-1])
    cal = profile.get("calendar")
    if isinstance(cal, dict):
        ex = cal.get("Ex-Dividend Date")
        # yfinance often returns the most recent (already past) ex-date here; only future dates count as "next"
        out["next_ex_date"] = str(ex) if ex and pd.Timestamp(ex).date() >= today else None
        pay = cal.get("Dividend Date")
        out["pay_date"] = str(pay) if pay and out["next_ex_date"] else None
    return out

def _synthetic(symbol: str, years: int) -> pd.DataFrame:
    rng = np.random.default_rng(sum(map(ord, symbol)))
    n = years * 252
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": vol}, index=idx)
