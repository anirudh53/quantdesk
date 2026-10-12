"""Data layer. Every record carries source, retrieval time and data-as-of date."""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_prices(symbol: str, years: int = 5, synthetic: bool = False) -> pd.DataFrame:
    """Daily OHLCV indexed by date. attrs: source, retrieved, asof. Delayed/EOD data, not live."""
    if synthetic:
        df = _synthetic(symbol, years)
        src = "synthetic (test only)"
    else:
        import yfinance as yf
        raw = yf.Ticker(symbol).history(period=f"{years}y", auto_adjust=True)
        if raw.empty:
            raise ValueError(f"no price data for {symbol}")
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df.dropna(subset=["close"])                    # Yahoo sometimes emits a price-less row for today (seen on NSE)
        src = "yfinance (delayed/EOD)"
    df.attrs.update(source=src, retrieved=now_utc(), asof=str(df.index[-1].date()))
    return df

def fetch_dividends(symbol: str, synthetic: bool = False) -> dict:
    """Trailing dividends and next ex-date if published. Missing stays None, never 0."""
    out = {"symbol": symbol, "trailing_12m": None, "last_amount": None, "last_ex_date": None, "next_ex_date": None, "pay_date": None,
           "source": "synthetic (test only)" if synthetic else "yfinance", "retrieved": now_utc()}
    if synthetic:
        return out
    import yfinance as yf
    t = yf.Ticker(symbol)
    d = t.dividends
    if d is not None and len(d):
        d.index = pd.to_datetime(d.index).tz_localize(None)
        out["trailing_12m"] = float(d[d.index > d.index[-1] - pd.Timedelta(days=365)].sum())
        out["last_ex_date"] = str(d.index[-1].date())
        out["last_amount"] = float(d.iloc[-1])
    try:
        cal = t.calendar
        ex = cal.get("Ex-Dividend Date") if isinstance(cal, dict) else None
        # yfinance often returns the most recent (already past) ex-date here; only future dates count as "next"
        out["next_ex_date"] = str(ex) if ex and pd.Timestamp(ex).date() >= dt.date.today() else None
        pay = cal.get("Dividend Date") if isinstance(cal, dict) else None
        out["pay_date"] = str(pay) if pay and out["next_ex_date"] else None
    except Exception:
        pass
    return out

def _synthetic(symbol: str, years: int) -> pd.DataFrame:
    rng = np.random.default_rng(sum(map(ord, symbol)))
    n = years * 252
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": vol}, index=idx)
