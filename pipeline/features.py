"""Features use only information available at each day's close (no look-ahead)."""
import numpy as np
import pandas as pd

FEATURES = ["r5", "r20", "r60", "vol20", "volratio", "rsi14", "dist_hi252"]

def build(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    c, v = df["close"], df["volume"]
    out = pd.DataFrame(index=df.index)
    out["r5"], out["r20"], out["r60"] = c.pct_change(5), c.pct_change(20), c.pct_change(60)
    out["vol20"] = c.pct_change().rolling(20).std()
    out["volratio"] = v / v.rolling(20).mean()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out["dist_hi252"] = c / c.rolling(252).max() - 1
    out["fwd_ret"] = c.shift(-horizon) / c - 1
    out["target"] = (out["fwd_ret"] > 0).astype(float).where(out["fwd_ret"].notna())
    return out
