"""Walk-forward baseline: expanding window, purged by label horizon, costs applied.
Output is a probability plus its OUT-OF-SAMPLE record. It is never an order."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURES

def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))

def walk_forward(feat: pd.DataFrame, horizon: int = 5, min_train: int = 504,
                 step: int = 21, cost_bps: float = 10.0) -> dict:
    labelled = feat.dropna(subset=FEATURES + ["target"])
    if len(labelled) < min_train + step * 3:
        return {"status": "insufficient_data", "n_labelled": int(len(labelled))}
    preds = pd.Series(np.nan, index=labelled.index)
    for start in range(min_train, len(labelled), step):
        train = labelled.iloc[: start - horizon]      # purge rows whose label window overlaps the test block
        test = labelled.iloc[start: start + step]
        if train["target"].nunique() < 2 or test.empty:
            continue
        m = _model().fit(train[FEATURES], train["target"])
        preds.loc[test.index] = m.predict_proba(test[FEATURES])[:, 1]
    oos = labelled.assign(p=preds).dropna(subset=["p"])
    y, p = oos["target"], oos["p"]
    n = len(oos)
    base = float(y.mean())
    acc = float(((p > 0.5) == (y == 1)).mean())
    guess = max(base, 1 - base)
    z = (acc - guess) / math.sqrt(guess * (1 - guess) / n)
    step_ret = oos["fwd_ret"].iloc[::horizon]               # non-overlapping periods
    pos = (p > 0.5).astype(float).iloc[::horizon]
    strat = pos * step_ret - pos.diff().abs().fillna(0) * cost_bps / 1e4
    return {"status": "ok", "n_oos": int(n), "base_rate_up": round(base, 3), "accuracy": round(acc, 3),
            "brier": round(float(((p - y) ** 2).mean()), 4),
            "brier_baseline": round(float(((base - y) ** 2).mean()), 4),
            "z_vs_baseline_optimistic": round(float(z), 2),   # overlapping labels inflate this: upper bound only
            "strategy_mean_per_period": round(float(strat.mean()), 5),
            "buyhold_mean_per_period": round(float(step_ret.mean()), 5),
            "beats_buyhold_after_costs": bool(strat.mean() > step_ret.mean())}

def latest_probability(feat: pd.DataFrame, min_train: int = 504):
    data = feat.dropna(subset=FEATURES)
    labelled = data.dropna(subset=["target"])
    if len(labelled) < min_train or data.empty:
        return None
    m = _model().fit(labelled[FEATURES], labelled["target"])
    return float(m.predict_proba(data[FEATURES].iloc[[-1]])[:, 1][0])
