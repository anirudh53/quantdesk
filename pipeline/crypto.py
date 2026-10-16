"""Daily crypto research run (24/7 market, daily bars close 00:00 UTC). Same rules as the stock pipeline:
evidence and model records, not buy/sell instructions, no position sizes.
  python -m pipeline.crypto [--synthetic]
Three statistical layers, each shown with its own historical record:
  1. Risk profile per coin (volatility, drawdown, tail loss, beta/correlation to BTC).
  2. Time-series trend rule (fixed, not fitted): hold when price is above most of its 20/50/100/200-day averages.
  3. Cross-sectional ranking (which coins beat the others next week): plain momentum vs ridge vs gradient boosting,
     walk-forward weekly, scored by rank IC and a top-third-vs-equal-weight test after costs.
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, pathlib, yaml
import numpy as np
import pandas as pd
from .data import now_utc, _retry

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "crypto"
DAYS = 365                                     # crypto trades every day
MAS = (20, 50, 100, 200)
FEATS = ["r7", "r28", "r90", "vol30", "vratio", "dist_hi365", "rsi14", "beta90"]
HORIZON = 7                                    # predict next 7 days
MIN_TRAIN_WEEKS = 52
MIN_COINS = 8

def load_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "crypto.yaml").read_text())

def fetch(symbols: list, years: int = 6, synthetic: bool = False):
    """Wide close/volume frames (dates x symbols). Delayed/EOD Yahoo data."""
    if synthetic:
        rng = np.random.default_rng(7)
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=years * DAYS)
        mkt = rng.normal(0.0005, 0.03, len(idx))
        close = pd.DataFrame({s: 100 * np.exp(np.cumsum(mkt * rng.uniform(.8, 1.5) + rng.normal(0, .03, len(idx))))
                              for s in symbols}, index=idx)
        vol = pd.DataFrame(rng.uniform(1e6, 5e6, close.shape), index=idx, columns=symbols)
        return close, vol, "synthetic (test only)"
    import yfinance as yf
    raw = _retry(lambda: yf.download(symbols, period=f"{years}y", auto_adjust=True, progress=False, threads=True))
    close, vol = raw["Close"].copy(), raw["Volume"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    vol.index = close.index
    # Yahoo's row for the current UTC day is a partial bar; drop it so every number uses completed days
    if close.index[-1].date() >= dt.datetime.now(dt.timezone.utc).date():
        close, vol = close.iloc[:-1], vol.iloc[:-1]
    return close, vol, "yfinance (daily, UTC close, delayed)"

# ---------- 1. risk profile ----------

def _maxdd(c: pd.Series) -> float:
    return float((c / c.cummax() - 1).min())

def coin_stats(c: pd.Series, v: pd.Series, btc: pd.Series) -> dict:
    c, v = c.dropna(), v.reindex(c.dropna().index)
    r = c.pct_change()
    last = float(c.iloc[-1])
    ret = lambda n: round(last / float(c.iloc[-n - 1]) - 1, 4) if len(c) > n else None
    y = r.iloc[-DAYS:]
    rb = btc.pct_change().reindex(r.index)
    pair = pd.concat([r, rb], axis=1).iloc[-90:].dropna()
    beta = float(np.cov(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1] / pair.iloc[:, 1].var()) if len(pair) > 30 else None
    vol20 = float(r.iloc[-21:-1].std())
    avgv = float(v.iloc[-31:-1].mean())
    above = {n: bool(last > c.iloc[-n:].mean()) for n in MAS if len(c) >= n}
    return {
        "close": round(last, 6), "history_days": int(len(c)),
        "ret_1d": ret(1), "ret_7d": ret(7), "ret_30d": ret(30), "ret_90d": ret(90), "ret_365d": ret(365),
        "move_vs_vol": round(float(r.iloc[-1]) / vol20, 2) if vol20 else None,
        "volume_ratio_30d": round(float(v.iloc[-1]) / avgv, 2) if avgv else None,
        "vol_ann_30d": round(float(r.iloc[-30:].std() * math.sqrt(DAYS)), 3),
        "vol_ann_1y": round(float(y.std() * math.sqrt(DAYS)), 3),
        "sharpe_1y": round(float(y.mean() / y.std() * math.sqrt(DAYS)), 2) if y.std() else None,
        "cvar5_daily_1y": round(float(y[y <= y.quantile(0.05)].mean()), 4),      # average of the worst 5% of days
        "dd_from_high": round(last / float(c.max()) - 1, 3),
        "maxdd_3y": round(_maxdd(c.iloc[-3 * DAYS:]), 3),
        "beta_btc_90d": round(beta, 2) if beta is not None else None,
        "corr_btc_90d": round(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), 2) if len(pair) > 30 else None,
        "above_ma": above, "trend_score": round(sum(above.values()) / len(above), 2) if above else None,
        "closes_120d": [[str(d.date()), round(float(x), 6)] for d, x in c.iloc[-120:].items()],
    }

# ---------- 2. time-series trend rule ----------

def trend_signal(c: pd.Series) -> pd.Series:
    """1 when price is above at least 3 of its 20/50/100/200-day averages (known at that day's close)."""
    votes = sum((c > c.rolling(n).mean()).astype(float) for n in MAS)
    return (votes >= 3).astype(float).where(c.rolling(max(MAS)).mean().notna())

def _perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < DAYS:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / DAYS
    return {"cagr": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 3), "vol": round(float(r.std() * math.sqrt(DAYS)), 3),
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(DAYS)), 2) if r.std() else None,
            "maxdd": round(_maxdd(eq), 3), "years": round(yrs, 1)}

def trend_backtest(c: pd.Series, cost_bps: float) -> dict:
    c = c.dropna()
    sig = trend_signal(c)
    r = c.pct_change()
    pos = sig.shift(1)                                     # trade at the next close after the signal
    strat = pos * r - pos.diff().abs().fillna(0) * cost_bps / 2 / 1e4
    keep = pos.notna()
    hold, rule = _perf(r[keep]), _perf(strat[keep])
    out = {"buy_hold": hold, "trend_rule": rule, "time_in_market": round(float(pos[keep].mean()), 2) if keep.any() else None,
           "signal_now": None if pd.isna(sig.iloc[-1]) else ("in (uptrend)" if sig.iloc[-1] else "out (downtrend)")}
    if hold and rule:
        out["better_risk_adjusted"] = bool((rule["sharpe"] or 0) > (hold["sharpe"] or 0))
        out["smaller_drawdown"] = bool(rule["maxdd"] > hold["maxdd"])
    return out

# ---------- 3. cross-sectional ranking ----------

def panel(close: pd.DataFrame, vol: pd.DataFrame, bench: str) -> pd.DataFrame:
    """Feature panel on weekly dates (every 7th day counting back from the last bar). Only past data per row."""
    r = close.pct_change()
    delta = close.diff()
    up, dn = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    rb = r[bench]
    beta = r.rolling(90).cov(rb).div(rb.rolling(90).var(), axis=0)
    f = {"r7": close.pct_change(7), "r28": close.pct_change(28), "r90": close.pct_change(90),
         "vol30": r.rolling(30).std(), "vratio": vol.rolling(7).mean() / vol.rolling(90).mean(),
         "dist_hi365": close / close.rolling(365).max() - 1, "rsi14": 100 - 100 / (1 + up / dn.replace(0, np.nan)),
         "beta90": beta, "fwd": close.shift(-HORIZON) / close - 1}
    dates = close.index[::-1][::HORIZON][::-1]
    long = pd.concat({k: v.loc[dates].stack(future_stack=True) for k, v in f.items()}, axis=1)
    long.index.names = ["date", "symbol"]
    long = long.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
    return long[long.groupby(level="date")["r7"].transform("size") >= MIN_COINS]

def _ranked(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks centred on 0: removes market-wide moves and outliers."""
    return df.groupby(level="date").rank(pct=True) - 0.5

def _models():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    return {"ridge": lambda: Ridge(alpha=10.0),
            "boosting": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05,
                                                              min_samples_leaf=50, random_state=0)}

def _top_vs_ew(score: pd.Series, fwd: pd.Series, cost_bps: float) -> pd.Series:
    """Weekly return of the top third by score minus the equal-weight universe, net of turnover cost."""
    out, prev = {}, pd.Series(dtype=float)
    for d, s in score.groupby(level="date"):
        s = s.droplevel("date")
        top = s[s >= s.quantile(2 / 3)].index
        w = pd.Series(1 / len(top), index=top)
        turn = w.subtract(prev, fill_value=0).abs().sum()
        fw = fwd.loc[d]
        out[d] = float(fw.reindex(top).mean() - fw.mean()) - turn * cost_bps / 2 / 1e4
        prev = w
    return pd.Series(out)

def _record(score: pd.Series, fwd: pd.Series, cost_bps: float) -> dict:
    df = pd.concat({"s": score, "f": fwd}, axis=1).dropna()
    ic = df.groupby(level="date").apply(lambda g: g["s"].corr(g["f"], method="spearman")).dropna()
    ex = _top_vs_ew(df["s"], df["f"], cost_bps)
    n = len(ic)
    t = float(ic.mean() / ic.std() * math.sqrt(n)) if n > 2 and ic.std() else 0.0
    last52 = ic.iloc[-52:]
    by_year = {str(y): round(float(v), 3) for y, v in ic.groupby(ic.index.year).mean().items()}
    return {"weeks": n, "mean_ic": round(float(ic.mean()), 4), "ic_t_stat": round(t, 2),
            "ic_positive_share": round(float((ic > 0).mean()), 2), "ic_last_52w": round(float(last52.mean()), 4),
            "ic_by_year": by_year,
            "top_third_excess_ann": round(float(ex.mean() * 52), 3),
            "top_third_excess_sharpe": round(float(ex.mean() / ex.std() * math.sqrt(52)), 2) if ex.std() else None,
            "top_third_hit_rate": round(float((ex > 0).mean()), 2)}

def cross_section(pnl: pd.DataFrame, cost_bps: float, refit_every: int = 4) -> dict:
    X = _ranked(pnl[FEATS])
    y = pnl["fwd"].groupby(level="date").rank(pct=True) - 0.5          # target: next-week rank, not raw return
    dates = pnl.index.get_level_values("date").unique().sort_values()
    if len(dates) < MIN_TRAIN_WEEKS + 26:
        return {"status": "insufficient_data", "weeks": int(len(dates))}
    preds = {k: pd.Series(np.nan, index=pnl.index) for k in _models()}
    fitted = {}
    for i in range(MIN_TRAIN_WEEKS, len(dates)):
        d = dates[i]
        # labels of week j end at week j+1, so at date d only weeks before d are fully known
        tr = y.index.get_level_values("date") < d
        tr = tr & y.notna().values
        if (i - MIN_TRAIN_WEEKS) % refit_every == 0:
            fitted = {k: mk().fit(X[tr], y[tr]) for k, mk in _models().items()}
        rows = X.index.get_level_values("date") == d
        for k, m in fitted.items():
            preds[k].loc[X.index[rows]] = m.predict(X[rows])
    fwd = pnl["fwd"]
    oos = {"momentum_28d (no model)": _record(X["r28"].where(preds["ridge"].notna()), fwd, cost_bps)}
    oos.update({k: _record(p, fwd, cost_bps) for k, p in preds.items()})
    # current scores: refit every model on all labelled weeks, score the latest date
    last = dates[-1]
    lab = y.notna()
    now_rows = X.xs(last, level="date")
    scores = pd.DataFrame({k: mk().fit(X[lab], y[lab]).predict(now_rows) for k, mk in _models().items()}, index=now_rows.index)
    scores["momentum_28d (no model)"] = now_rows["r28"]
    for k, rec in oos.items():
        rec["edge_evidence"] = ("candidate (still unvalidated)" if rec["ic_t_stat"] >= 3 and rec["top_third_excess_ann"] > 0
                                and rec["ic_last_52w"] > 0 else "weak/none")
    ranks = scores.rank(ascending=False).astype(int)
    return {"status": "ok", "asof": str(last.date()), "oos": oos,
            "latest_rank": {s: ranks.loc[s].to_dict() for s in ranks.index},
            "consensus_order": list(ranks.mean(axis=1).sort_values().index)}

# ---------- report ----------

def regime(close: pd.DataFrame, bench: str) -> dict:
    b = close[bench].dropna()
    above200 = (close.iloc[-1] > close.iloc[-200:].mean()).where(close.iloc[-200:].notna().all())
    vol = b.pct_change().rolling(30).std()
    return {"btc_above_200d": bool(b.iloc[-1] > b.iloc[-200:].mean()), "btc_dd_from_high": round(float(b.iloc[-1] / b.max() - 1), 3),
            "btc_vol30_percentile_5y": round(float((vol.iloc[-5 * DAYS:] <= vol.iloc[-1]).mean()), 2),
            "breadth_above_200d": round(float(above200.dropna().mean()), 2), "breadth_n": int(above200.notna().sum())}

def run(synthetic: bool = False) -> dict:
    cfg = load_config()
    coins = cfg["coins"]; bench = cfg.get("benchmark", "BTC-USD"); cost = float(cfg["costs"]["round_trip_bps"])
    syms = [c["symbol"] for c in coins]
    close, vol, src = fetch(syms, synthetic=synthetic)
    names = {c["symbol"]: c["name"] for c in coins}
    priced = [s for s in syms if s in close and close[s].notna().sum() > 60]
    unresolved = [s for s in syms if s not in priced]
    close, vol = close[priced], vol[priced]
    results = []
    for s in priced:
        try:
            r = {"symbol": s, "name": names[s], "status": "ok", **coin_stats(close[s], vol[s], close[bench])}
            r["trend"] = trend_backtest(close[s], cost)
        except Exception as e:
            r = {"symbol": s, "name": names[s], "status": "failed", "error": str(e)}
        results.append(r)
    try:
        xs = cross_section(panel(close, vol, bench), cost)
    except Exception as e:
        xs = {"status": "failed", "error": str(e)}
    return {"generated": now_utc(), "source": src, "asof": str(close.index[-1].date()), "cost_round_trip_bps": cost,
            "benchmark": bench, "regime": regime(close, bench), "results": results, "cross_section": xs,
            "unresolved": unresolved}

def to_markdown(doc: dict) -> str:
    pc = lambda x: f"{x:+.0%}" if isinstance(x, (int, float)) else "?"
    ok = [r for r in doc["results"] if r["status"] == "ok"]
    rg = doc["regime"]
    L = [f"# Crypto research brief - data as of {doc['asof']} (UTC close)", "",
         "Evidence and model records only. Not a buy/sell recommendation. Crypto can fall 70-90% and stay down for years.", "",
         "## Market regime", "",
         f"- BTC {'above' if rg['btc_above_200d'] else 'below'} its 200-day average; {pc(rg['btc_dd_from_high'])} from its high in this data.",
         f"- Breadth: {rg['breadth_above_200d']:.0%} of {rg['breadth_n']} coins above their 200-day average.",
         f"- BTC 30-day volatility is at the {rg['btc_vol30_percentile_5y']:.0%} percentile of the last 5 years.", "",
         "## Risk profile (past behaviour, not a forecast)", "",
         "| Coin | Close | 7d | 30d | 1y | Vol (1y ann.) | Sharpe 1y | Worst-5% day avg | From high | Max DD 3y | Beta to BTC | Trend |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ok, key=lambda r: r["vol_ann_1y"]):
        L.append(f"| {r['name']} | {r['close']:,.4g} | {pc(r['ret_7d'])} | {pc(r['ret_30d'])} | {pc(r['ret_365d'])} | "
                 f"{r['vol_ann_1y']:.0%} | {r['sharpe_1y']} | {r['cvar5_daily_1y']:.1%} | {pc(r['dd_from_high'])} | "
                 f"{pc(r['maxdd_3y'])} | {r['beta_btc_90d']} | {r['trend']['signal_now']} |")
    L += ["", "## Trend rule record (fixed rule: hold when above 3 of 4 moving averages, after costs)", "",
          "| Coin | Years | Buy-hold CAGR | Buy-hold max DD | Rule CAGR | Rule max DD | Buy-hold Sharpe | Rule Sharpe | Time in market |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        t = r["trend"]; h, s = t["buy_hold"], t["trend_rule"]
        if h and s:
            L.append(f"| {r['name']} | {h['years']} | {pc(h['cagr'])} | {pc(h['maxdd'])} | {pc(s['cagr'])} | {pc(s['maxdd'])} | "
                     f"{h['sharpe']} | {s['sharpe']} | {t['time_in_market']:.0%} |")
    xs = doc["cross_section"]
    L += ["", "## Cross-sectional ranking: which coins beat the others next week (walk-forward, out-of-sample)", ""]
    if xs.get("status") == "ok":
        L += ["| Method | Weeks | Mean rank IC | IC t-stat | Last 52w IC | Top-third excess / yr (after costs) | Hit rate | Edge evidence |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in xs["oos"].items():
            L.append(f"| {k} | {v['weeks']} | {v['mean_ic']} | {v['ic_t_stat']} | {v['ic_last_52w']} | {pc(v['top_third_excess_ann'])} | "
                     f"{v['top_third_hit_rate']:.0%} | {v['edge_evidence']} |")
        L += ["", f"Current ranks (1 = model expects it to do best vs the others next week; as of {xs['asof']}):", "",
              "| Coin | " + " | ".join(next(iter(xs["latest_rank"].values())).keys()) + " |",
              "|---" * (1 + len(next(iter(xs["latest_rank"].values())))) + "|"]
        for s in xs["consensus_order"]:
            L.append(f"| {s} | " + " | ".join(str(x) for x in xs["latest_rank"][s].values()) + " |")
    else:
        L.append(f"- not available: {xs.get('status')} {xs.get('error', '')}")
    L += ["", "## Gaps", ""] + [f"- {s}: Yahoo returned no usable prices, skipped" for s in doc["unresolved"]] + \
         [f"- {r['symbol']}: {r.get('error')}" for r in doc["results"] if r["status"] != "ok"]
    L += ["", "## Reading this",
          "- Universe = today's survivors. Coins that died are missing, so every backtest here is too kind.",
          "- Rank IC = correlation between the model's ranking and next week's actual ranking. 0 = useless; 0.05 sustained is good.",
          "- 'weak/none' means the out-of-sample record does not support relying on that ranking.",
          "- The trend rule mostly cuts drawdowns; it usually gives up some return in strong bull runs."]
    return "\n".join(L)

# ---------- phone alerts ----------

def changes(doc: dict, prev: dict | None) -> list:
    """What is new since the previous daily report: trend flips, BTC regime change, outsized moves, model edge appearing."""
    out = []
    old = {r["symbol"]: r for r in (prev or {}).get("results", []) if r.get("status") == "ok"}
    if prev and prev["regime"]["btc_above_200d"] != doc["regime"]["btc_above_200d"]:
        out.append(f"BTC crossed {'above' if doc['regime']['btc_above_200d'] else 'below'} its 200-day average")
    for r in doc["results"]:
        if r.get("status") != "ok":
            continue
        o, now = old.get(r["symbol"]), r["trend"]["signal_now"]
        if o and now and o["trend"]["signal_now"] and now != o["trend"]["signal_now"]:
            price = f"${r['close']:,.0f}" if r["close"] >= 1000 else f"${r['close']:.4g}"
            out.append(f"{r['name']}: trend turned {'UP' if now.startswith('in') else 'DOWN'} ({price})")
        if r.get("move_vs_vol") is not None and abs(r["move_vs_vol"]) >= 2.5:
            out.append(f"{r['name']}: {r['ret_1d']:+.1%} in a day ({r['move_vs_vol']:+.1f}x its normal move)")
    xs, oxs = doc.get("cross_section") or {}, (prev or {}).get("cross_section") or {}
    for k, v in (xs.get("oos") or {}).items():
        before = ((oxs.get("oos") or {}).get(k) or {}).get("edge_evidence")
        if v["edge_evidence"] != before and before is not None:
            out.append(f"Model '{k}' edge evidence: {before} -> {v['edge_evidence']}")
    return out

def send_alerts(doc: dict, prev: dict | None) -> list:
    """One ntfy digest per day, only when something changed. Re-runs for the same day do not repeat it."""
    from . import alerts
    items = changes(doc, prev)
    sent = alerts._load()
    key = f"CRYPTO:{doc['asof']}"
    if not items or key in sent:
        return items
    rg = doc["regime"]
    head = f"BTC {'above' if rg['btc_above_200d'] else 'below'} 200d, {rg['btc_dd_from_high']:+.0%} from high"
    alerts.notify(f"Crypto: {len(items)} change{'s' * (len(items) > 1)} ({doc['asof']})",
                  f"{head} · info only, not advice", "\n".join(f"- {x}" for x in items[:12]))
    sent[key] = doc["asof"]
    alerts.STATE.parent.mkdir(exist_ok=True)
    alerts.STATE.write_text(json.dumps(sent, indent=1))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-alerts", action="store_true", help="write the report but send no notifications")
    a = ap.parse_args()
    doc = run(a.synthetic)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{doc['asof']}_CRYPTO"
    older = sorted(p for p in OUT.glob("*_CRYPTO.json") if p.stem < stem)
    prev = json.loads(older[-1].read_text()) if older else None
    if not a.synthetic and not a.no_alerts:
        try:
            doc["alerts"] = send_alerts(doc, prev)
        except Exception as e:                       # an alert failure must not lose the report
            print(f"crypto alerts failed: {e}")
    md = to_markdown(doc)
    (OUT / f"{stem}.md").write_text(md)
    (OUT / f"{stem}.json").write_text(json.dumps(doc, indent=1, default=str))
    try:
        from . import dashboard
        dashboard.build()
    except Exception as e:
        print(f"dashboard build failed: {e}")
    print(md)

if __name__ == "__main__":
    main()
