"""Daily/midday research run for one market. Produces an ATTENTION LIST (what changed, what the evidence says),
not buy/sell instructions. Usage:
  python -m pipeline.run_daily --market US --mode open|midday|close [--synthetic]
"""
from __future__ import annotations
import argparse, datetime as dt, json, pathlib, yaml
import numpy as np
from zoneinfo import ZoneInfo
from . import alerts, data, features, model, sentiment

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKET_TZ = {"US": "America/New_York", "IN": "Asia/Kolkata"}

def market_today(market: str) -> dt.date:
    """Report dates follow the market's calendar, not the machine's (an India run at 23:00 New York is already tomorrow in IST)."""
    return dt.datetime.now(ZoneInfo(MARKET_TZ[market])).date()

def load_universe(market: str):
    cfg = yaml.safe_load((ROOT / "config" / "universe.yaml").read_text())
    rows = cfg["us" if market == "US" else "india"]
    ok = [r for r in rows if r.get("symbol")]
    unresolved = [r["name"] for r in rows if not r.get("symbol")]
    cost = cfg["costs"]["us_bps" if market == "US" else "in_bps"]
    return ok, unresolved, cost, cfg.get("watchlist", [])

def analyse(sym: str, cost_bps: float, synthetic: bool, feeds: list, today: dt.date | None = None) -> dict:
    out = {"symbol": sym, "flags": [], "gaps": []}
    try:
        px = data.fetch_prices(sym, synthetic=synthetic)
    except Exception as e:                                   # failed != quiet
        out["status"] = "failed_retryable"; out["gaps"].append(f"price fetch failed: {e}")
        return out
    out.update(status="ok", source=px.attrs["source"], retrieved=px.attrs["retrieved"], asof=px.attrs["asof"])
    f = features.build(px)
    last = f.iloc[-1]
    ret1 = float(px["close"].pct_change().iloc[-1])
    vol20 = float(last["vol20"])
    out["close"] = round(float(px["close"].iloc[-1]), 4)
    out["ret_1d"] = round(ret1, 4)
    out["move_vs_vol"] = round(ret1 / vol20, 2) if vol20 else None
    tail = px["close"].iloc[-60:]                            # recent history for the dashboard chart
    out["closes_60d"] = [[str(d.date()), round(float(c), 4)] for d, c in tail.items()]
    out["volume_ratio_20d"] = round(float(last["volratio"]), 2)
    out["dist_52w_high"] = round(float(last["dist_hi252"]), 3)
    age = ((today or dt.date.today()) - dt.date.fromisoformat(out["asof"])).days
    if age > 4: out["flags"].append(f"STALE data ({age} days old)")
    if out["move_vs_vol"] is not None and abs(out["move_vs_vol"]) >= 2.5:
        out["flags"].append(f"large 1-day move ({out['move_vs_vol']} x 20d vol)")
    if out["volume_ratio_20d"] >= 2.0:
        out["flags"].append(f"volume {out['volume_ratio_20d']}x 20-day average (anomaly; cause not identified)")
    if out["dist_52w_high"] > -0.01: out["flags"].append("at/near 52-week high")
    # dividends (missing stays unknown)
    try:
        dv = data.fetch_dividends(sym, synthetic=synthetic)
    except Exception as e:                                   # one bad fetch must not kill the run
        dv = {"symbol": sym, "trailing_12m": None, "next_ex_date": None, "error": str(e)}
        out["gaps"].append(f"dividend fetch failed: {e}")
    out["dividends"] = dv
    if dv.get("next_ex_date"): out["flags"].append(f"ex-dividend date {dv['next_ex_date']}")
    if dv.get("trailing_12m") is None: out["gaps"].append("dividend data unavailable or none paid (not distinguished)")
    # ML baseline with its own out-of-sample record
    wf = model.walk_forward(f, cost_bps=cost_bps)
    out["model"] = {"oos": wf, "p_up_5d": None, "label": "hypothesis, not a signal"}
    if wf["status"] == "ok":
        out["model"]["p_up_5d"] = round(model.latest_probability(f), 3)
        proven = wf["beats_buyhold_after_costs"] and wf["z_vs_baseline_optimistic"] >= 3
        out["model"]["edge_evidence"] = "weak/none" if not proven else "candidate (still unvalidated; check stability)"
    else:
        out["gaps"].append("model: insufficient history")
    # sentiment (press tone only)
    heads = sentiment.fetch_rss(sym, feeds) if (feeds and not synthetic) else []
    out["sentiment"] = sentiment.score_headlines(heads)
    if out["sentiment"]["status"] != "ok":
        out["gaps"].append("sentiment: no headlines (feeds not configured or none found) -> unknown, not neutral")
    return out

def attention(r: dict) -> float:
    s = 0.0
    if r.get("move_vs_vol") is not None: s += min(abs(r["move_vs_vol"]), 5)
    s += min(max(r.get("volume_ratio_20d", 1) - 1, 0), 3)
    s += 0.5 * len([x for x in r["flags"] if "ex-dividend" in x or "52-week" in x])
    return round(s, 2)

def to_markdown(market, mode, results, unresolved, now) -> str:
    L = [f"# {market} research brief ({mode}) - {now}",
         "", "Evidence and attention list only. Not a buy/sell recommendation. Data is delayed/end-of-day unless stated.", ""]
    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=attention, reverse=True)
    L += ["## Attention list (largest changes first)", "",
          "| Symbol | Close | 1d | Move vs vol | Vol ratio | Attention | Flags |", "|---|---|---|---|---|---|---|"]
    for r in ok:
        L.append(f"| {r['symbol']} | {r['close']} | {r['ret_1d']:+.2%} | {r['move_vs_vol']} | {r['volume_ratio_20d']} | "
                 f"{attention(r)} | {'; '.join(r['flags']) or 'none'} |")
    L += ["", "## Model record (hypothesis, out-of-sample, after costs)", "",
          "| Symbol | P(up 5d) | OOS n | Accuracy | Base rate | Beats buy-and-hold | Edge evidence |", "|---|---|---|---|---|---|---|"]
    for r in ok:
        w = r["model"]["oos"]
        if w["status"] == "ok":
            L.append(f"| {r['symbol']} | {r['model']['p_up_5d']} | {w['n_oos']} | {w['accuracy']} | {w['base_rate_up']} | "
                     f"{w['beats_buyhold_after_costs']} | {r['model']['edge_evidence']} |")
        else:
            L.append(f"| {r['symbol']} | n/a | - | - | - | - | insufficient data |")
    L += ["", "## Gaps and failures", ""]
    for r in results:
        if r.get("status") != "ok": L.append(f"- {r['symbol']}: {r.get('status')} - {'; '.join(r['gaps'])}")
        else:
            for g in r["gaps"]: L.append(f"- {r['symbol']}: {g}")
    for n in unresolved: L.append(f"- {n}: NSE/ticker symbol not resolved in config, not researched")
    L += ["", "## Reading this", "- Attention score ranks how unusual today's move and volume are. It says nothing about direction going forward.",
          "- Model probabilities are shown with their own out-of-sample record. 'weak/none' means the record does not support relying on it.",
          "- Sentiment (when present) is press tone, labelled hypothesis. Missing sentiment means unknown."]
    return "\n".join(L)

def diff_vs_previous(market: str, results: list) -> list:
    """Midday: what changed since the last saved run today."""
    prev = sorted((ROOT / "reports").glob(f"{market_today(market)}_{market}_*.json"))
    if not prev: return ["no earlier run today to compare"]
    old = {r["symbol"]: r for r in json.loads(prev[-1].read_text())["results"] if r.get("status") == "ok"}
    changes = []
    for r in results:
        o = old.get(r["symbol"])
        if not o or r.get("status") != "ok": continue
        new_flags = set(r["flags"]) - set(o["flags"])
        if new_flags: changes.append(f"{r['symbol']}: new flags -> {'; '.join(sorted(new_flags))}")
    return changes or ["no new flags since previous run"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["US", "IN"], required=True)
    ap.add_argument("--mode", choices=["open", "midday", "close"], required=True)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    ok, unresolved, cost, _ = load_universe(a.market)
    feeds_path = ROOT / "config" / "feeds.yaml"
    feeds = yaml.safe_load(feeds_path.read_text()).get(a.market, []) if feeds_path.exists() else []
    today = market_today(a.market)
    results = [dict(analyse(r["symbol"], cost, a.synthetic, feeds, today), name=r.get("name")) for r in ok]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = to_markdown(a.market, a.mode, results, unresolved, now)
    changes = diff_vs_previous(a.market, results) if a.mode == "midday" else None
    if changes:
        md += "\n\n## Changes since previous run today\n" + "\n".join(f"- {c}" for c in changes)
    divs = alerts.upcoming_dividends(results, today)
    if divs:
        md += "\n\n## Upcoming ex-dividend dates (next 14 days)\n" + "\n".join(
            f"- {d['symbol']}: ex {d['ex_date']}, own by close {d['last_buy_day']}, last paid {d['last_amount']}" for d in divs)
    if not a.synthetic:
        try:
            alerts.dividend_alerts(a.market, results, today)
        except Exception as e:                               # an alert failure must not lose the report
            print(f"dividend alerts failed: {e}")
    stem = f"{today}_{a.market}_{a.mode}"
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"{stem}.md").write_text(md)
    (ROOT / "reports" / f"{stem}.json").write_text(json.dumps({"generated": now, "market": a.market, "mode": a.mode, "date": str(today),
                                                         "results": results, "unresolved": unresolved, "changes": changes,
                                                         "upcoming_dividends": divs}, indent=1, default=str))
    try:
        from . import dashboard
        dashboard.build()
    except Exception as e:                                   # the report is the record; a dashboard failure must not hide it
        print(f"dashboard build failed: {e}")
    print(md)

if __name__ == "__main__":
    main()
