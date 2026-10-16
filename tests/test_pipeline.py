import numpy as np, pandas as pd
from pipeline import data, features, model

def test_no_lookahead_in_features():
    px = data.fetch_prices("TEST", synthetic=True)
    f1 = features.build(px)
    f2 = features.build(px.iloc[:-30])             # truncate future; earlier feature rows must be identical
    cols = features.FEATURES
    pd.testing.assert_frame_equal(f1[cols].iloc[:-30], f2[cols], check_exact=False)

def test_walk_forward_runs_and_reports_oos():
    px = data.fetch_prices("TEST", years=6, synthetic=True)
    r = model.walk_forward(features.build(px))
    assert r["status"] == "ok" and r["n_oos"] > 100
    assert 0 <= r["accuracy"] <= 1

def test_noise_has_no_edge():
    # random-walk data: the model should not look reliably predictive
    px = data.fetch_prices("NOISE", years=8, synthetic=True)
    r = model.walk_forward(features.build(px))
    assert r["z_vs_baseline_optimistic"] < 4

def test_insufficient_history_is_flagged_not_hidden():
    px = data.fetch_prices("SHORT", years=1, synthetic=True)
    assert model.walk_forward(features.build(px))["status"] == "insufficient_data"

def test_dividend_alert_dates():
    import datetime as dt
    from pipeline import alerts
    r = {"symbol": "X", "status": "ok", "close": 100.0,
         "dividends": {"next_ex_date": "2026-10-13", "trailing_12m": 2.0, "last_amount": 0.5, "pay_date": None}}
    d = alerts.upcoming_dividends([r], dt.date(2026, 10, 9))       # Friday
    assert d[0]["last_buy_day"] == "2026-10-12" and d[0]["yield_ttm"] == 0.02   # ex-date Tue -> must own by Mon close
    assert alerts.upcoming_dividends([r], dt.date(2026, 10, 14)) == []          # past ex-date is dropped

def test_dividend_record_streak_cut_and_growth():
    import datetime as dt
    from pipeline import research
    idx = pd.to_datetime([f"{y}-06-01" for y in range(2018, 2026)])
    rising = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], index=idx)
    r = research.dividend_record(rising, dt.date(2026, 10, 1))
    assert r["consecutive_increases"] == 7 and r["cut_in_5y"] is False and r["cagr_5y"] > 0
    cut = pd.Series([1.0, 1.1, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0], index=idx)
    r = research.dividend_record(cut, dt.date(2026, 10, 1))
    assert r["cut_in_5y"] is True and r["consecutive_increases"] == 4

def test_trailing_dividend_counts_back_from_today_not_last_payment():
    import datetime as dt
    from pipeline import data
    old = pd.Series([0.5, 0.5], index=pd.to_datetime(["2023-03-01", "2023-09-01"]))
    dv = data.fetch_dividends("X", profile={"dividends": old, "calendar": {}}, today=dt.date(2026, 10, 1))
    assert dv["trailing_12m"] == 0.0          # stopped paying: no trailing yield

def test_trend_risk_relative_to_market():
    from pipeline import research
    px = data.fetch_prices("TREND", years=2, synthetic=True)
    bench = px["close"] * 0 + 100.0            # flat market
    t = research.trend_risk(px, bench)
    assert abs(t["rel_12m"] - t["ret_12m"]) < 1e-9 and t["max_drawdown_1y"] <= 0

def test_session_fraction_partial_vs_complete():
    import datetime as dt
    from zoneinfo import ZoneInfo
    from pipeline import run_daily
    noon = dt.datetime(2026, 10, 12, 12, 45, tzinfo=ZoneInfo("America/New_York"))
    assert 0.4 < run_daily.session_fraction("US", "2026-10-12", noon) < 0.6
    assert run_daily.session_fraction("US", "2026-10-09", noon) is None                     # yesterday's bar is complete
    assert run_daily.session_fraction("US", "2026-10-12", noon.replace(hour=17)) is None    # after the close

def test_scan_valuation_from_cached_per_share_data():
    from pipeline import scan
    v = scan.valuation(100.0, {"eps_trailing": 5.0, "eps_forward": -1.0, "book_value": 20.0, "shares": 1e6, "payout_ratio": 0.4})
    assert v["pe_trailing"] == 20.0 and v["pe_forward"] is None and v["price_to_book"] == 5.0

def test_crypto_panel_has_no_lookahead():
    from pipeline import crypto
    syms = [f"C{i}" for i in range(10)]
    close, vol, _ = crypto.fetch(syms, years=3, synthetic=True)
    p1 = crypto.panel(close, vol, "C0")
    p2 = crypto.panel(close.iloc[:-70], vol.iloc[:-70], "C0")
    common = p1.index.intersection(p2.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(p1.loc[common, crypto.FEATS], p2.loc[common, crypto.FEATS], check_exact=False)

def test_crypto_random_coins_show_no_edge():
    from pipeline import crypto
    syms = [f"C{i}" for i in range(12)]
    close, vol, _ = crypto.fetch(syms, years=4, synthetic=True)
    xs = crypto.cross_section(crypto.panel(close, vol, "C0"), cost_bps=30, refit_every=8)
    assert xs["status"] == "ok"
    assert all(v["edge_evidence"] == "weak/none" for v in xs["oos"].values())
