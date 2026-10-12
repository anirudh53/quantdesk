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
