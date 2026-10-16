# quantdesk: daily US + India research pipeline (evidence, not trade signals)

## What it does
Per market, three runs a day (open / midday / close) writing reports/YYYY-MM-DD_<MKT>_<mode>.md and .json:
- Attention list: which names moved or traded unusually (move vs 20d vol, volume vs 20d average, 52-week-high proximity, ex-dividend dates).
- ML baseline (logistic regression, 5-day-up probability) with its own walk-forward out-of-sample record, after costs, vs buy-and-hold.
  It flags "weak/none" unless the record clearly supports it. Expect "weak/none" most of the time; that is the honest result.
- Press-tone sentiment from RSS headlines you configure (config/feeds.yaml). No headlines = unknown, not neutral.
- Dividends: trailing 12m, last and next ex-date when published; missing stays unknown.
- Midday run diffs against the earlier run and lists only NEW flags.
It deliberately does not output buy/sell, position sizes or price targets.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q                                  # 4 tests, synthetic data
    python -m pipeline.run_daily --market US --mode open        # live (yfinance, delayed/EOD)
    python -m pipeline.run_daily --market IN --mode open --synthetic   # test without network

1. Open config/universe.yaml: confirm the US tickers, and fill the empty India NSE symbols (format SYMBOL.NS). Empty ones are reported as unresolved and skipped.
2. Add RSS feeds to config/feeds.yaml (template with {q} for the symbol) if you want sentiment.
3. Schedule (macOS): `./schedule/install_macos.sh` installs a launchd agent that runs schedule/tick.py every 15 min.
   The tick runs each market's open/midday/close slot (08:30/12:30/16:30 in the market's own timezone) once, DST-safe;
   slots missed while the Mac slept are skipped once the next slot starts. Logs: logs/tick.log, logs/tick.err.
   Check what is due without running: `.venv/bin/python schedule/tick.py --dry-run`.
   Uninstall: `launchctl bootout gui/$(id -u)/com.quantdesk.tick && rm ~/Library/LaunchAgents/com.quantdesk.tick.plist`
   Linux: use schedule/crontab.txt instead (cronie, CRON_TZ per market).
   Report dates follow the market's calendar (an India run at 23:00 New York is filed under the IST date).
   Note: yfinance is delayed/end-of-day, so "midday" = re-run + diff, not tick data. For true intraday use a broker/market-data API.
4. Dashboard: dashboard.html is rebuilt after every run (all markets, last ~90 runs each, data embedded, works offline).
   Bookmark it in your browser: file://<path>/quantdesk/dashboard.html. Rebuild by hand: `.venv/bin/python -m pipeline.dashboard`.
   Layout lives in pipeline/dashboard_template.html.
5. Dividend alerts: macOS notifications for watchlist names going ex-dividend within 14 days, plus a reminder on the
   last day to own them (log: logs/alerts.log). If none appear, allow "Script Editor" in System Settings > Notifications.
6. Market scan (S&P 500 + Nifty 100, config/scan_universe.yaml): runs once a day at 17:30 market time; screens only, no model.
   One digest notification per scan for new ex-dividend dates with >= 3% trailing yield. Dashboard tab "Market scan".
   Refresh the constituent lists monthly: `.venv/bin/python -m pipeline.universe_refresh`. Run by hand: `.venv/bin/python -m pipeline.scan --market US`.
7. Crypto (config/crypto.yaml, 19 coins via Yahoo): `python -m pipeline.crypto`, once a day after the 00:00 UTC close, weekends too.
   Risk profile per coin, a fixed trend rule's record vs buy-and-hold, and a weekly walk-forward test of three rankers
   (28-day momentum, ridge, gradient boosting) scored by rank IC and top-third-vs-equal-weight after costs. Dashboard tab "Crypto".
8. Qualitative layer: open this folder in Claude Code (CLAUDE.md is the Chief). Ask it to read the latest reports and run the news / filings / industry / ownership desks on the flagged names, then bull/bear review.

## Validate before trusting anything
- Run for several weeks; read the "Model record" table. If edge evidence stays "weak/none", ignore the probabilities.
- The z-score is optimistic (overlapping 5-day labels). Treat any apparent edge as a candidate to stress-test (other periods, other names, higher costs), not a finding.
- Never put broker or API keys in chat or in files you share; use environment variables.

## Known limits
- Not tested against live data from my sandbox (no access to Yahoo). First live run may need fixes (symbol formats, rate limits).
- Free data can be wrong or delayed; no corporate-action or survivorship checks beyond adjusted prices.
- No fundamentals/EDGAR/NSE-filings ingestion yet; those are the next modules (Form 4, 8-K, results dates).
