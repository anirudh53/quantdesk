# System review – 2026-10-12

An outside-reviewer pass over quantdesk after the first cloud deployment. Grades are for the state **before** this round of fixes.

| Area | Grade | Verdict |
|---|---|---|
| Research hygiene (no look-ahead, out-of-sample testing, unknown ≠ neutral, sources labelled) | A- | Unusually careful for a personal tool. |
| Data correctness | C | Three real bugs (below), one of which sent a wrong-date alert. |
| Reliability / operations | C+ | Cloud schedule works, but failures were silent and a rate-limited scan could overwrite good data. |
| **Decision usefulness** | **D+** | Says *what moved*, not *what the company is worth, whether the dividend is safe, what is coming up, or why it moved*. The only forward-looking piece (the model) has no edge and its probabilities were still on screen. |
| Dashboard / UX | B+ | Clear and honest, but organised around price anomalies rather than the questions an investor has to answer. |
| **Overall** | **C+** | A trustworthy pipeline with too little in it to decide with. |

## Findings and what was done

| # | Finding | Severity | Fix |
|---|---|---|---|
| 1 | Yahoo calendar dates shifted a day early on machines west of UTC (MU ex-date showed Oct 13, really Oct 14). | High (wrong alert sent) | Pipeline runs in UTC (`pipeline/__init__.py`). |
| 2 | Trailing-12-month dividend counted back from the *last payment*, so a company that stopped paying still showed a yield. | High | Counts back from today; test added. |
| 3 | A mostly-failed scan (rate limit) overwrote the last good scan with an empty one. | High | Scan aborts and keeps the old file if under 50% of names load; dropped tickers retried once. |
| 4 | Failures were invisible: the cloud job showed green even when a scan failed. | High | Scheduler exits non-zero on any failure; workflow sends an ntfy "run failed" alert with a link to the log. |
| 5 | Midday runs compared a half day's volume against full-day averages, so volume always looked low. | Medium | Volume compared to expected pace for the time of day; bar labelled partial. |
| 6 | Watchlist "move vs normal" included today's move in its own baseline (damping exactly the big moves), unlike the scan. | Medium | Baseline excludes today in both. |
| 7 | 500 quote-summary calls per scan triggered Yahoo rate limits. | Medium | Per-share fundamentals cached and refreshed ~120 names/day; ratios computed from today's price. |
| 8 | Stock-list refresh in the cloud keyed on file age, which is always "today" on a fresh checkout, so it would never refresh. | Medium | Uses the `retrieved` date stored in the file. |
| 9 | No valuation, quality, dividend-safety, trend or event context. | High (usefulness) | Research card per watchlist name: P/E vs its own sector, margins/ROE/growth/debt, dividend streak/5-yr growth/payout/cuts, 1-12m returns vs index, 200-day trend, volatility, worst drawdown, beta. |
| 10 | No earnings dates, the single biggest scheduled price event. | High (usefulness) | Earnings date per name, flag at ≤14 days, phone alert ~7 days before. |
| 11 | "Why did it move?" unanswered; sentiment always "unknown". | Medium | Recent headlines (Google News, market-relevant terms) with source and date; press tone scored from them. |
| 12 | Model probabilities shown despite "weak/none" evidence invite misuse. | Medium | Hidden unless the out-of-sample record qualifies. |
| 13 | Scan dividend list ignored dividend safety. | Medium | Payout ratio column; warning when payout > 100% of earnings (REITs excluded; they're judged on FFO). |
| 14 | Research guidance was implicit. | Medium | "Questions to answer before deciding": evidence-driven prompts (earnings risk, premium/discount to sector, payout coverage, lagging trend, debt, shrinking revenue) plus "write down why you'd own it and what would make you sell". |

## Known limits that remain
- Yahoo is a secondary, sometimes wrong source; fundamentals should be checked against filings (SEC EDGAR / NSE-BSE) for anything that matters.
- Exchange holidays are not modelled (buy-by dates and scheduling assume weekdays are trading days).
- Sector valuation comparison fills in over ~4 trading days as the fundamentals cache builds.
- No insider/institutional ownership or filings ingestion yet (Form 4, 13F, NSE shareholding).
- Price/volume history does not predict 5-day moves for these names. That is a result, not a bug.
