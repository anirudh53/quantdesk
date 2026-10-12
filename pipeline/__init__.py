# Yahoo publishes calendar dates (ex-dividend, pay, earnings) as midnight UTC, and yfinance converts them with the
# machine's local timezone, which shifts them one day EARLY anywhere west of UTC (e.g. a Mac in New York showed
# MU's Oct 14 ex-date as Oct 13). Run the whole pipeline in UTC; market-local dates come from zoneinfo explicitly.
import os as _os
import time as _time

_os.environ["TZ"] = "UTC"
_time.tzset()
