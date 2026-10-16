"""Scheduler tick (macOS launchd calls this every 15 min). DST-safe and sleep-tolerant:
for each market, work out which slot is current in the MARKET's own timezone and run it once if its report is missing.
The wide market scan runs once after SCAN_TIME (market time) if that day's scan file is missing.
A slot missed while the Mac slept is skipped once the next slot starts (no stale back-fill).
Usage: python -m schedule.tick [--dry-run]
"""
from __future__ import annotations
import argparse, datetime as dt, pathlib, subprocess, sys
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKET_TZ = {"US": "America/New_York", "IN": "Asia/Kolkata"}
SLOTS = [("open", dt.time(8, 30)), ("midday", dt.time(12, 30)), ("close", dt.time(16, 30))]   # market local time
SCAN_TIME = dt.time(17, 30)          # wide market scan, once a day after the close
CRYPTO_TIME = dt.time(0, 30)         # crypto: once a day, every day, after the 00:00 UTC daily close

def due_slot(now_local: dt.datetime) -> str | None:
    if now_local.weekday() >= 5:                     # weekends; exchange holidays still run (report shows STALE)
        return None
    current = None
    for mode, t in SLOTS:
        if now_local.time() >= t:
            current = mode
    return current

FAILED = []

def run(cmd: list, label: str) -> None:
    r = subprocess.run([sys.executable, "-m", *cmd], cwd=ROOT, capture_output=True, text=True)
    print(f"{label}: exit {r.returncode}")
    if r.returncode:
        print(r.stderr[-3000:])
        FAILED.append(label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    uni = ROOT / "config" / "scan_universe.yaml"
    # age from the 'retrieved' stamp inside the file: in the cloud every checkout has a fresh file mtime
    stamp_line = next((l for l in uni.read_text().splitlines() if l.startswith("retrieved:")), "") if uni.exists() else ""
    retrieved = stamp_line.split(":", 1)[1].strip().strip("'\"")[:10] if stamp_line else "2000-01-01"
    if (dt.date.today() - dt.date.fromisoformat(retrieved)).days > 30:
        print("refreshing scan universe (index constituents older than 30 days)", flush=True)
        if not a.dry_run:
            run(["pipeline.universe_refresh"], "universe refresh")   # on failure the old list is kept
    utc = dt.datetime.now(dt.timezone.utc)
    # the report is named after the last completed UTC day, so after 00:30 today's job writes yesterday's date
    crypto_file = ROOT / "reports" / "crypto" / f"{utc.date() - dt.timedelta(days=1)}_CRYPTO.json"
    if utc.time() >= CRYPTO_TIME and not crypto_file.exists():
        print(f"{utc:%Y-%m-%d %H:%M} UTC crypto: running daily report", flush=True)
        if not a.dry_run:
            run(["pipeline.crypto"], "crypto")
    for mkt, tz in MARKET_TZ.items():
        now = dt.datetime.now(ZoneInfo(tz))
        mode = due_slot(now)
        stamp = now.strftime("%Y-%m-%d %H:%M %Z")
        scan_file = ROOT / "reports" / "scan" / f"{now.date()}_{mkt}_scan.json"
        if now.weekday() < 5 and now.time() >= SCAN_TIME and not scan_file.exists():
            print(f"{stamp} {mkt}: running market scan", flush=True)
            if not a.dry_run:
                run(["pipeline.scan", "--market", mkt], f"{mkt} scan")
        if mode is None:
            print(f"{stamp} {mkt}: nothing due"); continue
        report = ROOT / "reports" / f"{now.date()}_{mkt}_{mode}.md"
        if report.exists():
            print(f"{stamp} {mkt}: {mode} already done"); continue
        print(f"{stamp} {mkt}: running {mode}", flush=True)
        if a.dry_run:
            continue
        run(["pipeline.run_daily", "--market", mkt, "--mode", mode], f"{mkt} {mode}")

if __name__ == "__main__":
    main()
    if FAILED:                                       # non-zero exit so the cloud workflow shows red and sends a failure alert
        sys.exit(f"failed: {', '.join(FAILED)}")
