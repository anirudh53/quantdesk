#!/bin/zsh
# Installs (or reinstalls) the launchd agent that runs schedule/tick.py every 15 minutes.
# Uninstall: launchctl bootout gui/$(id -u)/com.quantdesk.tick && rm ~/Library/LaunchAgents/com.quantdesk.tick.plist
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=com.quantdesk.tick
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$ROOT/logs" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>$ROOT/schedule/tick.py</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>StartInterval</key><integer>900</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$ROOT/logs/tick.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/tick.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $PLIST"
