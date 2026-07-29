#!/bin/bash
# Runs one Polymarket odds poll cycle (live price/order-book snapshots for
# upcoming T1 matches + finalizing any that have since closed) and
# commits/pushes any resulting changes. Register this on the same schedule
# as scripts/run_closing_poll.sh (e.g. a LaunchAgent/cron entry) -- it is not
# registered automatically.
set -euo pipefail

REPO_DIR="/Users/tom/Documents/model"
PYTHON="$REPO_DIR/.venv/bin/python"

cd "$REPO_DIR"

SUMMARY=$("$PYTHON" cli.py polymarket run 2>&1 | tee /dev/stderr | grep -E "^Poll done:|^Wrote" | tr '\n' ' ')

if [ -n "$(git status --porcelain data/polymarket_odds)" ]; then
  git add data/polymarket_odds
  git commit -m "Update Polymarket odds: ${SUMMARY:-no summary captured}"
  git push origin main
else
  echo "No changes to commit."
fi
