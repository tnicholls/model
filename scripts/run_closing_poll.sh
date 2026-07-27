#!/bin/bash
# Runs one closing-lines poll cycle and commits/pushes any resulting changes.
# Invoked on a schedule by ~/Library/LaunchAgents/com.tnicholls.valorant-closing-lines.plist.
set -euo pipefail

REPO_DIR="/Users/tom/Documents/model"
PYTHON="$REPO_DIR/.venv/bin/python"

cd "$REPO_DIR"

SUMMARY=$("$PYTHON" cli.py closing run 2>&1 | tee /dev/stderr | grep -E "^Poll done:|^Wrote" | tr '\n' ' ')

if [ -n "$(git status --porcelain data/closing_lines)" ]; then
  git add data/closing_lines
  git commit -m "Update closing lines: ${SUMMARY:-no summary captured}"
  git push origin main
else
  echo "No changes to commit."
fi
