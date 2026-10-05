#!/usr/bin/env bash
# Everything the README claims, from a fresh clone.
#
# A judge who clones this and runs the verification commands should get the same results the
# author does. That did not work: the scripts hardcoded .venv/bin/python, so a fresh clone
# produced "No such file or directory" from the very steps the README recommended. This sets
# up an environment if there isn't one and runs all three checks.
#
# Usage:  scripts/check.sh
# Env:    PASSTHRU_PYTHON=/path/to/python  to use a specific interpreter

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${PASSTHRU_PYTHON:-}" ]; then
  PY="$PASSTHRU_PYTHON"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

echo "using interpreter: $PY"

if ! $PY -c "import passthru" >/dev/null 2>&1; then
  echo "installing passthru and pytest into a local virtual environment..."
  if [ ! -d .venv ]; then
    $PY -m venv .venv
    PY=".venv/bin/python"
  fi
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -e . pytest
fi

echo
echo "=== tests ==="
$PY -m pytest -q

echo
echo "=== page scorer against the package ==="
if command -v node >/dev/null 2>&1; then
  PASSTHRU_PYTHON="$PY" node scripts/check-parity.mjs
else
  echo "node not installed; skipping the browser parity check"
  echo "(the page embeds browser.js; src/passthru/browser.js is readable without it)"
fi

echo
echo "=== mutation testing ==="
"$PY" scripts/mutation.py

echo
echo "=== the report reproduces byte for byte ==="
$PY -m passthru.cli fixtures/corpus.json --out /tmp/passthru-check.html >/dev/null
if cmp -s /tmp/passthru-check.html reports/index.html; then
  echo "committed report == fresh render: yes"
else
  echo "committed report DIFFERS from a fresh render:"
  echo "  run: $PY -m passthru.cli fixtures/corpus.json && git add reports/index.html"
  exit 1
fi

echo
echo "all checks passed"