#!/usr/bin/env bash
# Runs the live pipeline. Requires a configured .env (Apify + Tavily/Exa).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
python app.py check
python app.py run-all
