#!/usr/bin/env bash
# Runs the full pipeline on labelled synthetic data. No credentials required.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
python app.py demo
