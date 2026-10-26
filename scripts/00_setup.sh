#!/usr/bin/env bash
# Create the environment and run the test-suite (CPU, ~1 min).
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[features,logging,dev]"
python -m pytest tests -q
