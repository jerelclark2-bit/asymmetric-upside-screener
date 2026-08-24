#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt
exec .venv/bin/streamlit run app.py --server.address 127.0.0.1
