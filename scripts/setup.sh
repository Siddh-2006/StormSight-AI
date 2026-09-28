#!/usr/bin/env bash
# One-shot environment setup (Git Bash / WSL / Linux). Run from repo root.
# Requires: uv, pip, Node.js, a C++17 compiler. No conda.
set -euo pipefail

echo "== Python venv (uv) =="
if [ ! -d ".venv" ]; then
  uv venv .venv --python 3.13
fi

echo "== Installing backend Python deps =="
uv pip install --python .venv/Scripts/python.exe -r backend/requirements-dev.txt \
  || uv pip install --python .venv/bin/python -r backend/requirements-dev.txt

echo "== Installing frontend deps =="
if [ -f "frontend/package.json" ]; then
  (cd frontend && npm install)
else
  echo "frontend/package.json not found yet — skipping."
fi

echo "== Done =="
echo "Next:"
echo "  1) Train a quick model:  (cd backend && ../.venv/Scripts/python -m app.ml.train --epochs 3)"
echo "  2) Run backend:          (cd backend && ../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000)"
echo "  3) Run frontend:         (cd frontend && npm run dev)"
