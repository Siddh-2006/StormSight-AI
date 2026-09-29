#!/usr/bin/env bash
# Launches backend + frontend together for local demo/judging (Git Bash/Linux).
# Run from repo root: bash scripts/run_dev.sh
# Ctrl+C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT/.venv/Scripts/python.exe"
[ -f "$VENV_PY" ] || VENV_PY="$ROOT/.venv/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "Venv not found — run scripts/setup.sh first." >&2
    exit 1
fi

trap 'echo "Stopping backend/frontend..."; kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

echo "Starting backend (http://localhost:8000, docs at /docs)..."
(cd "$ROOT/backend" && "$VENV_PY" -m uvicorn app.main:app --port 8000) &

sleep 2

echo "Starting frontend (Vite dev server)..."
(cd "$ROOT/frontend" && npm run dev) &

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: see Vite output above for the local URL (usually http://localhost:5173)"
echo "Press Ctrl+C to stop both."

wait
