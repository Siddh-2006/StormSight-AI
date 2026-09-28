# One-shot environment setup (Windows / PowerShell). Run from repo root.
# Requires: uv (present), pip (present), Node.js, a C++17 compiler (MSYS2 g++). No conda.

Write-Host "== Python venv (uv) ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    uv venv .venv --python 3.13
}

Write-Host "== Installing backend Python deps ==" -ForegroundColor Cyan
uv pip install -r backend/requirements-dev.txt -p .venv

Write-Host "== Building cpp_module (optional C++ baseline; backend falls back to OpenCV if this fails) ==" -ForegroundColor Cyan
Push-Location cpp_module
$env:CMAKE_GENERATOR = "MinGW Makefiles"
uv pip install --python "$PWD\..\.venv\Scripts\python.exe" --no-build-isolation .
Pop-Location

Write-Host "== Installing frontend deps ==" -ForegroundColor Cyan
if (Test-Path "frontend/package.json") {
    Push-Location frontend
    npm install
    Pop-Location
} else {
    Write-Host "frontend/package.json not found yet — skipping (frontend may still be building)." -ForegroundColor Yellow
}

Write-Host "== Done ==" -ForegroundColor Green
Write-Host "Next:"
Write-Host "  1) Train a quick model:  .venv\Scripts\python -m app.ml.train --epochs 3   (run from backend/)"
Write-Host "  2) Run backend:          .venv\Scripts\uvicorn app.main:app --reload --port 8000   (run from backend/)"
Write-Host "  3) Run frontend:         npm run dev   (run from frontend/)"
