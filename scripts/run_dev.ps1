# Launches backend + frontend together for local demo/judging (Windows/PowerShell).
# Run from repo root: powershell -ExecutionPolicy Bypass -File scripts/run_dev.ps1
# Ctrl+C stops both.

$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Venv not found at $venvPython — run scripts/setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting backend (http://localhost:8000, docs at /docs)..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $venvPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--port", "8000" `
    -WorkingDirectory (Join-Path $root "backend") `
    -PassThru -NoNewWindow

Start-Sleep -Seconds 2

Write-Host "Starting frontend (Vite dev server)..." -ForegroundColor Cyan
$frontend = Start-Process -FilePath "npm" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $root "frontend") `
    -PassThru -NoNewWindow

Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Green
Write-Host "Frontend: see Vite output above for the local URL (usually http://localhost:5173)" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop both." -ForegroundColor Yellow

try {
    Wait-Process -Id $backend.Id, $frontend.Id
} finally {
    Write-Host "Stopping backend/frontend..." -ForegroundColor Cyan
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $frontend.Id -ErrorAction SilentlyContinue
}
