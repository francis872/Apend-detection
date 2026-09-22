$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "Apend Detection - Localhost" -ForegroundColor Green

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creando entorno virtual..."
    python -m venv .venv
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -e .

Write-Host "Abriendo http://127.0.0.1:8000/ui/" -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8000/ui/"
& $python -m uvicorn geo_outliers.api:app --host 127.0.0.1 --port 8000
