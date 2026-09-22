@echo off
setlocal
cd /d "%~dp0"
title Apend Detection 1.1 - Localhost

echo.
echo ==========================================
echo        APEND DETECTION 1.1 - LOCAL
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python no esta disponible en PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creando entorno virtual...
  python -m venv .venv
  if errorlevel 1 goto :error
)

echo [2/3] Sincronizando dependencias...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -e .
if errorlevel 1 goto :error

echo [3/3] Iniciando backend con diagnostico...
echo.
".venv\Scripts\python.exe" run_local.py
if errorlevel 1 goto :error
goto :eof

:error
echo.
echo ==========================================
echo [ERROR] APEND DETECTION NO PUDO INICIAR
echo ==========================================
echo La causa exacta aparece arriba.
echo.
pause
exit /b 1
