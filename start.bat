@echo off
setlocal
cd /d "%~dp0"
title Apend Detection

echo.
echo ==========================================
echo        APEND DETECTION - LOCALHOST
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python no esta disponible en PATH.
  echo Instala Python 3.10 o superior y vuelve a intentarlo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creando entorno virtual...
  python -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo [1/4] Entorno virtual encontrado.
)

echo [2/4] Instalando o actualizando Apend Detection...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :error

echo [3/4] Iniciando servidor local...
echo [4/4] Abriendo http://127.0.0.1:8000/ui/
start "" "http://127.0.0.1:8000/ui/"

echo.
echo Apend Detection esta ejecutandose.
echo No cierres esta ventana mientras estes realizando el analisis.
echo Para detener el servidor presiona CTRL+C.
echo.
".venv\Scripts\python.exe" -m uvicorn geo_outliers.api:app --host 127.0.0.1 --port 8000
goto :eof

:error
echo.
echo [ERROR] No fue posible iniciar Apend Detection.
echo Revisa los mensajes anteriores.
pause
exit /b 1
