@echo off
setlocal
cd /d "%~dp0"
title Apend Detection - Localhost

set "HOST=127.0.0.1"
set "PORT=8000"
set "URL=http://%HOST%:%PORT%/ui/"
set "PY=.venv\Scripts\python.exe"

echo.
echo ==========================================
echo        APEND DETECTION 1.0 - LOCAL
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python no esta disponible en PATH.
  echo Instala Python 3.10 o superior.
  pause
  exit /b 1
)

if not exist "%PY%" (
  echo [1/5] Creando entorno virtual...
  python -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo [1/5] Entorno virtual listo.
)

echo [2/5] Instalando Apend Detection...
"%PY%" -m pip install -e .
if errorlevel 1 goto :error

echo [3/5] Verificando puerto %PORT%...
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo.
  echo El puerto %PORT% ya esta en uso.
  echo Si Apend Detection ya esta ejecutandose, abriendo la interfaz...
  start "" "%URL%"
  echo.
  echo URL: %URL%
  pause
  exit /b 0
)

echo [4/5] Preparando navegador...
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process '%URL%'"

echo [5/5] Iniciando servidor...
echo.
echo Apend Detection estara disponible en:
echo   %URL%
echo.
echo API:
echo   http://%HOST%:%PORT%/docs
echo Health:
echo   http://%HOST%:%PORT%/health
echo.
echo Mantenga esta ventana abierta.
echo Para detener el servidor: CTRL+C
echo.

"%PY%" -m uvicorn geo_outliers.api:app --host %HOST% --port %PORT%
if errorlevel 1 goto :error
goto :eof

:error
echo.
echo [ERROR] Apend Detection no pudo iniciar.
echo Revisa el mensaje anterior.
pause
exit /b 1
