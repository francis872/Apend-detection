@echo off
setlocal
cd /d "%~dp0"
title Apend Detection 1.1 - Localhost

set "HOST=127.0.0.1"
set "PORT=8000"
set "HEALTH=http://%HOST%:%PORT%/health"
set "URL=http://%HOST%:%PORT%/ui/"
set "PY=.venv\Scripts\python.exe"

echo.
echo ==========================================
echo        APEND DETECTION 1.1 - LOCAL
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
  echo El puerto %PORT% ya esta ocupado.
  echo Intentando abrir la interfaz existente...
  start "" "%URL%"
  pause
  exit /b 0
)

echo [4/5] El navegador se abrira cuando el backend responda correctamente.
start "" powershell -NoProfile -WindowStyle Hidden -Command "$u='%HEALTH%'; for($i=0;$i -lt 90;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 1; if($r.StatusCode -eq 200){ Start-Process '%URL%'; exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1"

echo [5/5] Iniciando FastAPI...
echo.
echo Cuando veas:
echo   Uvicorn running on http://%HOST%:%PORT%
echo el sistema esta listo.
echo.
echo Interfaz: %URL%
echo Health:   %HEALTH%
echo API docs: http://%HOST%:%PORT%/docs
echo.
echo Si ocurre un error, NO cierres esta ventana:
echo copia el mensaje rojo o el traceback y enviamelo.
echo Para detener el servidor: CTRL+C
echo.

"%PY%" -m uvicorn geo_outliers.api:app --host %HOST% --port %PORT%

if errorlevel 1 goto :error
goto :eof

:error
echo.
echo ==========================================
echo [ERROR] EL BACKEND NO PUDO INICIAR
echo ==========================================
echo.
echo Ejecuta manualmente este comando para ver el error:
echo   "%PY%" -m uvicorn geo_outliers.api:app --host %HOST% --port %PORT%
echo.
pause
exit /b 1
