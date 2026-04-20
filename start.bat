@echo off
setlocal EnableDelayedExpansion
chcp 65001 > nul 2>&1

:: ============================================================
::  FOREX SIGNAL SYSTEM — One-Command Local Startup (Windows)
::  Usage: start.bat
::         start.bat --skip-signals
::         start.bat --train
:: ============================================================

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

set SKIP_SIGNALS=0
set DO_TRAIN=0
for %%a in (%*) do (
  if "%%a"=="--skip-signals" set SKIP_SIGNALS=1
  if "%%a"=="--train"        set DO_TRAIN=1
)

echo.
echo   ===========================================================
echo     FOREX SIGNAL SYSTEM -- STARTING UP
echo   ===========================================================
echo.

:: Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo   [OK] Virtual environment activated
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo   [OK] Virtual environment (.venv) activated
)

:: 1. DB Migrations
echo.
echo   [1/5] Running database migrations...
python manage.py migrate --noinput
if errorlevel 1 (
    echo   [ERROR] Migration failed
    pause
    exit /b 1
)

:: 2. (Optional) Retrain
if "%DO_TRAIN%"=="1" (
    echo.
    echo   [2/5] Retraining models (5-15 minutes)...
    python manage.py train_models --fetch-data
)

:: 3. Fetch data + generate signals
if "%SKIP_SIGNALS%"=="0" (
    echo.
    echo   [3/5] Fetching market data and generating signals...
    python manage.py run_daily_signal --fetch-data --force
    if errorlevel 1 (
        echo   [!] Signal generation had issues. Models may need training.
        echo       Run: python manage.py train_models --fetch-data
    )
)

:: 4. Start Django backend in new window
echo.
echo   [4/5] Starting Django backend on http://localhost:8000 ...
start "Django Backend" /min cmd /c "python manage.py runserver 8000"
timeout /t 3 /nobreak > nul

:: 5. Start React frontend in new window
echo.
echo   [5/5] Starting React frontend on http://localhost:3000 ...
if not exist "frontend\node_modules" (
    echo         Installing npm packages (first run - may take a minute)...
    cd frontend
    call npm install
    cd ..
)
start "React Frontend" /min cmd /c "cd frontend && npm start"

echo.
echo   ===========================================================
echo     SYSTEM RUNNING
echo     Frontend -> http://localhost:3000
echo     Backend  -> http://localhost:8000
echo     Admin    -> http://localhost:8000/admin/
echo.
echo     Close the Django and React windows to stop.
echo   ===========================================================
echo.
echo   Press any key to open the app in your browser...
pause > nul
start http://localhost:3000
