@echo off
setlocal EnableDelayedExpansion
chcp 65001 > nul 2>&1

:: ============================================================
::  FOREX SIGNAL SYSTEM — One-Command Local Startup (Windows)
::
::  Usage:
::    start.bat                  - Full startup (recommended)
::    start.bat --skip-signals   - Skip signal generation
::    start.bat --train          - Retrain models before starting
::    start.bat --help           - Show this help
:: ============================================================

for %%a in (%*) do (
  if "%%a"=="--help" (
    echo.
    echo  Usage: start.bat [--skip-signals] [--train]
    echo.
    echo  --skip-signals   Skip fetching data and generating signals
    echo  --train          Retrain ML models before starting (takes 5-15 min)
    echo.
    exit /b 0
  )
)

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

set SKIP_SIGNALS=0
set DO_TRAIN=0
for %%a in (%*) do (
  if "%%a"=="--skip-signals" set SKIP_SIGNALS=1
  if "%%a"=="--train"        set DO_TRAIN=1
)

echo.
echo  ============================================================
echo    FOREX SIGNAL SYSTEM  --  Starting up...
echo  ============================================================
echo.

:: ── Activate virtual environment ──────────────────────────────
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo  [OK] Virtual environment activated (venv)
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo  [OK] Virtual environment activated (.venv)
) else (
    echo  [WARN] No virtual environment found - using system Python
)

:: ── Pre-flight: check Python is available ─────────────────────
python --version > nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python not found. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

:: ── 1. Database migrations ────────────────────────────────────
echo.
echo  [1/5] Running database migrations...
python manage.py migrate --noinput
if errorlevel 1 (
    echo.
    echo  [ERROR] Migration failed. Check error above.
    pause
    exit /b 1
)
echo  [OK] Database ready

:: ── 2. Optional model retraining ─────────────────────────────
if "%DO_TRAIN%"=="1" (
    echo.
    echo  [2/5] Retraining models -- this takes 5-15 minutes...
    python manage.py train_models --fetch-data
    if errorlevel 1 (
        echo  [WARN] Training encountered issues. Check output above.
    ) else (
        echo  [OK] Models retrained
    )
) else (
    echo  [2/5] Skipped retraining  (pass --train to retrain)
)

:: ── 3. Fetch data + generate signals ─────────────────────────
if "%SKIP_SIGNALS%"=="0" (
    echo.
    echo  [3/5] Generating today's trading signals...
    python manage.py run_daily_signal --fetch-data --force
    if errorlevel 1 (
        echo  [WARN] Signal generation had issues.
        echo         If models are missing, run: python manage.py train_models --fetch-data
    ) else (
        echo  [OK] Signals generated
    )
) else (
    echo  [3/5] Skipped signal generation  (pass nothing to generate)
)

:: ── 4. Start Django backend ───────────────────────────────────
echo.
echo  [4/5] Starting backend server (http://localhost:8000)...
start "Trading System - Backend" /min cmd /c "python manage.py runserver 8000 2>&1"

:: Wait for Django to be ready (poll /api/health/ endpoint)
set TRIES=0
:WAIT_DJANGO
timeout /t 2 /nobreak > nul
set /a TRIES+=1
curl -s -o nul -w "%%{http_code}" http://localhost:8000/api/health/ 2>nul | find "200" > nul
if not errorlevel 1 (
    echo  [OK] Backend is ready
    goto BACKEND_READY
)
if %TRIES% lss 10 goto WAIT_DJANGO
echo  [WARN] Backend taking longer than expected - continuing anyway

:BACKEND_READY

:: ── 5. Start React frontend ───────────────────────────────────
echo.
echo  [5/5] Starting dashboard (http://localhost:3000)...
if not exist "frontend\node_modules" (
    echo         Installing npm packages for first run - please wait...
    cd frontend
    call npm install --legacy-peer-deps
    cd ..
)
start "Trading System - Dashboard" /min cmd /c "cd frontend && npm start 2>&1"

:: Give React 5 seconds to start
timeout /t 5 /nobreak > nul

echo.
echo  ============================================================
echo    SYSTEM IS RUNNING
echo.
echo    Dashboard  ->  http://localhost:3000
echo    API        ->  http://localhost:8000/api/signals/
echo    Health     ->  http://localhost:8000/api/system-health/
echo    Admin      ->  http://localhost:8000/admin/
echo.
echo    To stop: close the two background windows
echo    To retrain models: run  start.bat --train
echo  ============================================================
echo.
echo  Opening dashboard in your browser...
timeout /t 2 /nobreak > nul
start http://localhost:3000
