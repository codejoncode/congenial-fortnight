#!/usr/bin/env bash
# ============================================================
#  FOREX SIGNAL SYSTEM — One-Command Local Startup
#  Usage: ./start.sh
#         ./start.sh --skip-signals     (don't regenerate signals)
#         ./start.sh --train            (retrain models first)
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

SKIP_SIGNALS=false
DO_TRAIN=false
for arg in "$@"; do
  case $arg in
    --skip-signals) SKIP_SIGNALS=true ;;
    --train)        DO_TRAIN=true ;;
  esac
done

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║    FOREX SIGNAL SYSTEM — STARTING UP     ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ── 1. Activate virtual environment ─────────────────────────
for venv_path in "venv/bin/activate" ".venv/bin/activate"; do
  if [ -f "$venv_path" ]; then
    source "$venv_path"
    echo "  [✓] Virtual environment activated ($venv_path)"
    break
  fi
done

# ── 2. Apply DB migrations ───────────────────────────────────
echo ""
echo "  [1/5] Running database migrations..."
python manage.py migrate --noinput 2>&1 | tail -3

# ── 3. (Optional) Retrain models ────────────────────────────
if [ "$DO_TRAIN" = "true" ]; then
  echo ""
  echo "  [2/5] Retraining models (this takes 5–15 minutes)..."
  python manage.py train_models --fetch-data
fi

# ── 4. Fetch data + generate signals ────────────────────────
if [ "$SKIP_SIGNALS" = "false" ]; then
  echo ""
  echo "  [3/5] Fetching market data and generating signals..."
  python manage.py run_daily_signal --fetch-data --force || {
    echo "  [!] Signal generation had issues — check models exist"
    echo "      Run:  python manage.py train_models --fetch-data"
  }
fi

# ── 5. Start Django backend ──────────────────────────────────
echo ""
echo "  [4/5] Starting Django backend on http://localhost:8000 ..."
python manage.py runserver 8000 &
DJANGO_PID=$!
sleep 2

# ── 6. Start React frontend ──────────────────────────────────
echo ""
echo "  [5/5] Starting React frontend on http://localhost:3000 ..."
cd frontend
if [ ! -d "node_modules" ]; then
  echo "        Installing npm packages (first run)..."
  npm install --silent
fi
npm start &
REACT_PID=$!
cd "$PROJECT_DIR"

echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  SYSTEM RUNNING                                  ║"
echo "  ║  Frontend → http://localhost:3000                ║"
echo "  ║  Backend  → http://localhost:8000                ║"
echo "  ║  Admin    → http://localhost:8000/admin/         ║"
echo "  ║                                                  ║"
echo "  ║  Press Ctrl+C to stop all processes              ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

# Cleanup on exit
cleanup() {
  echo ""
  echo "  Stopping processes..."
  kill "$DJANGO_PID" "$REACT_PID" 2>/dev/null || true
  echo "  Stopped. Goodbye."
}
trap cleanup SIGINT SIGTERM

wait "$DJANGO_PID" "$REACT_PID"
