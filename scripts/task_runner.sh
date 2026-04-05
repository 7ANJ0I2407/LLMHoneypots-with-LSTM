#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK_INPUT="${1:-help}"
# Support both positional tasks (alerts) and npm-pass-through style flags (--alerts).
if [[ "$TASK_INPUT" == --* ]]; then
  TASK_INPUT="${TASK_INPUT#--}"
fi

SERVER_STARTED_BY_RUNNER=0
SERVER_PID=""

show_help() {
  cat << 'EOF'
Unified task runner for LLM Honeypot

Usage:
  bash scripts/task_runner.sh <task> [options]

Tasks:
  setup            Install node + python dependencies
  start            Start honeypot API server
  demo             Run complete E2E demo (seed + preprocess + train + detect)
  seed             Seed only synthetic traffic
  pipeline         Run preprocess + train + detect
  validate         Validate latest model + dataset
  calibrate        Calibrate runtime threshold from recent windows
  trigger          Send 20-30 same-session requests to trigger LSTM window
                   options: --count N (20..30), --mode attack|mixed|adaptive
  trigger-detect   Run trigger then detect and show latest alerts
                   options: --count N (20..30), --mode attack|mixed|adaptive
  current-alerts   Show only alerts for latest trigger session
                   options: --last N (default 20)
  trigger-current  Run trigger + detect + current-alerts in one command
                   options: --count N (20..30), --mode attack|mixed|adaptive
  risk-timeline    Show runtime LSTM/hybrid risk timeline for latest trigger session
                   options: --last N (default 20), --follow
  session-gate     Show PASS/FAIL summary for latest trigger session
  trigger-risk     Run trigger + detect + current-alerts + risk-timeline
                   options: --count N (20..30), --mode attack|mixed|adaptive
  alerts           Show latest 10 alerts
  status           Show key artifact files
  help             Show this help
EOF
}

run_setup() {
  npm install
  npm run setup:py
}

server_is_healthy() {
  curl -sSf http://localhost:8080/health >/dev/null 2>&1
}

ensure_server() {
  if server_is_healthy; then
    return
  fi

  echo "[tasks] server not running, starting it in background..."
  ./.venv/bin/python scripts/scorer_server.py > logs/server.out.log 2>&1 &
  SERVER_PID=$!
  SERVER_STARTED_BY_RUNNER=1

  for _ in $(seq 1 25); do
    if server_is_healthy; then
      echo "[tasks] server is ready"
      return
    fi
    sleep 1
  done

  echo "[tasks] failed to start server. Check logs/server.out.log"
  exit 1
}

cleanup_server() {
  if [[ "$SERVER_STARTED_BY_RUNNER" -eq 1 && -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup_server EXIT

run_trigger() {
  local count="24"
  local mode="attack"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --count)
        count="$2"
        shift 2
        ;;
      --mode)
        mode="$2"
        shift 2
        ;;
      *)
        echo "Unknown option for trigger: $1"
        exit 1
        ;;
    esac
  done

  node scripts/trigger_lstm.js --count "$count" --mode "$mode"
}

run_current_alerts() {
  local last="20"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --last)
        last="$2"
        shift 2
        ;;
      *)
        echo "Unknown option for current-alerts: $1"
        exit 1
        ;;
    esac
  done

  node scripts/show_current_session_alerts.js --last "$last"
}

run_risk_timeline() {
  local last="20"
  local follow="0"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --last)
        last="$2"
        shift 2
        ;;
      --follow)
        follow="1"
        shift 1
        ;;
      *)
        echo "Unknown option for risk-timeline: $1"
        exit 1
        ;;
    esac
  done

  if [[ "$follow" -eq 1 ]]; then
    ./.venv/bin/python scripts/live_risk_timeline.py --last "$last" --follow
  else
    ./.venv/bin/python scripts/live_risk_timeline.py --last "$last"
  fi
}

case "$TASK_INPUT" in
  setup)
    run_setup
    ;;
  start)
    npm run start
    ;;
  demo)
    npm run demo:e2e
    ;;
  seed)
    npm run seed
    ;;
  pipeline)
    npm run pipeline
    ;;
  validate)
    npm run validate
    ;;
  calibrate)
    npm run calibrate:runtime
    ;;
  trigger)
    shift
    ensure_server
    run_trigger "$@"
    ;;
  trigger-detect)
    shift
    ensure_server
    run_trigger "$@"
    npm run detect
    tail -n 10 logs/alerts.jsonl || true
    ;;
  current-alerts)
    shift
    run_current_alerts "$@"
    ;;
  trigger-current)
    shift
    ensure_server
    run_trigger "$@"
    npm run detect
    run_current_alerts --last 20
    ;;
  risk-timeline)
    shift
    run_risk_timeline "$@"
    ;;
  trigger-risk)
    shift
    ensure_server
    run_trigger "$@"
    npm run calibrate:runtime
    npm run detect
    run_current_alerts --last 20
    run_risk_timeline --last 20
    npm run gate:session
    ;;
  session-gate)
    npm run gate:session
    ;;
  alerts)
    tail -n 10 logs/alerts.jsonl || true
    ;;
  status)
    ls -la models data logs
    ;;
  help)
    show_help
    ;;
  *)
    echo "Unknown task: $1"
    show_help
    exit 1
    ;;
esac
