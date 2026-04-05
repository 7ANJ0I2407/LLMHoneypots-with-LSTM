import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        return default


def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def run_pipeline(root: Path):
    steps = [
        ["npm", "run", "preprocess"],
        ["npm", "run", "train"],
        ["npm", "run", "train:novelty"],
        ["npm", "run", "calibrate:runtime"],
        ["npm", "run", "validate"],
    ]
    for cmd in steps:
        print(f"[feedback] running: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=root)
        if completed.returncode != 0:
            raise SystemExit(f"step_failed={' '.join(cmd)} code={completed.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Closed-loop retraining trigger from session-gate outcomes")
    parser.add_argument("--history", default="logs/session_gate_history.jsonl")
    parser.add_argument("--state", default="models/retrain_state.json")
    parser.add_argument("--queue-min", type=int, default=50)
    parser.add_argument("--cooldown-min", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(".").resolve()
    history_path = Path(args.history)
    state_path = Path(args.state)

    state = load_json(state_path, {"last_run": None, "last_seen_id": None})
    history = load_jsonl(history_path)

    failures = [h for h in history if h.get("session_gate") == "FAIL"]
    fail_count = len(failures)

    now = datetime.now(UTC)
    last_run_raw = state.get("last_run")
    cooldown_ok = True
    if last_run_raw:
        try:
            last_run = datetime.fromisoformat(last_run_raw.replace("Z", "+00:00"))
            delta_min = (now - last_run).total_seconds() / 60.0
            cooldown_ok = delta_min >= float(args.cooldown_min)
        except Exception:
            cooldown_ok = True

    should_run = args.force or (fail_count >= args.queue_min and cooldown_ok)

    print("=== FEEDBACK LOOP CHECK ===")
    print(f"history_records={len(history)}")
    print(f"failed_sessions={fail_count}")
    print(f"queue_min={args.queue_min}")
    print(f"cooldown_ok={cooldown_ok}")
    print(f"force={args.force}")
    print(f"should_retrain={should_run}")

    if not should_run:
        return

    if args.dry_run:
        print("dry_run=TRUE retraining not executed")
        return

    run_pipeline(root)
    state["last_run"] = now.isoformat().replace("+00:00", "Z")
    state["last_seen_id"] = history[-1].get("sessionId") if history else None
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf8")
    print("retraining_triggered=TRUE")


if __name__ == "__main__":
    main()
