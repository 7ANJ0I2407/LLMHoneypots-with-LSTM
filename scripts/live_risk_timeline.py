import argparse
import json
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Show latest runtime risk timeline for a session")
    p.add_argument("--raw", default="logs/raw_events.jsonl")
    p.add_argument("--trigger", default="logs/latest_trigger_session.json")
    p.add_argument("--session-id", default=None)
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--interval", type=float, default=1.5)
    return p.parse_args()


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
        except json.JSONDecodeError:
            continue
    return out


def pick_session_id(args):
    if args.session_id:
        return args.session_id

    tpath = Path(args.trigger)
    if not tpath.exists():
        raise SystemExit("No trigger metadata found. Run trigger command or pass --session-id.")

    payload = json.loads(tpath.read_text(encoding="utf8"))
    sid = payload.get("sessionId")
    if not sid:
        raise SystemExit("latest_trigger_session.json missing sessionId")
    return sid


def bar(value):
    ticks = int(max(0, min(1.0, float(value))) * 20)
    return "#" * ticks + "." * (20 - ticks)


def render_once(events, session_id, last_n):
    rows = []
    for e in events:
        if e.get("sessionId") != session_id:
            continue
        rd = e.get("runtimeDetection") or {}
        if not rd:
            continue
        lstm = (rd.get("lstm") or {}).get("score")
        hybrid = rd.get("hybridRisk")
        severity = rd.get("severity", "na")
        ts = e.get("timestamp", "?")
        if lstm is None and hybrid is None:
            continue
        rows.append(
            {
                "ts": ts,
                "lstm": float(lstm) if lstm is not None else None,
                "hybrid": float(hybrid) if hybrid is not None else None,
                "severity": severity,
            }
        )

    rows = rows[-last_n:]
    print("=== LIVE RISK TIMELINE ===")
    print(f"sessionId={session_id}")
    print(f"events_with_runtime_detection={len(rows)}")

    if not rows:
        print("No runtimeDetection records yet for this session.")
        return

    for r in rows:
        lstm_val = r["lstm"] if r["lstm"] is not None else 0.0
        hyb_val = r["hybrid"] if r["hybrid"] is not None else 0.0
        print(
            f"{r['ts']} | lstm={lstm_val:.4f} [{bar(lstm_val)}] "
            f"hybrid={hyb_val:.4f} [{bar(hyb_val)}] severity={r['severity']}"
        )


def main():
    args = parse_args()
    session_id = pick_session_id(args)
    raw_path = Path(args.raw)

    if not args.follow:
        render_once(load_jsonl(raw_path), session_id, args.last)
        return

    print("Following timeline. Press Ctrl+C to stop.")
    last_count = -1
    try:
        while True:
            events = load_jsonl(raw_path)
            if len(events) != last_count:
                print("\n" + "-" * 80)
                render_once(events, session_id, args.last)
                last_count = len(events)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
