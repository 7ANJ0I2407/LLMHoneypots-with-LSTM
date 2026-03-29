import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


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


def parse_ms(ts):
    if not ts:
        return None
    try:
        return int(__import__("datetime").datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def record_session_id(rec):
    if rec.get("event", {}).get("sessionId"):
        return rec["event"]["sessionId"]
    return rec.get("sessionId")


def record_ts_ms(rec):
    val = rec.get("createdAt") or rec.get("event", {}).get("timestamp")
    return parse_ms(val)


def main():
    trigger_path = Path("logs/latest_trigger_session.json")
    alerts_path = Path("logs/alerts.jsonl")
    raw_path = Path("logs/raw_events.jsonl")
    meta_path = Path("models/model_meta.json")

    if not trigger_path.exists():
        raise SystemExit("No latest trigger metadata found.")

    trig = load_json(trigger_path)
    sid = trig.get("sessionId")
    if not sid:
        raise SystemExit("latest trigger metadata missing sessionId.")

    start_ms = parse_ms(trig.get("startedAt") or trig.get("createdAt"))
    expected = int(trig.get("count", 0))

    alerts = load_jsonl(alerts_path)
    raw = load_jsonl(raw_path)
    meta = load_json(meta_path) if meta_path.exists() else {}

    session_events = [
        r
        for r in raw
        if r.get("sessionId") == sid and (start_ms is None or (parse_ms(r.get("timestamp")) or 0) >= start_ms)
    ]

    session_alerts = [
        a
        for a in alerts
        if record_session_id(a) == sid and (start_ms is None or (record_ts_ms(a) or 0) >= start_ms)
    ]

    rule_alerts = [a for a in session_alerts if a.get("type") == "rule_based_pre_alert"]
    lstm_alerts = [a for a in session_alerts if a.get("type") == "lstm_behavior_alert"]
    hybrid_alerts = [a for a in session_alerts if a.get("type") == "hybrid_online_alert"]

    max_lstm = 0.0
    max_hybrid = 0.0

    for e in session_events:
        rd = e.get("runtimeDetection") or {}
        lstm = (rd.get("lstm") or {}).get("score")
        hyb = rd.get("hybridRisk")
        if lstm is not None:
            max_lstm = max(max_lstm, float(lstm))
        if hyb is not None:
            max_hybrid = max(max_hybrid, float(hyb))

    runtime_threshold = float(meta.get("runtime_threshold", meta.get("threshold", 0.1)))

    g_events = len(session_events) >= max(20, expected)
    g_rules = len(rule_alerts) >= max(5, int(0.4 * max(expected, 1)))
    g_model = len(lstm_alerts) > 0 or len(hybrid_alerts) > 0 or max_hybrid >= 0.4 or max_lstm >= runtime_threshold

    overall = g_events and g_rules and g_model

    print("=== SESSION GATE SUMMARY ===")
    print(f"sessionId={sid}")
    print(f"expected_events={expected}")
    print(f"observed_events={len(session_events)}")
    print(f"rule_alerts={len(rule_alerts)}")
    print(f"lstm_alerts={len(lstm_alerts)}")
    print(f"hybrid_alerts={len(hybrid_alerts)}")
    print(f"max_runtime_lstm={max_lstm:.4f}")
    print(f"max_runtime_hybrid={max_hybrid:.4f}")
    print(f"runtime_threshold={runtime_threshold:.4f}")

    print(f"check_events={'PASS' if g_events else 'FAIL'}")
    print(f"check_rules={'PASS' if g_rules else 'FAIL'}")
    print(f"check_model_signal={'PASS' if g_model else 'FAIL'}")
    print(f"session_gate={'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
