import json
from datetime import UTC, datetime
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


def event_key_from_alert(rec):
    ev = rec.get("event") or {}
    sid = ev.get("sessionId") or rec.get("sessionId")
    ts = ev.get("timestamp") or rec.get("createdAt")
    if not sid or not ts:
        return None
    return f"{sid}|{ts}"


def event_key_from_event(rec):
    sid = rec.get("sessionId")
    ts = rec.get("timestamp")
    if not sid or not ts:
        return None
    return f"{sid}|{ts}"


def main():
    trigger_path = Path("logs/latest_trigger_session.json")
    alerts_path = Path("logs/alerts.jsonl")
    raw_path = Path("logs/raw_events.jsonl")
    meta_path = Path("models/model_meta_large.json")
    if not meta_path.exists():
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
    explicit_lstm_alerts = [a for a in session_alerts if a.get("type") == "lstm_behavior_alert"]
    explicit_hybrid_alerts = [a for a in session_alerts if a.get("type") == "hybrid_online_alert"]
    explicit_novelty_alerts = [a for a in session_alerts if a.get("type") == "novelty_behavior_alert"]

    # Runtime model signals are computed per request and can exist even when
    # explicit alert rows are not emitted for that session.
    runtime_lstm_signals = 0
    runtime_hybrid_signals = 0
    runtime_novelty_signals = 0

    explicit_lstm_keys = set()
    explicit_hybrid_keys = set()
    explicit_novelty_keys = set()

    max_lstm = 0.0
    max_hybrid = 0.0

    for a in explicit_lstm_alerts:
        key = event_key_from_alert(a)
        if key:
            explicit_lstm_keys.add(key)
    for a in explicit_hybrid_alerts:
        key = event_key_from_alert(a)
        if key:
            explicit_hybrid_keys.add(key)
    for a in explicit_novelty_alerts:
        key = event_key_from_alert(a)
        if key:
            explicit_novelty_keys.add(key)

    runtime_lstm_keys = set()
    runtime_hybrid_keys = set()
    runtime_novelty_keys = set()

    for e in session_events:
        rd = e.get("runtimeDetection") or {}
        lstm_obj = rd.get("lstm") or {}
        lstm = lstm_obj.get("score")
        lstm_decision = bool(lstm_obj.get("decision", False))
        hyb = rd.get("hybridRisk")
        novelty = rd.get("novelty") or {}
        if lstm_decision:
            runtime_lstm_signals += 1
            key = event_key_from_event(e)
            if key:
                runtime_lstm_keys.add(key)
        if hyb is not None and float(hyb) >= 0.4:
            runtime_hybrid_signals += 1
            key = event_key_from_event(e)
            if key:
                runtime_hybrid_keys.add(key)
        if bool(novelty.get("decision", False)):
            runtime_novelty_signals += 1
            key = event_key_from_event(e)
            if key:
                runtime_novelty_keys.add(key)
        if lstm is not None:
            max_lstm = max(max_lstm, float(lstm))
        if hyb is not None:
            max_hybrid = max(max_hybrid, float(hyb))

    runtime_threshold = float(meta.get("runtime_threshold", meta.get("threshold", 0.1)))

    lstm_alert_count = len(explicit_lstm_keys.union(runtime_lstm_keys))
    hybrid_alert_count = len(explicit_hybrid_keys.union(runtime_hybrid_keys))
    novelty_alert_count = len(explicit_novelty_keys.union(runtime_novelty_keys))

    g_events = len(session_events) >= max(20, expected)
    g_rules = len(rule_alerts) >= max(5, int(0.4 * max(expected, 1)))
    g_model = (
        lstm_alert_count > 0
        or hybrid_alert_count > 0
        or novelty_alert_count > 0
        or max_hybrid >= 0.4
        or max_lstm >= runtime_threshold
    )

    overall = g_events and g_rules and g_model

    history_path = Path("logs/session_gate_history.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_record = {
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "sessionId": sid,
        "expected_events": expected,
        "observed_events": len(session_events),
        "rule_alerts": len(rule_alerts),
        "lstm_alerts": lstm_alert_count,
        "hybrid_alerts": hybrid_alert_count,
        "novelty_alerts": novelty_alert_count,
        "max_runtime_lstm": max_lstm,
        "max_runtime_hybrid": max_hybrid,
        "runtime_threshold": runtime_threshold,
        "check_events": "PASS" if g_events else "FAIL",
        "check_rules": "PASS" if g_rules else "FAIL",
        "check_model_signal": "PASS" if g_model else "FAIL",
        "session_gate": "PASS" if overall else "FAIL",
    }
    with history_path.open("a", encoding="utf8") as f:
        f.write(json.dumps(history_record) + "\n")

    print("=== SESSION GATE SUMMARY ===")
    print(f"sessionId={sid}")
    print(f"expected_events={expected}")
    print(f"observed_events={len(session_events)}")
    print(f"rule_alerts={len(rule_alerts)}")
    print(f"lstm_alerts={lstm_alert_count}")
    print(f"hybrid_alerts={hybrid_alert_count}")
    print(f"novelty_alerts={novelty_alert_count}")
    print(f"explicit_lstm_alerts={len(explicit_lstm_alerts)}")
    print(f"explicit_hybrid_alerts={len(explicit_hybrid_alerts)}")
    print(f"explicit_novelty_alerts={len(explicit_novelty_alerts)}")
    print(f"runtime_lstm_signals={runtime_lstm_signals}")
    print(f"runtime_hybrid_signals={runtime_hybrid_signals}")
    print(f"runtime_novelty_signals={runtime_novelty_signals}")
    print(f"max_runtime_lstm={max_lstm:.4f}")
    print(f"max_runtime_hybrid={max_hybrid:.4f}")
    print(f"runtime_threshold={runtime_threshold:.4f}")

    print(f"check_events={'PASS' if g_events else 'FAIL'}")
    print(f"check_rules={'PASS' if g_rules else 'FAIL'}")
    print(f"check_model_signal={'PASS' if g_model else 'FAIL'}")
    print(f"session_gate={'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
