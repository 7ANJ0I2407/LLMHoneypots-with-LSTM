# Novelty Calibration and Benign Validation

## Goal

Validate novelty behavior on real benign traffic and ensure runtime scoring is active.

## Procedure

1. Reset runtime state.
2. Calibrate runtime threshold.
3. Run benign traffic.
4. Inspect generated benign report.

```bash
npm run reset:runtime-state
npm run calibrate:runtime
npm run trigger:realbenign
```

## Files to Inspect

- `benign_runs/benign_fp_report.json`
- `logs/raw_events.jsonl`
- `logs/alerts.jsonl`

## Required Checks

For the latest run in `benign_fp_report.json`:

- `successful_requests` matches planned request count
- `runtime_scored_events` is high and `runtime_timeout_events` is near zero
- `fp_rate_any_alert`, `fp_rate_model_alert`, `fp_rate_novelty_alert` are interpreted as event-level FP rates

## Interpretation Guide

- `alerts_total` and `model_alerts` are row-level counts and can exceed requests.
- `alerted_events_any/model/novelty` are deduplicated event counts.
- For publication tables, report both row-level and event-level values with clear labeling.

## Current Snapshot Source

Use latest numbers directly from `benign_runs/benign_fp_report.json` when writing reports.
