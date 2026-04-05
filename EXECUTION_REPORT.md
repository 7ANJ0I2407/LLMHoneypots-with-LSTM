# Execution Report (Latest Snapshot)

## Context

This report summarizes the latest measurable artifacts currently present in the workspace.

## Model Metadata (`models/model_meta.json`)

Current recorded values:

- `threshold`: 0.1
- `accuracy`: 0.9818181818181818
- `f1`: 0.988235294117647
- `samples`: 275
- `positive_samples`: 90
- `split_mode`: time

## Benign FP Report (`benign_runs/benign_fp_report.json`)

Latest run snapshot:

- `request_count`: 200
- `successful_requests`: 200
- `runtime_scored_events`: 200
- `runtime_timeout_events`: 0
- `alerts_total`: 257
- `alerted_events_any`: 200
- `alerted_events_model`: 200
- `alerted_events_novelty`: 200
- `fp_rate_any_alert`: 1.0
- `fp_rate_model_alert`: 1.0
- `fp_rate_novelty_alert`: 1.0

Interpretation:

- Row counts (`alerts_total`, `model_alerts`) and event-level rates are different metrics.
- Use `fp_rate_*` for event-level FP reporting.

## Fusion Holdout Report (`models/fusion_holdout_report.json`)

Latest available comparison artifact includes:

- OR mode metrics
- Blend mode metrics
- Tuned blend threshold under benign FPR constraint
- Delta metrics (`blend - or`)

## Reproducibility Notes

- Runtime recalibration behavior is controlled in calibration script options.
- Runtime state reset command exists and should be run before fresh report cycles.
- Dataset fingerprint utility is available and should be logged with experiment outputs.

## Recommended Re-run Sequence

```bash
npm run reset:runtime-state
npm run preprocess
npm run train
npm run train:novelty
npm run calibrate:runtime
npm run compare:fusion:holdout:tuned
npm run trigger:realbenign
```
