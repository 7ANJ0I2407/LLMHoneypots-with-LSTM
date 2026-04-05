# Publication Readiness (Current State)

## Status

This repository has implemented the key guardrails needed for defensible reporting:

- Reproducible runtime calibration defaults (`dataset` source + `keep-meta` novelty mode)
- Runtime state reset support before refresh flows
- Holdout-only OR vs Blend fusion comparison script
- Dataset fingerprint utility for provenance checks
- Benign FP reporting with runtime evidence and event-level deduplicated rates

## What to Report from This Repo

Use measured values from these artifacts only:

- `models/model_meta.json`
- `models/fusion_holdout_report.json`
- `benign_runs/benign_fp_report.json`

Avoid projected metrics not backed by the above files.

## Recommended Publication Pipeline

```bash
npm run reset:runtime-state
npm run preprocess
npm run train
npm run train:novelty
npm run calibrate:runtime
npm run compare:fusion:holdout:tuned
npm run trigger:realbenign
```

## Large-Artifact Pipeline

```bash
npm run preprocess:large
npm run train:large
npm run train:novelty:large
npm run calibrate:runtime:large
npm run validate:group10:large
npm run report:refresh:large
```

## Required Interpretation Notes

- Runtime FP interpretation should use `alerted_events_*` and `fp_rate_*` fields.
- `alerts_total` may exceed request count due to multiple alert rows on the same event.
- Holdout OR-vs-Blend comparisons should be taken from `models/fusion_holdout_report.json` only.

## Remaining Caveats

- Metric values are data-state dependent; retraining or reseeding changes outcomes.
- Always include dataset fingerprint and artifact timestamps in paper logs.
