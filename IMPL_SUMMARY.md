# Implementation Summary (Current)

## Server and Logging

Implemented:

- Express honeypot API endpoint
- Event feature extraction and session fingerprinting
- Raw event logging and alert logging
- Runtime scoring integration hooks

Primary files:

- `src/server.js`
- `src/featureExtractor.js`
- `src/logger.js`

## ML Pipeline

Implemented scripts:

- `scripts/preprocess.py`
- `scripts/train_lstm.py`
- `scripts/train_autoencoder.py`
- `scripts/detect_from_log.py`
- `scripts/validate_run.py`

Validation modes available now:

- Time split (`--dataset/--model/--meta`)
- Group K-Fold (`--group-kfold --n-splits N`)

## Runtime Calibration and Safety

Implemented:

- Runtime threshold calibration with explicit source modes
- Runtime novelty threshold mode control
- Runtime state reset utility
- Optional session persistence behavior for contamination control

Primary files:

- `scripts/calibrate_runtime_threshold.py`
- `scripts/reset_runtime_state.js`
- `src/server.js`

## Evaluation and Reporting

Implemented:

- Holdout-only fusion OR-vs-Blend comparison
- Dataset fingerprint tool
- Real benign runner with auto report generation
- Event-level deduplicated FP rate computation

Primary files:

- `scripts/compare_fusion_holdout.py`
- `scripts/dataset_fingerprint.py`
- `scripts/realbenign.js`

## Canonical npm Commands

```bash
npm run preprocess
npm run train
npm run train:novelty
npm run compare:fusion:holdout:tuned
npm run trigger:realbenign
```

## Artifacts to Cite

- `models/model_meta.json`
- `models/fusion_holdout_report.json`
- `benign_runs/benign_fp_report.json`
