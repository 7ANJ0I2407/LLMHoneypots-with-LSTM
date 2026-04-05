# LLM Honeypot with LSTM + Novelty Detection

A local honeypot that mimics an OpenAI-style chat endpoint, logs behavior, and applies ML-based behavioral detection.

## Current Scope (April 2026)

- Node/Express server at `POST /v1/chat/completions`
- Structured telemetry logging to `logs/raw_events.jsonl`
- Alert logging to `logs/alerts.jsonl`
- Runtime scoring support (LSTM + novelty, with calibrated thresholds)
- Offline validation modes: time split, GroupKFold
- Benign false-positive reporting with automatic JSON generation

## Project Layout

- `src/server.js`: API server + runtime scoring integration
- `src/featureExtractor.js`: event features and session fingerprinting
- `scripts/preprocess.py`: dataset generation from raw logs
- `scripts/train_lstm.py`: LSTM train and model metadata output
- `scripts/train_autoencoder.py`: novelty model training
- `scripts/detect_from_log.py`: offline detection pass from logs
- `scripts/validate_run.py`: validation workflows
- `scripts/calibrate_runtime_threshold.py`: runtime threshold calibration
- `scripts/realbenign.js`: real benign traffic generator + report writer

## Setup

```bash
npm install
npm run setup:py
```

## Standard Workflow

### 1) Start server

```bash
npm start
```

### 2) Generate traffic

```bash
npm run demo:e2e
# or
npm run trigger:lstm
npm run trigger:realbenign
```

### 3) Build dataset and train

```bash
npm run preprocess
npm run train
npm run train:novelty
```

### 4) Validate

```bash
npm run validate
npm run validate:group10
```

### 5) Runtime calibration + refresh

```bash
npm run reset:runtime-state
npm run calibrate:runtime
npm run report:refresh
```

## Large-Artifact Workflow

```bash
npm run preprocess:large
npm run train:large
npm run train:novelty:large
npm run validate:large
npm run report:refresh:large
```

## Benign FP Reporting

Run benign traffic:

```bash
npm run trigger:realbenign
```

Outputs:

- `benign_runs/run_latest.json`
- `benign_runs/benign_fp_report.json`

Interpretation:

- `alerts_total` and `model_alerts` are alert-row counts (can be > requests)
- `alerted_events_any/model/novelty` are deduplicated event counts
- `fp_rate_*` fields are the event-level FP rates

## Current Notes

- Preprocessing currently uses sliding windows and scaling only (no SMOTE/augmentation flags in CLI).
- Runtime fusion defaults are configured via `models/model_meta.json` and calibration scripts.
- Use measured artifacts for reporting, not projected metrics.
