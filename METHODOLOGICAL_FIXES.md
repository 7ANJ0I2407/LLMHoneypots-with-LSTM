# Methodological Fixes Log

## Scope

This document captures the major method corrections now present in code.

## Implemented Fixes

### 1) Reproducible Runtime Calibration

- Script: `scripts/calibrate_runtime_threshold.py`
- Added explicit calibration source mode (`dataset/live/auto`)
- Added runtime novelty mode (`keep-meta/recompute`)
- Defaults now favor reproducibility (`dataset` + `keep-meta`)

### 2) Runtime State Hygiene

- Script: `scripts/reset_runtime_state.js`
- Clears stale state files before report refresh runs
- Prevents accidental cross-run session contamination

### 3) Session Persistence Control

- Server behavior made opt-in for persistence (environment-gated)
- Reduces hidden state carry-over in repeated evaluations

### 4) Holdout-Only Fusion Comparison

- Script: `scripts/compare_fusion_holdout.py`
- Compares OR and Blend decisions on holdout partitions
- Optional tuning with benign-FPR constraint

### 5) Dataset Provenance

- Script: `scripts/dataset_fingerprint.py`
- Validation outputs include dataset hash to track exact input state

### 6) Benign FP Report Corrections

- Script: `scripts/realbenign.js`
- Benign report auto-generated after runs
- Includes runtime evidence fields (`runtime_scored_events`, timeouts)
- FP-rate logic deduplicated at event level

## Deprecated or Removed Doc Claims

These are no longer valid in the current CLI and should not be used:

- `preprocess:augment`
- `preprocess:smote`
- `preprocess:full`
- `pipeline:augmented`
- `publication:test`
- `demo:large`

## Reference Artifacts

- `models/model_meta.json`
- `models/fusion_holdout_report.json`
- `benign_runs/benign_fp_report.json`
