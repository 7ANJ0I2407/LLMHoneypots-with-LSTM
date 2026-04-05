# Ready for Validation Checklist

## Current Validation Modes

- Time split: `npm run validate`
- GroupKFold: `npm run validate:group10`
- Holdout fusion compare: `npm run compare:fusion:holdout:tuned`

## Pre-Validation Steps

```bash
npm run reset:runtime-state
npm run data:fingerprint
npm run calibrate:runtime
```

## Standard Validation Run

```bash
npm run preprocess
npm run train
npm run train:novelty
npm run compare:fusion:holdout:tuned
```

## Benign Validation Run

```bash
npm run trigger:realbenign
```

## Required Output Files

- `models/model_meta.json`
- `models/fusion_holdout_report.json`
- `benign_runs/benign_fp_report.json`

## Sign-Off Criteria

- Commands execute without runtime errors
- Validation mode and dataset are explicitly recorded
- Benign report shows runtime scoring evidence
- Reported FP rates use event-level metrics (`fp_rate_*`)
