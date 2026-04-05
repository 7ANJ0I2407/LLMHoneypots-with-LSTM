# Quick Reference

## Core Commands

```bash
npm start
npm run dev
npm run setup:py
```

## Data + Training

```bash
npm run preprocess
npm run train
npm run train:novelty
npm run detect
```

## Validation

```bash
npm run validate
npm run validate:group10
```

## Runtime Calibration

```bash
npm run reset:runtime-state
npm run calibrate:runtime
npm run calibrate:novelty
npm run gate:session
npm run risk:timeline
```

## Triggering / Traffic

```bash
npm run seed
npm run trigger:lstm
npm run trigger:detect
npm run trigger:benign
npm run trigger:realbenign
npm run trigger:llm-agent
```

## Large-Artifacts

```bash
npm run preprocess:large
npm run train:large
npm run train:novelty:large
npm run validate:large
npm run validate:group10:large
npm run calibrate:runtime:large
npm run report:refresh:large
```

## Reporting and Comparison

```bash
npm run report:refresh
npm run compare:fusion:holdout
npm run compare:fusion:holdout:tuned
npm run ablation
```

## Current Artifact Files

- `models/model_meta.json`
- `models/fusion_holdout_report.json`
- `benign_runs/benign_fp_report.json`
