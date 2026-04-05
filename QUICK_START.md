# Quick Start (Current)

This is the shortest reliable path to run the system with current scripts.

## 1) Install once

```bash
npm install
npm run setup:py
```

## 2) Start honeypot server

```bash
npm start
```

## 3) Seed traffic

```bash
npm run demo:e2e
```

## 4) Train models

```bash
npm run preprocess
npm run train
npm run train:novelty
```

## 5) Validate

```bash
npm run validate
npm run validate:group10
```

## 6) Calibrate runtime + refresh report

```bash
npm run reset:runtime-state
npm run calibrate:runtime
npm run report:refresh
```

## 7) Benign FP check

```bash
npm run trigger:realbenign
```

Review:

- `benign_runs/benign_fp_report.json`
- `logs/alerts.jsonl`
- `logs/raw_events.jsonl`

## Important

- Use only scripts present in `package.json`.
- If you need large-artifact runs, use `*:large` commands.
- For per-event FP interpretation, prefer `fp_rate_*` over raw `alerts_total`.
