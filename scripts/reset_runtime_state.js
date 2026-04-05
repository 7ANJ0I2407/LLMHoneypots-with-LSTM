#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const args = new Set(process.argv.slice(2));
const fullReset = args.has("--full") || args.has("--clear-logs");

const targets = [
  path.join(ROOT, "logs", "session_state.json"),
  path.join(ROOT, "logs", "latest_trigger_session.json"),
  path.join(ROOT, "logs", "session_gate_history.jsonl"),
];

if (fullReset) {
  targets.push(
    path.join(ROOT, "logs", "raw_events.jsonl"),
    path.join(ROOT, "logs", "alerts.jsonl"),
  );
}

let removed = 0;
for (const target of targets) {
  try {
    if (fs.existsSync(target)) {
      fs.rmSync(target, { force: true });
      removed += 1;
      console.log(`removed=${path.relative(ROOT, target)}`);
    }
  } catch (err) {
    console.error(`failed=${path.relative(ROOT, target)} error=${err.message}`);
    process.exitCode = 1;
  }
}

if (!process.exitCode) {
  console.log(`runtime_state_reset=OK removed_files=${removed}`);
}
