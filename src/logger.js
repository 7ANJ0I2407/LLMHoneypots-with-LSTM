const fs = require("fs");
const path = require("path");

const LOG_DIR = path.join(__dirname, "..", "logs");
const RAW_LOG_PATH = path.join(LOG_DIR, "raw_events.jsonl");
const ALERT_LOG_PATH = path.join(LOG_DIR, "alerts.jsonl");

function ensureLogDir() {
  if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
  }
}

function appendJsonLine(filePath, payload) {
  ensureLogDir();
  fs.appendFileSync(filePath, `${JSON.stringify(payload)}\n`, "utf8");
}

function logEvent(event) {
  appendJsonLine(RAW_LOG_PATH, event);
}

function logAlert(alert) {
  appendJsonLine(ALERT_LOG_PATH, alert);
}

module.exports = {
  RAW_LOG_PATH,
  ALERT_LOG_PATH,
  logEvent,
  logAlert,
};
