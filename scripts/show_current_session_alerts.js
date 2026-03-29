const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const ALERTS_FILE = path.join(ROOT, "logs", "alerts.jsonl");
const LAST_TRIGGER_FILE = path.join(ROOT, "logs", "latest_trigger_session.json");

function parseArgs(argv) {
  const out = { last: 20 };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--last" && argv[i + 1]) {
      out.last = Number(argv[++i]);
    }
  }

  if (!Number.isFinite(out.last) || out.last < 1 || out.last > 200) {
    throw new Error("--last must be between 1 and 200");
  }

  return out;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function parseJsonl(filePath) {
  const raw = fs.readFileSync(filePath, "utf8");
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch (_err) {
        return null;
      }
    })
    .filter(Boolean);
}

function getEventSessionId(record) {
  if (record?.event?.sessionId) {
    return record.event.sessionId;
  }
  if (record?.sessionId) {
    return record.sessionId;
  }
  return null;
}

function parseTs(value) {
  if (!value) {
    return null;
  }
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? null : ms;
}

function getRecordTimeMs(record) {
  return parseTs(record?.createdAt) ?? parseTs(record?.event?.timestamp);
}

function main() {
  try {
    const args = parseArgs(process.argv.slice(2));

    if (!fs.existsSync(LAST_TRIGGER_FILE)) {
      throw new Error("No latest trigger session found. Run trigger first.");
    }
    if (!fs.existsSync(ALERTS_FILE)) {
      throw new Error("No alerts file found. Run detect first.");
    }

    const lastTrigger = readJson(LAST_TRIGGER_FILE);
    const sessionId = lastTrigger.sessionId;
    const triggerStartMs = parseTs(lastTrigger.startedAt || lastTrigger.createdAt);

    const allAlerts = parseJsonl(ALERTS_FILE);
    const filtered = allAlerts.filter((record) => {
      if (getEventSessionId(record) !== sessionId) {
        return false;
      }
      if (triggerStartMs == null) {
        return true;
      }
      const recordMs = getRecordTimeMs(record);
      if (recordMs == null) {
        return false;
      }
      return recordMs >= triggerStartMs;
    });
    const tail = filtered.slice(-args.last);

    const lstmCount = filtered.filter((r) => r.type === "lstm_behavior_alert").length;
    const ruleCount = filtered.filter((r) => r.type === "rule_based_pre_alert").length;

    console.log("=== CURRENT SESSION ALERTS ===");
    console.log(`sessionId=${sessionId}`);
    console.log(`ip=${lastTrigger.ip}`);
    console.log(`userAgent=${lastTrigger.userAgent}`);
    console.log(`total_filtered=${filtered.length}`);
    console.log(`rule_based_count=${ruleCount}`);
    console.log(`lstm_count=${lstmCount}`);

    if (tail.length === 0) {
      console.log("No alerts found for this session yet.");
      return;
    }

    console.log(`showing_last=${tail.length}`);
    tail.forEach((entry) => {
      console.log(JSON.stringify(entry));
    });
  } catch (err) {
    console.error(`[current-alerts] error: ${err.message}`);
    process.exitCode = 1;
  }
}

main();
