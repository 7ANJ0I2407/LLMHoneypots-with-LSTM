#!/usr/bin/env node
/**
 * Generate real benign traffic using genuine curl-like patterns.
 * This validates false-positive rate on traffic similar to real API usage,
 * not synthetically-generated benign requests.
 * 
 * Patterns tested:
 * 1. Curl POST requests (structured, terse)
 * 2. Python Requests library patterns
 * 3. Documentation/tutorial queries
 * 4. API debugging queries
 * 5. Normal application logs
 * 
 * Usage: node scripts/realbenign.js --count 200 --output benign_runs/run_1.json
 */

const fs = require("fs");
const path = require("path");
const { buildSessionId } = require("../src/featureExtractor");

const BASE_URL = process.env.BASE_URL || "http://localhost:8080";
const BENIGN_RUNS_DIR = "benign_runs";
const ALERTS_LOG_PATH = path.join("logs", "alerts.jsonl");
const RAW_EVENTS_LOG_PATH = path.join("logs", "raw_events.jsonl");
const DEFAULT_OUTPUT_FILE = path.join(BENIGN_RUNS_DIR, "run_latest.json");
const BENIGN_FP_REPORT_FILE = path.join(BENIGN_RUNS_DIR, "benign_fp_report.json");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkServerHealth(timeoutMs = 1500) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE_URL}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    return res.ok;
  } catch (_err) {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function postChat(content, temperature, maxTokens, sessionHeaders = {}, timeoutMs = 6000, label = "benign") {
  const payload = {
    model: "gpt-4o-mini",
    messages: [{ role: "user", content }],
    temperature,
    max_tokens: maxTokens,
    label,
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  let res;
  try {
    res = await fetch(`${BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-honeypot-label": label,
        ...sessionHeaders,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed: ${res.status} ${text}`);
  }

  // We only need status/latency for FP measurement.
  await res.arrayBuffer();
}

/**
 * Curl-like queries: Short, precise, structured API testing
 */
const curlLikeQueries = [
  "GET endpoint status check",
  "POST endpoint payload format check",
  "Health endpoint JSON verification",
  "Connectivity check for local API service",
  "Response time under 500ms expected",
  "Expected not-found behavior on unknown route",
  "Expected created response on valid create call",
  "Check content length header behavior",
  "Validate response schema structure",
  "Test timeout handling behavior",
];

/**
 * Python dev queries: Exploratory, documentation-focused
 */
const pythonDevQueries = [
  "import requests module guide",
  "how to set request headers",
  "timeout parameter in requests.get()",
  "handle unauthorized response in client",
  "parse JSON response from API",
  "retry logic for flaky endpoints",
  "mock requests library for testing",
  "write unit test for http client",
  "async requests with aiohttp",
  "gzip compression on requests",
];

/**
 * Documentation/Learning queries: Educational, benign
 */
const docQueries = [
  "explain REST API basics",
  "HTTP status codes reference table",
  "difference between POST and PUT",
  "when to use PATCH vs PUT",
  "CORS headers explained",
  "authentication best practices",
  "rate limiting strategies",
  "webhook implementation guide",
  "API versioning patterns",
  "caching with etag headers",
];

/**
 * Application logs/health queries: Operational
 */
const opsQueries = [
  "Startup: initializing database connection pool",
  "Health check: all services operational",
  "Request latency p95: 123ms",
  "Failure threshold: 0.1% acceptable",
  "Memory usage trending down",
  "CPU idle time: 67%",
  "Disk space available: 85GB",
  "Network throughput stable",
  "No alerts in prometheus",
  "Backup completed successfully",
];

/**
 * Browser/GUI-like queries: Typical end-user patternsh
 */
const guiQueries = [
  "Search results for python tutorial",
  "Show more details about that",
  "Can you explain that simpler",
  "Give me an example please",
  "How does that work",
  "What does this error mean",
  "How do I fix this",
  "Show me step by step guide",
  "What are best practices here",
  "Any gotchas I should know about",
];

function pick(arr, seed) {
  return arr[seed % arr.length];
}

function safeReadJson(filePath, fallback = null) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (_err) {
    return fallback;
  }
}

function safeReadJsonl(filePath) {
  if (!fs.existsSync(filePath)) {
    return [];
  }
  return fs
    .readFileSync(filePath, "utf8")
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

function buildBenignFpReport(runsDir, alertsPath, rawEventsPath, reportPath) {
  const files = fs
    .readdirSync(runsDir)
    .filter((name) => name.endsWith(".json") && name !== path.basename(reportPath))
    .sort();

  const alerts = safeReadJsonl(alertsPath);
  const rawEvents = safeReadJsonl(rawEventsPath);

  const runs = files
    .map((name) => {
      const filePath = path.join(runsDir, name);
      const data = safeReadJson(filePath, null);
      if (!data) {
        return null;
      }

      const sessionId = data.session_id;
      const runtimeSessionIds = Array.isArray(data.runtime_session_ids) && data.runtime_session_ids.length
        ? data.runtime_session_ids
        : [sessionId];
      const results = Array.isArray(data.results) ? data.results : [];
      const requestCount = Number.isFinite(data.request_count) ? data.request_count : results.length;
      const successfulRequests = results.filter((r) => r && r.success === true).length;

      const sessionAlerts = alerts.filter((a) => a && runtimeSessionIds.includes(a.sessionId));
      const sessionEvents = rawEvents.filter((e) => e && runtimeSessionIds.includes(e.sessionId));
      const noveltyAlerts = sessionAlerts.filter((a) => a.type === "novelty_behavior_alert" || a.type === "novelty_alert");
      const modelAlerts = sessionAlerts.filter((a) => (
        a.type === "lstm_behavior_alert"
        || a.type === "hybrid_online_alert"
        || a.type === "fusion_behavior_alert"
        || a.type === "novelty_behavior_alert"
        || a.type === "novelty_alert"
      ));

      const eventKey = (rec) => {
        const ev = rec.event || {};
        const sid = rec.sessionId || ev.sessionId || "";
        const ts = ev.timestamp || rec.createdAt || "";
        const preview = ev.promptPreview || "";
        return `${sid}|${ts}|${preview}`;
      };
      const uniqueAnyAlertEvents = new Set(sessionAlerts.map(eventKey));
      const uniqueNoveltyAlertEvents = new Set(noveltyAlerts.map(eventKey));
      const uniqueModelAlertEvents = new Set(modelAlerts.map(eventKey));

      const runtimeDetection = sessionEvents
        .map((e) => e.runtimeDetection)
        .filter((rd) => rd && typeof rd === "object");
      const runtimeScored = runtimeDetection.filter((rd) => rd.lstm && Number.isFinite(rd.lstm.score));
      const runtimeTimeout = runtimeDetection.filter((rd) => {
        const err = String(rd.error || "").toLowerCase();
        return err.includes("timeout");
      });

      const denom = Math.max(successfulRequests, 1);

      return {
        file: path.join(runsDir, name),
        session_id: sessionId,
        runtime_session_ids: runtimeSessionIds,
        request_count: requestCount,
        successful_requests: successfulRequests,
        runtime_events: sessionEvents.length,
        runtime_detection_events: runtimeDetection.length,
        runtime_scored_events: runtimeScored.length,
        runtime_timeout_events: runtimeTimeout.length,
        alerts_total: sessionAlerts.length,
        alerted_events_any: uniqueAnyAlertEvents.size,
        alerted_events_model: uniqueModelAlertEvents.size,
        alerted_events_novelty: uniqueNoveltyAlertEvents.size,
        novelty_alerts: noveltyAlerts.length,
        model_alerts: modelAlerts.length,
        fp_rate_any_alert: uniqueAnyAlertEvents.size / denom,
        fp_rate_model_alert: uniqueModelAlertEvents.size / denom,
        fp_rate_novelty_alert: uniqueNoveltyAlertEvents.size / denom,
      };
    })
    .filter(Boolean);

  const report = {
    generated_at: new Date().toISOString(),
    source: {
      benign_runs_dir: runsDir,
      alerts_log: alertsPath,
      raw_events_log: rawEventsPath,
    },
    runs,
  };

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  return report;
}

function buildSessionHeaders() {
  return {
    "x-forwarded-for": `10.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
    "user-agent": `Mozilla/5.0 (${["Linux", "Macintosh", "Windows"][Math.floor(Math.random() * 3)]}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36`,
  };
}

async function sendRealBenignSession(sessionId, count = 50, options = {}) {
  const {
    concurrency = 1,
    delayMinMs = 120,
    delayMaxMs = 350,
    timeoutMs = 15000,
    maxTokensMin = 64,
    maxTokensMax = 128,
    sessionPool = 2,
    progressEvery = 25,
  } = options;

  const safeDelayMin = Math.max(0, Math.min(delayMinMs, delayMaxMs));
  const safeDelayMax = Math.max(safeDelayMin, delayMaxMs);
  const safeMaxMin = Math.max(16, Math.min(maxTokensMin, maxTokensMax));
  const safeMaxMax = Math.max(safeMaxMin, maxTokensMax);

  const sessionHeadersPool = Array.from({ length: Math.max(1, sessionPool) }, () => buildSessionHeaders());

  const results = {
    session_id: sessionId,
    ip: sessionHeadersPool[0]["x-forwarded-for"],
    runtime_session_ids: sessionHeadersPool.map((headers) => buildSessionId(headers["x-forwarded-for"], headers["user-agent"])),
    request_count: count,
    results: [],
  };

  console.log(`[realbenign-${sessionId}] Starting ${count} requests...`);

  const effectiveConcurrency = Math.min(Math.max(1, concurrency), count);
  let nextRequest = 1;
  let completed = 0;

  async function worker() {
    while (nextRequest <= count) {
      const requestNum = nextRequest;
      nextRequest += 1;

      // Random mix of all benign patterns
      const pattern = Math.floor(Math.random() * 100);
      let content;
      if (pattern < 20) {
        content = pick(curlLikeQueries, requestNum);
      } else if (pattern < 40) {
        content = pick(pythonDevQueries, requestNum);
      } else if (pattern < 60) {
        content = pick(docQueries, requestNum);
      } else if (pattern < 80) {
        content = pick(opsQueries, requestNum);
      } else {
        content = pick(guiQueries, requestNum);
      }

      // Benign patterns: lower temperature, shorter responses.
      const temperature = 0.4 + Math.random() * 0.3;
      const maxTokens = safeMaxMin + Math.floor(Math.random() * (safeMaxMax - safeMaxMin + 1));

      // Keep jitter realistic but bounded so test does not stall local machine.
      const interArrivalMs = safeDelayMin + Math.floor(Math.random() * (safeDelayMax - safeDelayMin + 1));
      await sleep(interArrivalMs);

      const sessionHeaders = sessionHeadersPool[requestNum % sessionHeadersPool.length];

      try {
        const startTime = Date.now();
        await postChat(content, temperature, maxTokens, sessionHeaders, timeoutMs, "benign");
        const latency = Date.now() - startTime;

        results.results[requestNum - 1] = {
          request_num: requestNum,
          pattern: pattern < 20 ? "curl" : pattern < 40 ? "python_dev" : pattern < 60 ? "docs" : pattern < 80 ? "ops" : "gui",
          content_preview: content.substring(0, 40),
          temperature,
          max_tokens: maxTokens,
          latency_ms: latency,
          inter_arrival_ms: interArrivalMs,
          source_session_ip: sessionHeaders["x-forwarded-for"],
          success: true,
        };
      } catch (err) {
        results.results[requestNum - 1] = {
          request_num: requestNum,
          success: false,
          error: err.message,
        };
      }

      completed += 1;
      if (completed % progressEvery === 0 || completed === count) {
        const lastChunk = results.results.slice(Math.max(0, completed - progressEvery), completed).filter(Boolean);
        const avgLatency = lastChunk.length
          ? (lastChunk.filter((r) => r.success && r.latency_ms).reduce((a, r) => a + r.latency_ms, 0) / Math.max(lastChunk.filter((r) => r.success && r.latency_ms).length, 1))
          : 0;
        console.log(`[realbenign-${sessionId}] Sent ${completed}/${count} requests (avg latency ${avgLatency.toFixed(0)}ms)`);
      }
    }
  }

  await Promise.all(Array.from({ length: effectiveConcurrency }, () => worker()));

  console.log(`[realbenign-${sessionId}] Completed ${count} requests`);
  return results;
}

async function main() {
  let count = 200;
  let output_file = null;
  let concurrency = 1;
  let delayMinMs = 120;
  let delayMaxMs = 350;
  let timeoutMs = 15000;
  let sessionPool = 2;
  let maxTokensMin = 64;
  let maxTokensMax = 128;

  for (let i = 0; i < process.argv.length; i++) {
    if (process.argv[i] === "--count" && process.argv[i + 1]) {
      count = parseInt(process.argv[++i], 10);
    } else if (process.argv[i] === "--output" && process.argv[i + 1]) {
      output_file = process.argv[++i];
    } else if (process.argv[i] === "--concurrency" && process.argv[i + 1]) {
      concurrency = parseInt(process.argv[++i], 10);
    } else if (process.argv[i] === "--delay-min" && process.argv[i + 1]) {
      delayMinMs = parseInt(process.argv[++i], 10);
    } else if (process.argv[i] === "--delay-max" && process.argv[i + 1]) {
      delayMaxMs = parseInt(process.argv[++i], 10);
    } else if (process.argv[i] === "--timeout-ms" && process.argv[i + 1]) {
      timeoutMs = parseInt(process.argv[++i], 10);
    } else if (process.argv[i] === "--session-pool" && process.argv[i + 1]) {
      sessionPool = parseInt(process.argv[++i], 10);
    } else if (process.argv[i] === "--max-tokens-min" && process.argv[i + 1]) {
      maxTokensMin = parseInt(process.argv[++i], 10);
    } else if (process.argv[i] === "--max-tokens-max" && process.argv[i + 1]) {
      maxTokensMax = parseInt(process.argv[++i], 10);
    }
  }

  if (count < 1 || count > 5000) {
    console.error("Error: --count must be between 1 and 5000");
    process.exitCode = 1;
    return;
  }

  try {
    console.log(`=== REAL BENIGN TRAFFIC GENERATOR ===`);
    console.log(`Generating ${count} real benign API queries...`);
    console.log(`Patterns: curl, python-dev, documentation, ops, GUI`);
    console.log(`Load profile: concurrency=${concurrency}, delay=${delayMinMs}-${delayMaxMs}ms, timeout=${timeoutMs}ms, session_pool=${sessionPool}`);
    console.log();

    const healthy = await checkServerHealth();
    if (!healthy) {
      console.error(`Server check failed at ${BASE_URL}/health`);
      console.error(`Start the server first: npm start`);
      process.exitCode = 1;
      return;
    }

    const sessionId = `realbenign_${Date.now()}`;
    const results = await sendRealBenignSession(sessionId, count, {
      concurrency,
      delayMinMs,
      delayMaxMs,
      timeoutMs,
      sessionPool,
      maxTokensMin,
      maxTokensMax,
    });

    // Save results and refresh benign FP report every run.
    const resolvedOutputFile = output_file || DEFAULT_OUTPUT_FILE;
    const outputDir = path.dirname(resolvedOutputFile);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    fs.writeFileSync(resolvedOutputFile, JSON.stringify(results, null, 2));
    console.log(`\nResults saved: ${resolvedOutputFile}`);

    const report = buildBenignFpReport(BENIGN_RUNS_DIR, ALERTS_LOG_PATH, RAW_EVENTS_LOG_PATH, BENIGN_FP_REPORT_FILE);
    console.log(`Report updated: ${BENIGN_FP_REPORT_FILE} (runs=${report.runs.length})`);

    // Quick stats
    const successful = results.results.filter((r) => r.success).length;
    const avgLatency = results.results
      .filter((r) => r.success && r.latency_ms)
      .reduce((a, r) => a + r.latency_ms, 0) / successful || 0;

    console.log(`\nStats:`);
    console.log(`  Successful requests: ${successful}/${count}`);
    console.log(`  Average latency: ${avgLatency.toFixed(0)}ms`);
    console.log(
      `  Requests IP: ${results.ip}`
    );
    console.log();
    console.log(`Next steps:`);
    console.log(`  1. Open ${BENIGN_FP_REPORT_FILE} for current FP summary`);
    console.log(`  2. npm run detect  (if you want to refresh alerts from logs)`);
  } catch (err) {
    console.error("Error:", err.message);
    process.exitCode = 1;
  }
}

main();
