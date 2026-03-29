const express = require("express");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { logEvent, logAlert } = require("./logger");
const { buildEventRecord } = require("./featureExtractor");

const app = express();
const PORT = process.env.PORT || 8080;
const ROOT = path.join(__dirname, "..");
const MODEL_PATH = path.join(ROOT, "models", "lstm_detector.pt");
const META_PATH = path.join(ROOT, "models", "model_meta.json");
const SCALER_PATH = path.join(ROOT, "models", "scaler.json");
const ONLINE_SCORER = path.join(ROOT, "scripts", "online_score.py");
const PYTHON_BIN = path.join(ROOT, ".venv", "bin", "python");

const sessionState = new Map();
const sessionMaxWindow = 30;

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value)));
}

function hasModelArtifacts() {
  return (
    fs.existsSync(MODEL_PATH)
    && fs.existsSync(META_PATH)
    && fs.existsSync(SCALER_PATH)
    && fs.existsSync(ONLINE_SCORER)
    && fs.existsSync(PYTHON_BIN)
  );
}

function buildVectorFromEvent(event, interArrivalSec) {
  const current = new Date(event.timestamp);
  const hourFloat = current.getUTCHours() + current.getUTCMinutes() / 60;
  const angle = (2 * Math.PI * hourFloat) / 24;

  return [
    Number(event.promptLength || 0),
    Number(event.messageCount || 0),
    Number(event.signalScore || 0),
    Number(event.latencyMs || 0),
    Number(event.temperature || 1),
    Number(event.maxTokens || 0),
    Number(interArrivalSec || 0),
    Number((event.signalHits || []).length),
    Math.sin(angle),
    Math.cos(angle),
  ];
}

function updateSessionWindow(event) {
  const sid = event.sessionId;
  const currentMs = Date.parse(event.timestamp);
  const state = sessionState.get(sid) || { vectors: [], lastMs: null };

  const interArrivalSec = state.lastMs == null ? 0 : Math.max(0, (currentMs - state.lastMs) / 1000);
  const vec = buildVectorFromEvent(event, interArrivalSec);

  state.vectors.push(vec);
  if (state.vectors.length > sessionMaxWindow) {
    state.vectors.shift();
  }
  state.lastMs = currentMs;
  sessionState.set(sid, state);

  return state.vectors;
}

function computeHybridRisk(event, lstmScore) {
  const ruleProxy = clamp01((event.signalScore || 0) / 4);
  const burstProxy = clamp01(1 - Math.min(Number(event.latencyMs || 0), 500) / 500);
  const hybridRisk = clamp01((lstmScore * 0.55) + (ruleProxy * 0.35) + (burstProxy * 0.1));

  let severity = "low";
  if (hybridRisk >= 0.85) {
    severity = "critical";
  } else if (hybridRisk >= 0.65) {
    severity = "high";
  } else if (hybridRisk >= 0.4) {
    severity = "medium";
  }

  const factors = [
    {
      name: "lstm_sequence_score",
      impact: Number(lstmScore.toFixed(4)),
      detail: "Window-level anomaly confidence",
    },
    {
      name: "rule_signal_proxy",
      impact: Number(ruleProxy.toFixed(4)),
      detail: "Pattern-based suspiciousness in latest event",
    },
    {
      name: "signal_hit_count_proxy",
      impact: Number(clamp01((event.signalHits || []).length / 4).toFixed(4)),
      detail: "Number of direct attack indicators in prompt",
    },
  ];

  return { hybridRisk, severity, topFactors: factors };
}

function scoreWindowOnline(window) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, [
      ONLINE_SCORER,
      "--model", MODEL_PATH,
      "--meta", META_PATH,
      "--scaler", SCALER_PATH,
    ], {
      cwd: ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `online scorer exited with code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch (_err) {
        reject(new Error("online scorer returned invalid JSON"));
      }
    });

    child.stdin.write(JSON.stringify({ window }));
    child.stdin.end();
  });
}

app.use(express.json({ limit: "2mb" }));

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "llm-honeypot" });
});

app.post("/v1/chat/completions", async (req, res) => {
  const started = Date.now();

  const fallbackReply =
    "I can help with that. Please provide more context so I can generate a precise response.";

  const firstMessage = Array.isArray(req.body?.messages)
    ? req.body.messages[0]?.content
    : "";

  const baitReply = firstMessage
    ? `Processing request: ${String(firstMessage).slice(0, 80)}...`
    : fallbackReply;

  const responsePayload = {
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: req.body?.model || "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: baitReply,
        },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };

  const latencyMs = Date.now() - started;
  const event = buildEventRecord(req, req.body, baitReply, latencyMs);

  const vectors = updateSessionWindow(event);
  const runtimeDetection = {
    onlineScoringEnabled: hasModelArtifacts(),
    sessionEventCount: vectors.length,
  };

  if (runtimeDetection.onlineScoringEnabled) {
    try {
      const result = await scoreWindowOnline(vectors);
      if (result?.ok) {
        runtimeDetection.lstm = {
          score: result.score,
          threshold: result.threshold,
          decision: result.decision,
          windowSize: result.windowSize,
          confidenceBand: Math.abs(result.score - result.threshold) <= 0.07 ? "uncertain" : "confident",
        };

        const hybrid = computeHybridRisk(event, result.score);
        runtimeDetection.hybridRisk = hybrid.hybridRisk;
        runtimeDetection.severity = hybrid.severity;
        runtimeDetection.topFactors = hybrid.topFactors;

        if (hybrid.hybridRisk >= 0.65) {
          logAlert({
            type: "hybrid_online_alert",
            createdAt: new Date().toISOString(),
            sessionId: event.sessionId,
            severity: hybrid.severity,
            hybridRisk: hybrid.hybridRisk,
            threshold: result.threshold,
            lstmScore: result.score,
            confidenceBand: runtimeDetection.lstm.confidenceBand,
            topFactors: hybrid.topFactors,
            event,
          });
        }
      }
    } catch (err) {
      runtimeDetection.error = String(err.message || err);
    }
  }

  event.runtimeDetection = runtimeDetection;
  logEvent(event);

  if (event.signalScore >= 2) {
    logAlert({
      type: "rule_based_pre_alert",
      severity: event.signalScore >= 3 ? "high" : "medium",
      createdAt: new Date().toISOString(),
      event,
    });
  }

  res.status(200).json(responsePayload);
});

app.use((err, _req, res, _next) => {
  const status = err.status || 500;
  res.status(status).json({ error: "Invalid request" });
});

app.listen(PORT, () => {
  console.log(`[honeypot] listening on port ${PORT}`);
});
