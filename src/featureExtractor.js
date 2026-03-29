const crypto = require("crypto");

const ATTACK_PATTERNS = [
  /ignore\s+all\s+previous\s+instructions/i,
  /reveal\s+system\s+prompt/i,
  /bypass/i,
  /jailbreak/i,
  /sudo|rm\s+-rf|wget|curl|powershell/i,
  /drop\s+table|union\s+select|or\s+1=1/i,
  /<script>|javascript:/i,
  /api[_-]?key|token|password|secret/i,
];

function getClientIp(req) {
  const forwarded = req.headers["x-forwarded-for"];
  if (forwarded) {
    return String(forwarded).split(",")[0].trim();
  }
  return req.socket.remoteAddress || "unknown";
}

function getPromptText(messages = []) {
  if (!Array.isArray(messages)) {
    return "";
  }

  return messages
    .map((m) => (typeof m?.content === "string" ? m.content : ""))
    .join("\n");
}

function buildSessionId(ip, userAgent) {
  return crypto
    .createHash("sha256")
    .update(`${ip}:${userAgent}`)
    .digest("hex")
    .slice(0, 16);
}

function attackSignalScore(promptText) {
  let score = 0;
  const hits = [];

  ATTACK_PATTERNS.forEach((regex) => {
    if (regex.test(promptText)) {
      score += 1;
      hits.push(regex.source);
    }
  });

  const upperRatio =
    promptText.length > 0
      ? (promptText.match(/[A-Z]/g) || []).length / promptText.length
      : 0;

  if (upperRatio > 0.35) {
    score += 0.5;
    hits.push("high_uppercase_ratio");
  }

  if (promptText.length > 2000) {
    score += 0.5;
    hits.push("very_long_prompt");
  }

  return { score, hits };
}

function buildEventRecord(req, body, replyText, latencyMs) {
  const timestamp = new Date().toISOString();
  const ip = getClientIp(req);
  const userAgent = req.headers["user-agent"] || "unknown";

  const messages = Array.isArray(body?.messages) ? body.messages : [];
  const promptText = getPromptText(messages);

  const { score: signalScore, hits: signalHits } = attackSignalScore(promptText);

  return {
    timestamp,
    ip,
    userAgent,
    sessionId: buildSessionId(ip, userAgent),
    endpoint: req.path,
    method: req.method,
    model: body?.model || "unknown",
    messageCount: messages.length,
    promptLength: promptText.length,
    temperature: Number.isFinite(body?.temperature)
      ? Number(body.temperature)
      : 1.0,
    maxTokens: Number.isFinite(body?.max_tokens)
      ? Number(body.max_tokens)
      : 0,
    latencyMs,
    signalScore,
    signalHits,
    promptPreview: promptText.slice(0, 280),
    responsePreview: String(replyText || "").slice(0, 160),
  };
}

module.exports = {
  buildEventRecord,
};
