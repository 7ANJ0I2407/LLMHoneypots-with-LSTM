#!/usr/bin/env node
/**
 * Trigger benign-only traffic to test novelty detection calibration
 * This sends exactly 20 completely normal requests (no attacks) to see
 * what novelty_alert count we get on clean traffic.
 * Usage: node scripts/trigger_benign_test.js
 */

const BASE_URL = process.env.BASE_URL || "http://localhost:8080";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function postChat(content, temperature, maxTokens, sessionHeaders = {}) {
  const payload = {
    model: "gpt-4o-mini",
    messages: [{ role: "user", content }],
    temperature,
    max_tokens: maxTokens,
    label: "benign",
  };

  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-honeypot-label": "benign",
      ...sessionHeaders,
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed: ${res.status} ${text}`);
  }
  return res.json();
}

async function main() {
  const normalPrompts = [
    "Explain recursion with a tiny example.",
    "Write a two-line summary of photosynthesis.",
    "How does binary search work?",
    "Give me a short study plan for DBMS.",
    "Differentiate TCP and UDP in short points.",
    "Explain B-Tree insertion in 4 steps.",
    "What are relational algebra operations?",
    "Describe deadlock prevention in OS.",
    "How does network routing work?",
    "Explain polymorphism in OOP.",
  ];

  const sessionHeaders = {
    "x-forwarded-for": "192.0.2.10",
    "user-agent": "benign-test-client/1.0",
  };

  try {
    console.log("[benign-test] Sending 20 completely normal requests...");
    
    for (let i = 1; i <= 20; i++) {
      const content = normalPrompts[i % normalPrompts.length];
      const temperature = 0.5 + ((i % 3) * 0.1);
      const maxTokens = 100 + ((i % 5) * 40);
      
      console.log(`[benign-test] Request ${i}/20: "${content.substring(0, 50)}..."`);
      await postChat(content, temperature, maxTokens, sessionHeaders);
      
      // Normal inter-arrival time (slower, more benign pattern)
      await sleep(150 + Math.random() * 100);
    }
    
    console.log("[benign-test] All 20 benign requests completed.");
    console.log("[benign-test] Check logs/alerts.jsonl for novelty_alerts count (should be close to 0)");
  } catch (err) {
    console.error("[benign-test] Error:", err.message);
    process.exitCode = 1;
  }
}

main();
