const { EventEmitter } = require("events");

const bus = new EventEmitter();
const RING_LIMIT = 500;
const ring = [];

function publish(topic, payload) {
  const record = {
    topic,
    createdAt: new Date().toISOString(),
    payload,
  };
  ring.push(record);
  if (ring.length > RING_LIMIT) {
    ring.shift();
  }
  bus.emit(topic, record);
}

function subscribe(topic, handler) {
  bus.on(topic, handler);
  return () => bus.off(topic, handler);
}

function snapshot(topic = null) {
  if (!topic) {
    return ring.slice();
  }
  return ring.filter((r) => r.topic === topic);
}

module.exports = {
  publish,
  subscribe,
  snapshot,
};
