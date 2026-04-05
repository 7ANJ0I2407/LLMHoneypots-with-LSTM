const { subscribe } = require('../src/eventBus');
const { logEvent } = require('../src/logger');

let pass = false;
let unsubscribe = () => {};
unsubscribe = subscribe('raw_event', () => {
  pass = true;
  console.log('event_bus_raw_event=PASS');
  unsubscribe();
});

logEvent({ timestamp: new Date().toISOString(), sessionId: 'selftest', signalScore: 0 });

setTimeout(() => {
  if (!pass) {
    console.log('event_bus_raw_event=FAIL');
    process.exit(1);
  }
  process.exit(0);
}, 250);
