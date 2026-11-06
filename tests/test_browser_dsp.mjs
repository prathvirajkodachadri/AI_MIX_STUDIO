// Browser DSP tests for index.html — run with: node --test tests/test_browser_dsp.mjs
// Loads the page's <script> with a minimal DOM stub and exercises the pure
// DSP functions it exposes as globalThis.MixDSP.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

const elements = new Map();
// Run in THIS realm (like a browser tab). A separate vm context makes
// typed-array access artificially slow and would fake a performance bug.
globalThis.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, { id, textContent: "", innerHTML: "", style: {}, onclick: null, onchange: null });
    return elements.get(id);
  },
  createElement() { return { textContent: "", className: "", onclick: null }; },
};
new vm.Script(script).runInThisContext();
const MixDSP = globalThis.MixDSP;
assert.ok(MixDSP, "index.html should expose globalThis.MixDSP");

function tone(freq, seconds, amp = 0.5, sr = 48000) {
  const n = Math.floor(sr * seconds);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

test("stereo 440 Hz tone: levels, loudness, spectrum", () => {
  const L = tone(440, 1), R = tone(440, 1);
  const r = MixDSP.analyzeTrack({ name: "t", left: L, right: R, sr: 48000, duration: 1, channels: 2 });
  assert.ok(Math.abs(r.peak - -6.02) < 0.1, `peak ${r.peak}`);
  assert.ok(r.tp >= r.peak - 0.1, `true peak ${r.tp}`);
  assert.ok(Math.abs(r.rms - -9.03) < 0.2, `rms ${r.rms}`);
  assert.ok(r.corr > 0.999, `corr ${r.corr}`);
  assert.ok(r.lufs > -14 && r.lufs < -6, `lufs ${r.lufs}`);
  assert.ok(Math.abs(r.centroid - 440) < 30, `centroid ${r.centroid}`);
  const loudest = r.bands.indexOf(Math.max(...r.bands));
  assert.ok(Math.abs(r.centers[loudest] - 445) < 1, `loudest band ${r.centers[loudest]}`);
  assert.ok(Math.abs(r.bands[loudest] - -9.03) < 1.5, `band dB ${r.bands[loudest]}`);
});

test("anti-phase stereo is not silence", () => {
  const L = tone(440, 1);
  const R = Float32Array.from(L, v => -v);
  const r = MixDSP.analyzeTrack({ name: "t", left: L, right: R, sr: 48000, duration: 1, channels: 2 });
  assert.ok(Math.abs(r.corr - -1) < 1e-6, `corr ${r.corr}`);
  assert.ok(Math.abs(r.peak - -6.02) < 0.1, `peak ${r.peak}`);
  assert.ok(r.tp >= -6.2, `true peak ${r.tp}`);
});

test("mono track analyzes (was: no spectrum)", () => {
  const L = tone(440, 1);
  const r = MixDSP.analyzeTrack({ name: "m", left: L, right: L, sr: 48000, duration: 1, channels: 1 });
  assert.equal(r.corr, 1);
  assert.ok(Math.abs(r.centroid - 440) < 30, `centroid ${r.centroid}`);
});

test("full-length 3-minute stem analyzes fast (was ~76 s / frozen page)", () => {
  const L = tone(440, 180, 0.4), R = tone(440, 180, 0.4);
  const t0 = Date.now();
  const r = MixDSP.analyzeTrack({ name: "long", left: L, right: R, sr: 48000, duration: 180, channels: 2 });
  const ms = Date.now() - t0;
  assert.ok(ms < 10000, `took ${ms} ms`);
  assert.ok(r.lufs > -16 && r.lufs < -6, `lufs ${r.lufs}`);
});
