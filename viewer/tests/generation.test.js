"use strict";
const test = require("node:test");
const assert = require("node:assert");

const MTJ = require("../mtj.js");

// Build a synthetic .mtj in memory (mirrors the writer layout in the spec).
function buildMtj({ manifest, blob = new Uint8Array(0), version = 1, magic = "MTRJ" }) {
  let json = Buffer.from(JSON.stringify(manifest), "utf-8");
  const pad = (16 - ((12 + json.length) % 16)) % 16;
  json = Buffer.concat([json, Buffer.alloc(pad, 0x20)]);
  const header = Buffer.alloc(12);
  header.write(magic, 0, "ascii");
  header.writeUInt32LE(version, 4);
  header.writeUInt32LE(json.length, 8);
  return Buffer.concat([header, json, Buffer.from(blob.buffer || blob)]);
}

// Minimal scene: 2x2 terrain plus one run of N trajectories x 2 layers.
function sceneWithRun(runFields, nTrajs) {
  const pts = [];
  for (let i = 0; i < nTrajs * 2; i++) pts.push(i, i, i);
  const blob = new Uint8Array(64 + Math.ceil((pts.length * 4) / 16) * 16);
  const put = (off, vals) => new Uint8Array(blob.buffer, off, vals.length * 4)
    .set(new Uint8Array(new Float32Array(vals).buffer));
  put(0, [0, 1]); put(16, [0, 1]); put(32, [0, 0.5, 0.5, 1]); put(64, pts);
  return buildMtj({
    manifest: {
      format: "mottled-trajectory", version: 1, kind: "scene",
      terrain: { x: "x", y: "y", z: "z" },
      runs: [Object.assign({ label: "A", prompt: "p", points: "pts" }, runFields)],
      arrays: {
        x: { dtype: "float32", shape: [2], offset: 0, length: 8 },
        y: { dtype: "float32", shape: [2], offset: 16, length: 8 },
        z: { dtype: "float32", shape: [2, 2], offset: 32, length: 16 },
        pts: { dtype: "float32", shape: [nTrajs, 2, 3], offset: 64, length: nTrajs * 24 },
      },
    },
    blob,
  });
}

const GEN = {
  prompt_tokens: 2, new_tokens: 2, mode: "greedy", temperature: 0.0, seed: null,
  steps: [{ token: " is", id: 318, p: 0.42, entropy: 2.31 },
          { token: " paris", id: 6342, p: 0.87, entropy: 0.55 }],
};

test("isGeneratedToken splits prompt from decode axis and is null-safe", () => {
  assert.strictEqual(MTJ.isGeneratedToken(GEN, 0), false);
  assert.strictEqual(MTJ.isGeneratedToken(GEN, 1), false);
  assert.strictEqual(MTJ.isGeneratedToken(GEN, 2), true);
  assert.strictEqual(MTJ.isGeneratedToken(GEN, 3), true);
  assert.strictEqual(MTJ.isGeneratedToken(null, 3), false);
  assert.strictEqual(MTJ.isGeneratedToken(undefined, 3), false);
  assert.strictEqual(MTJ.isGeneratedToken({}, 3), false); // no prompt_tokens
  assert.strictEqual(MTJ.isGeneratedToken({ prompt_tokens: "2" }, 3), false);
  assert.strictEqual(MTJ.isGeneratedToken(GEN, "2"), false);
});

test("generationStep returns the per-step record for generated tokens only", () => {
  assert.strictEqual(MTJ.generationStep(GEN, 0), null);
  assert.strictEqual(MTJ.generationStep(GEN, 1), null);
  assert.deepStrictEqual(MTJ.generationStep(GEN, 2), GEN.steps[0]);
  assert.deepStrictEqual(MTJ.generationStep(GEN, 3), GEN.steps[1]);
  assert.strictEqual(MTJ.generationStep(GEN, 4), null);   // past the record
  assert.strictEqual(MTJ.generationStep(null, 2), null);
  assert.strictEqual(MTJ.generationStep({ prompt_tokens: 2 }, 2), null); // no steps
});

test("continuationText joins step tokens (space-joined for synthetic)", () => {
  assert.strictEqual(MTJ.continuationText(GEN), " is paris");
  assert.strictEqual(MTJ.continuationText(GEN, "torch"), " is paris");
  const bare = { steps: [{ token: "the" }, { token: "capital" }] };
  assert.strictEqual(MTJ.continuationText(bare, "synthetic"), "the capital");
  assert.strictEqual(MTJ.continuationText(bare), "thecapital");
  assert.strictEqual(MTJ.continuationText(null), "");
  assert.strictEqual(MTJ.continuationText({}), "");
  assert.strictEqual(MTJ.continuationText({ steps: [{}, { token: "x" }] }), "x");
});

test("decodeSummary shows mode, count, and temperature only when sampling", () => {
  assert.strictEqual(MTJ.decodeSummary(GEN), "greedy · +2 tokens");
  assert.strictEqual(
    MTJ.decodeSummary({ mode: "sample", temperature: 0.8, new_tokens: 5 }),
    "sample · T=0.8 · +5 tokens");
  assert.strictEqual(MTJ.decodeSummary({ mode: "greedy", new_tokens: 1 }),
                     "greedy · +1 token");
  // missing new_tokens falls back to the number of recorded steps
  assert.strictEqual(MTJ.decodeSummary({ mode: "greedy", steps: [{}, {}, {}] }),
                     "greedy · +3 tokens");
  assert.strictEqual(MTJ.decodeSummary(null), "");
  assert.strictEqual(MTJ.decodeSummary("greedy"), "");
});

test("generationBoundary needs an aligned all-tokens run", () => {
  const scene = MTJ.loadScene(sceneWithRun({
    tokens: ["the", "cat", "+sat", "+down"],
    trajectory_labels: ["the", "cat", "+sat", "+down"],
    generation: { prompt_tokens: 2, new_tokens: 2, mode: "greedy", steps: [] },
  }, 4));
  // labels must literally equal tokens, whatever the exporter wrote
  assert.strictEqual(MTJ.generationBoundary(scene.runs[0]), 2);

  // misaligned labels (e.g. mean-trajectory export): no boundary
  const off = MTJ.loadScene(sceneWithRun({
    tokens: ["the", "cat", "sat", "down"],
    trajectory_labels: ["mean", "cat", "sat", "down"],
    generation: { prompt_tokens: 2 },
  }, 4));
  assert.strictEqual(MTJ.generationBoundary(off.runs[0]), -1);

  // single-token export: fewer trajectories than tokens
  const single = MTJ.loadScene(sceneWithRun({
    tokens: ["the", "cat", "sat", "down"],
    trajectory_labels: ["down"],
    generation: { prompt_tokens: 2 },
  }, 1));
  assert.strictEqual(MTJ.generationBoundary(single.runs[0]), -1);

  // degenerate boundaries
  const all = MTJ.loadScene(sceneWithRun({
    tokens: ["a", "b"], trajectory_labels: ["a", "b"],
    generation: { prompt_tokens: 2 },  // nothing generated
  }, 2));
  assert.strictEqual(MTJ.generationBoundary(all.runs[0]), -1);
  assert.strictEqual(MTJ.generationBoundary(null), -1);
  assert.strictEqual(MTJ.generationBoundary({ generation: null }), -1);
});

test("loadScene passes generation through and defaults it to null", () => {
  const withGen = MTJ.loadScene(sceneWithRun({
    tokens: ["a", "+b"], trajectory_labels: ["a", "+b"], generation: GEN,
  }, 2));
  assert.deepStrictEqual(withGen.runs[0].generation, GEN);

  const without = MTJ.loadScene(sceneWithRun({
    tokens: ["a", "b"], trajectory_labels: ["a", "b"],
  }, 2));
  assert.strictEqual(without.runs[0].generation, null);

  // malformed records (wrong JSON type) are dropped, not passed through
  const bad = MTJ.loadScene(sceneWithRun({
    tokens: ["a", "b"], trajectory_labels: ["a", "b"], generation: [1, 2],
  }, 2));
  assert.strictEqual(bad.runs[0].generation, null);
});
