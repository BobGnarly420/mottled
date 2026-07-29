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

// Minimal scene: 2x2 terrain plus one run of 2 trajectories x 2 layers, and
// SAE feature arrays for L=2 layers x T=2 tokens laid out after the points.
//   fre  = recon_error (2,)   float32  [0.22, 0.80]
//   fid  = top_id      (2,2)  int32    [[10, 11], [12, 13]]
//   fact = top_act     (2,2)  float32  [[1.5, 2.5], [3.5, 4.5]]
function sceneWithFeatures(featFields) {
  const pts = [];
  for (let i = 0; i < 2 * 2; i++) pts.push(i, i, i);
  const blob = new Uint8Array(144);
  const putF = (off, vals) => new Uint8Array(blob.buffer, off, vals.length * 4)
    .set(new Uint8Array(new Float32Array(vals).buffer));
  const putI = (off, vals) => new Uint8Array(blob.buffer, off, vals.length * 4)
    .set(new Uint8Array(new Int32Array(vals).buffer));
  putF(0, [0, 1]); putF(16, [0, 1]); putF(32, [0, 0.5, 0.5, 1]); putF(48, pts);
  putF(96, [0.22, 0.8]); putI(112, [10, 11, 12, 13]); putF(128, [1.5, 2.5, 3.5, 4.5]);
  return buildMtj({
    manifest: {
      format: "mottled-trajectory", version: 1, kind: "scene",
      terrain: { x: "x", y: "y", z: "z" },
      runs: [Object.assign({ label: "A", prompt: "p", points: "pts",
                             tokens: ["a", "b"] }, featFields)],
      arrays: {
        x: { dtype: "float32", shape: [2], offset: 0, length: 8 },
        y: { dtype: "float32", shape: [2], offset: 16, length: 8 },
        z: { dtype: "float32", shape: [2, 2], offset: 32, length: 16 },
        pts: { dtype: "float32", shape: [2, 2, 3], offset: 48, length: 48 },
        fre: { dtype: "float32", shape: [2], offset: 96, length: 8 },
        fid: { dtype: "int32", shape: [2, 2], offset: 112, length: 16 },
        fact: { dtype: "float32", shape: [2, 2], offset: 128, length: 16 },
      },
    },
    blob,
  });
}

const FULL_FEATURES = {
  source: "jbloom/GPT2-Small-SAEs-Reformatted", hook: "blocks.8.hook_resid_pre",
  best_layer: 0, recon_error: "fre", top_id: "fid", top_act: "fact",
};

// A resolved-record shape for the pure helpers (what loadScene hands back).
const FEATS = {
  source: "jbloom/GPT2-Small-SAEs-Reformatted", hook: "blocks.8.hook_resid_pre",
  best_layer: 0,
  recon_error: { dtype: "float32", shape: [2], data: new Float32Array([0.22, 0.8]) },
  top_id: { dtype: "int32", shape: [2, 3], data: new Int32Array([10, 11, 12, 13, 14, 15]) },
  top_act: { dtype: "float32", shape: [2, 3], data: new Float32Array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]) },
};

test("featureAt reads the dominant feature and is bounds-checked", () => {
  assert.deepStrictEqual(MTJ.featureAt(FEATS, 0, 0), { id: 10, act: 1.5 });
  assert.deepStrictEqual(MTJ.featureAt(FEATS, 0, 2), { id: 12, act: 3.5 });
  assert.deepStrictEqual(MTJ.featureAt(FEATS, 1, 1), { id: 14, act: 5.5 });
  assert.strictEqual(MTJ.featureAt(FEATS, 2, 0), null);   // layer out of range
  assert.strictEqual(MTJ.featureAt(FEATS, 0, 3), null);   // token out of range
  assert.strictEqual(MTJ.featureAt(FEATS, -1, 0), null);
  assert.strictEqual(MTJ.featureAt(FEATS, 0, -1), null);
  assert.strictEqual(MTJ.featureAt(FEATS, 0.5, 0), null); // non-integer indices
  assert.strictEqual(MTJ.featureAt(FEATS, "0", 0), null);
});

test("featureAt is null-safe against absent or malformed records", () => {
  assert.strictEqual(MTJ.featureAt(null, 0, 0), null);
  assert.strictEqual(MTJ.featureAt(undefined, 0, 0), null);
  assert.strictEqual(MTJ.featureAt({}, 0, 0), null);
  assert.strictEqual(MTJ.featureAt({ top_id: FEATS.top_id }, 0, 0), null); // no top_act
  const flat = { top_id: { shape: [6], data: FEATS.top_id.data }, top_act: FEATS.top_act };
  assert.strictEqual(MTJ.featureAt(flat, 0, 0), null); // wrong rank
});

test("featureFitSummary reports dictionary, best layer, and fit error", () => {
  assert.strictEqual(
    MTJ.featureFitSummary(FEATS),
    "features: jbloom/GPT2-Small-SAEs-Reformatted · best fit layer 0 · 22% err");
  // recon_error above 0.5 at the best layer flags extrapolation
  assert.strictEqual(
    MTJ.featureFitSummary(Object.assign({}, FEATS, { best_layer: 1 })),
    "features: jbloom/GPT2-Small-SAEs-Reformatted · best fit layer 1 · 80% err · EXTRAPOLATION");
  // missing/out-of-range best_layer falls back to the argmin
  assert.strictEqual(
    MTJ.featureFitSummary(Object.assign({}, FEATS, { best_layer: null })),
    "features: jbloom/GPT2-Small-SAEs-Reformatted · best fit layer 0 · 22% err");
  assert.strictEqual(
    MTJ.featureFitSummary(Object.assign({}, FEATS, { best_layer: 99 })),
    "features: jbloom/GPT2-Small-SAEs-Reformatted · best fit layer 0 · 22% err");
  // a null source still summarizes
  assert.strictEqual(
    MTJ.featureFitSummary(Object.assign({}, FEATS, { source: null })),
    "features: ? · best fit layer 0 · 22% err");
});

test("featureFitSummary is empty for null/malformed input", () => {
  assert.strictEqual(MTJ.featureFitSummary(null), "");
  assert.strictEqual(MTJ.featureFitSummary(undefined), "");
  assert.strictEqual(MTJ.featureFitSummary("features"), "");
  assert.strictEqual(MTJ.featureFitSummary({}), "");
  assert.strictEqual(MTJ.featureFitSummary({ recon_error: { data: new Float32Array(0) } }), "");
});

test("loadScene resolves the features record's array refs", () => {
  const scene = MTJ.loadScene(sceneWithFeatures({ features: FULL_FEATURES }));
  const f = scene.runs[0].features;
  assert.ok(f);
  assert.strictEqual(f.source, "jbloom/GPT2-Small-SAEs-Reformatted");
  assert.strictEqual(f.hook, "blocks.8.hook_resid_pre");
  assert.strictEqual(f.best_layer, 0);
  assert.deepStrictEqual(f.recon_error.shape, [2]);
  assert.deepStrictEqual(f.top_id.shape, [2, 2]);
  assert.deepStrictEqual(f.top_act.shape, [2, 2]);
  assert.ok(f.top_id.data instanceof Int32Array);
  assert.deepStrictEqual(MTJ.featureAt(f, 1, 1), { id: 13, act: 4.5 });
  assert.ok(Math.abs(f.recon_error.data[0] - 0.22) < 1e-6);
});

test("loadScene defaults features to null when absent", () => {
  const scene = MTJ.loadScene(sceneWithFeatures({}));
  assert.strictEqual(scene.runs[0].features, null);
});

test("loadScene drops malformed features records", () => {
  for (const bad of [[1, 2], "features", 7, true]) {
    const scene = MTJ.loadScene(sceneWithFeatures({ features: bad }));
    assert.strictEqual(scene.runs[0].features, null);
  }
});

test("loadScene drops features with missing or dangling array refs", () => {
  // dangling: names that resolve to no array in the container
  const dangling = MTJ.loadScene(sceneWithFeatures({
    features: Object.assign({}, FULL_FEATURES, { top_act: "nope" }),
  }));
  assert.strictEqual(dangling.runs[0].features, null);
  // missing members drop the whole record too
  const missing = MTJ.loadScene(sceneWithFeatures({
    features: { source: "s", hook: "h", best_layer: 0, recon_error: "fre" },
  }));
  assert.strictEqual(missing.runs[0].features, null);
  // non-string refs never resolve
  const nonString = MTJ.loadScene(sceneWithFeatures({
    features: Object.assign({}, FULL_FEATURES, { top_id: 3 }),
  }));
  assert.strictEqual(nonString.runs[0].features, null);
});
