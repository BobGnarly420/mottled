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
// inspector arrays for L=2 layers x T=2 tokens x k=2 neighbors laid out after
// the points.
//   nidx   = idx              (2,2,2) int32   into ["alpha", "beta", "gamma"]
//   nsim   = sim              (2,2,2) float32 cosine, nearest first
//   shares = component_shares (1,2,2) float32 — ONE block, writing layer 1
function sceneWithInspector(inspectorFields) {
  const pts = [];
  for (let i = 0; i < 2 * 2; i++) pts.push(i, i, i);
  const blob = new Uint8Array(176);
  const putF = (off, vals) => new Uint8Array(blob.buffer, off, vals.length * 4)
    .set(new Uint8Array(new Float32Array(vals).buffer));
  const putI = (off, vals) => new Uint8Array(blob.buffer, off, vals.length * 4)
    .set(new Uint8Array(new Int32Array(vals).buffer));
  putF(0, [0, 1]); putF(16, [0, 1]); putF(32, [0, 0.5, 0.5, 1]); putF(48, pts);
  putI(96, [0, 1, 1, 2, 2, 0, 0, 2]);
  putF(128, [0.9, 0.5, 0.8, 0.4, 0.7, 0.3, 0.6, 0.2]);
  putF(160, [0.75, 0.25, 0.25, 0.75]);
  return buildMtj({
    manifest: {
      format: "mottled-trajectory", version: 1, kind: "scene",
      terrain: { x: "x", y: "y", z: "z" },
      runs: [Object.assign({ label: "A", prompt: "p", points: "pts",
                             tokens: ["a", "b"] }, inspectorFields)],
      arrays: {
        x: { dtype: "float32", shape: [2], offset: 0, length: 8 },
        y: { dtype: "float32", shape: [2], offset: 16, length: 8 },
        z: { dtype: "float32", shape: [2, 2], offset: 32, length: 16 },
        pts: { dtype: "float32", shape: [2, 2, 3], offset: 48, length: 48 },
        nidx: { dtype: "int32", shape: [2, 2, 2], offset: 96, length: 32 },
        nsim: { dtype: "float32", shape: [2, 2, 2], offset: 128, length: 32 },
        shares: { dtype: "float32", shape: [1, 2, 2], offset: 160, length: 16 },
      },
    },
    blob,
  });
}

const FULL_INSPECTOR = {
  tokens: ["alpha", "beta", "gamma"],
  idx: "nidx", sim: "nsim", component_shares: "shares",
};

// A resolved-record shape for the pure helpers (what loadScene hands back):
// L=2 layers, T=2 tokens, k=3 neighbors; component_shares has 2 blocks, so it
// describes the states at layers 1 and 2.
const INSP = {
  tokens: ["alpha", "beta", "gamma", "delta"],
  idx: { dtype: "int32", shape: [2, 2, 3],
         data: new Int32Array([0, 1, 2, 1, 2, 3, 3, 2, 1, 2, 0, 1]) },
  sim: { dtype: "float32", shape: [2, 2, 3],
         data: new Float32Array([0.91, 0.62, 0.4, 0.88, 0.5, 0.31,
                                 0.77, 0.55, 0.2, 0.7, 0.44, -0.1]) },
  component_shares: { dtype: "float32", shape: [2, 2, 2],
                      data: new Float32Array([0.7, 0.3, 0.4, 0.6,
                                              0.25, 0.75, 0.5, 0.5]) },
};

const sims = (rows) => rows.map((r) => Number(r.sim.toFixed(2)));
const toks = (rows) => rows.map((r) => r.token);

test("neighborsAt reads the neighbor list nearest-first", () => {
  const n = MTJ.neighborsAt(INSP, 0, 0, 3);
  assert.deepStrictEqual(toks(n), ["alpha", "beta", "gamma"]);
  assert.deepStrictEqual(sims(n), [0.91, 0.62, 0.4]);
  // similarities are non-increasing: the export orders them nearest-first
  for (let i = 1; i < n.length; i++) assert.ok(n[i].sim <= n[i - 1].sim);
  assert.deepStrictEqual(toks(MTJ.neighborsAt(INSP, 0, 1, 3)), ["beta", "gamma", "delta"]);
  assert.deepStrictEqual(toks(MTJ.neighborsAt(INSP, 1, 0, 3)), ["delta", "gamma", "beta"]);
  assert.deepStrictEqual(sims(MTJ.neighborsAt(INSP, 1, 1, 3)), [0.7, 0.44, -0.1]);
});

test("neighborsAt honors the limit and clamps it to k", () => {
  assert.deepStrictEqual(toks(MTJ.neighborsAt(INSP, 0, 0, 2)), ["alpha", "beta"]);
  assert.deepStrictEqual(MTJ.neighborsAt(INSP, 0, 0, 0), []);
  assert.strictEqual(MTJ.neighborsAt(INSP, 0, 0, 99).length, 3); // clamped to k
  assert.strictEqual(MTJ.neighborsAt(INSP, 0, 0, -4).length, 0);
  assert.strictEqual(MTJ.neighborsAt(INSP, 0, 0).length, 3);     // no limit: all k
  assert.strictEqual(MTJ.neighborsAt(INSP, 0, 0, 2.5).length, 3); // non-integer: all k
});

test("neighborsAt is bounds-checked", () => {
  assert.deepStrictEqual(MTJ.neighborsAt(INSP, 2, 0, 3), []);   // layer out of range
  assert.deepStrictEqual(MTJ.neighborsAt(INSP, 0, 2, 3), []);   // token out of range
  assert.deepStrictEqual(MTJ.neighborsAt(INSP, -1, 0, 3), []);
  assert.deepStrictEqual(MTJ.neighborsAt(INSP, 0, -1, 3), []);
  assert.deepStrictEqual(MTJ.neighborsAt(INSP, 0.5, 0, 3), []); // non-integer indices
  assert.deepStrictEqual(MTJ.neighborsAt(INSP, "0", 0, 3), []);
});

test("neighborsAt is null-safe against absent, partial, or malformed records", () => {
  assert.deepStrictEqual(MTJ.neighborsAt(null, 0, 0, 3), []);
  assert.deepStrictEqual(MTJ.neighborsAt(undefined, 0, 0, 3), []);
  assert.deepStrictEqual(MTJ.neighborsAt({}, 0, 0, 3), []);
  assert.deepStrictEqual(MTJ.neighborsAt("inspector", 0, 0, 3), []);
  // one half of the pair is not a neighbor list
  assert.deepStrictEqual(MTJ.neighborsAt({ tokens: INSP.tokens, idx: INSP.idx }, 0, 0, 3), []);
  assert.deepStrictEqual(MTJ.neighborsAt({ tokens: INSP.tokens, sim: INSP.sim }, 0, 0, 3), []);
  // a shares-only record (no embedding matrix at capture) reads as no neighbors
  assert.deepStrictEqual(
    MTJ.neighborsAt({ tokens: [], idx: null, sim: null,
                      component_shares: INSP.component_shares }, 1, 0, 3), []);
  // wrong rank
  const flat = { tokens: INSP.tokens, sim: INSP.sim,
                 idx: { shape: [12], data: INSP.idx.data } };
  assert.deepStrictEqual(MTJ.neighborsAt(flat, 0, 0, 3), []);
  // no token table, or indices that point off the end of it: those rows drop
  assert.deepStrictEqual(MTJ.neighborsAt(Object.assign({}, INSP, { tokens: [] }), 0, 0, 3), []);
  assert.deepStrictEqual(
    toks(MTJ.neighborsAt(Object.assign({}, INSP, { tokens: ["alpha", "beta"] }), 0, 0, 3)),
    ["alpha", "beta"]);
  // a truncated data buffer stops the walk rather than reading undefined
  const short = Object.assign({}, INSP, {
    sim: { shape: [2, 2, 3], data: INSP.sim.data.subarray(0, 2) },
  });
  assert.deepStrictEqual(toks(MTJ.neighborsAt(short, 0, 0, 3)), ["alpha", "beta"]);
});

test("componentShareAt maps a layer to the block that wrote it", () => {
  // block 0 writes layer 1, block 1 writes layer 2 — the shares for layer l
  // live at index l-1
  const s10 = MTJ.componentShareAt(INSP, 1, 0);
  assert.deepStrictEqual(Object.keys(s10), ["attn", "mlp"]);
  assert.ok(Math.abs(s10.attn - 0.7) < 1e-6 && Math.abs(s10.mlp - 0.3) < 1e-6);
  const s11 = MTJ.componentShareAt(INSP, 1, 1);
  assert.ok(Math.abs(s11.attn - 0.4) < 1e-6 && Math.abs(s11.mlp - 0.6) < 1e-6);
  const s20 = MTJ.componentShareAt(INSP, 2, 0);
  assert.ok(Math.abs(s20.attn - 0.25) < 1e-6 && Math.abs(s20.mlp - 0.75) < 1e-6);
  const s21 = MTJ.componentShareAt(INSP, 2, 1);
  assert.ok(Math.abs(s21.attn - 0.5) < 1e-6 && Math.abs(s21.mlp - 0.5) < 1e-6);
  // each pair sums to 1
  for (const s of [s11, s20, s21]) assert.ok(Math.abs(s.attn + s.mlp - 1) < 1e-6);
});

test("componentShareAt returns null for layer 0 and out-of-range indices", () => {
  assert.strictEqual(MTJ.componentShareAt(INSP, 0, 0), null); // embedding stream
  assert.strictEqual(MTJ.componentShareAt(INSP, 0, 1), null);
  assert.strictEqual(MTJ.componentShareAt(INSP, 3, 0), null); // past the last block
  assert.strictEqual(MTJ.componentShareAt(INSP, -1, 0), null);
  assert.strictEqual(MTJ.componentShareAt(INSP, 1, 2), null); // token out of range
  assert.strictEqual(MTJ.componentShareAt(INSP, 1, -1), null);
  assert.strictEqual(MTJ.componentShareAt(INSP, 1.5, 0), null); // non-integer
  assert.strictEqual(MTJ.componentShareAt(INSP, 1, "0"), null);
});

test("componentShareAt is null-safe against absent, partial, or malformed records", () => {
  assert.strictEqual(MTJ.componentShareAt(null, 1, 0), null);
  assert.strictEqual(MTJ.componentShareAt(undefined, 1, 0), null);
  assert.strictEqual(MTJ.componentShareAt({}, 1, 0), null);
  assert.strictEqual(MTJ.componentShareAt("inspector", 1, 0), null);
  // a neighbors-only record (capture without residual components)
  assert.strictEqual(
    MTJ.componentShareAt({ tokens: INSP.tokens, idx: INSP.idx, sim: INSP.sim,
                           component_shares: null }, 1, 0), null);
  // wrong rank / too few components / truncated buffer
  assert.strictEqual(MTJ.componentShareAt(
    { component_shares: { shape: [8], data: INSP.component_shares.data } }, 1, 0), null);
  assert.strictEqual(MTJ.componentShareAt(
    { component_shares: { shape: [2, 2, 1], data: INSP.component_shares.data } }, 1, 0), null);
  assert.strictEqual(MTJ.componentShareAt(
    { component_shares: { shape: [2, 2, 2],
                          data: INSP.component_shares.data.subarray(0, 1) } }, 1, 0), null);
});

test("loadScene resolves the inspector record's array refs", () => {
  const scene = MTJ.loadScene(sceneWithInspector({ inspector: FULL_INSPECTOR }));
  const e = scene.runs[0].inspector;
  assert.ok(e);
  assert.deepStrictEqual(e.tokens, ["alpha", "beta", "gamma"]);
  assert.deepStrictEqual(e.idx.shape, [2, 2, 2]);
  assert.deepStrictEqual(e.sim.shape, [2, 2, 2]);
  assert.deepStrictEqual(e.component_shares.shape, [1, 2, 2]);
  assert.ok(e.idx.data instanceof Int32Array);
  assert.ok(e.sim.data instanceof Float32Array);
  // ... and the helpers read straight off the resolved record
  const n = MTJ.neighborsAt(e, 0, 0, 2);
  assert.deepStrictEqual(toks(n), ["alpha", "beta"]);
  assert.ok(Math.abs(n[0].sim - 0.9) < 1e-6);
  assert.deepStrictEqual(toks(MTJ.neighborsAt(e, 1, 1, 2)), ["alpha", "gamma"]);
  const share = MTJ.componentShareAt(e, 1, 0);
  assert.ok(Math.abs(share.attn - 0.75) < 1e-6 && Math.abs(share.mlp - 0.25) < 1e-6);
  // the single block writes layer 1 only
  assert.strictEqual(MTJ.componentShareAt(e, 0, 0), null);
  assert.strictEqual(MTJ.componentShareAt(e, 2, 0), null);
});

test("loadScene defaults inspector to null when absent", () => {
  const scene = MTJ.loadScene(sceneWithInspector({}));
  assert.strictEqual(scene.runs[0].inspector, null);
});

test("loadScene drops malformed inspector records", () => {
  for (const bad of [[1, 2], "inspector", 7, true, null]) {
    const scene = MTJ.loadScene(sceneWithInspector({ inspector: bad }));
    assert.strictEqual(scene.runs[0].inspector, null);
  }
});

test("loadScene keeps whichever inspector members resolve", () => {
  // a run captured without residual components: neighbors only
  const noShares = MTJ.loadScene(sceneWithInspector({
    inspector: { tokens: ["alpha", "beta", "gamma"], idx: "nidx", sim: "nsim" },
  })).runs[0].inspector;
  assert.ok(noShares);
  assert.strictEqual(noShares.component_shares, null);
  assert.strictEqual(MTJ.componentShareAt(noShares, 1, 0), null);
  assert.strictEqual(MTJ.neighborsAt(noShares, 0, 0, 2).length, 2);

  // a producer with no embedding matrix: shares only
  const noNeighbors = MTJ.loadScene(sceneWithInspector({
    inspector: { component_shares: "shares" },
  })).runs[0].inspector;
  assert.ok(noNeighbors);
  assert.strictEqual(noNeighbors.idx, null);
  assert.strictEqual(noNeighbors.sim, null);
  assert.deepStrictEqual(noNeighbors.tokens, []);
  assert.deepStrictEqual(MTJ.neighborsAt(noNeighbors, 0, 0, 2), []);
  assert.ok(MTJ.componentShareAt(noNeighbors, 1, 0));

  // a dangling ref drops just that member
  const dangling = MTJ.loadScene(sceneWithInspector({
    inspector: Object.assign({}, FULL_INSPECTOR, { component_shares: "nope" }),
  })).runs[0].inspector;
  assert.ok(dangling);
  assert.strictEqual(dangling.component_shares, null);
  assert.ok(dangling.idx && dangling.sim);

  // non-string refs never resolve
  const nonString = MTJ.loadScene(sceneWithInspector({
    inspector: Object.assign({}, FULL_INSPECTOR, { idx: 3, sim: { name: "nsim" } }),
  })).runs[0].inspector;
  assert.ok(nonString);
  assert.strictEqual(nonString.idx, null);
  assert.strictEqual(nonString.sim, null);
  assert.ok(nonString.component_shares);
});

test("loadScene drops an inspector record whose refs all dangle", () => {
  const scene = MTJ.loadScene(sceneWithInspector({
    inspector: { tokens: ["alpha"], idx: "gone", sim: "gone", component_shares: "gone" },
  }));
  assert.strictEqual(scene.runs[0].inspector, null);
  // an empty record (no members at all) is nothing to resolve either
  assert.strictEqual(
    MTJ.loadScene(sceneWithInspector({ inspector: { tokens: ["alpha"] } })).runs[0].inspector,
    null);
});

test("a scene without an inspector record still loads and reads null", () => {
  const scene = MTJ.loadScene(sceneWithInspector({}));
  assert.strictEqual(scene.runs[0].inspector, null);
  assert.deepStrictEqual(MTJ.neighborsAt(scene.runs[0].inspector, 0, 0, 5), []);
  assert.strictEqual(MTJ.componentShareAt(scene.runs[0].inspector, 1, 0), null);
});

test("loadScene surfaces the per-run model, null when absent", () => {
  // the scene-level meta describes run 0, which is wrong for a scene whose
  // runs *are* different models — so each run carries its own
  const fs = require("node:fs"), path = require("node:path");
  const load = (name) => MTJ.loadScene(new Uint8Array(fs.readFileSync(
    path.join(__dirname, "..", "samples", name))).buffer);

  assert.deepEqual(load("models-qwen-gpt2.mtj").runs.map((r) => r.model),
                   ["Qwen/Qwen2.5-1.5B-Instruct", "gpt2"]);
  // a scene written before the field existed reads null, not undefined
  assert.ok(load("self-portrait.mtj").runs.every((r) => r.model === null));
});
