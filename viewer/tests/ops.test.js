"use strict";
const test = require("node:test");
const assert = require("node:assert");

const M = require("../model.js");
const GPU = require("../ops-webgpu.js");

/* Two things are checked here, and a third deliberately is not.
 *
 * 1. `cpuOps` arithmetic, against values computed by hand. The Python-side
 *    `tests/test_model_conformance.py` already pins the whole forward pass
 *    against HuggingFace; this is the fast, dependency-free layer that says
 *    *which* op broke when that one goes red.
 *
 * 2. The WGSL shaders against the JavaScript that binds them. `dispatch()`
 *    binds three storage buffers and one uniform at bindings 0-3 and requests
 *    the entry point "main"; a shader edited out of step with that fails at
 *    pipeline creation in the browser and nowhere else. Parsing the WGSL
 *    properly needs a parser, and CI runs these tests with no npm
 *    dependencies, so this checks the interface contract rather than the
 *    grammar. (The shaders were validated against `wgsl_reflect` when
 *    written; the structural check is what survives into CI.)
 *
 * 3. NOT checked here: that the GPU kernels compute the right answers. That
 *    needs a GPU, which CI does not have. `parity.html` in this directory is
 *    where it is checked — both backends over identical inputs in a real
 *    browser — and it has been run and passed. Because that check is manual
 *    by necessity, re-run it after touching a kernel: nothing in CI will
 *    catch an arithmetic regression there.
 */

const near = (a, b, eps = 1e-6) =>
  assert.ok(Math.abs(a - b) <= eps, `expected ${a} ≈ ${b} (within ${eps})`);

test("matmul computes x W^T with W stored (out, in)", () => {
  // x is 2x3, W is 2x3 (out=2, in=3) -> y is 2x2
  const x = Float32Array.from([1, 2, 3, 4, 5, 6]);
  const w = Float32Array.from([1, 0, -1, 2, 1, 0]);
  const y = M.cpuOps.matmul(x, w, 2, 3, 2);
  // row0 . w0 = 1-3 = -2 ; row0 . w1 = 2+2 = 4
  // row1 . w0 = 4-6 = -2 ; row1 . w1 = 8+5 = 13
  assert.deepStrictEqual(Array.from(y), [-2, 4, -2, 13]);
});

test("rmsNorm scales by the root mean square and applies the weight", () => {
  const x = Float32Array.from([3, 4]);          // mean square = 12.5
  const w = Float32Array.from([1, 2]);
  const y = M.cpuOps.rmsNorm(x, w, 1, 2, 0);
  const scale = 1 / Math.sqrt(12.5);
  near(y[0], 3 * scale);
  near(y[1], 4 * scale * 2);
});

test("swiglu is silu(gate) * up", () => {
  const gate = Float32Array.from([0, 1]);
  const up = Float32Array.from([5, 2]);
  const y = M.cpuOps.swiglu(gate, up, 1, 2);
  near(y[0], 0);                                 // silu(0) = 0
  near(y[1], (1 / (1 + Math.exp(-1))) * 2);
});

test("rotary rotation pairs j with j + headDim/2, not adjacent lanes", () => {
  // At position 1 with headDim 4 the first pair rotates by 1 radian.
  const { cos, sin } = M.ropeTable(2, 4, 10000.0, null);
  const x = Float32Array.from([0, 0, 0, 0, 1, 0, 0, 0]);   // t=1, x[0] = 1
  const out = M.applyRope(x, 1, 2, 4, cos, sin);
  near(out[4], Math.cos(1), 1e-6);
  // The partner lane is index 0 + half = 2, so that is where the sine lands.
  near(out[6], Math.sin(1), 1e-6);
  near(out[5], 0, 1e-6);
});

test("attention is causal: the first token attends only to itself", () => {
  const T = 3, H = 1, D = 2;
  const q = Float32Array.from([1, 0, 0, 1, 1, 1]);
  const k = Float32Array.from([1, 0, 0, 1, 1, 1]);
  const v = Float32Array.from([10, 20, 30, 40, 50, 60]);
  const out = M.attention(q, k, v, T, H, H, D);
  assert.deepStrictEqual(Array.from(out.slice(0, 2)), [10, 20]);
});

test("grouped-query attention maps query heads onto kv heads by division", () => {
  // 4 query heads over 2 kv heads: heads 0,1 -> kv 0 and heads 2,3 -> kv 1.
  const T = 1, H = 4, KV = 2, D = 1;
  const q = Float32Array.from([1, 1, 1, 1]);
  const k = Float32Array.from([1, 1]);
  const v = Float32Array.from([7, 9]);
  const out = M.attention(q, k, v, T, H, KV, D);
  assert.deepStrictEqual(Array.from(out), [7, 7, 9, 9]);
});

test("every shader matches the bindings dispatch() supplies", () => {
  for (const [name, code] of Object.entries(GPU.SHADERS)) {
    const storage = [...code.matchAll(/@group\(0\)\s*@binding\((\d+)\)\s*var<storage/g)]
      .map((m) => Number(m[1]));
    const uniform = [...code.matchAll(/@group\(0\)\s*@binding\((\d+)\)\s*var<uniform/g)]
      .map((m) => Number(m[1]));

    assert.deepStrictEqual(storage, [0, 1, 2],
      `${name}: dispatch() binds three storage buffers at 0-2`);
    assert.deepStrictEqual(uniform, [3],
      `${name}: dispatch() binds the dims uniform at 3`);
    assert.ok(/@compute\s+@workgroup_size\([^)]+\)\s*\nfn main\(/.test(code),
      `${name}: the pipeline requests entryPoint "main"`);
    // The third storage buffer is the output and is written to.
    assert.ok(/@binding\(2\)\s*var<storage,\s*read_write>/.test(code),
      `${name}: binding 2 is the output and must be read_write`);
  }
});

test("the WebGPU backend refuses clearly where WebGPU is absent", async () => {
  // Node has no navigator.gpu; the failure has to name the reason rather than
  // throwing "cannot read property requestAdapter of undefined".
  await assert.rejects(() => GPU.create(), /WebGPU unavailable/);
});

test("forward is async so one implementation serves both backends", () => {
  assert.strictEqual(M.forward.constructor.name, "AsyncFunction");
  assert.strictEqual(M.logitLens.constructor.name, "AsyncFunction");
});
