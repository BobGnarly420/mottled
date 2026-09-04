/* ops-webgpu.js — the WebGPU backend for viewer/model.js.
 *
 * This implements the same four operations `cpuOps` does. It is deliberately
 * *only* an accelerator: `model.js` owns the forward pass, the CPU reference
 * is the source of truth, and `tests/test_model_ops_parity.js` runs both
 * backends over identical inputs so the GPU path cannot quietly diverge.
 *
 * Why hand-written kernels rather than a runtime: Mottled needs the residual
 * stream after every block. Chat-oriented runtimes return text (and sometimes
 * a final hidden state); none expose per-layer activations, and bolting that
 * onto a minified inference bundle is worse than owning four small kernels.
 *
 * Precision note: WebGPU shaders are f32, and the accumulation order in a
 * tiled matmul differs from the CPU loop's, so parity is checked to a
 * tolerance rather than bit-exactly — the same standard applied to any
 * GPU/CPU comparison. That parity check (`tests/parity.html`) has been run on
 * real hardware and passed. It cannot run in CI, which has no GPU, so it is
 * the manual step that must be repeated after changing a kernel here.
 *
 * UMD: `window.MottledOpsWebGPU` in the browser, `module.exports` under Node
 * (where it exports the shader sources so they can be parsed and checked
 * without a GPU). */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MottledOpsWebGPU = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const TILE = 16;

  /* y[m, n] = sum_k x[m, k] * w[n, k]
   * Weights are (out, in) row-major, matching HuggingFace's Linear layout, so
   * both operands are read along k and the inner loop is contiguous for w. */
  const MATMUL_WGSL = `
struct Dims { m: u32, k: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileX: array<f32, ${TILE * TILE}>;
var<workgroup> tileW: array<f32, ${TILE * TILE}>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  var acc = 0.0;
  let tiles = (dims.k + ${TILE}u - 1u) / ${TILE}u;

  for (var t = 0u; t < tiles; t = t + 1u) {
    let kx = t * ${TILE}u + lid.x;
    let kw = t * ${TILE}u + lid.y;
    if (row < dims.m && kx < dims.k) {
      tileX[lid.y * ${TILE}u + lid.x] = x[row * dims.k + kx];
    } else {
      tileX[lid.y * ${TILE}u + lid.x] = 0.0;
    }
    if (col < dims.n && kw < dims.k) {
      tileW[lid.y * ${TILE}u + lid.x] = w[col * dims.k + kw];
    } else {
      tileW[lid.y * ${TILE}u + lid.x] = 0.0;
    }
    workgroupBarrier();

    for (var i = 0u; i < ${TILE}u; i = i + 1u) {
      acc = acc + tileX[lid.y * ${TILE}u + i] * tileW[i * ${TILE}u + lid.x];
    }
    workgroupBarrier();
  }

  if (row < dims.m && col < dims.n) {
    y[row * dims.n + col] = acc;
  }
}`;

  /* One workgroup per row: reduce the sum of squares, then scale. */
  const RMSNORM_WGSL = `
struct Dims { rows: u32, dim: u32, eps: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;
  if (row >= dims.rows) { return; }
  let base = row * dims.dim;

  var acc = 0.0;
  var i = lid.x;
  loop {
    if (i >= dims.dim) { break; }
    let v = x[base + i];
    acc = acc + v * v;
    i = i + 256u;
  }
  partial[lid.x] = acc;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let scale = 1.0 / sqrt(partial[0] / f32(dims.dim) + dims.eps);
  var j = lid.x;
  loop {
    if (j >= dims.dim) { break; }
    y[base + j] = x[base + j] * scale * weight[j];
    j = j + 256u;
  }
}`;

  /* SwiGLU: silu(gate) * up, elementwise. */
  const SWIGLU_WGSL = `
struct Dims { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) { return; }
  let g = gate[i];
  y[i] = (g / (1.0 + exp(-g))) * up[i];
}`;

  const ADD_WGSL = `
struct Dims { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) { return; }
  y[i] = a[i] + b[i];
}`;

  const SHADERS = {
    matmul: MATMUL_WGSL, rmsNorm: RMSNORM_WGSL,
    swiglu: SWIGLU_WGSL, add: ADD_WGSL,
  };

  /* Create the backend. Returns an object with the same shape as `cpuOps`,
   * except every method returns a Promise — which is why `model.forward` is
   * async and awaits each op even on the CPU path. */
  async function create(options = {}) {
    if (typeof navigator === "undefined" || !navigator.gpu)
      throw new Error("WebGPU unavailable: this browser exposes no navigator.gpu");
    const adapter = await navigator.gpu.requestAdapter(options.adapter);
    if (!adapter) throw new Error("WebGPU: no adapter (software fallback is off?)");
    const device = await adapter.requestDevice(options.device);

    const pipelines = {};
    for (const [name, code] of Object.entries(SHADERS)) {
      const shader = device.createShaderModule({ code, label: `mottled:${name}` });
      pipelines[name] = device.createComputePipeline({
        layout: "auto", label: `mottled:${name}`,
        compute: { module: shader, entryPoint: "main" },
      });
    }

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
                  GPUBufferUsage.COPY_DST;

    function upload(data) {
      const buf = device.createBuffer({ size: Math.max(data.byteLength, 4), usage });
      device.queue.writeBuffer(buf, 0, data);
      return buf;
    }

    function uniform(values) {
      const buf = device.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, values);
      return buf;
    }

    async function readback(buf, bytes) {
      const staging = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return out;
    }

    function dispatch(name, buffers, dims, groups) {
      const pipeline = pipelines[name];
      const entries = buffers.map((buffer, i) => ({ binding: i, resource: { buffer } }));
      entries.push({ binding: buffers.length, resource: { buffer: dims } });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0), entries });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(groups[0], groups[1] ?? 1, groups[2] ?? 1);
      pass.end();
      device.queue.submit([enc.finish()]);
    }

    const ceil = (a, b) => Math.ceil(a / b);

    return {
      device,

      async matmul(x, w, m, k, n) {
        const bx = upload(x), bw = upload(w);
        const by = device.createBuffer({ size: Math.max(m * n * 4, 4), usage });
        const dims = uniform(new Uint32Array([m, k, n, 0]));
        dispatch("matmul", [bx, bw, by], dims, [ceil(n, TILE), ceil(m, TILE)]);
        const out = await readback(by, m * n * 4);
        bx.destroy(); bw.destroy(); by.destroy(); dims.destroy();
        return out;
      },

      async rmsNorm(x, weight, rows, dim, eps) {
        const bx = upload(x), bw = upload(weight);
        const by = device.createBuffer({ size: Math.max(rows * dim * 4, 4), usage });
        const dims = device.createBuffer({
          size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const view = new ArrayBuffer(16);
        new Uint32Array(view, 0, 2).set([rows, dim]);
        new Float32Array(view, 8, 1)[0] = eps;
        device.queue.writeBuffer(dims, 0, view);
        dispatch("rmsNorm", [bx, bw, by], dims, [rows]);
        const out = await readback(by, rows * dim * 4);
        bx.destroy(); bw.destroy(); by.destroy(); dims.destroy();
        return out;
      },

      async swiglu(gate, up, rows, inter) {
        const n = rows * inter;
        const bg = upload(gate), bu = upload(up);
        const by = device.createBuffer({ size: Math.max(n * 4, 4), usage });
        const dims = uniform(new Uint32Array([n, 0, 0, 0]));
        dispatch("swiglu", [bg, bu, by], dims, [ceil(n, 64)]);
        const out = await readback(by, n * 4);
        bg.destroy(); bu.destroy(); by.destroy(); dims.destroy();
        return out;
      },

      async add(a, b) {
        const ba = upload(a), bb = upload(b);
        const by = device.createBuffer({ size: Math.max(a.length * 4, 4), usage });
        const dims = uniform(new Uint32Array([a.length, 0, 0, 0]));
        dispatch("add", [ba, bb, by], dims, [ceil(a.length, 64)]);
        const out = await readback(by, a.length * 4);
        ba.destroy(); bb.destroy(); by.destroy(); dims.destroy();
        return out;
      },
    };
  }

  return { create, SHADERS, TILE };
});
