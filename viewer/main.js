/* Mottled web viewer — raw WebGL2, no dependencies. World is z-up to match
 * the .mtj scene convention (terrain z = height, trajectory xyz draped). */
"use strict";

// ---------------------------------------------------------------- constants
// Incision data palette — accent blue, live teal, risk amber, loss red,
// payout green, then lightened variants. Mirrors ui.py's _MARBLE_COLORS.
const PALETTE = ["#4B7CF3", "#00CCA8", "#D4934A", "#E05050", "#38B07A",
                 "#8FA7F7", "#5CE0C6", "#E6B884", "#F08A8A", "#7FD0AC"];
// draw/skip fine-segment run lengths, cycled per run: solid, dash, dot, longdash, dashdot
const DASH_CYCLE = [null, [5, 3], [1, 2], [9, 3], [5, 2, 1, 2]];
const SEG = 8;              // Catmull-Rom subdivisions per layer span
const OVERLAY_ALPHA = 0.55; // runs after the first
const PICK_RADIUS = 14;     // px
// Terrain potential ramp: void -> surfaces -> precision blue -> light blue.
// Mirrors ui.py's _TERRAIN_COLORSCALE.
const TERRAIN_RAMP = [[0.016, 0.024, 0.055], [0.031, 0.047, 0.102], [0.047, 0.063, 0.125],
                      [0.078, 0.118, 0.271], [0.110, 0.165, 0.333], [0.184, 0.333, 0.722],
                      [0.294, 0.486, 0.953], [0.475, 0.616, 0.965], [0.784, 0.831, 0.984]];

function terrainColor(t) {
  t = Math.min(1, Math.max(0, t)) * (TERRAIN_RAMP.length - 1);
  const i = Math.min(TERRAIN_RAMP.length - 2, Math.floor(t)), f = t - i;
  const a = TERRAIN_RAMP[i], b = TERRAIN_RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function hexRGB(h) {
  return [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255,
          parseInt(h.slice(5, 7), 16) / 255];
}

// ---------------------------------------------------------------- mat4 (column-major)
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0,
                           0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(eye, c, up) {
  let zx = eye[0] - c[0], zy = eye[1] - c[1], zz = eye[2] - c[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
                           -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
                           -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
                           -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1]);
}
function matMul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
function projectPoint(m, p, w, h) { // -> [px, py, clipW] in CSS pixels
  const cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  const x = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / cw;
  const y = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / cw;
  return [(x * 0.5 + 0.5) * w, (0.5 - y * 0.5) * h, cw];
}

// ---------------------------------------------------------------- GL helpers
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", { antialias: true });

function makeProgram(vsSrc, fsSrc) {
  const mk = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
function makeVAO(program, buffers) { // buffers: [{name, size, data|length}]
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const out = { vao, bufs: {} };
  for (const b of buffers) {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, b.data || b.length * 4, b.data ? gl.STATIC_DRAW : gl.DYNAMIC_DRAW);
    const loc = gl.getAttribLocation(program, b.name);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, b.size, gl.FLOAT, false, 0, 0);
    out.bufs[b.name] = buf;
  }
  gl.bindVertexArray(null);
  return out;
}

const meshProg = gl ? makeProgram(
  `#version 300 es
   in vec3 pos; in vec3 nrm; in vec3 col;
   uniform mat4 mvp; out vec3 vc; out vec3 vn;
   void main() { gl_Position = mvp * vec4(pos, 1.0); vc = col; vn = nrm; }`,
  `#version 300 es
   precision highp float; in vec3 vc; in vec3 vn; out vec4 o;
   void main() {
     float d = 0.55 + 0.45 * max(dot(normalize(vn), normalize(vec3(0.35, 0.25, 1.0))), 0.0);
     o = vec4(vc * d, 1.0);
   }`) : null;

const lineProg = gl ? makeProgram(
  `#version 300 es
   in vec3 pos; in vec4 col;
   uniform mat4 mvp; out vec4 vc;
   void main() { gl_Position = mvp * vec4(pos, 1.0); vc = col; }`,
  `#version 300 es
   precision highp float; in vec4 vc; out vec4 o;
   void main() { o = vc; }`) : null;

const pointProg = gl ? makeProgram(
  `#version 300 es
   in vec3 pos; in vec4 col; in float size;
   uniform mat4 mvp; out vec4 vc;
   void main() { gl_Position = mvp * vec4(pos, 1.0); gl_PointSize = size; vc = col; }`,
  `#version 300 es
   precision highp float; in vec4 vc; uniform float rim; out vec4 o;
   void main() {
     float r = length(gl_PointCoord - 0.5);
     if (r > 0.5) discard;
     vec3 c = (rim > 0.5 && r > 0.40) ? vec3(1.0) : vc.rgb;
     o = vec4(c, vc.a * smoothstep(0.5, 0.45, r));
   }`) : null;

// ---------------------------------------------------------------- app state
const state = {
  scene: null,        // MTJ.loadScene result
  runs: [],           // per-run render data
  terrain: null,
  attn: null,         // attention line buffer
  bounds: null,
  layerF: 0, L: 1,
  playing: false, speed: 2,
  showAttention: false,
  visible: [],
  pick: null, hover: null,
  cam: { theta: -2.2, phi: 0.65, dist: 10, target: [0, 0, 0] },
  mouse: null,
};

function catmullRom(pts, L) { // pts: Float32Array (L*3) -> densified Float32Array
  const n = (L - 1) * SEG + 1, out = new Float32Array(n * 3);
  const P = (i) => { i = Math.min(L - 1, Math.max(0, i)); return [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]]; };
  for (let s = 0; s < L - 1; s++) {
    const p0 = P(s - 1), p1 = P(s), p2 = P(s + 1), p3 = P(s + 2);
    for (let k = 0; k < SEG; k++) {
      const t = k / SEG, t2 = t * t, t3 = t2 * t;
      for (let d = 0; d < 3; d++)
        out[(s * SEG + k) * 3 + d] = 0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t +
          (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3);
    }
  }
  out.set(P(L - 1), (n - 1) * 3);
  return out;
}

function buildTerrain(t) {
  const W = t.x.shape[0], H = t.y.shape[0], xs = t.x.data, ys = t.y.data, zs = t.z.data;
  let zmin = Infinity, zmax = -Infinity;
  for (const v of zs) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
  const zr = (zmax - zmin) || 1;
  const pos = new Float32Array(W * H * 3), col = new Float32Array(W * H * 3), nrm = new Float32Array(W * H * 3);
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    const v = i * W + j, z = zs[v];
    pos.set([xs[j], ys[i], z], v * 3);
    const shade = terrainColor((z - zmin) / zr);
    // faint contour-band feel: darken near iso-lines of height
    const iso = Math.abs(((z - zmin) / zr * 14) % 1 - 0.5);
    const k = iso < 0.06 ? 0.82 : 1.0;
    col.set([shade[0] * k, shade[1] * k, shade[2] * k], v * 3);
    const dzx = (zs[i * W + Math.min(j + 1, W - 1)] - zs[i * W + Math.max(j - 1, 0)]) /
                ((xs[Math.min(j + 1, W - 1)] - xs[Math.max(j - 1, 0)]) || 1);
    const dzy = (zs[Math.min(i + 1, H - 1) * W + j] - zs[Math.max(i - 1, 0) * W + j]) /
                ((ys[Math.min(i + 1, H - 1)] - ys[Math.max(i - 1, 0)]) || 1);
    const nl = Math.hypot(dzx, dzy, 1);
    nrm.set([-dzx / nl, -dzy / nl, 1 / nl], v * 3);
  }
  const idx = new Uint32Array((W - 1) * (H - 1) * 6);
  let q = 0;
  for (let i = 0; i < H - 1; i++) for (let j = 0; j < W - 1; j++) {
    const a = i * W + j;
    idx.set([a, a + 1, a + W, a + 1, a + W + 1, a + W], q); q += 6;
  }
  const vaoInfo = makeVAO(meshProg, [{ name: "pos", size: 3, data: pos },
                                     { name: "nrm", size: 3, data: nrm },
                                     { name: "col", size: 3, data: col }]);
  gl.bindVertexArray(vaoInfo.vao);
  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  gl.bindVertexArray(null);

  // sparse wireframe, lifted a hair above the surface
  const lift = zr * 0.004;
  const wire = [];
  for (let i = 0; i < H; i += 8) for (let j = 0; j < W - 1; j++)
    wire.push(xs[j], ys[i], zs[i * W + j] + lift, xs[j + 1], ys[i], zs[i * W + j + 1] + lift);
  for (let j = 0; j < W; j += 8) for (let i = 0; i < H - 1; i++)
    wire.push(xs[j], ys[i], zs[i * W + j] + lift, xs[j], ys[i + 1], zs[(i + 1) * W + j] + lift);
  const wirePos = new Float32Array(wire);
  const wireCol = new Float32Array(wirePos.length / 3 * 4);
  for (let i = 0; i < wireCol.length; i += 4) wireCol.set([1, 1, 1, 0.06], i);
  const wireVao = makeVAO(lineProg, [{ name: "pos", size: 3, data: wirePos },
                                     { name: "col", size: 4, data: wireCol }]);
  return { vao: vaoInfo.vao, count: idx.length, wireVao: wireVao.vao, wireCount: wirePos.length / 3,
           zmin, zmax };
}

function buildRun(run, runIdx, colorBase) {
  const [N, L] = run.points.shape, pts = run.points.data;
  const alpha = runIdx === 0 ? 1.0 : OVERLAY_ALPHA;
  const dash = DASH_CYCLE[runIdx % DASH_CYCLE.length];
  const trajs = [], lineVerts = [], lineCols = [];
  for (let j = 0; j < N; j++) {
    const rgb = hexRGB(PALETTE[(colorBase + j) % PALETTE.length]);
    const fine = catmullRom(pts.subarray(j * L * 3, (j + 1) * L * 3), L);
    const nFine = fine.length / 3 - 1;
    let phase = 0;
    for (let s = 0; s < nFine; s++) {
      let draw = true;
      if (dash) { // dash by segment skipping over the draw/skip pattern
        let p = phase % dash.reduce((a, b) => a + b, 0), di = 0;
        while (p >= dash[di]) { p -= dash[di]; di++; }
        draw = di % 2 === 0;
        phase++;
      }
      if (draw) {
        lineVerts.push(fine[s * 3], fine[s * 3 + 1], fine[s * 3 + 2],
                       fine[s * 3 + 3], fine[s * 3 + 4], fine[s * 3 + 5]);
        lineCols.push(rgb[0], rgb[1], rgb[2], alpha, rgb[0], rgb[1], rgb[2], alpha);
      }
    }
    trajs.push({ fine, rgb, label: run.trajectoryLabels[j] != null ? String(run.trajectoryLabels[j]) : `#${j}` });
  }
  const linePos = new Float32Array(lineVerts), lineCol = new Float32Array(lineCols);
  const lineVao = makeVAO(lineProg, [{ name: "pos", size: 3, data: linePos },
                                     { name: "col", size: 4, data: lineCol }]);
  // small dot at every stored layer point (the "markers" of the reference)
  const dotPos = new Float32Array(N * L * 3), dotCol = new Float32Array(N * L * 4), dotSize = new Float32Array(N * L);
  for (let j = 0; j < N; j++) for (let l = 0; l < L; l++) {
    dotPos.set(pts.subarray((j * L + l) * 3, (j * L + l) * 3 + 3), (j * L + l) * 3);
    dotCol.set([...trajs[j].rgb, alpha], (j * L + l) * 4);
    dotSize[j * L + l] = 5;
  }
  const dotVao = makeVAO(pointProg, [{ name: "pos", size: 3, data: dotPos },
                                     { name: "col", size: 4, data: dotCol },
                                     { name: "size", size: 1, data: dotSize }]);
  return { run, N, L, trajs, alpha,
           lineVao: lineVao.vao, lineCount: linePos.length / 3,
           dotVao: dotVao.vao, dotCount: N * L };
}

// dynamic point buffer (marbles + hover highlight)
let dynPoints = null;
function ensureDynPoints(maxPts) {
  dynPoints = { max: maxPts, ...makeVAO(pointProg, [
    { name: "pos", size: 3, length: maxPts * 3 },
    { name: "col", size: 4, length: maxPts * 4 },
    { name: "size", size: 1, length: maxPts }]) };
}
// dynamic attention line buffer
let attnBuf = null;
function ensureAttnBuf(maxVerts) {
  attnBuf = { max: maxVerts, count: 0, ...makeVAO(lineProg, [
    { name: "pos", size: 3, length: maxVerts * 3 },
    { name: "col", size: 4, length: maxVerts * 4 }]) };
}

function setScene(scene) {
  state.scene = scene;
  state.L = scene.runs[0].points.shape[1];
  state.layerF = 0;
  state.playing = false;
  state.pick = null;
  state.visible = scene.runs.map(() => true);
  state.terrain = buildTerrain(scene.terrain);
  state.runs = [];
  let colorBase = 0, totalTrajs = 0, maxAttn = 0;
  scene.runs.forEach((run, i) => {
    state.runs.push(buildRun(run, i, colorBase));
    colorBase += run.points.shape[0];
    totalTrajs += run.points.shape[0];
    if (run.attention) maxAttn += run.attention.shape[1] * 3 * 2;
  });
  ensureDynPoints(totalTrajs + 1);
  ensureAttnBuf(Math.max(maxAttn, 2));

  // bounds: terrain footprint + trajectory extents
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const acc = (x, y, z) => {
    lo[0] = Math.min(lo[0], x); hi[0] = Math.max(hi[0], x);
    lo[1] = Math.min(lo[1], y); hi[1] = Math.max(hi[1], y);
    lo[2] = Math.min(lo[2], z); hi[2] = Math.max(hi[2], z);
  };
  const t = scene.terrain;
  acc(t.x.data[0], t.y.data[0], state.terrain.zmin);
  acc(t.x.data[t.x.data.length - 1], t.y.data[t.y.data.length - 1], state.terrain.zmax);
  // frame the trajectories (the terrain usually extends far beyond the action)
  const tlo = [Infinity, Infinity, Infinity], thi = [-Infinity, -Infinity, -Infinity];
  for (const r of scene.runs) {
    const p = r.points.data;
    for (let i = 0; i < p.length; i += 3) {
      acc(p[i], p[i + 1], p[i + 2]);
      for (let d = 0; d < 3; d++) {
        tlo[d] = Math.min(tlo[d], p[i + d]); thi[d] = Math.max(thi[d], p[i + d]);
      }
    }
  }
  state.bounds = { lo, hi };
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  const tdiag = Math.hypot(thi[0] - tlo[0], thi[1] - tlo[1], thi[2] - tlo[2]) || 1;
  const target = [(tlo[0] + thi[0]) / 2, (tlo[1] + thi[1]) / 2, (tlo[2] + thi[2]) / 2];
  state.cam = { theta: -2.35, phi: 0.55, dist: Math.min(diag * 1.05, tdiag * 1.7), target };
  state.diag = diag;
  state.zEps = (state.terrain.zmax - state.terrain.zmin) * 0.02 || 0.02;

  buildUI(scene);
  rebuildAttention();
  hideMessage();
}

function rebuildAttention() {
  if (!attnBuf) return;
  attnBuf.count = 0;
  const layer = Math.floor(state.layerF);
  if (!state.showAttention || layer < 1) return;
  const pos = [], col = [];
  state.runs.forEach((rd, i) => {
    const att = rd.run.attention;
    if (!state.visible[i] || !att || layer - 1 >= att.shape[0]) return;
    const T = att.shape[1], w = att.data, base = (layer - 1) * T * T;
    const pts = rd.run.points.data, L = rd.L;
    for (let dst = 0; dst < T && dst < rd.N; dst++) {
      const row = Array.from({ length: T }, (_, s) => s).sort((a, b) => w[base + dst * T + b] - w[base + dst * T + a]);
      for (const src of row.slice(0, 3)) {
        if (src === dst || w[base + dst * T + src] < 0.1 || src >= rd.N) continue;
        pos.push(pts[(src * L + layer) * 3], pts[(src * L + layer) * 3 + 1], pts[(src * L + layer) * 3 + 2],
                 pts[(dst * L + layer) * 3], pts[(dst * L + layer) * 3 + 1], pts[(dst * L + layer) * 3 + 2]);
        col.push(0.506, 0.561, 0.722, 0.55 * rd.alpha,
                 0.506, 0.561, 0.722, 0.55 * rd.alpha);
      }
    }
  });
  attnBuf.count = Math.min(pos.length / 3, attnBuf.max);
  gl.bindBuffer(gl.ARRAY_BUFFER, attnBuf.bufs.pos);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(pos.slice(0, attnBuf.max * 3)));
  gl.bindBuffer(gl.ARRAY_BUFFER, attnBuf.bufs.col);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(col.slice(0, attnBuf.max * 4)));
}

// ---------------------------------------------------------------- rendering
function currentMVP() {
  const c = state.cam;
  const eye = [c.target[0] + c.dist * Math.cos(c.phi) * Math.cos(c.theta),
               c.target[1] + c.dist * Math.cos(c.phi) * Math.sin(c.theta),
               c.target[2] + c.dist * Math.sin(c.phi)];
  const proj = perspective(0.9, canvas.clientWidth / Math.max(canvas.clientHeight, 1), c.dist * 0.01, c.dist * 20);
  return matMul(proj, lookAt(eye, c.target, [0, 0, 1]));
}

function marblePositions() {
  const out = [];
  const f = state.layerF * SEG;
  state.runs.forEach((rd, i) => {
    if (!state.visible[i]) return;
    for (const t of rd.trajs) {
      const n = t.fine.length / 3;
      const i0 = Math.min(Math.floor(f), n - 1), i1 = Math.min(i0 + 1, n - 1), fr = f - Math.floor(f);
      out.push({
        p: [t.fine[i0 * 3] + (t.fine[i1 * 3] - t.fine[i0 * 3]) * fr,
            t.fine[i0 * 3 + 1] + (t.fine[i1 * 3 + 1] - t.fine[i0 * 3 + 1]) * fr,
            t.fine[i0 * 3 + 2] + (t.fine[i1 * 3 + 2] - t.fine[i0 * 3 + 2]) * fr + state.zEps],
        rgb: t.rgb, alpha: rd.alpha,
      });
    }
  });
  return out;
}

let lastT = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - lastT) / 1000, 0.1); lastT = now;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.031, 0.043, 0.094, 1);  // --color-base #080B18
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!state.scene) return;

  if (state.playing) {
    state.layerF += dt * state.speed;
    if (state.layerF > state.L - 1) state.layerF = 0;
    ui.layerSlider.value = state.layerF;
    onLayerChanged(false);
  }

  const mvp = currentMVP();
  gl.enable(gl.DEPTH_TEST);

  gl.useProgram(meshProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(meshProg, "mvp"), false, mvp);
  gl.bindVertexArray(state.terrain.vao);
  gl.drawElements(gl.TRIANGLES, state.terrain.count, gl.UNSIGNED_INT, 0);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.depthMask(false);
  gl.useProgram(lineProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "mvp"), false, mvp);
  gl.bindVertexArray(state.terrain.wireVao);
  gl.drawArrays(gl.LINES, 0, state.terrain.wireCount);
  state.runs.forEach((rd, i) => {
    if (!state.visible[i]) return;
    gl.bindVertexArray(rd.lineVao);
    gl.drawArrays(gl.LINES, 0, rd.lineCount);
  });
  if (attnBuf.count) {
    gl.bindVertexArray(attnBuf.vao);
    gl.drawArrays(gl.LINES, 0, attnBuf.count);
  }

  gl.useProgram(pointProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(pointProg, "mvp"), false, mvp);
  const rimLoc = gl.getUniformLocation(pointProg, "rim");
  gl.uniform1f(rimLoc, 0);
  state.runs.forEach((rd, i) => {
    if (!state.visible[i]) return;
    gl.bindVertexArray(rd.dotVao);
    gl.drawArrays(gl.POINTS, 0, rd.dotCount);
  });

  // marbles + hover highlight through the dynamic buffer
  const marbles = marblePositions();
  const pick = updatePick(mvp);
  const pos = new Float32Array(dynPoints.max * 3), col = new Float32Array(dynPoints.max * 4),
        size = new Float32Array(dynPoints.max);
  let n = 0;
  for (const m of marbles) {
    pos.set(m.p, n * 3); col.set([...m.rgb, m.alpha], n * 4); size[n] = 13 * dpr; n++;
  }
  if (pick) {
    pos.set([pick.p[0], pick.p[1], pick.p[2] + state.zEps], n * 3);
    col.set([...pick.rgb, 1], n * 4); size[n] = 17 * dpr; n++;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, dynPoints.bufs.pos); gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
  gl.bindBuffer(gl.ARRAY_BUFFER, dynPoints.bufs.col); gl.bufferSubData(gl.ARRAY_BUFFER, 0, col);
  gl.bindBuffer(gl.ARRAY_BUFFER, dynPoints.bufs.size); gl.bufferSubData(gl.ARRAY_BUFFER, 0, size);
  gl.uniform1f(rimLoc, 1);
  gl.bindVertexArray(dynPoints.vao);
  gl.drawArrays(gl.POINTS, 0, n);

  gl.depthMask(true);
  gl.disable(gl.BLEND);
  gl.bindVertexArray(null);
}

// ---------------------------------------------------------------- picking
function updatePick(mvp) {
  if (!state.mouse) { setPickInfo(null); return null; }
  const w = canvas.clientWidth, h = canvas.clientHeight;
  let best = null, bestD = PICK_RADIUS;
  const multi = state.scene.runs.length > 1;
  state.runs.forEach((rd, i) => {
    if (!state.visible[i]) return;
    const pts = rd.run.points.data, L = rd.L;
    for (let j = 0; j < rd.N; j++) for (let l = 0; l < L; l++) {
      const p = [pts[(j * L + l) * 3], pts[(j * L + l) * 3 + 1], pts[(j * L + l) * 3 + 2]];
      const s = projectPoint(mvp, p, w, h);
      if (s[2] <= 0) continue;
      const d = Math.hypot(s[0] - state.mouse[0], s[1] - state.mouse[1]);
      if (d < bestD) {
        bestD = d;
        best = { runIdx: i, traj: j, layer: l, p, rgb: rd.trajs[j].rgb,
                 label: (multi ? rd.run.label + " · " : "") + rd.trajs[j].label };
      }
    }
  });
  setPickInfo(best);
  return best;
}

function setPickInfo(pick) {
  const same = (a, b) => a && b && a.runIdx === b.runIdx && a.traj === b.traj && a.layer === b.layer;
  if (same(pick, state.pick) || (!pick && !state.pick)) { state.pick = pick; return; }
  state.pick = pick;
  if (!pick) { ui.infoPanel.hidden = true; return; }
  const run = state.scene.runs[pick.runIdx];
  const T = run.tokens.length;
  const tok = pick.traj < T ? pick.traj : T - 1; // trajectory j reads out token j (last token if fewer)
  let html = `<div class="ip-title">${esc(pick.label)}</div>` +
             `<div>layer <b>${pick.layer}</b> · token <span class="mono">'${esc(run.tokens[tok] ?? "?")}'</span></div>`;
  if (run.entropy) {
    const [L, Te] = run.entropy.shape;
    if (pick.layer < L && tok < Te)
      html += `<div>entropy ${run.entropy.data[pick.layer * Te + tok].toFixed(2)} nats</div>`;
  }
  if (run.topk && run.topk[pick.layer] && run.topk[pick.layer][tok]) {
    const rows = run.topk[pick.layer][tok].slice(0, 5).map(([t, p]) =>
      `<div><span class="bar" style="width:${Math.max(2, p * 120)}px"></span>` +
      `<span class="mono">'${esc(t)}'</span> <span class="dim">${(p * 100).toFixed(1)}%</span></div>`).join("");
    html += `<div class="topk"><span class="dim">top-k readout</span>${rows}</div>`;
  }
  ui.infoPanel.innerHTML = html;
  ui.infoPanel.hidden = false;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- UI
const ui = {
  overlay: document.getElementById("overlay"),
  overlayMsg: document.getElementById("overlay-msg"),
  dropShade: document.getElementById("drop-shade"),
  meta: document.getElementById("meta"),
  runsPanel: document.getElementById("runs-panel"),
  runsList: document.getElementById("runs-list"),
  comparisons: document.getElementById("comparisons"),
  attnRow: document.getElementById("attn-row"),
  attnToggle: document.getElementById("attnToggle"),
  infoPanel: document.getElementById("info-panel"),
  hudBottom: document.getElementById("hud-bottom"),
  playBtn: document.getElementById("playBtn"),
  speedSel: document.getElementById("speedSel"),
  layerSlider: document.getElementById("layerSlider"),
  layerLabel: document.getElementById("layerLabel"),
  openBtn: document.getElementById("openBtn"),
  fileInput: document.getElementById("fileInput"),
};

function showMessage(msg, isError) {
  ui.overlayMsg.textContent = msg;
  ui.overlay.classList.toggle("error", !!isError);
  ui.overlay.style.display = "flex";
}
function hideMessage() { ui.overlay.style.display = "none"; }

function buildUI(scene) {
  const meta = scene.meta || {};
  ui.meta.textContent = [meta.model && `model: ${meta.model}`, meta.backend && `backend: ${meta.backend}`]
    .filter(Boolean).join("  ·  ");
  ui.runsPanel.hidden = false;
  ui.hudBottom.hidden = false;
  ui.layerSlider.max = state.L - 1;
  ui.layerSlider.value = 0;
  onLayerChanged(false);

  ui.runsList.innerHTML = "";
  let colorBase = 0;
  scene.runs.forEach((run, i) => {
    const sw = PALETTE[colorBase % PALETTE.length];
    colorBase += run.points.shape[0];
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = true;
    cb.addEventListener("change", () => { state.visible[i] = cb.checked; rebuildAttention(); });
    label.appendChild(cb);
    label.insertAdjacentHTML("beforeend",
      `<span class="swatch" style="background:${sw}"></span><b>${esc(run.label)}</b>` +
      `<span class="prompt" title="${esc(run.prompt)}">${esc(run.prompt)}</span>`);
    ui.runsList.appendChild(label);
  });

  if (scene.comparisons.length) {
    ui.comparisons.innerHTML = "<table><tr><th></th><th>hausdorff</th><th>dtw</th><th>shared</th></tr>" +
      scene.comparisons.map((c) =>
        `<tr><td><b>${esc(c.label ?? "?")}</b></td><td>${fmt(c.hausdorff)}</td>` +
        `<td>${fmt(c.dtw_normalized)}</td><td>${c.shared_tokens ?? "–"}</td></tr>`).join("") + "</table>";
  } else ui.comparisons.innerHTML = "";

  const hasAttn = scene.runs.some((r) => r.attention);
  ui.attnRow.hidden = !hasAttn;
  ui.attnToggle.checked = state.showAttention = false;
}
const fmt = (v) => (typeof v === "number" ? v.toFixed(3) : "–");

function onLayerChanged(fromSlider) {
  if (fromSlider) state.layerF = parseFloat(ui.layerSlider.value);
  const l = Math.floor(state.layerF);
  ui.layerLabel.textContent = `layer ${l} / ${state.L - 1}`;
  if (l !== state._lastAttnLayer) { state._lastAttnLayer = l; rebuildAttention(); }
}

ui.layerSlider.addEventListener("input", () => { state.playing = false; ui.playBtn.innerHTML = "&#9654;"; onLayerChanged(true); });
ui.playBtn.addEventListener("click", () => {
  state.playing = !state.playing;
  ui.playBtn.innerHTML = state.playing ? "&#10074;&#10074;" : "&#9654;";
});
ui.speedSel.addEventListener("change", () => { state.speed = parseFloat(ui.speedSel.value); });
ui.attnToggle.addEventListener("change", () => { state.showAttention = ui.attnToggle.checked; rebuildAttention(); });
ui.openBtn.addEventListener("click", () => ui.fileInput.click());
ui.fileInput.addEventListener("change", () => {
  if (ui.fileInput.files[0]) loadBlob(ui.fileInput.files[0], ui.fileInput.files[0].name);
});

// camera controls
let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  state.mouse = [e.clientX, e.clientY];
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  const c = state.cam;
  if (drag.pan) {
    const k = c.dist * 0.0012;
    const st = Math.sin(c.theta), ct = Math.cos(c.theta), sp = Math.sin(c.phi), cp = Math.cos(c.phi);
    const right = [-st, ct, 0], up = [-sp * ct, -sp * st, cp]; // camera basis, world z up
    for (let d = 0; d < 3; d++) c.target[d] += (-dx * right[d] + dy * up[d]) * k;
  } else {
    c.theta -= dx * 0.006;
    c.phi = Math.min(1.55, Math.max(-1.55, c.phi + dy * 0.006));
  }
});
canvas.addEventListener("pointerup", () => { drag = null; });
canvas.addEventListener("pointerleave", () => { state.mouse = null; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const d = state.diag || 10;
  state.cam.dist = Math.min(d * 12, Math.max(d * 0.03, state.cam.dist * Math.exp(e.deltaY * 0.0012)));
}, { passive: false });

// drag & drop
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; ui.dropShade.hidden = false; });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; ui.dropShade.hidden = true; } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0; ui.dropShade.hidden = true;
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) loadBlob(f, f.name);
});

// ---------------------------------------------------------------- loading
function loadBuffer(buf, name) {
  try {
    setScene(MTJ.loadScene(buf));
  } catch (err) {
    showMessage(`${name}: ${err.message}`, err.kind !== "trajectory");
    console.warn(err);
  }
}
function loadBlob(blob, name) {
  blob.arrayBuffer().then((buf) => loadBuffer(buf, name))
    .catch((err) => showMessage(`could not read ${name}: ${err.message}`, true));
}
async function loadURL(url, quiet) {
  try {
    const res = await fetch(url); // relative to the page
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    loadBuffer(await res.arrayBuffer(), url);
    return true;
  } catch (err) {
    if (!quiet) showMessage(`could not fetch ${url}: ${err.message}`, true);
    return false;
  }
}

// Capture backend (serve.py) discovery: when /api/health answers, the
// browser can generate new scenes directly instead of only loading files.
(async function discoverBackend() {
  let health;
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    health = await res.json();
  } catch { return; }  // plain static hosting: no backend, no form
  const row = document.getElementById("capture-row");
  const promptsEl = document.getElementById("capturePrompts");
  const btn = document.getElementById("captureBtn");
  row.hidden = false;
  document.getElementById("captureModel").textContent = `model: ${health.model}`;
  btn.addEventListener("click", async () => {
    const prompts = promptsEl.value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!prompts.length) return;
    btn.disabled = true; btn.textContent = "Capturing";
    try {
      const res = await fetch("/api/scene", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompts }),
      });
      if (!res.ok) throw new Error((await res.json()).error || `HTTP ${res.status}`);
      loadBuffer(await res.arrayBuffer(), prompts[0]);
    } catch (err) {
      showMessage(`capture failed: ${err.message}`, true);
    } finally {
      btn.disabled = false; btn.textContent = "Capture";
    }
  });
})();

(async function boot() {
  if (!gl) { showMessage("WebGL2 is not available in this browser", true); return; }
  requestAnimationFrame(frame);
  const param = new URLSearchParams(location.search).get("file");
  if (param) { await loadURL(param, false); return; }
  await loadURL("samples/scene-abc.mtj", true); // graceful fallback to the drop prompt
})();
