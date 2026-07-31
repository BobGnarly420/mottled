"use strict";
const test = require("node:test");
const assert = require("node:assert");

const BVH = require("../bvh.js");

// The Python `bvh.py` is the reference implementation; these tests pin the
// semantics the port has to keep (front-most wins, radius is inclusive,
// `point` is on the ray, the build is deterministic). Numerical parity with
// bvh.py itself is checked from the Python side.

const near = (a, b, eps = 1e-12) =>
  assert.ok(Math.abs(a - b) <= eps, `expected ${a} ≈ ${b} (within ${eps})`);
const nearPoint = (p, q, eps = 1e-12) => {
  assert.strictEqual(p.length, 3);
  for (let d = 0; d < 3; d++) near(p[d], q[d], eps);
};

// A single unit segment along +y at x = 0, z = 0..0 — the simplest thing a
// ray can hit or miss.
const oneSeg = () => BVH.build([0, 0, 0], [0, 1, 0], null, 8);

test("an empty BVH answers null and reports zero extent", () => {
  const b = BVH.build([], [], null, 8);
  assert.strictEqual(b.length, 0);
  assert.strictEqual(b.diagonal, 0);
  assert.deepStrictEqual(b.boundsMin, [0, 0, 0]);
  assert.deepStrictEqual(b.boundsMax, [0, 0, 0]);
  assert.strictEqual(b.order.length, 0);
  assert.strictEqual(BVH.rayPick(b, [0, 0, 0], [0, 0, 1]), null);
  assert.strictEqual(BVH.nearest(b, [0, 0, 0]), null);
  // a zero-length run has no segments to connect, so it builds empty too
  assert.strictEqual(BVH.fromPoints([], 0, 0).length, 0);
  assert.strictEqual(BVH.fromPoints([1, 2, 3], 1, 1).length, 0);
});

test("build exposes the segment arrays, order and bounds", () => {
  const b = BVH.build([0, 0, 0, 2, 2, 2], [1, 0, 0, 3, 2, 2], [7, 1, 7, 2], 8);
  assert.strictEqual(b.length, 2);
  assert.ok(b.A instanceof Float64Array && b.B instanceof Float64Array);
  assert.ok(b.meta instanceof Int32Array);
  assert.deepStrictEqual(Array.from(b.order), [0, 1]);
  assert.deepStrictEqual(b.boundsMin, [0, 0, 0]);
  assert.deepStrictEqual(b.boundsMax, [3, 2, 2]);
  near(b.diagonal, Math.hypot(3, 2, 2));
  assert.deepStrictEqual(BVH.metaOf(b, 1), [7, 2]);
  assert.strictEqual(BVH.metaOf(b, 9), null);            // out of range
  assert.strictEqual(BVH.metaOf(BVH.build([0, 0, 0], [1, 0, 0], null), 0), null);
});

test("a single segment is hit head-on and missed when out of radius", () => {
  const b = oneSeg();
  // ray along +x at y = 0.5 passes 0 away from the segment at t = 4
  const hit = BVH.rayPick(b, [-4, 0.5, 0], [1, 0, 0], { radius: 0.1 });
  assert.ok(hit);
  assert.strictEqual(hit.index, 0);
  near(hit.t, 4);
  near(hit.distance, 0);
  nearPoint(hit.point, [0, 0.5, 0]);      // the closest point ON THE RAY
  // the same ray offset in z by more than the radius misses
  assert.strictEqual(BVH.rayPick(b, [-4, 0.5, 0.5], [1, 0, 0], { radius: 0.1 }), null);
  // ... and pointing away from the segment misses (t would be negative)
  assert.strictEqual(BVH.rayPick(b, [4, 0.5, 0], [1, 0, 0], { radius: 0.1 }), null);
  // beyond maxDist it misses too
  assert.strictEqual(BVH.rayPick(b, [-4, 0.5, 0], [1, 0, 0], { radius: 0.1, maxDist: 3 }), null);
});

test("the ray direction is normalised internally and must be non-zero", () => {
  const b = oneSeg();
  const unit = BVH.rayPick(b, [-4, 0.5, 0], [1, 0, 0], { radius: 0.1 });
  const long = BVH.rayPick(b, [-4, 0.5, 0], [17, 0, 0], { radius: 0.1 });
  assert.strictEqual(long.index, unit.index);
  near(long.t, unit.t);                    // t is in world units, not |d| units
  assert.throws(() => BVH.rayPick(b, [0, 0, 0], [0, 0, 0]), /non-zero/);
  // an empty BVH short-circuits before the direction is ever validated
  assert.strictEqual(BVH.rayPick(BVH.build([], []), [0, 0, 0], [0, 0, 0]), null);
});

test("the front-most of several collinear segments wins", () => {
  // five parallel segments stacked along the ray's axis (+x); a ray down the
  // axis grazes all of them and must return the nearest
  const A = [], B = [];
  for (let i = 0; i < 5; i++) { A.push(i * 2, -1, 0); B.push(i * 2, 1, 0); }
  const b = BVH.build(A, B, null, 2);       // leafSize 2 forces a real tree
  const hit = BVH.rayPick(b, [-3, 0, 0], [1, 0, 0], { radius: 0.25 });
  assert.strictEqual(hit.index, 0);
  near(hit.t, 3);
  // walking the origin forward drops the segments it has passed, one by one
  for (let i = 0; i < 5; i++) {
    const h = BVH.rayPick(b, [i * 2 - 0.5, 0, 0], [1, 0, 0], { radius: 0.25 });
    assert.strictEqual(h.index, i);
    near(h.t, 0.5);
  }
  // from behind the last one there is nothing left in front
  assert.strictEqual(BVH.rayPick(b, [9, 0, 0], [1, 0, 0], { radius: 0.25 }), null);
  // maxDist clips to the segments in front of it
  assert.strictEqual(BVH.rayPick(b, [-3, 0, 0], [1, 0, 0], { radius: 0.25, maxDist: 2 }), null);
  assert.strictEqual(
    BVH.rayPick(b, [-3, 0, 0], [1, 0, 0], { radius: 0.25, maxDist: 4 }).index, 0);
});

test("the radius tolerance is inclusive at its boundary", () => {
  // segment sits exactly 1.0 away from the ray
  const b = BVH.build([0, 1, 0], [1, 1, 0], null, 8);
  const ray = [[-1, 0, 0], [1, 0, 0]];
  assert.strictEqual(BVH.rayPick(b, ray[0], ray[1], { radius: 1 }).index, 0);   // <= radius
  assert.strictEqual(BVH.rayPick(b, ray[0], ray[1], { radius: 1.0000001 }).index, 0);
  assert.strictEqual(BVH.rayPick(b, ray[0], ray[1], { radius: 0.9999999 }), null);
  near(BVH.rayPick(b, ray[0], ray[1], { radius: 1 }).distance, 1);
  // radius 0 is honoured as given, not treated as "unset"
  assert.strictEqual(BVH.rayPick(b, ray[0], ray[1], { radius: 0 }), null);
  assert.strictEqual(
    BVH.rayPick(BVH.build([0, 0, 0], [1, 0, 0]), [-1, 0, 0], [1, 0, 0], { radius: 0 }).index, 0);
});

test("the default radius is 1% of the scene diagonal", () => {
  // diagonal 100 -> default tolerance 1.0
  const b = BVH.build([0, 0, 0, 0, 1, 0], [100, 0, 0, 100, 1, 0], null, 8);
  near(b.diagonal, Math.hypot(100, 1, 0));
  const r = 0.01 * b.diagonal;
  // a ray offset just inside the default tolerance still hits
  assert.ok(BVH.rayPick(b, [50, -r * 0.9, -5], [0, 0, 1]));
  assert.strictEqual(BVH.rayPick(b, [50, -r * 1.1 - 0.0001, -5], [0, 0, 1]), null);
  // a degenerate (single-point) scene falls back to 1e-6 rather than 0
  const d = BVH.build([1, 1, 1], [1, 1, 1], null, 8);
  assert.strictEqual(d.diagonal, 0);
  assert.ok(BVH.rayPick(d, [1, 1, -5], [0, 0, 1]));
  assert.strictEqual(BVH.rayPick(d, [1, 1.001, -5], [0, 0, 1]), null);
});

test("nearest returns the closest point on the closest segment", () => {
  const b = BVH.build([0, 0, 0, 5, 5, 5], [1, 0, 0, 6, 5, 5], null, 8);
  const h = BVH.nearest(b, [0.5, 2, 0]);
  assert.strictEqual(h.index, 0);
  assert.strictEqual(h.t, 0);
  near(h.distance, 2);
  nearPoint(h.point, [0.5, 0, 0]);
  // past an endpoint the answer clamps to that endpoint
  const e = BVH.nearest(b, [-3, 0, 0]);
  assert.strictEqual(e.index, 0);
  near(e.distance, 3);
  nearPoint(e.point, [0, 0, 0]);
  // and the far segment wins when the query point is near it
  const f = BVH.nearest(b, [5.5, 5, 6]);
  assert.strictEqual(f.index, 1);
  near(f.distance, 1);
  nearPoint(f.point, [5.5, 5, 5]);
  // a zero-length segment still answers (distance to the point itself)
  const z = BVH.nearest(BVH.build([2, 2, 2], [2, 2, 2]), [2, 2, 5]);
  near(z.distance, 3);
  nearPoint(z.point, [2, 2, 2]);
});

test("fromPoints maps trajectory/layer to segment index and meta", () => {
  // 3 trajectories x 4 layers; trajectory j runs along +x at y = j
  const N = 3, L = 4, pts = [];
  for (let j = 0; j < N; j++) for (let l = 0; l < L; l++) pts.push(l, j, 0);
  const b = BVH.fromPoints(pts, N, L, 8);
  assert.strictEqual(b.length, N * (L - 1));
  for (let j = 0; j < N; j++) for (let l = 0; l < L - 1; l++) {
    const s = j * (L - 1) + l;                     // trajectory-major, like bvh.py
    assert.deepStrictEqual(BVH.metaOf(b, s), [j, l]);
    assert.deepStrictEqual(Array.from(b.A.slice(s * 3, s * 3 + 3)), [l, j, 0]);
    assert.deepStrictEqual(Array.from(b.B.slice(s * 3, s * 3 + 3)), [l + 1, j, 0]);
  }
  // a ray dropped onto trajectory 2 between layers 1 and 2 reports exactly that
  const hit = BVH.rayPick(b, [1.5, 2, -3], [0, 0, 1], { radius: 0.1 });
  assert.deepStrictEqual(BVH.metaOf(b, hit.index), [2, 1]);
  near(hit.t, 3);
  near(BVH.segmentParam(b, hit.index, hit.point), 0.5);
  // segmentParam clamps to the span and is 0 for an unknown index
  near(BVH.segmentParam(b, hit.index, [99, 2, 0]), 1);
  assert.strictEqual(BVH.segmentParam(b, -1, [0, 0, 0]), 0);
  // float32 input (the viewer's draped points) is widened, not rejected
  const f32 = BVH.fromPoints(Float32Array.from(pts), N, L, 8);
  assert.strictEqual(f32.length, b.length);
  assert.deepStrictEqual(Array.from(f32.order), Array.from(b.order));
  assert.throws(() => BVH.fromPoints(pts, N, L + 4, 8), /expected/);
});

test("the build is deterministic and its order is a permutation", () => {
  // pseudo-random but fixed: the same input must always give the same tree
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5) * 20;
  const A = [], B = [];
  for (let i = 0; i < 400; i++) {
    const x = rnd(), y = rnd(), z = rnd();
    A.push(x, y, z); B.push(x + rnd() * 0.05, y + rnd() * 0.05, z + rnd() * 0.05);
  }
  const b1 = BVH.build(A, B, null, 8), b2 = BVH.build(A, B, null, 8);
  assert.deepStrictEqual(Array.from(b1.order), Array.from(b2.order));
  assert.deepStrictEqual(Array.from(b1.left), Array.from(b2.left));
  assert.deepStrictEqual(Array.from(b1.start), Array.from(b2.start));
  assert.deepStrictEqual(Array.from(b1.count), Array.from(b2.count));
  // order is a permutation of 0..N-1, and leaves partition it exactly once
  assert.deepStrictEqual(Array.from(b1.order).slice().sort((p, q) => p - q),
                         Array.from({ length: 400 }, (_, i) => i));
  const covered = new Int32Array(400);
  let leaves = 0;
  for (let n = 0; n < b1.left.length; n++) {
    if (b1.left[n] >= 0) continue;
    leaves++;
    assert.ok(b1.count[n] <= 8 || b1.count[n] === 0, "leaves respect leafSize");
    for (let k = b1.start[n]; k < b1.start[n] + b1.count[n]; k++) covered[b1.order[k]]++;
  }
  assert.ok(leaves > 1);
  assert.ok(Array.from(covered).every((c) => c === 1), "every segment lives in exactly one leaf");
  // identical centroids can't be split, so they fall into one oversized leaf
  const dup = BVH.build(new Array(120).fill(0), new Array(40).fill(0).flatMap(() => [0, 0, 1]), null, 8);
  assert.strictEqual(dup.left.length, 1);
  assert.strictEqual(dup.count[0], 40);
});

test("a brute-force scan agrees with the accelerated queries", () => {
  // the tree is only an accelerator: over a random scene it must return the
  // same segment a linear scan would
  let seed = 99;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5) * 10;
  const A = [], B = [];
  for (let i = 0; i < 300; i++) {
    const x = rnd(), y = rnd(), z = rnd();
    A.push(x, y, z); B.push(x + rnd() * 0.3, y + rnd() * 0.3, z + rnd() * 0.3);
  }
  const b = BVH.build(A, B, null, 8);
  const flat = BVH.build(A, B, null, 100000);   // one leaf: no pruning at all
  for (let k = 0; k < 60; k++) {
    // Cast from outside the scene and aim in. An origin sitting *inside* the
    // geometry clamps every segment behind it to t = 0 at once, and which of
    // those exactly-tied hits wins is decided by traversal order — so the two
    // tree shapes are entitled to disagree there (bvh.py does the same).
    const o = [30, rnd(), rnd()];                       // outside, off to +x
    const d = [-30, rnd() - o[1], rnd() - o[2]];        // aimed back at the cloud
    if (Math.hypot(d[0], d[1], d[2]) < 1e-9) continue;
    for (const radius of [0.2, 1.0]) {
      const fast = BVH.rayPick(b, o, d, { radius }), slow = BVH.rayPick(flat, o, d, { radius });
      assert.strictEqual(!!fast, !!slow);
      if (fast) { assert.strictEqual(fast.index, slow.index); near(fast.t, slow.t); }
    }
    const nf = BVH.nearest(b, o), ns = BVH.nearest(flat, o);
    assert.strictEqual(nf.index, ns.index);
    near(nf.distance, ns.distance);
  }
});

test("build validates its inputs", () => {
  assert.throws(() => BVH.build([0, 0, 0], [0, 0, 0, 1, 1, 1]), /same shape/);
  assert.throws(() => BVH.build([0, 0], [1, 1]), /flat \(N, 3\)/);
  assert.throws(() => BVH.build([0, 0, 0], [1, 1, 1], [1, 2, 3]), /flat \(N, 2\)/);
  assert.throws(() => BVH.build(null, null), /flat array/);
  // leafSize is clamped to >= 1 and defaults to 8
  assert.strictEqual(BVH.build([0, 0, 0], [1, 1, 1], null, 0).leafSize, 1);
  assert.strictEqual(BVH.build([0, 0, 0], [1, 1, 1], null, -3).leafSize, 1);
  assert.strictEqual(BVH.build([0, 0, 0], [1, 1, 1]).leafSize, 8);
});
