/* bvh.js — bounding-volume hierarchy over trajectory segments.
 * A port of the repo's `bvh.py` (read that module for the design rationale:
 * trajectories are polylines, so the acceleration structure is over the
 * *curve segments*, never a grid over the empty volume around them).
 *
 * Build order, split rule and query semantics mirror `bvh.py` exactly — the
 * same segment set, ray and radius must yield the same hit index and the same
 * `t`/`distance` to float64 precision (tests/test_bvh_conformance.py checks
 * this). Every deviation from the Python is either impossible to observe
 * (NaN-only ray components) or called out in a comment.
 *
 * UMD: `window.BVH` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.BVH = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const EPS = 1e-12;         // bvh.py's _EPS — same value, same role
  const DEFAULT_LEAF = 8;

  // Every coordinate lives in a Float64Array: numpy does the Python side in
  // float64 and the conformance test compares the two directly, so f32 inputs
  // (the viewer's draped points) are widened once here, at build time.
  function toF64(a, what) {
    if (a == null || typeof a === "number")
      throw new TypeError(`${what} must be a flat array of numbers`);
    return new Float64Array(a);   // copies typed arrays and plain arrays alike
  }

  // ------------------------------------------------------------------ build
  // Median-split BVH over segments A[i] -> B[i]. `A`/`B` are flat (N, 3)
  // arrays (length 3N), `meta` an optional flat (N, 2) int array, `leafSize`
  // the maximum segment count of a leaf (>= 1, default 8).
  function build(A, B, meta, leafSize) {
    const a = toF64(A, "A"), b = toF64(B, "B");
    if (a.length !== b.length) throw new Error("A and B must have the same shape");
    if (a.length % 3 !== 0) throw new Error("A and B must be flat (N, 3) arrays");
    const n = a.length / 3;
    const leaf = Math.max(1, Math.trunc(leafSize == null ? DEFAULT_LEAF : leafSize));

    let metaArr = null;
    if (meta != null) {
      metaArr = meta instanceof Int32Array ? new Int32Array(meta) : Int32Array.from(meta);
      if (metaArr.length !== 2 * n)
        throw new Error(`meta must be a flat (N, 2) array; got length ${metaArr.length} for N=${n}`);
    }

    // per-segment AABB and centroid, the quantities the split rule reads
    const segMin = new Float64Array(3 * n), segMax = new Float64Array(3 * n),
          centroid = new Float64Array(3 * n);
    for (let i = 0; i < 3 * n; i++) {
      const p = a[i], q = b[i];
      segMin[i] = p < q ? p : q;
      segMax[i] = p > q ? p : q;
      centroid[i] = 0.5 * (p + q);
    }

    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;

    // node arrays, appended to during the recursive build
    const nmin = [], nmax = [], left = [], right = [], start = [], count = [];
    const st = { a, b, segMin, segMax, centroid, order, leaf,
                 nmin, nmax, left, right, start, count };
    if (n > 0) buildNode(st, 0, n);

    const bvh = {
      A: a, B: b, meta: metaArr, order, leafSize: leaf, length: n,
      segMin, segMax, centroid,
      nmin: Float64Array.from(nmin), nmax: Float64Array.from(nmax),
      left: Int32Array.from(left), right: Int32Array.from(right),
      start: Int32Array.from(start), count: Int32Array.from(count),
      boundsMin: [0, 0, 0], boundsMax: [0, 0, 0], diagonal: 0,
    };
    if (n > 0) {
      bvh.boundsMin = [bvh.nmin[0], bvh.nmin[1], bvh.nmin[2]];
      bvh.boundsMax = [bvh.nmax[0], bvh.nmax[1], bvh.nmax[2]];
      bvh.diagonal = Math.hypot(bvh.boundsMax[0] - bvh.boundsMin[0],
                                bvh.boundsMax[1] - bvh.boundsMin[1],
                                bvh.boundsMax[2] - bvh.boundsMin[2]);
    }
    return bvh;
  }

  // Recursively build the node covering order[start, end); returns its index.
  // Deterministic: the widest *centroid* axis, split at the count median of a
  // STABLE sort along it — same rule, same tie-breaks as bvh.py's `_build`.
  function buildNode(st, start, end) {
    const node = st.left.length;
    st.nmin.push(0, 0, 0); st.nmax.push(0, 0, 0);
    st.left.push(-1); st.right.push(-1); st.start.push(-1); st.count.push(0);

    const cnt = end - start;
    const bmin = [Infinity, Infinity, Infinity], bmax = [-Infinity, -Infinity, -Infinity];
    const cmin = [Infinity, Infinity, Infinity], cmax = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i++) {
      const o = st.order[i] * 3;
      for (let d = 0; d < 3; d++) {
        if (st.segMin[o + d] < bmin[d]) bmin[d] = st.segMin[o + d];
        if (st.segMax[o + d] > bmax[d]) bmax[d] = st.segMax[o + d];
        if (st.centroid[o + d] < cmin[d]) cmin[d] = st.centroid[o + d];
        if (st.centroid[o + d] > cmax[d]) cmax[d] = st.centroid[o + d];
      }
    }
    for (let d = 0; d < 3; d++) { st.nmin[node * 3 + d] = bmin[d]; st.nmax[node * 3 + d] = bmax[d]; }

    // widest centroid extent; ties take the lowest axis, like np.argmax
    let axis = 0, extent = cmax[0] - cmin[0];
    for (let d = 1; d < 3; d++) {
      const e = cmax[d] - cmin[d];
      if (e > extent) { extent = e; axis = d; }
    }
    if (cnt <= st.leaf || extent <= EPS) {
      st.start[node] = start;
      st.count[node] = cnt;
      return node;
    }

    // stable sort of the slice by centroid along `axis` (positions break ties,
    // which is exactly what numpy's kind="stable" argsort does)
    const pos = new Array(cnt), key = new Float64Array(cnt);
    for (let i = 0; i < cnt; i++) { pos[i] = i; key[i] = st.centroid[st.order[start + i] * 3 + axis]; }
    pos.sort((p, q) => (key[p] - key[q]) || (p - q));
    const reordered = new Int32Array(cnt);
    for (let i = 0; i < cnt; i++) reordered[i] = st.order[start + pos[i]];
    st.order.set(reordered, start);

    const mid = start + (cnt >>> 1);
    st.left[node] = buildNode(st, start, mid);
    st.right[node] = buildNode(st, mid, end);
    return node;
  }

  // Segments of a run's flat (N, L, 3) trajectory points: segment
  // j * (L - 1) + l joins layer l to layer l+1 of trajectory j, meta [j, l].
  //
  // bvh.py's `segments_from_coords` takes (steps, units, C) — step-major — and
  // transposes to unit-major before flattening, so its segment order is
  // unit * (steps - 1) + step. The viewer's points are already
  // trajectory-major ((N, L, 3), trajectory = unit, layer = step), so the
  // translation is just "drop the transpose": the resulting index and meta
  // ordering is identical to the Python's for the same trajectories.
  function fromPoints(points, N, L, leafSize) {
    const p = toF64(points, "points");
    N = Math.trunc(N); L = Math.trunc(L);
    if (N < 0 || L < 0) throw new Error("N and L must be non-negative");
    if (p.length < N * L * 3)
      throw new Error(`points has ${p.length} values, expected ${N * L * 3} for (${N}, ${L}, 3)`);
    if (N === 0 || L < 2) return build([], [], null, leafSize);  // nothing to connect
    const segs = N * (L - 1);
    const A = new Float64Array(segs * 3), B = new Float64Array(segs * 3),
          meta = new Int32Array(segs * 2);
    let s = 0;
    for (let j = 0; j < N; j++) for (let l = 0; l < L - 1; l++, s++) {
      const o = (j * L + l) * 3;
      A[s * 3] = p[o]; A[s * 3 + 1] = p[o + 1]; A[s * 3 + 2] = p[o + 2];
      B[s * 3] = p[o + 3]; B[s * 3 + 1] = p[o + 4]; B[s * 3 + 2] = p[o + 5];
      meta[s * 2] = j; meta[s * 2 + 1] = l;
    }
    return build(A, B, meta, leafSize);
  }

  // ------------------------------------------------------- geometry kernels
  // Scratch result for the per-segment solves; reused so a leaf test allocates
  // nothing (the Python vectorises the same maths over the whole leaf).
  const _sol = { dist: 0, t: 0, s: 0, px: 0, py: 0, pz: 0 };

  // Closest approach between the ray o + t d (t >= 0, |d| = 1) and segment
  // A->B. Fills `_sol` with the perpendicular distance, the ray parameter t,
  // the segment parameter s, and the closest point ON THE RAY — bvh.py's
  // `_closest_ray_segments` returns that point (`rp`), not the segment's.
  function closestRaySegment(ox, oy, oz, dx, dy, dz, ax, ay, az, bx, by, bz) {
    const vx = bx - ax, vy = by - ay, vz = bz - az;
    const w0x = ox - ax, w0y = oy - ay, w0z = oz - az;
    const b = vx * dx + vy * dy + vz * dz;
    const c = vx * vx + vy * vy + vz * vz;
    const dd = w0x * dx + w0y * dy + w0z * dz;
    const e = vx * w0x + vy * w0y + vz * w0z;
    const denom = c - b * b;                      // >= 0
    let s = denom > EPS ? (e - b * dd) / denom : 0;
    s = s < 0 ? 0 : (s > 1 ? 1 : s);
    let t = s * b - dd;
    if (t < 0) {                                  // closest point is behind the
      t = 0;                                      // origin: clamp t and re-solve s
      s = c > EPS ? e / c : 0;
      s = s < 0 ? 0 : (s > 1 ? 1 : s);
    }
    const rx = ox + t * dx, ry = oy + t * dy, rz = oz + t * dz;
    const ex = rx - (ax + s * vx), ey = ry - (ay + s * vy), ez = rz - (az + s * vz);
    _sol.dist = Math.sqrt(ex * ex + ey * ey + ez * ez);
    _sol.t = t; _sol.s = s; _sol.px = rx; _sol.py = ry; _sol.pz = rz;
    return _sol;
  }

  // Distance from point p to segment A->B; fills `_sol` with the distance and
  // the closest point ON THE SEGMENT (bvh.py's `_point_segments`).
  function closestPointSegment(px, py, pz, ax, ay, az, bx, by, bz) {
    const vx = bx - ax, vy = by - ay, vz = bz - az;
    const wx = px - ax, wy = py - ay, wz = pz - az;
    const c = vx * vx + vy * vy + vz * vz;
    let s = c > EPS ? (wx * vx + wy * vy + wz * vz) / c : 0;
    s = s < 0 ? 0 : (s > 1 ? 1 : s);
    const qx = ax + s * vx, qy = ay + s * vy, qz = az + s * vz;
    const ex = px - qx, ey = py - qy, ez = pz - qz;
    _sol.dist = Math.sqrt(ex * ex + ey * ey + ez * ez);
    _sol.t = 0; _sol.s = s; _sol.px = qx; _sol.py = qy; _sol.pz = qz;
    return _sol;
  }

  // Entry distance of the ray into node's box grown by `radius`, or null when
  // the ray misses it. `invd` components are +Infinity where d is ~0, so a
  // slab the ray is parallel to contributes ±Infinity (miss) or NaN (the
  // origin sits exactly on the slab plane) — NaN is dropped by the
  // nan-ignoring reduction below, mirroring np.nanmax / np.nanmin.
  function rayAabbEnter(bvh, node, ox, oy, oz, ivx, ivy, ivz, radius) {
    let tmin = NaN, tmax = NaN;
    const o3 = node * 3;
    for (let d = 0; d < 3; d++) {
      const org = d === 0 ? ox : d === 1 ? oy : oz;
      const inv = d === 0 ? ivx : d === 1 ? ivy : ivz;
      const t1 = (bvh.nmin[o3 + d] - radius - org) * inv;
      const t2 = (bvh.nmax[o3 + d] + radius - org) * inv;
      const lo = Math.min(t1, t2), hi = Math.max(t1, t2);
      if (lo === lo) tmin = tmin === tmin ? Math.max(tmin, lo) : lo;   // x === x
      if (hi === hi) tmax = tmax === tmax ? Math.min(tmax, hi) : hi;   // is !isNaN
    }
    // NaN compares false either way, exactly as the numpy/Python original does
    if (tmax < Math.max(tmin, 0)) return null;
    return Math.max(tmin, 0);
  }

  // -------------------------------------------------------------- ray pick
  // Front-most segment within `radius` of the ray (the grab gesture).
  //   opts.radius  : pick tolerance in world units; default 1% of the scene
  //                  diagonal (1e-6 for a degenerate scene)
  //   opts.maxDist : ignore hits at or beyond this ray parameter (default Inf)
  // Returns {index, t, point, distance} for the smallest t, or null. `point`
  // is the closest point on the ray, matching bvh.py's RayHit.
  function rayPick(bvh, origin, direction, opts) {
    if (!bvh || bvh.length === 0) return null;
    const o = opts || {};
    const ox = origin[0], oy = origin[1], oz = origin[2];
    let dx = direction[0], dy = direction[1], dz = direction[2];
    const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(norm >= EPS)) throw new Error("direction must be non-zero");
    dx /= norm; dy /= norm; dz /= norm;
    const radius = o.radius == null
      ? (bvh.diagonal > 0 ? 0.01 * bvh.diagonal : 1e-6)
      : o.radius;
    const ivx = Math.abs(dx) > EPS ? 1 / dx : Infinity;
    const ivy = Math.abs(dy) > EPS ? 1 / dy : Infinity;
    const ivz = Math.abs(dz) > EPS ? 1 / dz : Infinity;

    let bestT = o.maxDist == null ? Infinity : o.maxDist;
    let best = null;
    const stack = [0];
    while (stack.length) {
      const node = stack.pop();
      const enter = rayAabbEnter(bvh, node, ox, oy, oz, ivx, ivy, ivz, radius);
      if (enter === null || enter > bestT) continue;
      if (bvh.left[node] < 0) {                   // leaf: test its segments
        const s0 = bvh.start[node], s1 = s0 + bvh.count[node];
        for (let k = s0; k < s1; k++) {
          const seg = bvh.order[k], p = seg * 3;
          const h = closestRaySegment(ox, oy, oz, dx, dy, dz,
                                      bvh.A[p], bvh.A[p + 1], bvh.A[p + 2],
                                      bvh.B[p], bvh.B[p + 1], bvh.B[p + 2]);
          // strictly-smaller t keeps the first of several equal-t candidates,
          // which is what the Python's argmin over the leaf's mask returns
          if (h.dist <= radius && h.t >= 0 && h.t < bestT) {
            bestT = h.t;
            best = { index: seg, t: h.t, point: [h.px, h.py, h.pz], distance: h.dist };
          }
        }
      } else {                                    // push left then right, so
        stack.push(bvh.left[node]);               // the right child is visited
        stack.push(bvh.right[node]);              // first — as in bvh.py
      }
    }
    return best;
  }

  // -------------------------------------------------------------- nearest
  // Closest segment to a 3-D point (hover / snap). Returns
  // {index, t: 0, point, distance} where `point` is on the segment, or null.
  function nearest(bvh, point) {
    if (!bvh || bvh.length === 0) return null;
    const px = point[0], py = point[1], pz = point[2];
    let bestD = Infinity, best = null;
    const stack = [0];
    while (stack.length) {
      const node = stack.pop();
      // distance to the node box (0 inside): prune when it can't improve
      const qx = Math.min(Math.max(px, bvh.nmin[node * 3]), bvh.nmax[node * 3]);
      const qy = Math.min(Math.max(py, bvh.nmin[node * 3 + 1]), bvh.nmax[node * 3 + 1]);
      const qz = Math.min(Math.max(pz, bvh.nmin[node * 3 + 2]), bvh.nmax[node * 3 + 2]);
      if (Math.hypot(qx - px, qy - py, qz - pz) >= bestD) continue;
      if (bvh.left[node] < 0) {
        const s0 = bvh.start[node], s1 = s0 + bvh.count[node];
        for (let k = s0; k < s1; k++) {
          const seg = bvh.order[k], p = seg * 3;
          const h = closestPointSegment(px, py, pz,
                                        bvh.A[p], bvh.A[p + 1], bvh.A[p + 2],
                                        bvh.B[p], bvh.B[p + 1], bvh.B[p + 2]);
          if (h.dist < bestD) {
            bestD = h.dist;
            best = { index: seg, t: 0, point: [h.px, h.py, h.pz], distance: h.dist };
          }
        }
      } else {
        stack.push(bvh.left[node]);
        stack.push(bvh.right[node]);
      }
    }
    return best;
  }

  // Segment metadata [traj, layer] for a hit index, or null when the BVH was
  // built without meta.
  function metaOf(bvh, index) {
    if (!bvh || !bvh.meta || index < 0 || index >= bvh.length) return null;
    return [bvh.meta[index * 2], bvh.meta[index * 2 + 1]];
  }

  // Parameter (0..1) of the closest point on segment `index` to a world point
  // — the fraction of the layer span a ray hit landed at.
  function segmentParam(bvh, index, point) {
    if (!bvh || index < 0 || index >= bvh.length) return 0;
    const p = index * 3;
    closestPointSegment(point[0], point[1], point[2],
                        bvh.A[p], bvh.A[p + 1], bvh.A[p + 2],
                        bvh.B[p], bvh.B[p + 1], bvh.B[p + 2]);
    return _sol.s;
  }

  return { build, fromPoints, rayPick, nearest, metaOf, segmentParam, EPS };
});
