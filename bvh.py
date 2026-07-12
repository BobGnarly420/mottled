"""Bounding-volume hierarchy over trajectory segments.

The fly-through canvas renders trajectories as polylines (curves), not voxels:
in any 3-D projection the states occupy a vanishing fraction of the volume, so
a spatial *acceleration structure over the curve segments* — not a grid over
empty space — is the right primitive. This BVH answers the queries the
interaction grammar needs:

    ray_pick   — the grab gesture: camera ray -> front-most segment within a
                 pick radius (mouse-over / click-to-grab a state)
    nearest    — closest segment to a 3-D point (hover, snapping)
    query_box  — segments overlapping an axis-aligned box (region select)
    query_frustum — segments in a view frustum (fly-through culling)

It is deliberately backend-agnostic: it consumes 3-D points (the output of a
projection / Reduction), never anything transformer-specific. A Trace of any
substrate that has been projected to <=3-D can be picked with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------
# Segment construction from projected coordinates
# --------------------------------------------------------------------------
def segments_from_coords(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-unit polyline segments from projected coordinates.

    coords : (n_steps, n_units, C) with C in {2, 3} — e.g. projected hidden
             states, step 0..L-1 for each unit (token / neuron / agent).
    Returns (A, B, meta):
        A, B : (N, 3) segment endpoints (2-D input is lifted to z=0).
        meta : (N, 2) int, each row (unit, step) — the segment connects
               step -> step+1 of that unit.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] not in (2, 3):
        raise ValueError(f"coords must be (steps, units, 2|3); got {coords.shape}")
    L, T, C = coords.shape
    if C == 2:
        coords = np.concatenate([coords, np.zeros((L, T, 1))], axis=2)
    if L < 2:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), dtype=int)
    # unit-major: for each unit, its consecutive-step segments
    A = coords[:-1].transpose(1, 0, 2).reshape(-1, 3)
    B = coords[1:].transpose(1, 0, 2).reshape(-1, 3)
    units = np.repeat(np.arange(T), L - 1)
    steps = np.tile(np.arange(L - 1), T)
    return A, B, np.stack([units, steps], axis=1)


# --------------------------------------------------------------------------
# Geometry primitives (vectorised over a batch of segments)
# --------------------------------------------------------------------------
def _closest_ray_segments(o: np.ndarray, d: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Closest approach between a ray (o + t d, t>=0, |d|=1) and segments A->B.

    Returns (dist, t, point): perpendicular distance, ray parameter t of the
    closest point, and that point on the ray — each of length N.
    """
    v = B - A
    w0 = o - A
    b = v @ d                                   # (N,)
    c = np.einsum("ij,ij->i", v, v)             # |v|^2
    dd = w0 @ d
    e = np.einsum("ij,ij->i", v, w0)
    denom = c - b * b                           # >= 0
    safe = denom > _EPS
    s = np.where(safe, (e - b * dd) / np.where(safe, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = s * b - dd
    # if the closest point falls behind the ray origin, clamp t=0 and re-solve s
    behind = t < 0.0
    if behind.any():
        t = np.where(behind, 0.0, t)
        cpos = c > _EPS
        s_at0 = np.where(cpos, e / np.where(cpos, c, 1.0), 0.0)
        s = np.where(behind, np.clip(s_at0, 0.0, 1.0), s)
    rp = o + t[:, None] * d
    sp = A + s[:, None] * v
    diff = rp - sp
    return np.sqrt(np.einsum("ij,ij->i", diff, diff)), t, rp


def _point_segments(p: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Distance from point p to segments A->B, vectorised. Returns (dist, point)."""
    v = B - A
    w = p - A
    c = np.einsum("ij,ij->i", v, v)
    s = np.where(c > _EPS, np.einsum("ij,ij->i", w, v) / np.where(c > _EPS, c, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    proj = A + s[:, None] * v
    diff = p - proj
    return np.sqrt(np.einsum("ij,ij->i", diff, diff)), proj


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class RayHit:
    """A ray-pick result."""

    index: int            # segment index into the BVH's A/B/meta arrays
    t: float              # distance along the ray to the closest point
    point: np.ndarray     # closest point on the ray (3,)
    distance: float       # perpendicular distance from ray to the segment


# --------------------------------------------------------------------------
# BVH
# --------------------------------------------------------------------------
class BVH:
    """Median-split bounding-volume hierarchy over line segments.

    Deterministic build (stable median split on the widest centroid axis).
    Leaves hold up to ``leaf_size`` segments and are tested vectorised.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, meta: np.ndarray | None = None,
                 leaf_size: int = 8):
        self.A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
        self.B = np.asarray(B, dtype=np.float64).reshape(-1, 3)
        if self.A.shape != self.B.shape:
            raise ValueError("A and B must have the same shape")
        self.meta = None if meta is None else np.asarray(meta)
        self.leaf_size = max(1, int(leaf_size))
        n = len(self.A)

        self._seg_min = np.minimum(self.A, self.B)
        self._seg_max = np.maximum(self.A, self.B)
        self._centroid = 0.5 * (self.A + self.B)
        self.order = np.arange(n)

        # node arrays, filled during build
        self._nmin: list = []
        self._nmax: list = []
        self._left: list = []
        self._right: list = []
        self._start: list = []
        self._count: list = []
        if n > 0:
            self._build(0, n)

        self.nmin = np.array(self._nmin, dtype=np.float64).reshape(-1, 3)
        self.nmax = np.array(self._nmax, dtype=np.float64).reshape(-1, 3)
        self.left = np.array(self._left, dtype=np.int64) if self._left else np.zeros(0, np.int64)
        self.right = np.array(self._right, dtype=np.int64) if self._right else np.zeros(0, np.int64)
        self.start = np.array(self._start, dtype=np.int64) if self._start else np.zeros(0, np.int64)
        self.count = np.array(self._count, dtype=np.int64) if self._count else np.zeros(0, np.int64)
        del self._nmin, self._nmax, self._left, self._right, self._start, self._count

        if n > 0:
            self.bounds_min = self.nmin[0].copy()
            self.bounds_max = self.nmax[0].copy()
            self.diagonal = float(np.linalg.norm(self.bounds_max - self.bounds_min))
        else:
            self.bounds_min = np.zeros(3)
            self.bounds_max = np.zeros(3)
            self.diagonal = 0.0

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_coords(cls, coords: np.ndarray, leaf_size: int = 8) -> "BVH":
        A, B, meta = segments_from_coords(coords)
        return cls(A, B, meta=meta, leaf_size=leaf_size)

    def __len__(self) -> int:
        return len(self.A)

    # ------------------------------------------------------------------ build
    def _build(self, start: int, end: int) -> int:
        node = len(self._nmin)
        self._nmin.append(None)
        self._nmax.append(None)
        self._left.append(-1)
        self._right.append(-1)
        self._start.append(-1)
        self._count.append(0)

        idx = self.order[start:end]
        self._nmin[node] = self._seg_min[idx].min(axis=0)
        self._nmax[node] = self._seg_max[idx].max(axis=0)
        count = end - start

        cen = self._centroid[idx]
        extent = cen.max(axis=0) - cen.min(axis=0)
        axis = int(np.argmax(extent))
        if count <= self.leaf_size or extent[axis] <= _EPS:
            self._start[node] = start
            self._count[node] = count
            return node

        order_slice = idx[np.argsort(cen[:, axis], kind="stable")]
        self.order[start:end] = order_slice
        mid = start + count // 2
        self._left[node] = self._build(start, mid)
        self._right[node] = self._build(mid, end)
        return node

    def _is_leaf(self, node: int) -> bool:
        return self.left[node] < 0

    # ------------------------------------------------------------- ray pick
    def _ray_aabb_enter(self, o, invd, node, radius):
        """Entry distance of the ray into node's box expanded by `radius`, or None."""
        lo = self.nmin[node] - radius - o
        hi = self.nmax[node] + radius - o
        with np.errstate(invalid="ignore"):
            t1 = lo * invd
            t2 = hi * invd
        tmin = np.nanmax(np.minimum(t1, t2))
        tmax = np.nanmin(np.maximum(t1, t2))
        if tmax < max(tmin, 0.0):
            return None
        return max(tmin, 0.0)

    def ray_pick(self, origin, direction, radius: float | None = None,
                 max_dist: float = np.inf) -> RayHit | None:
        """Front-most segment within `radius` of the ray (the grab gesture).

        origin, direction : camera ray (direction need not be normalised).
        radius : pick tolerance in world units; defaults to 1% of the scene
                 diagonal. Returns the hit with the smallest ray parameter t
                 (closest to the camera) whose distance to the ray <= radius,
                 or None.
        """
        if len(self) == 0:
            return None
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm < _EPS:
            raise ValueError("direction must be non-zero")
        d = d / norm
        if radius is None:
            radius = 0.01 * self.diagonal if self.diagonal > 0 else 1e-6
        with np.errstate(divide="ignore"):
            invd = np.where(np.abs(d) > _EPS, 1.0 / d, np.inf)

        best_t = float(max_dist)
        best: RayHit | None = None
        stack = [0]
        while stack:
            node = stack.pop()
            enter = self._ray_aabb_enter(o, invd, node, radius)
            if enter is None or enter > best_t:
                continue
            if self._is_leaf(node):
                s, e = self.start[node], self.start[node] + self.count[node]
                seg = self.order[s:e]
                dist, t, pts = _closest_ray_segments(o, d, self.A[seg], self.B[seg])
                ok = (dist <= radius) & (t >= 0.0) & (t < best_t)
                if ok.any():
                    j = np.flatnonzero(ok)[np.argmin(t[ok])]
                    best_t = float(t[j])
                    best = RayHit(int(seg[j]), best_t, pts[j].copy(), float(dist[j]))
            else:
                stack.append(self.left[node])
                stack.append(self.right[node])
        return best

    # -------------------------------------------------------------- nearest
    def nearest(self, point) -> RayHit | None:
        """Closest segment to a 3-D point (hover / snap).

        Returns a RayHit whose `t` is 0, `point` is the closest point on the
        segment, and `distance` is the point-to-segment distance.
        """
        if len(self) == 0:
            return None
        p = np.asarray(point, dtype=np.float64)
        best_d = np.inf
        best: RayHit | None = None
        stack = [0]
        while stack:
            node = stack.pop()
            q = np.minimum(np.maximum(p, self.nmin[node]), self.nmax[node])
            if np.linalg.norm(q - p) >= best_d:
                continue
            if self._is_leaf(node):
                s, e = self.start[node], self.start[node] + self.count[node]
                seg = self.order[s:e]
                dist, pts = _point_segments(p, self.A[seg], self.B[seg])
                j = int(np.argmin(dist))
                if dist[j] < best_d:
                    best_d = float(dist[j])
                    best = RayHit(int(seg[j]), 0.0, pts[j].copy(), best_d)
            else:
                stack.append(self.left[node])
                stack.append(self.right[node])
        return best

    # ------------------------------------------------------------- box query
    def query_box(self, box_min, box_max) -> np.ndarray:
        """Indices of segments whose AABB overlaps [box_min, box_max]."""
        if len(self) == 0:
            return np.zeros(0, dtype=np.int64)
        lo = np.asarray(box_min, dtype=np.float64)
        hi = np.asarray(box_max, dtype=np.float64)
        out: list[int] = []
        stack = [0]
        while stack:
            node = stack.pop()
            if (self.nmax[node] < lo).any() or (self.nmin[node] > hi).any():
                continue
            if self._is_leaf(node):
                s, e = self.start[node], self.start[node] + self.count[node]
                seg = self.order[s:e]
                overlap = ~((self._seg_max[seg] < lo).any(axis=1) |
                            (self._seg_min[seg] > hi).any(axis=1))
                out.extend(seg[overlap].tolist())
            else:
                stack.append(self.left[node])
                stack.append(self.right[node])
        return np.array(sorted(out), dtype=np.int64)

    # --------------------------------------------------------- frustum query
    def query_frustum(self, planes) -> np.ndarray:
        """Indices of segments not culled by the view frustum.

        planes : (6, 4) array of (a, b, c, d); a point is inside a plane when
        a*x + b*y + c*z + d >= 0. A node is culled when its AABB lies entirely
        on the negative side of any plane (conservative: may keep a few
        segments just outside, never drops a visible one).
        """
        if len(self) == 0:
            return np.zeros(0, dtype=np.int64)
        planes = np.asarray(planes, dtype=np.float64).reshape(-1, 4)
        out: list[int] = []
        stack = [0]
        while stack:
            node = stack.pop()
            if self._node_outside(self.nmin[node], self.nmax[node], planes):
                continue
            if self._is_leaf(node):
                s, e = self.start[node], self.start[node] + self.count[node]
                for i in self.order[s:e]:
                    if not self._node_outside(self._seg_min[i], self._seg_max[i], planes):
                        out.append(int(i))
            else:
                stack.append(self.left[node])
                stack.append(self.right[node])
        return np.array(sorted(out), dtype=np.int64)

    @staticmethod
    def _node_outside(bmin, bmax, planes) -> bool:
        # positive-vertex test: the box corner furthest along the plane normal
        n = planes[:, :3]
        pos = np.where(n >= 0, bmax, bmin)          # (P, 3)
        return bool((np.einsum("ij,ij->i", n, pos) + planes[:, 3] < 0).any())
