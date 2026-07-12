"""BVH correctness: verified against O(N) brute force + analytic cases.

A spatial acceleration structure is only trustworthy if it never *misses* a
hit the brute-force scan would find. Every query here is checked against an
unaccelerated reference over the same geometry primitive, so the tests
isolate the traversal/pruning logic; separate analytic cases pin the geometry
primitive itself.
"""

import numpy as np
import pytest

from bvh import (
    BVH,
    RayHit,
    _closest_ray_segments,
    _point_segments,
    segments_from_coords,
)


# ------------------------------------------------------------------ fixtures
def random_segments(rng, n, scale=10.0, seg_len=1.0):
    A = rng.uniform(-scale, scale, size=(n, 3))
    B = A + rng.uniform(-seg_len, seg_len, size=(n, 3))
    return A, B


# ---------------------------------------------------------- geometry primitive
def test_closest_ray_segment_perpendicular():
    # ray along +x at origin; segment crossing it at x=5, offset z=2
    o, d = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    A = np.array([[5.0, -1, 2]])
    B = np.array([[5.0, 1, 2]])
    dist, t, pt = _closest_ray_segments(o, d, A, B)
    assert dist[0] == pytest.approx(2.0)      # z-offset
    assert t[0] == pytest.approx(5.0)          # meets at x=5
    assert pt[0] == pytest.approx([5.0, 0, 0])


def test_closest_ray_segment_endpoint_clamp():
    # closest approach is past the segment end -> clamps to the endpoint
    o, d = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    A = np.array([[5.0, 1, 0]])
    B = np.array([[5.0, 3, 0]])
    dist, t, _ = _closest_ray_segments(o, d, A, B)
    assert dist[0] == pytest.approx(1.0)       # nearest point is A at y=1
    assert t[0] == pytest.approx(5.0)


def test_closest_ray_segment_behind_origin():
    # segment sits behind the ray origin -> t clamps to 0
    o, d = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    A = np.array([[-5.0, 0, 0]])
    B = np.array([[-5.0, 0, 4]])
    dist, t, _ = _closest_ray_segments(o, d, A, B)
    assert t[0] == pytest.approx(0.0)
    assert dist[0] == pytest.approx(5.0)


def test_point_segment_distance():
    p = np.array([0.0, 0, 0])
    A = np.array([[1.0, -1, 0], [3.0, 3, 0]])
    B = np.array([[1.0, 1, 0], [5.0, 5, 0]])
    dist, proj = _point_segments(p, A, B)
    assert dist[0] == pytest.approx(1.0)          # perpendicular to first
    assert proj[0] == pytest.approx([1.0, 0, 0])
    assert dist[1] == pytest.approx(np.hypot(3, 3))  # clamps to endpoint (3,3)


# ------------------------------------------------------------ build invariants
@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 50, 257])
def test_build_covers_all_segments(n):
    rng = np.random.default_rng(n)
    bvh = BVH(*random_segments(rng, n), leaf_size=8)
    # every segment appears exactly once across the leaves
    seen = []
    for node in range(len(bvh.nmin)):
        if bvh.left[node] < 0:
            s, e = bvh.start[node], bvh.start[node] + bvh.count[node]
            seen.extend(bvh.order[s:e].tolist())
    assert sorted(seen) == list(range(n))


def test_parent_boxes_contain_children():
    rng = np.random.default_rng(0)
    bvh = BVH(*random_segments(rng, 200), leaf_size=8)
    for node in range(len(bvh.nmin)):
        if bvh.left[node] >= 0:
            for child in (bvh.left[node], bvh.right[node]):
                assert (bvh.nmin[node] <= bvh.nmin[child] + 1e-9).all()
                assert (bvh.nmax[node] >= bvh.nmax[child] - 1e-9).all()


def test_leaf_boxes_contain_their_segments():
    rng = np.random.default_rng(1)
    A, B = random_segments(rng, 120)
    bvh = BVH(A, B, leaf_size=4)
    for node in range(len(bvh.nmin)):
        if bvh.left[node] < 0:
            s, e = bvh.start[node], bvh.start[node] + bvh.count[node]
            for i in bvh.order[s:e]:
                assert (bvh.nmin[node] <= np.minimum(A[i], B[i]) + 1e-9).all()
                assert (bvh.nmax[node] >= np.maximum(A[i], B[i]) - 1e-9).all()


# --------------------------------------------------- ray_pick vs brute force
def _brute_ray(o, d, A, B, radius, max_dist=np.inf):
    d = np.asarray(d, float) / np.linalg.norm(d)
    dist, t, _ = _closest_ray_segments(np.asarray(o, float), d, A, B)
    ok = (dist <= radius) & (t >= 0) & (t < max_dist)
    if not ok.any():
        return None
    return int(np.flatnonzero(ok)[np.argmin(t[ok])])


@pytest.mark.parametrize("seed", range(8))
def test_ray_pick_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    A, B = random_segments(rng, 400, scale=10, seg_len=2)
    bvh = BVH(A, B, leaf_size=8)
    radius = 1.0
    for _ in range(30):
        o = rng.uniform(-15, 15, size=3)
        d = rng.normal(size=3)
        hit = bvh.ray_pick(o, d, radius=radius)
        ref = _brute_ray(o, d, A, B, radius)
        if ref is None:
            assert hit is None
        else:
            assert hit is not None
            # allow a tie: BVH may pick an equally-close-in-t segment
            _, t_hit, _ = _closest_ray_segments(o, d / np.linalg.norm(d),
                                                 A[hit.index:hit.index + 1],
                                                 B[hit.index:hit.index + 1])
            _, t_ref, _ = _closest_ray_segments(o, d / np.linalg.norm(d),
                                                 A[ref:ref + 1], B[ref:ref + 1])
            assert t_hit[0] == pytest.approx(t_ref[0], abs=1e-6)


def test_ray_pick_front_most():
    # two parallel segments straddling the ray; the nearer in t wins
    A = np.array([[3.0, -1, 0], [6.0, -1, 0]])
    B = np.array([[3.0, 1, 0], [6.0, 1, 0]])
    bvh = BVH(A, B, leaf_size=1)
    hit = bvh.ray_pick([0, 0, 0], [1, 0, 0], radius=0.5)
    assert hit.index == 0 and hit.t == pytest.approx(3.0)


def test_ray_pick_respects_radius_and_maxdist():
    A = np.array([[5.0, -1, 0]])
    B = np.array([[5.0, 1, 0]])
    bvh = BVH(A, B)
    assert bvh.ray_pick([0, 0, 2], [1, 0, 0], radius=1.0) is None   # 2 away, radius 1
    assert bvh.ray_pick([0, 0, 2], [1, 0, 0], radius=3.0) is not None
    assert bvh.ray_pick([0, 0, 0], [1, 0, 0], radius=0.5, max_dist=4.0) is None


def test_ray_pick_zero_direction_rejected():
    bvh = BVH(*random_segments(np.random.default_rng(0), 10))
    with pytest.raises(ValueError):
        bvh.ray_pick([0, 0, 0], [0, 0, 0])


# ----------------------------------------------------- nearest vs brute force
@pytest.mark.parametrize("seed", range(6))
def test_nearest_matches_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    A, B = random_segments(rng, 300)
    bvh = BVH(A, B, leaf_size=8)
    for _ in range(30):
        p = rng.uniform(-12, 12, size=3)
        hit = bvh.nearest(p)
        dist, _ = _point_segments(p, A, B)
        assert hit.distance == pytest.approx(dist.min(), abs=1e-9)


# -------------------------------------------------------- box query vs brute
@pytest.mark.parametrize("seed", range(6))
def test_query_box_matches_brute_force(seed):
    rng = np.random.default_rng(200 + seed)
    A, B = random_segments(rng, 300)
    bvh = BVH(A, B, leaf_size=8)
    lo = rng.uniform(-8, 0, size=3)
    hi = lo + rng.uniform(1, 8, size=3)
    got = bvh.query_box(lo, hi)
    seg_min = np.minimum(A, B)
    seg_max = np.maximum(A, B)
    ref = np.flatnonzero(~((seg_max < lo).any(axis=1) | (seg_min > hi).any(axis=1)))
    assert np.array_equal(got, ref)


# ------------------------------------------------------------ frustum query
def test_query_frustum_keeps_inside_drops_outside():
    # unit box frustum around the origin: |x|,|y|,|z| <= 1
    planes = np.array([
        [1, 0, 0, 1], [-1, 0, 0, 1],
        [0, 1, 0, 1], [0, -1, 0, 1],
        [0, 0, 1, 1], [0, 0, -1, 1],
    ], dtype=float)
    A = np.array([[0.0, 0, 0], [5.0, 5, 5], [0.5, 0, 0]])
    B = np.array([[0.2, 0, 0], [6.0, 5, 5], [0.6, 0, 0]])
    bvh = BVH(A, B, leaf_size=1)
    got = set(bvh.query_frustum(planes).tolist())
    assert 0 in got and 2 in got     # inside the box
    assert 1 not in got              # far outside


def test_query_frustum_is_conservative_superset():
    # a random frustum: every truly-visible segment must be kept
    rng = np.random.default_rng(7)
    A, B = random_segments(rng, 200)
    bvh = BVH(A, B)
    planes = np.array([
        [1, 0, 0, 3], [-1, 0, 0, 3],
        [0, 1, 0, 3], [0, -1, 0, 3],
        [0, 0, 1, 3], [0, 0, -1, 3],
    ], dtype=float)
    kept = set(bvh.query_frustum(planes).tolist())

    def inside(pt):
        return bool((planes[:, :3] @ pt + planes[:, 3] >= 0).all())

    for i in range(len(A)):
        if inside(A[i]) or inside(B[i]):
            assert i in kept


# ------------------------------------------------------- coords -> segments
def test_segments_from_coords_topology():
    # 3 steps, 2 units -> each unit contributes 2 segments
    coords = np.arange(3 * 2 * 3, dtype=float).reshape(3, 2, 3)
    A, B, meta = segments_from_coords(coords)
    assert A.shape == B.shape == (4, 3)
    assert meta.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    # unit 0 segment 0 connects step0 -> step1 of unit 0
    assert np.array_equal(A[0], coords[0, 0]) and np.array_equal(B[0], coords[1, 0])


def test_segments_from_coords_lifts_2d():
    coords = np.zeros((4, 3, 2))
    A, B, meta = segments_from_coords(coords)
    assert A.shape[1] == 3 and (A[:, 2] == 0).all()
    assert len(A) == (4 - 1) * 3


def test_from_coords_roundtrip_pick():
    rng = np.random.default_rng(3)
    coords = rng.normal(size=(13, 5, 3))     # 13 steps, 5 units (a StateTrajectory shape)
    bvh = BVH.from_coords(coords)
    assert len(bvh) == 12 * 5
    # a ray straight at a known vertex should pick one of its incident segments
    target = coords[6, 2]
    hit = bvh.ray_pick(target + np.array([0, 0, 5.0]), [0, 0, -1], radius=1e-3)
    assert hit is not None
    assert 2 in bvh.meta[hit.index]          # belongs to unit 2


# --------------------------------------------------------------- degenerate
def test_empty_bvh():
    bvh = BVH(np.zeros((0, 3)), np.zeros((0, 3)))
    assert len(bvh) == 0
    assert bvh.ray_pick([0, 0, 0], [1, 0, 0]) is None
    assert bvh.nearest([0, 0, 0]) is None
    assert bvh.query_box([-1, -1, -1], [1, 1, 1]).size == 0


def test_single_and_zero_length_segments():
    # a zero-length segment (a point) must still be pickable / measurable
    A = np.array([[2.0, 0, 0]])
    B = np.array([[2.0, 0, 0]])
    bvh = BVH(A, B)
    hit = bvh.ray_pick([0, 0, 0], [1, 0, 0], radius=0.1)
    assert hit is not None and hit.t == pytest.approx(2.0)
    assert bvh.nearest([2, 0, 3]).distance == pytest.approx(3.0)


def test_deterministic_build():
    rng = np.random.default_rng(5)
    A, B = random_segments(rng, 150)
    b1, b2 = BVH(A, B), BVH(A, B)
    assert np.array_equal(b1.order, b2.order)
    assert np.array_equal(b1.left, b2.left)
