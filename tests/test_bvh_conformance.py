"""BVH conformance: the Python spatial index and the JS viewer agree.

`bvh.py` is the reference implementation (tested in test_bvh.py); the viewer
picks in the browser with `viewer/bvh.js`. Picking the *same* segment for the
same ray is the contract that makes the two surfaces one tool, so it is pinned
here the way `.mtj` decoding is pinned in test_mtj_conformance.py: run both
implementations over identical data and compare.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bvh import BVH, segments_from_coords

ROOT = Path(__file__).resolve().parent.parent
BVH_JS = ROOT / "viewer" / "bvh.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")


def _random_segments(n=200, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, 3))
    B = A + 0.35 * rng.normal(size=(n, 3))
    return A, B


def _rays(n=40, seed=1):
    """Rays aimed through the segment cloud from outside it."""
    rng = np.random.default_rng(seed)
    origins = rng.normal(size=(n, 3)) * 0.5 + np.array([0.0, 0.0, 6.0])
    targets = rng.normal(size=(n, 3)) * 0.8
    return origins, targets - origins


def _run_js(script: str, payload: dict) -> dict:
    proc = subprocess.run(["node", "-e", script, str(BVH_JS)],
                          input=json.dumps(payload), capture_output=True,
                          text=True, check=True)
    return json.loads(proc.stdout)


_PICK_JS = r"""
  const BVH = require(process.argv[1]);
  let raw = ""; process.stdin.on("data", (c) => (raw += c));
  process.stdin.on("end", () => {
    const p = JSON.parse(raw);
    const bvh = BVH.build(p.A, p.B, p.meta || null, p.leafSize || 8);
    const out = p.rays.map((r) =>
      BVH.rayPick(bvh, r.origin, r.direction,
                  r.radius == null ? {} : { radius: r.radius }));
    process.stdout.write(JSON.stringify({
      hits: out.map((h) => (h ? { index: h.index, t: h.t, distance: h.distance,
                                  point: h.point } : null)),
      diagonal: bvh.diagonal, length: bvh.length,
    }));
  });
"""


def test_ray_pick_matches_python_reference():
    """Same ray, same segment, same geometry — across languages."""
    A, B = _random_segments()
    origins, directions = _rays()
    radius = 0.25

    ref = BVH(A, B)
    expected = [ref.ray_pick(o, d, radius=radius)
                for o, d in zip(origins, directions)]

    js = _run_js(_PICK_JS, {
        "A": A.ravel().tolist(), "B": B.ravel().tolist(),
        "rays": [{"origin": o.tolist(), "direction": d.tolist(), "radius": radius}
                 for o, d in zip(origins, directions)],
    })

    assert js["length"] == len(ref)
    assert js["diagonal"] == pytest.approx(ref.diagonal, rel=1e-9)
    assert any(h is not None for h in expected), "fixture should produce hits"

    for i, (want, got) in enumerate(zip(expected, js["hits"])):
        if want is None:
            assert got is None, f"ray {i}: JS hit segment {got} where Python missed"
            continue
        assert got is not None, f"ray {i}: JS missed segment {want.index}"
        assert got["index"] == want.index, f"ray {i}: different segment"
        assert got["t"] == pytest.approx(want.t, rel=1e-9, abs=1e-9)
        assert got["distance"] == pytest.approx(want.distance, rel=1e-9, abs=1e-9)
        np.testing.assert_allclose(got["point"], want.point, rtol=1e-9, atol=1e-9)


def test_default_radius_matches_python():
    """The pick tolerance itself is part of the contract (1% of the diagonal)."""
    A, B = _random_segments(n=120, seed=5)
    origins, directions = _rays(n=25, seed=6)

    ref = BVH(A, B)
    expected = [ref.ray_pick(o, d) for o, d in zip(origins, directions)]
    js = _run_js(_PICK_JS, {
        "A": A.ravel().tolist(), "B": B.ravel().tolist(),
        "rays": [{"origin": o.tolist(), "direction": d.tolist()}
                 for o, d in zip(origins, directions)],
    })
    for i, (want, got) in enumerate(zip(expected, js["hits"])):
        assert (want is None) == (got is None), f"ray {i}: hit/miss disagreement"
        if want is not None:
            assert got["index"] == want.index


_NEAREST_JS = r"""
  const BVH = require(process.argv[1]);
  let raw = ""; process.stdin.on("data", (c) => (raw += c));
  process.stdin.on("end", () => {
    const p = JSON.parse(raw);
    const bvh = BVH.build(p.A, p.B, null, 8);
    process.stdout.write(JSON.stringify(p.points.map((q) => {
      const h = BVH.nearest(bvh, q);
      return h ? { index: h.index, distance: h.distance, point: h.point } : null;
    })));
  });
"""


def test_nearest_matches_python_reference():
    A, B = _random_segments(n=150, seed=2)
    rng = np.random.default_rng(3)
    points = rng.normal(size=(30, 3))

    ref = BVH(A, B)
    expected = [ref.nearest(p) for p in points]
    got = _run_js(_NEAREST_JS, {"A": A.ravel().tolist(), "B": B.ravel().tolist(),
                                "points": points.tolist()})

    for i, (want, hit) in enumerate(zip(expected, got)):
        assert hit is not None and want is not None
        assert hit["index"] == want.index, f"point {i}: different segment"
        assert hit["distance"] == pytest.approx(want.distance, rel=1e-9, abs=1e-9)


_FROM_POINTS_JS = r"""
  const BVH = require(process.argv[1]);
  let raw = ""; process.stdin.on("data", (c) => (raw += c));
  process.stdin.on("end", () => {
    const p = JSON.parse(raw);
    const bvh = BVH.fromPoints(p.points, p.N, p.L, 8);
    const meta = [];
    for (let i = 0; i < bvh.length; i++) meta.push([bvh.meta[2 * i], bvh.meta[2 * i + 1]]);
    process.stdout.write(JSON.stringify({
      length: bvh.length, meta,
      A: Array.from(bvh.A), B: Array.from(bvh.B),
    }));
  });
"""


def test_from_points_matches_segments_from_coords():
    """The viewer builds from (N, L, 3) run points; Python from (L, T, C)
    coords. Same segments, same (trajectory, layer) meta, same order."""
    rng = np.random.default_rng(4)
    N, L = 6, 9
    points = rng.normal(size=(N, L, 3))            # viewer layout
    coords = points.transpose(1, 0, 2).copy()      # (L, T, C) Python layout

    A, B, meta = segments_from_coords(coords)
    js = _run_js(_FROM_POINTS_JS, {"points": points.ravel().tolist(),
                                   "N": N, "L": L})

    assert js["length"] == len(A) == N * (L - 1)
    np.testing.assert_allclose(np.array(js["A"]).reshape(-1, 3), A, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.array(js["B"]).reshape(-1, 3), B, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(np.array(js["meta"], dtype=int), meta)


def test_empty_index_agrees():
    ref = BVH(np.zeros((0, 3)), np.zeros((0, 3)))
    assert ref.ray_pick([0, 0, 1], [0, 0, -1]) is None
    js = _run_js(_PICK_JS, {"A": [], "B": [],
                            "rays": [{"origin": [0, 0, 1], "direction": [0, 0, -1],
                                      "radius": 1.0}]})
    assert js["hits"] == [None] and js["length"] == 0
