"""StateTrajectory data model, extraction modes, animation continuity (spec test 7)."""

import numpy as np
import pytest

from models import synthetic
from trajectory import StateTrajectory, densify, extract


@pytest.fixture(scope="module")
def traj():
    return synthetic.capture("the marble rolls downhill fast")


@pytest.fixture(scope="module")
def coords(traj):
    from projection import project

    c, _ = project(traj.hidden, method="pca")
    return c


def test_validate_catches_bad_shapes(traj):
    traj.validate()
    broken = StateTrajectory(hidden=traj.hidden, tokens=traj.tokens[:-1])
    with pytest.raises(ValueError):
        broken.validate()
    nan = StateTrajectory(hidden=np.full((2, 2, 4), np.nan), tokens=["a", "b"])
    with pytest.raises(ValueError):
        nan.validate()


def test_state_accessor(traj):
    s = traj.state(3, 1)
    assert s.layer == 3 and s.token == 1 and s.text == traj.tokens[1]
    assert s.vector.shape == (traj.dim,)
    assert np.isfinite(s.entropy) and s.norm > 0
    assert len(s.topk) > 0


def test_extraction_modes(traj, coords):
    L, T = traj.n_layers, traj.n_tokens
    assert len(extract(coords, traj.tokens, "all_tokens")) == T
    single = extract(coords, traj.tokens, "token", token=2)[0]
    assert single.points.shape == (L, 2)
    assert np.array_equal(single.points, coords[:, 2, :])
    mean = extract(coords, traj.tokens, "mean")[0]
    assert np.allclose(mean.points, coords.mean(axis=1))
    cls = extract(coords, traj.tokens, "cls")[0]
    assert cls.token == T - 1
    with pytest.raises(ValueError):
        extract(coords, traj.tokens, "warp-drive")


def test_densify_passes_through_endpoints(coords):
    path = coords[:, 0, :]
    fine = densify(path, steps_per_segment=5)
    assert len(fine) == (len(path) - 1) * 5 + 1
    assert np.allclose(fine[0], path[0], atol=1e-9)
    assert np.allclose(fine[-1], path[-1], atol=1e-9)
    # spline passes through every original layer point
    assert np.allclose(fine[::5], path, atol=1e-6)


def test_animation_continuity(coords):
    """Consecutive animation samples stay close: no jumps, no gaps."""
    path = coords[:, 0, :]
    fine = densify(path, steps_per_segment=6)
    steps = np.linalg.norm(np.diff(fine, axis=0), axis=1)
    assert np.isfinite(fine).all()
    longest_layer_hop = np.linalg.norm(np.diff(path, axis=0), axis=1).max()
    assert steps.max() < longest_layer_hop  # densified motion is finer than layer hops
