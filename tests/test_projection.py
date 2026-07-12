"""Projection deterministic (spec test 3) + incremental support."""

import numpy as np
import pytest

from projection import PROJECTIONS, get_projection, project


@pytest.fixture(scope="module")
def hidden():
    rng = np.random.default_rng(7)
    return rng.normal(size=(13, 6, 64)).astype(np.float32)


def test_pca_shape_and_determinism(hidden):
    coords1, _ = project(hidden, method="pca")
    coords2, _ = project(hidden, method="pca")
    assert coords1.shape == (13, 6, 2)
    assert np.array_equal(coords1, coords2)
    assert np.isfinite(coords1).all()


def test_pca_incremental_transform(hidden):
    coords, proj = project(hidden, method="pca")
    new_states = hidden[0] + 0.01  # "new" states projected into fitted space
    out = proj.transform(new_states)
    assert out.shape == (6, 2)
    # transform of the training data reproduces the fitted coordinates
    flat = hidden.reshape(-1, 64)
    assert np.allclose(proj.transform(flat), coords.reshape(-1, 2), atol=1e-4)


@pytest.mark.skipif("umap" not in PROJECTIONS, reason="umap not registered")
def test_umap_shape(hidden):
    pytest.importorskip("umap")
    coords, _ = project(hidden[:4, :3], method="umap")  # small input: fast
    assert coords.shape == (4, 3, 2)
    assert np.isfinite(coords).all()


def test_unknown_projection_rejected(hidden):
    with pytest.raises(ValueError, match="unknown projection"):
        project(hidden, method="tsne-magic")


def test_registry_plugin():
    class Identity2D:
        def __init__(self, n_components=2, seed=0):
            pass

        def fit_transform(self, X):
            return np.asarray(X)[:, :2]

    PROJECTIONS["identity"] = Identity2D
    try:
        p = get_projection("identity")
        X = np.arange(12.0).reshape(3, 4)
        assert np.array_equal(p.fit_transform(X), X[:, :2])
    finally:
        del PROJECTIONS["identity"]
