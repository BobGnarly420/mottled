"""Projection deterministic (spec test 3) + incremental support."""

import numpy as np
import pytest

from projection import (
    PROJECTIONS,
    get_projection,
    neighborhood_preservation,
    project,
    projection_quality,
)


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


def test_neighborhood_preservation_perfect_when_structure_kept():
    # A projection that keeps the first two (already dominant) coordinates
    # preserves neighborhoods exactly.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    X = np.column_stack([X, 0.001 * rng.normal(size=(40, 3))])  # near-2D cloud
    Y = X[:, :2]
    pres = neighborhood_preservation(X, Y, k=5)
    assert pres.shape == (40,)
    assert (pres >= 0).all() and (pres <= 1).all()
    assert pres.mean() > 0.9


def test_projection_quality_reports_pca_residual_and_variance(hidden):
    coords, proj = project(hidden, method="pca")
    q = projection_quality(hidden, coords, proj, k=8)
    assert q.preservation.shape == hidden.shape[:2]
    assert (q.preservation >= 0).all() and (q.preservation <= 1).all()
    # PCA exposes both residual and explained variance
    assert q.residual is not None and q.residual.shape == hidden.shape[:2]
    assert (q.residual >= 0).all()
    assert q.explained_variance is not None and 0.0 <= q.explained_variance <= 1.0


def test_projection_quality_residual_matches_variance_direction():
    # A cloud that lives almost entirely in a 2-plane: low residual, high var.
    rng = np.random.default_rng(9)
    flat = np.column_stack([rng.normal(size=(30, 2)) * 10,
                            rng.normal(size=(30, 6)) * 0.01])
    hidden = flat.reshape(5, 6, 8)
    coords, proj = project(hidden, method="pca")
    q = projection_quality(hidden, coords, proj)
    assert q.explained_variance > 0.98
    assert q.residual.mean() < 0.1


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
