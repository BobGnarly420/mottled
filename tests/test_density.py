"""Density finite (spec test 5)."""

import numpy as np
import pytest

from density import DENSITY_ESTIMATORS, compute_density


@pytest.fixture(scope="module")
def coords():
    rng = np.random.default_rng(11)
    return rng.normal(size=(80, 2)).astype(np.float32)


@pytest.mark.parametrize("method", sorted(DENSITY_ESTIMATORS))
def test_density_finite_and_normalised(coords, method):
    land = compute_density(coords, method=method, grid_size=32)
    assert land.density.shape == (32, 32)
    assert np.isfinite(land.density).all()
    assert np.isfinite(land.point_density).all()
    assert land.density.max() == pytest.approx(1.0)
    assert land.density.min() >= 0.0


@pytest.mark.parametrize("method", sorted(DENSITY_ESTIMATORS))
def test_degenerate_inputs_stay_finite(method):
    # collinear points (singular covariance) and duplicated points
    line = np.column_stack([np.linspace(0, 1, 20), np.zeros(20)])
    dupes = np.zeros((10, 2))
    for pts in (line, dupes):
        land = compute_density(pts, method=method, grid_size=16)
        assert np.isfinite(land.density).all()


def test_grid_covers_points(coords):
    land = compute_density(coords, grid_size=24, padding=0.2)
    assert land.grid_x.min() < coords[:, 0].min() and land.grid_x.max() > coords[:, 0].max()
    assert land.grid_y.min() < coords[:, 1].min() and land.grid_y.max() > coords[:, 1].max()


def test_density_peaks_where_points_cluster():
    rng = np.random.default_rng(5)
    cluster = rng.normal(0, 0.05, size=(60, 2))
    outlier = np.array([[4.0, 4.0]])
    land = compute_density(np.vstack([cluster, outlier]), method="kde", grid_size=48)
    assert land.point_density[:60].mean() > land.point_density[-1] * 5


def test_accepts_layer_token_rank(coords):
    land = compute_density(coords.reshape(8, 10, 2), grid_size=16)
    assert land.coordinates.shape == (80, 2)
