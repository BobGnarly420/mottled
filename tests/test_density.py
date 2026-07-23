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


def test_no_bootstrap_leaves_se_none(coords):
    land = compute_density(coords, grid_size=16)
    assert land.density_se is None


@pytest.mark.parametrize("method", sorted(DENSITY_ESTIMATORS))
def test_bootstrap_produces_finite_se(coords, method):
    land = compute_density(coords, method=method, grid_size=16, bootstrap=12, seed=1)
    assert land.density_se is not None
    assert land.density_se.shape == land.density.shape
    assert np.isfinite(land.density_se).all()
    assert (land.density_se >= 0).all()


def test_bootstrap_is_seed_deterministic(coords):
    a = compute_density(coords, grid_size=16, bootstrap=10, seed=42)
    b = compute_density(coords, grid_size=16, bootstrap=10, seed=42)
    assert np.array_equal(a.density_se, b.density_se)


def test_se_reflects_resampling_variability():
    # A single small sample is a noisy density estimate: the bootstrap SE must
    # be a non-trivial field (not identically zero) concentrated where the
    # estimate actually has mass — the far empty corners barely move.
    rng = np.random.default_rng(4)
    pts = rng.normal(0, 1.0, size=(40, 2))
    land = compute_density(pts, method="kde", grid_size=48, bootstrap=48, seed=0)
    se = land.density_se
    assert se.max() > 0.0                      # resampling moved the estimate
    # SE where the cloud has density vs. the low-density tail of the grid
    has_mass = land.density > 0.3
    tail = land.density < 0.02
    if has_mass.any() and tail.any():
        assert se[has_mass].mean() > se[tail].mean()


def test_more_points_reduce_se():
    # The estimate stabilises as the sample grows: a big sample's mean SE is
    # below a small one's (both drawn from the same generator).
    rng = np.random.default_rng(7)
    small = compute_density(rng.normal(size=(30, 2)), method="kde",
                            grid_size=32, bootstrap=32, seed=0)
    big = compute_density(rng.normal(size=(400, 2)), method="kde",
                          grid_size=32, bootstrap=32, seed=0)
    assert big.density_se.mean() < small.density_se.mean()
