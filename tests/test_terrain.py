"""Terrain mesh manifold (spec test 6): consistent, finite, smooth surface."""

import numpy as np
import pytest

from density import compute_density
from terrain import drape, height_map, mesh, surface_height


@pytest.fixture(scope="module")
def landscape():
    rng = np.random.default_rng(2)
    return compute_density(rng.normal(size=(60, 2)), grid_size=32)


def test_mesh_shapes(landscape):
    m = mesh(landscape)
    assert m.z.shape == (len(m.y), len(m.x)) == (32, 32)
    assert np.isfinite(m.z).all()
    assert m.z.min() >= 0.0 and m.z.max() <= 1.0 + 1e-6


def test_smoothing_reduces_roughness(landscape):
    rough = height_map(landscape.density, smooth_sigma=0.0)
    smooth = height_map(landscape.density, smooth_sigma=3.0)

    def roughness(z):  # second-difference energy: scale-robust, unlike raw TV
        return np.abs(np.diff(z, n=2, axis=0)).sum() + np.abs(np.diff(z, n=2, axis=1)).sum()

    assert roughness(smooth) < roughness(rough)


def test_invert_flips_relief(landscape):
    z = height_map(landscape.density, invert=False)
    zi = height_map(landscape.density, invert=True)
    # densest cell is the global peak in one orientation, a floor in the other
    peak = np.unravel_index(np.argmax(z), z.shape)
    assert zi[peak] == pytest.approx(zi.min(), abs=1e-6)


def test_surface_height_interpolation(landscape):
    m = mesh(landscape)
    # exactly on grid nodes the interpolant reproduces the mesh
    pts = np.array([[m.x[3], m.y[5]], [m.x[10], m.y[20]]])
    h = surface_height(m, pts)
    assert h[0] == pytest.approx(m.z[5, 3], abs=1e-6)
    assert h[1] == pytest.approx(m.z[20, 10], abs=1e-6)
    # off-grid and out-of-bounds queries stay finite and within range
    wild = np.array([[0.0, 0.0], [1e6, -1e6]])
    hw = surface_height(m, wild)
    assert np.isfinite(hw).all()
    assert (hw >= m.z.min() - 1e-6).all() and (hw <= m.z.max() + 1e-6).all()


def test_drape_lifts_points(landscape):
    m = mesh(landscape)
    pts = landscape.coordinates[:10]
    xyz = drape(m, pts, lift=0.05)
    assert xyz.shape == (10, 3)
    assert np.allclose(xyz[:, :2], pts, atol=1e-6)
    assert (xyz[:, 2] >= m.z.min() + 0.05 - 1e-6).all()
