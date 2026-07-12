"""Manifold density estimation over projected coordinates.

The density of projected hidden states is the scalar potential that terrain.py
turns into a landscape.  Two estimators are registered: Gaussian KDE (default)
and kNN inverse-distance.  Both are evaluated on a regular grid spanning the
point cloud (plus padding) and guaranteed to return finite values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DENSITY_ESTIMATORS: dict[str, type] = {}


def register_density(name: str):
    def deco(cls):
        DENSITY_ESTIMATORS[name] = cls
        return cls

    return deco


@dataclass
class Landscape:
    """Density field over the projected manifold."""

    coordinates: np.ndarray            # (N, 2) the points the field was fit on
    grid_x: np.ndarray                 # (W,)
    grid_y: np.ndarray                 # (H,)
    density: np.ndarray                # (H, W) normalised to [0, 1]
    point_density: np.ndarray          # (N,) density at each input point
    neighbors: dict = field(default_factory=dict)  # optional neighbor annotations


@register_density("kde")
class KDEEstimator:
    """Gaussian kernel density (Scott's-rule bandwidth)."""

    def __init__(self, points: np.ndarray):
        from sklearn.neighbors import KernelDensity

        points = np.asarray(points, dtype=np.float64)
        n, d = points.shape
        scale = float(points.std(axis=0).mean()) or 1.0
        bandwidth = scale * n ** (-1.0 / (d + 4)) if n > 1 else 1.0
        self._kde = KernelDensity(bandwidth=max(bandwidth, 1e-6)).fit(points)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return np.exp(self._kde.score_samples(np.asarray(points, dtype=np.float64)))


@register_density("knn")
class KNNEstimator:
    """kNN inverse mean-distance density."""

    def __init__(self, points: np.ndarray, k: int = 8):
        from sklearn.neighbors import NearestNeighbors

        self._points = np.asarray(points, dtype=np.float64)
        self._k = int(min(k, len(self._points)))
        self._nn = NearestNeighbors(n_neighbors=self._k).fit(self._points)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        dist, _ = self._nn.kneighbors(np.asarray(points, dtype=np.float64))
        return 1.0 / (dist.mean(axis=1) + 1e-6)


def get_estimator(name: str, points: np.ndarray):
    try:
        cls = DENSITY_ESTIMATORS[name]
    except KeyError:
        raise ValueError(f"unknown density estimator {name!r}; available: {sorted(DENSITY_ESTIMATORS)}") from None
    return cls(points)


def compute_density(
    coords: np.ndarray,
    method: str = "kde",
    grid_size: int = 64,
    padding: float = 0.2,
) -> Landscape:
    """Estimate the density field of projected states on a regular grid.

    coords: (N, 2) or (L, T, 2) — higher-rank inputs are flattened.
    """
    pts = np.asarray(coords, dtype=np.float64).reshape(-1, coords.shape[-1])[:, :2]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo, hi = lo - padding * span, hi + padding * span

    grid_x = np.linspace(lo[0], hi[0], grid_size)
    grid_y = np.linspace(lo[1], hi[1], grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    est = get_estimator(method, pts)
    dens = np.nan_to_num(est.evaluate(grid_pts), nan=0.0, posinf=0.0, neginf=0.0)
    point_dens = np.nan_to_num(est.evaluate(pts), nan=0.0, posinf=0.0, neginf=0.0)

    peak = dens.max()
    if peak > 0:
        dens = dens / peak
        point_dens = point_dens / peak
    return Landscape(
        coordinates=pts.astype(np.float32),
        grid_x=grid_x.astype(np.float32),
        grid_y=grid_y.astype(np.float32),
        density=dens.reshape(grid_size, grid_size).astype(np.float32),
        point_density=point_dens.astype(np.float32),
    )
