"""Density -> height map -> renderable terrain mesh.

The landscape's density field becomes a smoothed height map; trajectories are
draped over it by interpolating the surface height at their (x, y) positions,
so the marble visibly rolls over the semantic terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from density import Landscape


@dataclass
class TerrainMesh:
    x: np.ndarray  # (W,)
    y: np.ndarray  # (H,)
    z: np.ndarray  # (H, W) heights, z[i, j] at (x[j], y[i])

    def interpolator(self) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.y.astype(np.float64), self.x.astype(np.float64)),
            self.z.astype(np.float64),
            bounds_error=False,
            fill_value=float(self.z.min()),
        )


def height_map(
    density: np.ndarray,
    smooth_sigma: float = 1.5,
    scale: float = 1.0,
    invert: bool = False,
) -> np.ndarray:
    """Smooth the density field and scale it into heights.

    invert=True turns dense regions into valleys (the marble settles into
    semantic attractors); by default dense regions are peaks.
    """
    z = gaussian_filter(np.asarray(density, dtype=np.float64), sigma=smooth_sigma)
    if invert:
        z = z.max() - z
    z -= z.min()
    peak = z.max()
    if peak > 0:
        z /= peak
    return (z * scale).astype(np.float32)


def mesh(
    landscape: Landscape,
    smooth_sigma: float = 1.5,
    height_scale: float = 1.0,
    invert: bool = False,
) -> TerrainMesh:
    """Build the terrain mesh for a landscape."""
    z = height_map(landscape.density, smooth_sigma=smooth_sigma, scale=height_scale, invert=invert)
    return TerrainMesh(x=landscape.grid_x.copy(), y=landscape.grid_y.copy(), z=z)


def surface_height(terrain: TerrainMesh, points_xy: np.ndarray) -> np.ndarray:
    """Terrain height at arbitrary (x, y) positions (clipped to the grid)."""
    pts = np.atleast_2d(np.asarray(points_xy, dtype=np.float64))[:, :2]
    interp = terrain.interpolator()
    return interp(np.column_stack([pts[:, 1], pts[:, 0]])).astype(np.float32)


def drape(terrain: TerrainMesh, points_xy: np.ndarray, lift: float = 0.04) -> np.ndarray:
    """Lift 2-D trajectory points onto the terrain: (N, 2) -> (N, 3)."""
    pts = np.atleast_2d(np.asarray(points_xy, dtype=np.float32))[:, :2]
    z = surface_height(terrain, pts) + lift
    return np.column_stack([pts, z.astype(np.float32)])
