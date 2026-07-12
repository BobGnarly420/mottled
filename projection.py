"""Projection of hidden vectors to low-dimensional coordinates.

Plugin registry: any object with fit_transform / transform can be registered
as a projection.  PCA is the deterministic default; UMAP is available when
umap-learn is installed.  `transform` enables incremental projection of new
states into an already-fitted space.
"""

from __future__ import annotations

import numpy as np

PROJECTIONS: dict[str, type] = {}


def register_projection(name: str):
    def deco(cls):
        PROJECTIONS[name] = cls
        return cls

    return deco


def get_projection(name: str, n_components: int = 2, seed: int = 0):
    try:
        cls = PROJECTIONS[name]
    except KeyError:
        raise ValueError(f"unknown projection {name!r}; available: {sorted(PROJECTIONS)}") from None
    return cls(n_components=n_components, seed=seed)


@register_projection("pca")
class PCAProjection:
    """Deterministic linear projection (exact full SVD)."""

    def __init__(self, n_components: int = 2, seed: int = 0):
        from sklearn.decomposition import PCA

        self._pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
        self.fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        out = self._pca.fit_transform(np.asarray(X, dtype=np.float64))
        self.fitted = True
        return out.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._pca.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)


@register_projection("umap")
class UMAPProjection:
    """Nonlinear manifold projection; deterministic for a fixed seed."""

    def __init__(self, n_components: int = 2, seed: int = 0):
        import umap  # optional dependency

        self._umap = umap.UMAP(n_components=n_components, random_state=seed, n_jobs=1)
        self.fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # UMAP needs more samples than neighbors; shrink for tiny inputs.
        self._umap.n_neighbors = int(min(self._umap.n_neighbors, max(2, len(X) - 1)))
        out = self._umap.fit_transform(X)
        self.fitted = True
        return np.asarray(out, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._umap.transform(np.asarray(X, dtype=np.float32)), dtype=np.float32)


def project(
    hidden: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    seed: int = 0,
):
    """Project (L, T, D) hidden states to (L, T, n_components) coordinates.

    All states across layers and tokens are embedded in one shared space so
    that distances between layers are meaningful.  Returns (coords, projector);
    the fitted projector supports incremental `transform` for new states.
    """
    hidden = np.asarray(hidden)
    L, T, D = hidden.shape
    proj = get_projection(method, n_components=n_components, seed=seed)
    coords = proj.fit_transform(hidden.reshape(L * T, D))
    return coords.reshape(L, T, n_components), proj
