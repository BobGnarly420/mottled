"""Phase 3 — sparse-autoencoder features over the residual stream.

An SAE re-expresses a hidden state as a sparse combination of learned
dictionary directions ("features"):

    f = ReLU((h - b_dec) @ w_enc + b_enc)        # (…, F) activations
    h ≈ f @ w_dec + b_dec                        # reconstruction

This module *applies* SAEs — it never trains them (a non-goal).  Weights
come from `load_npz` (export any pretrained SAE — SAELens, dictionary-
learning runs — to the four arrays below), or from `demo_sae`, an untrained
random dictionary that exercises the whole feature pipeline without a
download.  Demo activations are sparse projections, NOT interpretable
features; they exist so overlays, tests and UI development work offline.

Everything is a pure numpy function over StateTrajectories, consistent with
the architecture: no torch, no transformer internals, any backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trajectory import StateTrajectory


@dataclass
class SAE:
    """A sparse autoencoder as plain arrays.

    w_enc : (D, F)   encoder weights
    b_enc : (F,)     encoder bias
    w_dec : (F, D)   decoder dictionary (rows are feature directions)
    b_dec : (D,)     decoder bias (subtracted before encoding)
    labels: optional human-readable name per feature
    """

    w_enc: np.ndarray
    b_enc: np.ndarray
    w_dec: np.ndarray
    b_dec: np.ndarray
    labels: list[str] | None = None

    @property
    def dim(self) -> int:
        return self.w_enc.shape[0]

    @property
    def n_features(self) -> int:
        return self.w_enc.shape[1]

    def validate(self) -> None:
        D, F = self.w_enc.shape
        if self.b_enc.shape != (F,):
            raise ValueError(f"b_enc shape {self.b_enc.shape} != {(F,)}")
        if self.w_dec.shape != (F, D):
            raise ValueError(f"w_dec shape {self.w_dec.shape} != {(F, D)}")
        if self.b_dec.shape != (D,):
            raise ValueError(f"b_dec shape {self.b_dec.shape} != {(D,)}")
        if self.labels is not None and len(self.labels) != F:
            raise ValueError(f"labels ({len(self.labels)}) != F ({F})")

    # ------------------------------------------------------------- transforms
    def encode(self, hidden: np.ndarray) -> np.ndarray:
        """(…, D) states -> (…, F) non-negative sparse activations."""
        h = np.asarray(hidden, dtype=np.float32)
        pre = (h - self.b_dec) @ self.w_enc + self.b_enc
        return np.maximum(pre, 0.0)

    def decode(self, acts: np.ndarray) -> np.ndarray:
        """(…, F) activations -> (…, D) reconstructed states."""
        return np.asarray(acts, dtype=np.float32) @ self.w_dec + self.b_dec

    def reconstruct(self, hidden: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(hidden))

    def reconstruction_error(self, hidden: np.ndarray) -> np.ndarray:
        """Relative L2 reconstruction error per state: (…,)."""
        h = np.asarray(hidden, dtype=np.float32)
        err = np.linalg.norm(h - self.reconstruct(h), axis=-1)
        return err / np.maximum(np.linalg.norm(h, axis=-1), 1e-12)

    def feature_label(self, i: int) -> str:
        if self.labels is not None:
            return self.labels[i]
        return f"f{i}"


# --------------------------------------------------------------------- IO
def save_npz(sae: SAE, path: str | Path) -> None:
    """Persist an SAE as a portable .npz (the interchange format)."""
    sae.validate()
    arrays = {"w_enc": sae.w_enc, "b_enc": sae.b_enc,
              "w_dec": sae.w_dec, "b_dec": sae.b_dec}
    if sae.labels is not None:
        arrays["labels"] = np.asarray(sae.labels)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> SAE:
    """Load an SAE saved by `save_npz` (or exported from any trainer)."""
    with np.load(path, allow_pickle=False) as z:
        sae = SAE(
            w_enc=z["w_enc"].astype(np.float32),
            b_enc=z["b_enc"].astype(np.float32),
            w_dec=z["w_dec"].astype(np.float32),
            b_dec=z["b_dec"].astype(np.float32),
            labels=[str(s) for s in z["labels"]] if "labels" in z else None,
        )
    sae.validate()
    return sae


def demo_sae(dim: int, n_features: int = 256, seed: int = 0, bias: float = 0.05) -> SAE:
    """Untrained random dictionary: tied weights, unit decoder rows.

    A negative encoder bias keeps activations sparse.  Deterministic for a
    (dim, n_features, seed) triple.  For development and tests only — the
    "features" are arbitrary directions, not learned structure.
    """
    rng = np.random.default_rng(seed)
    w_dec = rng.normal(size=(n_features, dim)).astype(np.float32)
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    return SAE(
        w_enc=w_dec.T.copy(),
        b_enc=np.full(n_features, -float(bias), dtype=np.float32),
        w_dec=w_dec,
        b_dec=np.zeros(dim, dtype=np.float32),
    )


# ---------------------------------------------------------------- features
def feature_trajectory(traj: StateTrajectory, sae: SAE) -> np.ndarray:
    """Feature activations for every captured state: (L, T, F).

    The SAE dictionary must match the trajectory's hidden dimension —
    features are directions in that state space.
    """
    if sae.dim != traj.dim:
        raise ValueError(f"SAE dim {sae.dim} != trajectory dim {traj.dim}")
    return sae.encode(traj.hidden)


def top_features(acts: np.ndarray, layer: int, token: int, k: int = 5) -> list[tuple[int, float]]:
    """Strongest-activating features at one state: [(feature_id, activation)].

    Only features that actually fire (activation > 0) are returned, so the
    list may be shorter than k.
    """
    a = np.asarray(acts)[layer, token]
    order = np.argsort(-a)[:k]
    return [(int(i), float(a[i])) for i in order if a[i] > 0]


def active_features(acts: np.ndarray, k: int = 20) -> np.ndarray:
    """Feature ids ranked by peak activation anywhere in the trajectory."""
    peak = np.asarray(acts).max(axis=(0, 1))
    order = np.argsort(-peak)[:k]
    return order[peak[order] > 0]
