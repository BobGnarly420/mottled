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


def _to_numpy(x) -> np.ndarray:
    """Array from a numpy array or a torch tensor, without importing torch."""
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _orient(w, rows: int, cols: int, name: str) -> np.ndarray:
    """Return `w` as (rows, cols), transposing if it arrived the other way.

    Trainers disagree on weight orientation; we pin ours to Mottled's
    convention (w_enc is (D, F), w_dec is (F, D)) and accept either layout.
    """
    w = _to_numpy(w)
    if w.shape == (rows, cols):
        return w
    if w.shape == (cols, rows):
        return np.ascontiguousarray(w.T)
    raise ValueError(f"{name} shape {w.shape} is neither {(rows, cols)} nor its transpose")


def from_state_dict(state_dict, labels: list[str] | None = None) -> SAE:
    """Build an SAE from a pretrained weight mapping (the real-weights path).

    Accepts the standard four-array convention shared by SAELens and most
    dictionary-learning runs — keys `W_enc`/`b_enc`/`W_dec`/`b_dec` (case
    tolerant) — as numpy arrays or torch tensors. `b_dec` fixes the hidden
    dimension D, `b_enc` the feature count F; encoder/decoder weights are
    oriented to Mottled's (D, F) / (F, D) layout automatically. This applies
    a *trained* dictionary — the whole point of the feature overlay: real
    features instead of `demo_sae`'s random directions.
    """
    def _pick(*names):
        for n in names:
            for key in (n, n.lower(), n.upper()):
                if key in state_dict:
                    return state_dict[key]
        raise KeyError(f"state dict is missing any of {names}; keys: {list(state_dict)[:8]}")

    b_dec = _to_numpy(_pick("b_dec"))
    b_enc = _to_numpy(_pick("b_enc"))
    D, F = int(b_dec.shape[0]), int(b_enc.shape[0])
    w_enc = _orient(_pick("W_enc", "w_enc"), D, F, "W_enc")
    w_dec = _orient(_pick("W_dec", "w_dec"), F, D, "W_dec")
    sae = SAE(w_enc=w_enc, b_enc=b_enc, w_dec=w_dec, b_dec=b_dec, labels=labels)
    sae.validate()
    return sae


def from_sae_lens(sae, labels: list[str] | None = None) -> SAE:
    """Convert a loaded SAELens SAE object into a Mottled SAE.

    SAELens' *standard* SAE stores `W_enc` (d_in, d_sae), `b_enc`, `W_dec`
    (d_sae, d_in), `b_dec` and runs the same ReLU forward Mottled applies, so
    conversion is a direct array copy. Two config options change that forward,
    and we handle each explicitly rather than silently mis-encoding a nominally
    "standard" SAE:

    * ``apply_b_dec_to_input=False`` — the source encodes ``h @ W_enc + b_enc``
      without the ``b_dec`` subtraction Mottled always does. That constant folds
      exactly into ``b_enc`` (``b_enc += b_dec @ W_enc``), so the converted
      encoder stays numerically identical.
    * ``normalize_activations`` — a runtime input rescaling Mottled does not do
      and cannot fold into a fixed dictionary, so it is rejected loudly.

    Non-standard architectures (gated / jumprelu / top-k) use a different
    nonlinearity than our plain ReLU and are rejected — detected from
    ``cfg.architecture`` / ``activation_fn`` and, so the gate cannot fail open
    when a config is missing, from the tell-tale weights they carry.
    """
    cfg = getattr(sae, "cfg", None)

    def _cfg(name):
        v = getattr(cfg, name, None) if cfg is not None else None
        return v() if callable(v) else v

    arch = _cfg("architecture")
    if arch is not None and str(arch).lower() not in {"standard", "relu"}:
        raise ValueError(
            f"unsupported SAE architecture {arch!r}: Mottled applies a plain ReLU "
            "(standard) SAE. Gated / JumpReLU / top-k SAEs encode differently and "
            "must be converted with their own tooling first.")
    act = _cfg("activation_fn") or _cfg("activation_fn_str")
    if act is not None and str(act).lower() not in {"relu", "", "none"}:
        raise ValueError(
            f"unsupported SAE activation_fn {act!r}: Mottled applies a plain ReLU. "
            "Top-k / other activations must be converted with their own tooling.")
    # Fail closed even if the config is absent: these weights exist only on
    # gated SAEs, whose forward is not our ReLU.
    for attr, kind in (("W_gate", "gated"), ("r_mag", "gated")):
        if getattr(sae, attr, None) is not None:
            raise ValueError(
                f"this looks like a {kind} SAE (carries {attr!r}); Mottled applies "
                "a plain ReLU. Convert it with its own tooling first.")
    norm = _cfg("normalize_activations")
    if norm not in (None, False, "none", "None"):
        raise ValueError(
            f"this SAE normalizes activations (normalize_activations={norm!r}), an "
            "input rescaling Mottled does not apply; its features would not match. "
            "Export it without activation normalization first.")

    out = from_state_dict({"W_enc": sae.W_enc, "b_enc": sae.b_enc,
                           "W_dec": sae.W_dec, "b_dec": sae.b_dec}, labels=labels)
    if _cfg("apply_b_dec_to_input") is False:
        # Source skipped the b_dec subtraction on the encoder input; Mottled's
        # forward always subtracts it, so fold the constant back into b_enc:
        # (h - b_dec) @ W_enc + (b_enc + b_dec @ W_enc) == h @ W_enc + b_enc.
        out.b_enc = (out.b_enc + out.b_dec @ out.w_enc).astype(np.float32)
        out.validate()
    return out


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


# ------------------------------------------------------------ feature field
@dataclass
class FeatureField:
    """The SAE evaluated over the projection plane itself.

    The domain-coloring analogue: where a complex-plane plot evaluates f(z)
    at every point of the plane and shows arg(f) as hue and |f| as
    brightness, this evaluates the SAE at every grid point of the projected
    manifold — `dominant` (which feature decodes strongest) plays the role
    of the phase, `magnitude` (its activation) the modulus.
    """

    grid_x: np.ndarray      # (W,)
    grid_y: np.ndarray      # (H,)
    magnitude: np.ndarray   # (H, W) strongest activation at each plane point
    dominant: np.ndarray    # (H, W) id of the strongest feature; -1 where none fires
    n_active: np.ndarray    # (H, W) how many features fire at each point

    @property
    def features(self) -> np.ndarray:
        """Sorted ids of every feature that dominates somewhere."""
        ids = np.unique(self.dominant)
        return ids[ids >= 0]


def feature_field(sae: SAE, projector, grid_x: np.ndarray, grid_y: np.ndarray) -> FeatureField:
    """Evaluate `sae` over a grid on the projection plane: the domain coloring.

    Each (x, y) grid point is inverse-projected back to hidden space and
    encoded; the field records, per point, the strongest feature and its
    activation.  Requires an invertible projector (PCA is exact — grid
    points map onto the fitted affine 2-plane in hidden space, so the field
    shows what the SAE dictionary sees *along the plane you are looking
    at*; UMAP's inverse is approximate).
    """
    if not hasattr(projector, "inverse_transform"):
        raise ValueError(
            f"projection {type(projector).__name__} has no inverse_transform; "
            "the feature field needs plane -> hidden-space inversion (use pca)")
    gx, gy = np.meshgrid(np.asarray(grid_x), np.asarray(grid_y))
    plane = np.column_stack([gx.ravel(), gy.ravel()])
    hidden = np.asarray(projector.inverse_transform(plane))
    if hidden.shape[-1] != sae.dim:
        raise ValueError(f"projector inverts to dim {hidden.shape[-1]}, SAE dim is {sae.dim}")

    acts = sae.encode(hidden)                      # (H*W, F)
    magnitude = acts.max(axis=-1)
    dominant = acts.argmax(axis=-1).astype(np.int32)
    dominant[magnitude <= 0] = -1
    shape = gx.shape
    return FeatureField(
        grid_x=np.asarray(grid_x, dtype=np.float32),
        grid_y=np.asarray(grid_y, dtype=np.float32),
        magnitude=magnitude.reshape(shape).astype(np.float32),
        dominant=dominant.reshape(shape),
        n_active=(acts > 0).sum(axis=-1).reshape(shape).astype(np.int32),
    )
