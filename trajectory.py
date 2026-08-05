"""Core data model: StateTrajectory and trajectory extraction.

StateTrajectory is the common abstraction every analysis in Mottled
operates on.  A backend (transformer forward pass, synthetic generator, in
the future RNNs / Mamba / biological recordings) produces one; projection,
density, terrain, metrics and the UI are pure functions over it.  Nothing
above this module may reach into transformer internals.

Shapes
------
hidden : (L, T, D)  float32 — L captured layers (layer 0 = initial residual
         stream / embeddings), T tokens, D hidden dimension.
logits : (L, T, V)  optional logit-lens logits per state.
entropy: (L, T)     predictive entropy (nats) per state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HiddenState:
    """A single point of the residual stream (one layer, one token)."""

    layer: int
    token: int
    text: str
    vector: np.ndarray
    logits: np.ndarray | None
    entropy: float
    topk: list[tuple[str, float]]

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))


@dataclass
class Trajectory:
    """An ordered path (layer 0 -> layer N) through some coordinate space."""

    token: int | str            # token index, or a label such as "mean"
    label: str = ""
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Routing:
    """Which experts each token was sent to, in a sparse-MoE model.

    In a dense model every token takes the same path and the interesting
    variation is all in the activations. In a sparse MoE it is not: the
    router makes a *discrete* choice per token per layer, and that choice is
    a legible, low-dimensional trace of how the model is treating this token
    — arguably the most interpretable signal such a model emits, and one that
    needs no dictionary learning to read.

    experts : (M, T, k) int32   — the k expert ids chosen per token
    weights : (M, T, k) float32 — their router weights (rows sum to ~1)
    layers  : (M,) int32        — which block each row came from, because MoE
              layers are usually interleaved with dense ones rather than
              being every layer
    n_experts : the pool each choice was made from
    """

    experts: np.ndarray
    weights: np.ndarray
    layers: np.ndarray
    n_experts: int

    @property
    def n_moe_layers(self) -> int:
        return self.experts.shape[0]

    @property
    def top_k(self) -> int:
        return self.experts.shape[-1]

    def validate(self, n_tokens: int) -> None:
        M, T, k = self.experts.shape
        if self.weights.shape != (M, T, k):
            raise ValueError(f"weights {self.weights.shape} != experts {(M, T, k)}")
        if self.layers.shape != (M,):
            raise ValueError(f"layers {self.layers.shape} != {(M,)}")
        if T != n_tokens:
            raise ValueError(f"routing covers {T} tokens, trajectory has {n_tokens}")
        if self.experts.size and (self.experts.min() < 0
                                  or self.experts.max() >= self.n_experts):
            raise ValueError("expert ids outside the pool")

    def path(self, token: int) -> list[tuple[int, list[int]]]:
        """[(layer, [expert ids])] for one token — its route through the model."""
        t = int(token) % self.experts.shape[1]
        return [(int(l), [int(e) for e in self.experts[m, t]])
                for m, l in enumerate(self.layers)]

    def agreement(self, other: "Routing") -> np.ndarray:
        """(M, T) fraction of experts shared with another routing, per state.

        The natural cross-run comparison for a sparse model: two prompts that
        route the same way are being processed the same way, whatever their
        activations look like.
        """
        if other.experts.shape != self.experts.shape:
            raise ValueError("routings must have the same shape to compare")
        M, T, k = self.experts.shape
        out = np.empty((M, T), dtype=np.float32)
        for m in range(M):
            for t in range(T):
                out[m, t] = len(set(self.experts[m, t]) &
                                set(other.experts[m, t])) / k
        return out


@dataclass
class StateTrajectory:
    """Hidden-state evolution of a full forward pass."""

    hidden: np.ndarray                       # (L, T, D)
    tokens: list[str]                        # length T
    logits: np.ndarray | None = None         # (L, T, V)
    entropy: np.ndarray | None = None        # (L, T)
    topk: list[list[list[tuple[str, float]]]] | None = None  # [L][T][k]
    vocab: list[str] | None = None            # id -> token string
    embedding_matrix: np.ndarray | None = None  # (V, D) token embeddings
    components: dict[str, np.ndarray] | None = None  # residual decomposition,
    # e.g. {"attn": (L-1, T, D), "mlp": (L-1, T, D)} — each block's additive
    # writes to the stream: hidden[l+1] ≈ hidden[l] + attn[l] + mlp[l]
    attention: np.ndarray | None = None      # (L-1, T, T) head-averaged
    # attention: attention[b, t, s] is how much block b's destination token t
    # reads from source token s (causal: zero for s > t, rows sum to 1)
    routing: "Routing | None" = None          # sparse-MoE expert routing
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ shape
    @property
    def n_layers(self) -> int:
        return self.hidden.shape[0]

    @property
    def n_tokens(self) -> int:
        return self.hidden.shape[1]

    @property
    def dim(self) -> int:
        return self.hidden.shape[2]

    def validate(self) -> None:
        """Raise if internal shapes are inconsistent."""
        L, T, _ = self.hidden.shape
        if len(self.tokens) != T:
            raise ValueError(f"tokens ({len(self.tokens)}) != T ({T})")
        if self.entropy is not None and self.entropy.shape != (L, T):
            raise ValueError(f"entropy shape {self.entropy.shape} != {(L, T)}")
        if self.logits is not None and self.logits.shape[:2] != (L, T):
            raise ValueError(f"logits shape {self.logits.shape[:2]} != {(L, T)}")
        if self.components is not None:
            for name, arr in self.components.items():
                if arr.shape != (L - 1, T, self.dim):
                    raise ValueError(
                        f"component {name!r} shape {arr.shape} != {(L - 1, T, self.dim)}")
        if self.attention is not None and self.attention.shape != (L - 1, T, T):
            raise ValueError(
                f"attention shape {self.attention.shape} != {(L - 1, T, T)}")
        if self.routing is not None:
            self.routing.validate(T)
        if not np.isfinite(self.hidden).all():
            raise ValueError("hidden contains non-finite values")

    # ------------------------------------------------------------------ access
    def state(self, layer: int, token: int) -> HiddenState:
        """Materialise one HiddenState record."""
        return HiddenState(
            layer=layer,
            token=token,
            text=self.tokens[token],
            vector=self.hidden[layer, token],
            logits=None if self.logits is None else self.logits[layer, token],
            entropy=float(self.entropy[layer, token]) if self.entropy is not None else float("nan"),
            topk=self.topk[layer][token] if self.topk is not None else [],
        )

    def flat_hidden(self) -> np.ndarray:
        """(L*T, D) view, layer-major, for projection / indexing."""
        return self.hidden.reshape(-1, self.dim)


# --------------------------------------------------------------------------
# Trajectory extraction: connect states layer 0 -> layer N in a coordinate
# space (usually projected R^2/R^3 coords of shape (L, T, C)).
# --------------------------------------------------------------------------

def token_trajectory(coords: np.ndarray, token: int, label: str = "") -> Trajectory:
    """Path of one token position through the layers."""
    token = int(token) % coords.shape[1]
    return Trajectory(token=token, label=label, points=coords[:, token, :].copy())


def mean_trajectory(coords: np.ndarray) -> Trajectory:
    """Mean of all token positions per layer (sequence centroid)."""
    return Trajectory(token="mean", label="mean", points=coords.mean(axis=1))


def cls_trajectory(coords: np.ndarray, tokens: list[str]) -> Trajectory:
    """CLS surrogate: the final token, which aggregates the causal context."""
    t = coords.shape[1] - 1
    return Trajectory(token=t, label=f"cls:{tokens[t]}", points=coords[:, t, :].copy())


def extract(
    coords: np.ndarray,
    tokens: list[str],
    mode: str = "all_tokens",
    token: int = -1,
) -> list[Trajectory]:
    """Build trajectories from (L, T, C) coordinates.

    mode: "all_tokens" | "token" | "mean" | "cls"
    """
    if mode == "all_tokens":
        return [token_trajectory(coords, t, label=tokens[t]) for t in range(coords.shape[1])]
    if mode == "token":
        t = int(token) % coords.shape[1]
        return [token_trajectory(coords, t, label=tokens[t])]
    if mode == "mean":
        return [mean_trajectory(coords)]
    if mode == "cls":
        return [cls_trajectory(coords, tokens)]
    raise ValueError(f"unknown trajectory mode: {mode!r}")


def densify(points: np.ndarray, steps_per_segment: int = 4) -> np.ndarray:
    """Interpolate a polyline for smooth animation.

    Uses a cubic spline through the layer points when there are enough of
    them, otherwise linear interpolation.  The returned path passes through
    every original point; consecutive samples are close together, which is
    what makes the marble animation continuous.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2 or steps_per_segment <= 1:
        return points.copy()
    t = np.arange(n, dtype=np.float64)
    fine = np.linspace(0.0, n - 1.0, (n - 1) * steps_per_segment + 1)
    if n >= 4:
        from scipy.interpolate import CubicSpline

        return CubicSpline(t, points, axis=0)(fine)
    out = np.empty((len(fine), points.shape[1]))
    for d in range(points.shape[1]):
        out[:, d] = np.interp(fine, t, points[:, d])
    return out
