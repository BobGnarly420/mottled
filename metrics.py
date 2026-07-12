"""Metrics over StateTrajectories.

Every metric is a function over a StateTrajectory (and optionally projected
coordinates) — never over transformer internals.  A registry maps metric
names to summary functions so new metrics plug in without touching the UI.
"""

from __future__ import annotations

import numpy as np

from trajectory import StateTrajectory

_EPS = 1e-12


# ------------------------------------------------------------- distributions
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=axis, keepdims=True)


def entropy(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Predictive entropy (nats) of logit rows."""
    p = softmax(logits, axis=axis)
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=axis)


def topk_predictions(logits: np.ndarray, vocab: list[str], k: int = 5) -> list[tuple[str, float]]:
    p = softmax(np.asarray(logits, dtype=np.float64))
    idx = np.argsort(-p)[:k]
    return [(vocab[i], float(p[i])) for i in idx]


def kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray, axis: int = -1) -> np.ndarray:
    """KL(P || Q) between two logit rows (nats)."""
    p = softmax(logits_p, axis=axis)
    q = softmax(logits_q, axis=axis)
    return (p * (np.log(p + _EPS) - np.log(q + _EPS))).sum(axis=axis)


# ---------------------------------------------------------------- kinematics
def velocity(points: np.ndarray) -> np.ndarray:
    """First differences along a path: (N-1, D)."""
    return np.diff(np.asarray(points, dtype=np.float64), axis=0)


def acceleration(points: np.ndarray) -> np.ndarray:
    """Second differences along a path: (N-2, D)."""
    return np.diff(np.asarray(points, dtype=np.float64), n=2, axis=0)


def speed(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocity(points), axis=-1)


def path_length(points: np.ndarray) -> float:
    return float(speed(points).sum())


def turning_angles(points: np.ndarray) -> np.ndarray:
    """Angle (radians) between consecutive segments: (N-2,)."""
    v = velocity(points)
    if len(v) < 2:
        return np.zeros(0)
    a, b = v[:-1], v[1:]
    cos = (a * b).sum(axis=-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), _EPS
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))


def integrated_curvature(points: np.ndarray) -> float:
    """Total turning along the path (radians)."""
    return float(turning_angles(points).sum())


# --------------------------------------------------------- layerwise dynamics
def layer_displacement(hidden: np.ndarray) -> np.ndarray:
    """||h_{l+1} - h_l|| in full hidden space: (L-1, T)."""
    return np.linalg.norm(np.diff(np.asarray(hidden, dtype=np.float64), axis=0), axis=-1)


def semantic_drift(hidden: np.ndarray) -> np.ndarray:
    """Cosine distance between consecutive layers' states: (L-1, T)."""
    h = np.asarray(hidden, dtype=np.float64)
    a, b = h[:-1], h[1:]
    cos = (a * b).sum(axis=-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), _EPS
    )
    return 1.0 - np.clip(cos, -1.0, 1.0)


def entropy_collapse(entropy_series: np.ndarray) -> float:
    """Entropy drop from first to last layer (positive = collapsing)."""
    e = np.asarray(entropy_series, dtype=np.float64)
    return float(e[0] - e[-1])


def neighbor_stability(neighbor_ids_per_layer: list[list[int]]) -> np.ndarray:
    """Jaccard overlap of neighbor sets between consecutive layers: (L-1,)."""
    out = []
    for a, b in zip(neighbor_ids_per_layer[:-1], neighbor_ids_per_layer[1:]):
        sa, sb = set(a), set(b)
        union = len(sa | sb)
        out.append(len(sa & sb) / union if union else 1.0)
    return np.asarray(out)


def layerwise_kl(logits: np.ndarray) -> np.ndarray:
    """KL(layer l -> l+1) of the logit-lens distributions: (L-1, T)."""
    lg = np.asarray(logits, dtype=np.float64)
    return kl_divergence(lg[:-1], lg[1:])


def branch_divergence(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Pointwise distance between two aligned trajectories (Phase-2 preview)."""
    n = min(len(points_a), len(points_b))
    return np.linalg.norm(
        np.asarray(points_a[:n], dtype=np.float64) - np.asarray(points_b[:n], dtype=np.float64),
        axis=-1,
    )


# ------------------------------------------------------------------ summary
def summarize(traj: StateTrajectory, coords: np.ndarray, token: int = -1) -> dict[str, float]:
    """Research metrics for one token's trajectory over the manifold."""
    token = int(token) % traj.n_tokens
    path = coords[:, token, :]
    h = traj.hidden[:, token, :]
    out = {
        "trajectory_length": path_length(path),
        "integrated_curvature": integrated_curvature(path),
        "avg_semantic_drift": float(semantic_drift(h[:, None, :]).mean()),
        "avg_layer_displacement": float(layer_displacement(h[:, None, :]).mean()),
        "final_vector_norm": float(np.linalg.norm(h[-1])),
    }
    if traj.entropy is not None:
        out["entropy_collapse"] = entropy_collapse(traj.entropy[:, token])
        out["final_entropy"] = float(traj.entropy[-1, token])
    if traj.embedding_matrix is not None:
        from neighbors import neighbor_ids_per_layer

        ids = neighbor_ids_per_layer(traj.hidden, traj.embedding_matrix, token, k=5)
        out["nn_stability"] = float(neighbor_stability(ids).mean())
    return out


METRICS = {
    "trajectory_length": lambda traj, coords, token=-1: path_length(coords[:, int(token) % traj.n_tokens, :]),
    "integrated_curvature": lambda traj, coords, token=-1: integrated_curvature(coords[:, int(token) % traj.n_tokens, :]),
    "summary": summarize,
}
