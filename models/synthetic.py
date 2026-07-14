"""Synthetic backend: dependency-free StateTrajectory generator.

Produces deterministic, smooth hidden-state trajectories with a plausible
structure (momentum random walk, per-token attractors, entropy collapse over
depth) so the entire pipeline — projection, density, terrain, metrics, UI —
can run without torch, transformers, or a model download.  It doubles as the
proof of the architectural principle: everything above `capture` only ever
sees a StateTrajectory.
"""

from __future__ import annotations

import zlib

import numpy as np

from trajectory import StateTrajectory

VOCAB = [
    "the", "of", "and", "a", "to", "in", "is", "was", "it", "for", "on",
    "are", "as", "with", "his", "they", "at", "be", "this", "from", "have",
    "or", "one", "had", "by", "word", "but", "not", "what", "all", "were",
    "when", "your", "can", "said", "there", "use", "an", "each", "which",
    "she", "do", "how", "their", "if", "will", "way", "about", "many",
    "then", "them", "would", "like", "so", "these", "her", "long", "make",
    "thing", "see", "him", "two", "has", "look", "more", "day", "could",
    "go", "come", "did", "number", "sound", "no", "most", "people", "my",
    "over", "know", "water", "than", "call", "first", "who", "may", "down",
    "side", "been", "now", "find", "capital", "france", "paris", "city",
    "country", "europe", "london", "berlin", "rome", "madrid", "river",
    "king", "language", "world", "history", "north", "south", "east", "west",
]

DIM = 64
N_LAYERS = 13  # layer 0 (embeddings) + 12 blocks


def _seed_for(prompt: str) -> int:
    return zlib.crc32(prompt.encode("utf-8"))


def embedding_matrix(dim: int = DIM) -> np.ndarray:
    """Fixed (V, D) token embedding table shared across prompts."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(VOCAB), dim)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def _tokenize(prompt: str) -> list[str]:
    tokens = prompt.strip().split()
    return tokens if tokens else ["<empty>"]


def capture(
    prompt: str,
    n_layers: int = N_LAYERS,
    dim: int = DIM,
    top_k: int = 5,
    keep_logits: bool = True,
    capture_components: bool = False,
) -> StateTrajectory:
    """Generate a deterministic synthetic StateTrajectory for `prompt`."""
    tokens = _tokenize(prompt)
    T, L = len(tokens), n_layers
    rng = np.random.default_rng(_seed_for(prompt))
    emb = embedding_matrix(dim)
    vocab_index = {w: i for i, w in enumerate(VOCAB)}

    # Layer 0: token embedding (known words) or a random vector (unknown).
    hidden = np.zeros((L, T, dim), dtype=np.float32)
    for t, tok in enumerate(tokens):
        idx = vocab_index.get(tok.lower().strip(".,!?"))
        base = emb[idx] if idx is not None else rng.normal(size=dim).astype(np.float32)
        hidden[0, t] = base + 0.05 * rng.normal(size=dim)

    # Momentum random walk drifting toward a per-token attractor: smooth
    # trajectories whose late layers settle near a semantic target.
    attractors = rng.normal(size=(T, dim)).astype(np.float32)
    attractors /= np.linalg.norm(attractors, axis=1, keepdims=True)
    velocity = np.zeros((T, dim), dtype=np.float32)
    for layer in range(1, L):
        pull = attractors - hidden[layer - 1]
        noise = rng.normal(size=(T, dim)).astype(np.float32)
        velocity = 0.7 * velocity + 0.15 * pull + 0.08 * noise
        hidden[layer] = hidden[layer - 1] + velocity

    # Residual decomposition analog: split each layer's update into an
    # "attention" and an "MLP" write that sum to it exactly, mirroring a
    # pre-norm transformer.  The pull term plays attention (context-driven),
    # the momentum+noise remainder plays the MLP.
    components = None
    if capture_components:
        updates = np.diff(hidden, axis=0)                       # (L-1, T, D)
        share = rng.uniform(0.3, 0.7, size=(L - 1, T, 1)).astype(np.float32)
        attn = share * updates
        components = {"attn": attn, "mlp": updates - attn}

    # Logit lens: similarity to token embeddings, sharpened with depth so
    # entropy collapses like a real network committing to a prediction.
    sharpness = 1.0 + 4.0 * np.arange(L, dtype=np.float32) / max(L - 1, 1)
    logits = np.einsum("ltd,vd->ltv", hidden, emb) * sharpness[:, None, None]
    logits = logits.astype(np.float32)

    ent, topk = _entropy_and_topk(logits, top_k)
    return StateTrajectory(
        hidden=hidden,
        tokens=tokens,
        logits=logits.astype(np.float16) if keep_logits else None,
        entropy=ent,
        topk=topk,
        vocab=list(VOCAB),
        embedding_matrix=emb,
        components=components,
        meta={"backend": "synthetic", "prompt": prompt, "model": "synthetic"},
    )


def _entropy_and_topk(logits: np.ndarray, k: int):
    """Shared softmax/entropy/top-k computation (also used by tests)."""
    x = logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    ent = (-(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)).astype(np.float32)

    L, T, _ = logits.shape
    order = np.argsort(-p, axis=-1)[..., :k]
    topk = [
        [
            [(VOCAB[j], float(p[layer, t, j])) for j in order[layer, t]]
            for t in range(T)
        ]
        for layer in range(L)
    ]
    return ent, topk
