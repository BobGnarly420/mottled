"""Nearest-neighbor search over hidden vectors and token embeddings.

Cosine similarity throughout (vectors are L2-normalised, search is inner
product).  FAISS is used when available; a NumPy brute-force backend is the
fallback and is also preferred for very large vocabularies to avoid copying
the embedding matrix into an index.
"""

from __future__ import annotations

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:  # pragma: no cover
    HAS_FAISS = False

# Above this many vectors, "auto" prefers the copy-free NumPy backend.
_FAISS_AUTO_LIMIT = 50_000


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class NeighborIndex:
    """Cosine-similarity index over a set of vectors."""

    def __init__(self, vectors: np.ndarray, backend: str = "auto"):
        self.vectors = _normalize(np.atleast_2d(vectors))
        if backend == "auto":
            backend = "faiss" if HAS_FAISS and len(self.vectors) <= _FAISS_AUTO_LIMIT else "numpy"
        if backend == "faiss" and not HAS_FAISS:
            backend = "numpy"
        self.backend = backend
        if backend == "faiss":
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, queries: np.ndarray, k: int = 5):
        """Return (similarities, indices), each (n_queries, k)."""
        q = _normalize(np.atleast_2d(queries))
        k = int(min(k, len(self.vectors)))
        if self.backend == "faiss":
            sims, idx = self._index.search(q, k)
            return sims, idx
        sims_full = q @ self.vectors.T
        idx = np.argsort(-sims_full, axis=1)[:, :k]
        sims = np.take_along_axis(sims_full, idx, axis=1)
        return sims, idx


class TokenNeighbors:
    """Nearest token embeddings: the semantic neighborhood of a hidden state."""

    def __init__(self, embedding_matrix: np.ndarray, vocab: list[str], backend: str = "auto"):
        self.index = NeighborIndex(embedding_matrix, backend=backend)
        self.vocab = vocab

    def nearest(self, vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        sims, idx = self.index.search(vector, k)
        return [(self.vocab[j], float(s)) for j, s in zip(idx[0], sims[0])]


class StateNeighbors:
    """Nearest hidden states across all (layer, token) positions."""

    def __init__(self, hidden: np.ndarray, tokens: list[str], backend: str = "auto"):
        L, T, D = hidden.shape
        self.shape = (L, T)
        self.tokens = tokens
        self.index = NeighborIndex(hidden.reshape(L * T, D), backend=backend)

    def nearest(self, vector: np.ndarray, k: int = 5, exclude: tuple[int, int] | None = None):
        """Return [(layer, token, token_text, similarity)] for the k nearest states."""
        extra = 1 if exclude is not None else 0
        sims, idx = self.index.search(vector, k + extra)
        out = []
        for j, s in zip(idx[0], sims[0]):
            layer, token = divmod(int(j), self.shape[1])
            if exclude is not None and (layer, token) == tuple(exclude):
                continue
            out.append((layer, token, self.tokens[token], float(s)))
        return out[:k]


def neighbor_ids_per_layer(traj_hidden: np.ndarray, embedding_matrix: np.ndarray,
                           token: int, k: int = 5, backend: str = "auto") -> list[list[int]]:
    """Token-embedding neighbor ids of one token's state at every layer.

    Feeds the nearest-neighbor-stability metric.
    """
    index = NeighborIndex(embedding_matrix, backend=backend)
    _, idx = index.search(traj_hidden[:, token, :], k)
    return [list(map(int, row)) for row in idx]
