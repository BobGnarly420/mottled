"""Neighbor lookup valid (spec test 4)."""

import numpy as np
import pytest

from neighbors import HAS_FAISS, NeighborIndex, StateNeighbors, TokenNeighbors, neighbor_ids_per_layer

BACKENDS = ["numpy"] + (["faiss"] if HAS_FAISS else [])


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(3)
    return rng.normal(size=(50, 16)).astype(np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_self_is_nearest(vectors, backend):
    index = NeighborIndex(vectors, backend=backend)
    sims, idx = index.search(vectors[:10], k=3)
    assert idx.shape == sims.shape == (10, 3)
    assert (idx[:, 0] == np.arange(10)).all()
    assert np.allclose(sims[:, 0], 1.0, atol=1e-5)
    # similarities are sorted descending and are valid cosines
    assert (np.diff(sims, axis=1) <= 1e-6).all()
    assert (sims <= 1.0 + 1e-5).all() and (sims >= -1.0 - 1e-5).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_agree(vectors, backend):
    ref_sims, ref_idx = NeighborIndex(vectors, backend="numpy").search(vectors[:5], k=4)
    sims, idx = NeighborIndex(vectors, backend=backend).search(vectors[:5], k=4)
    assert np.array_equal(idx, ref_idx)
    assert np.allclose(sims, ref_sims, atol=1e-5)


def test_cosine_ignores_scale(vectors):
    index = NeighborIndex(vectors)
    _, idx1 = index.search(vectors[7], k=5)
    _, idx2 = index.search(vectors[7] * 100.0, k=5)
    assert np.array_equal(idx1, idx2)


def test_token_neighbors(vectors):
    vocab = [f"tok{i}" for i in range(len(vectors))]
    tn = TokenNeighbors(vectors, vocab)
    hits = tn.nearest(vectors[4], k=3)
    assert hits[0] == ("tok4", pytest.approx(1.0, abs=1e-5))
    assert len(hits) == 3


def test_state_neighbors_exclude():
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(5, 4, 8)).astype(np.float32)
    sn = StateNeighbors(hidden, tokens=["a", "b", "c", "d"])
    hits = sn.nearest(hidden[2, 1], k=3, exclude=(2, 1))
    assert all((layer, token) != (2, 1) for layer, token, _, _ in hits)
    assert len(hits) == 3


def test_neighbor_ids_per_layer(vectors):
    hidden = np.stack([vectors[:4], vectors[4:8], vectors[8:12]])  # (3, 4, 16)
    ids = neighbor_ids_per_layer(hidden, vectors, token=0, k=3)
    assert len(ids) == 3 and all(len(row) == 3 for row in ids)
    assert ids[0][0] == 0  # layer 0 of token 0 IS vectors[0]
