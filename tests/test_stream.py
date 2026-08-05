"""Layer-by-layer streaming capture, and sparse-MoE expert routing.

The reason this module exists is models too large to hold in memory (Kimi K3
is 93 layers and ~1.5 TB). Two properties make it trustworthy and both are
pinned here: a streamed pass returns **exactly** what an in-memory pass
returns, and only the configured number of blocks is ever resident.
"""
import zlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from capture import capture  # noqa: E402
from stream import ShardIndex, StreamedBlocks, stream_capture  # noqa: E402

PROMPT = "the capital of france is"
VOCAB_SIZE = 128


class DummyTokenizer:
    """Deterministic word-level tokenizer (mirrors tests/test_capture.py)."""

    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._names = {}

    def _id(self, word):
        i = zlib.crc32(word.encode()) % (self.vocab_size - 1) + 1
        self._names[i] = word
        return i

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[self._id(w) for w in text.split()]])}

    def convert_ids_to_tokens(self, ids):
        return [self._names.get(int(i), f"<{int(i)}>") for i in ids]


def _dense(tmp_path, layers=6):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(cfg).eval()
    path = tmp_path / "dense"
    model.save_pretrained(path, safe_serialization=True)
    return model, path


def _moe(tmp_path, layers=4, n_experts=8, k=2):
    torch.manual_seed(0)
    cfg = transformers.Qwen3MoeConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        moe_intermediate_size=32, num_experts=n_experts, num_experts_per_tok=k,
        decoder_sparse_step=1, max_position_embeddings=64)
    model = transformers.Qwen3MoeForCausalLM(cfg).eval()
    path = tmp_path / "moe"
    model.save_pretrained(path, safe_serialization=True)
    return model, path


# ------------------------------------------------------------- streaming
def test_streamed_capture_is_exactly_the_in_memory_capture(tmp_path):
    """The whole correctness claim: streaming changes *when* weights exist,
    not what the model computes.

    Bit-exactness here is only safe to assert because both paths make the
    *same calls* — same weights, same forward, same shapes, and one shared
    `logit_lens`. An earlier version reimplemented the lens with different
    chunking; it was bit-identical locally and off by one float16 bucket on
    CI. Equal arithmetic is not equal bits.
    """
    model, path = _dense(tmp_path)
    tok = DummyTokenizer()
    mem = capture(model, PROMPT, tokenizer=tok)
    streamed = stream_capture(path, PROMPT, tokenizer=DummyTokenizer())

    assert streamed.hidden.shape == mem.hidden.shape
    np.testing.assert_array_equal(streamed.hidden, mem.hidden)
    np.testing.assert_array_equal(streamed.logits, mem.logits)
    assert streamed.tokens == mem.tokens
    assert [t[0][0] for t in streamed.topk[-1]] == [t[0][0] for t in mem.topk[-1]]


def test_only_the_configured_number_of_blocks_is_resident(tmp_path):
    """The memory bound is the point; assert it rather than trusting it."""
    _, path = _dense(tmp_path, layers=6)
    seen = []

    original = StreamedBlocks._load

    def spy(self, i):
        original(self, i)
        seen.append(sum(1 for b in self.blocks
                        if next(b.parameters()).device.type != "meta"))

    StreamedBlocks._load = spy
    try:
        one = stream_capture(path, PROMPT, tokenizer=DummyTokenizer(), keep_resident=1)
        assert max(seen) == 1
        seen.clear()
        three = stream_capture(path, PROMPT, tokenizer=DummyTokenizer(), keep_resident=3)
        assert max(seen) == 3
    finally:
        StreamedBlocks._load = original
    # residency is a memory knob, never a numerical one
    np.testing.assert_array_equal(one.hidden, three.hidden)


def test_each_block_is_loaded_once(tmp_path):
    _, path = _dense(tmp_path, layers=6)
    traj = stream_capture(path, PROMPT, tokenizer=DummyTokenizer())
    assert traj.meta["block_loads"] == traj.meta["n_blocks"] == 6
    assert traj.meta["streamed"] is True
    # a streamed pass says what it cannot give you
    assert "attention patterns" in traj.meta["absent"]


def test_shard_index_finds_tensors(tmp_path):
    _, path = _dense(tmp_path, layers=3)
    idx = ShardIndex(path)
    assert len(idx) > 0
    block0 = idx.prefixed("model.layers.0.")
    assert block0 and all(not k.startswith("model.layers.0.") for k in block0)
    assert idx.prefixed("model.layers.99.") == {}


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        ShardIndex(tmp_path / "nope")


# ------------------------------------------------------- MoE and routing
def test_routing_capture_records_the_experts_each_token_used(tmp_path):
    model, _ = _moe(tmp_path)
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_routing=True)
    traj.validate()

    r = traj.routing
    assert r is not None
    assert r.experts.shape == (4, traj.n_tokens, 2)   # layers × tokens × top-k
    assert r.n_experts == 8 and r.top_k == 2
    assert r.experts.min() >= 0 and r.experts.max() < 8
    np.testing.assert_allclose(r.weights.sum(-1), 1.0, rtol=1e-5)
    # a token's route is readable without any dictionary learning
    path = r.path(0)
    assert len(path) == 4 and all(len(experts) == 2 for _, experts in path)


def test_routing_agreement_compares_two_runs(tmp_path):
    model, _ = _moe(tmp_path)
    a = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_routing=True)
    b = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_routing=True)
    same = a.routing.agreement(b.routing)
    assert same.shape == (4, a.n_tokens)
    np.testing.assert_allclose(same, 1.0)            # same prompt, same route


def test_routing_refuses_on_a_dense_model(tmp_path):
    """A dense model has no experts. Returning an empty routing would read as
    'this token used no experts', so it refuses instead — the same contract
    attention capture has on Mamba."""
    model, _ = _dense(tmp_path, layers=2)
    with pytest.raises(ValueError, match="dense|router"):
        capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_routing=True)
    with pytest.raises(ValueError, match="dense"):
        capture("synthetic", PROMPT, capture_routing=True)


def test_streamed_moe_matches_in_memory_including_routing(tmp_path):
    """MoE checkpoints store experts per-expert while the runtime wants them
    fused; getting that wrong leaves the experts empty and the capture silently
    wrong. Pin both the states and the routing."""
    model, path = _moe(tmp_path)
    mem = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_routing=True)
    streamed = stream_capture(path, PROMPT, tokenizer=DummyTokenizer(),
                              capture_routing=True)

    np.testing.assert_array_equal(streamed.hidden, mem.hidden)
    np.testing.assert_array_equal(streamed.routing.experts, mem.routing.experts)
    np.testing.assert_allclose(streamed.routing.weights, mem.routing.weights,
                               rtol=1e-5, atol=1e-6)


def test_unloadable_weights_fail_loudly(tmp_path, monkeypatch):
    """The failure mode that matters: a checkpoint whose layout does not match
    the module leaves parameters on meta, a fused kernel takes the meta path,
    and the capture comes back as plausible-looking nonsense. It must raise."""
    import stream as S

    _, path = _moe(tmp_path)
    monkeypatch.setattr(S, "_fuse_experts", lambda sd, block: sd)  # break the mapping
    with pytest.raises(RuntimeError, match="never loaded"):
        stream_capture(path, PROMPT, tokenizer=DummyTokenizer())


def test_routing_validates_against_the_trajectory(tmp_path):
    from trajectory import Routing

    model, _ = _moe(tmp_path)
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_routing=True)
    bad = Routing(experts=traj.routing.experts[:, :1],
                  weights=traj.routing.weights[:, :1],
                  layers=traj.routing.layers, n_experts=8)
    traj.routing = bad
    with pytest.raises(ValueError, match="routing covers"):
        traj.validate()
