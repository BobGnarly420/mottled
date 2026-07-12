"""Hidden tensors captured correctly + shape consistency (spec tests 1-2)."""

import zlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from capture import HookCapture, capture, logit_lens  # noqa: E402
from models.families import resolve_family  # noqa: E402

VOCAB_SIZE = 128
PROMPT = "the capital of france is"


class DummyTokenizer:
    """Deterministic word-level tokenizer for locally-built test models."""

    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._names = {}

    def _id(self, word):
        i = zlib.crc32(word.encode()) % (self.vocab_size - 1) + 1
        self._names[i] = word
        return i

    def __call__(self, text, return_tensors="pt"):
        ids = [self._id(w) for w in text.split()]
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids):
        return [self._names.get(int(i), f"<{int(i)}>") for i in ids]


@pytest.fixture(scope="module")
def tiny_llama():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def test_hooks_match_output_hidden_states(tiny_llama):
    """Hook-captured residual stream equals HF's own hidden_states."""
    tok = DummyTokenizer()
    ids = tok(PROMPT)["input_ids"]
    with HookCapture(tiny_llama) as cap, torch.no_grad():
        out = tiny_llama(ids, output_hidden_states=True)
    hidden = cap.stacked()  # (L+1, T, D)

    ref = torch.stack([h[0] for h in out.hidden_states]).float()
    n_blocks = len(cap.adapter.blocks)
    # HF's tuple is (embeddings, block1..block_{N-1}, final_norm(block_N)):
    # every pre-norm entry must match our capture exactly...
    assert torch.allclose(hidden[:n_blocks], ref[:n_blocks], atol=1e-5)
    # ...and our raw final-block output must match after applying final norm.
    with torch.no_grad():
        final = cap.adapter.final_norm(hidden[-1])
    assert torch.allclose(final, ref[-1], atol=1e-5)


def test_capture_shapes_and_stats(tiny_llama):
    traj = capture(tiny_llama, PROMPT, tokenizer=DummyTokenizer(), top_k=3)
    traj.validate()
    L, T, D = traj.hidden.shape
    assert (L, D) == (4, 32) and T == len(PROMPT.split())
    assert traj.logits.shape == (L, T, VOCAB_SIZE)
    assert traj.entropy.shape == (L, T)
    assert np.isfinite(traj.entropy).all()
    assert len(traj.topk) == L and len(traj.topk[0]) == T and len(traj.topk[0][0]) == 3
    assert all(0.0 <= p <= 1.0 for _, p in traj.topk[0][0])
    assert traj.embedding_matrix.shape == (VOCAB_SIZE, D)
    assert traj.tokens[1] == "capital"


def test_logit_lens_final_layer_matches_model(tiny_llama):
    """Logit lens at the last layer must reproduce the model's real logits."""
    tok = DummyTokenizer()
    ids = tok(PROMPT)["input_ids"]
    with HookCapture(tiny_llama) as cap, torch.no_grad():
        out = tiny_llama(ids)
    lens = logit_lens(cap.stacked(), cap.adapter)
    assert torch.allclose(lens[-1], out.logits[0].float(), atol=1e-4)


def test_synthetic_backend_shapes():
    traj = capture("synthetic", PROMPT, top_k=5)
    traj.validate()
    assert traj.n_tokens == len(PROMPT.split())
    assert traj.n_layers > 1
    assert traj.entropy is not None and np.isfinite(traj.entropy).all()
    # deterministic per prompt
    again = capture("synthetic", PROMPT, top_k=5)
    assert np.array_equal(traj.hidden, again.hidden)


def test_gpt2_style_layout_resolves():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=VOCAB_SIZE, n_embd=32, n_layer=2, n_head=4, n_positions=64)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    adapter = resolve_family(model)
    assert adapter.n_layers == 2
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), top_k=2)
    assert traj.hidden.shape[0] == 3
