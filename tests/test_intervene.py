"""Resumable forward pass / perturb-and-replay.

Mechanism tests run on a tiny locally-built Llama (no network): they pin the
exact propagation semantics of a write-edit. A separate semantic test on real
GPT-2 demonstrates causal control — a state edit flips the prediction.
"""

import zlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from capture import capture  # noqa: E402
from intervene import (  # noqa: E402
    FreezeLayer,
    InjectNoise,
    Perturb,
    SetState,
    divergence,
    intervene,
)

VOCAB_SIZE = 128
PROMPT = "the capital of france is"


class DummyTokenizer:
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


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def baseline(tiny):
    return capture(tiny, PROMPT, tokenizer=DummyTokenizer(), top_k=3)


# --------------------------------------------------------------- propagation
def test_perturb_propagates_forward_only(tiny, baseline):
    k, tok = 2, -1
    D = baseline.dim
    delta = np.full(D, 0.5, dtype=np.float32)
    branch = intervene(tiny, PROMPT, [Perturb(layer=k, delta=delta, token=tok)],
                       tokenizer=DummyTokenizer(), top_k=3)

    # layers before the edit are untouched
    assert np.allclose(branch.hidden[:k], baseline.hidden[:k], atol=1e-5)
    # the edited state == baseline + delta at the target token only
    assert np.allclose(branch.hidden[k, tok], baseline.hidden[k, tok] + delta, atol=1e-4)
    assert np.allclose(branch.hidden[k, 0], baseline.hidden[k, 0], atol=1e-5)  # other token
    # every later layer is changed by the perturbation
    assert not np.allclose(branch.hidden[k + 1], baseline.hidden[k + 1], atol=1e-4)


def test_perturb_layer_zero(tiny, baseline):
    D = baseline.dim
    delta = np.full(D, 1.0, dtype=np.float32)
    branch = intervene(tiny, PROMPT, [Perturb(layer=0, delta=delta, token=-1)],
                       tokenizer=DummyTokenizer())
    assert np.allclose(branch.hidden[0, -1], baseline.hidden[0, -1] + delta, atol=1e-4)
    assert not np.allclose(branch.hidden[1], baseline.hidden[1], atol=1e-4)


def test_readout_responds_to_perturbation(tiny, baseline):
    # a large perturbation must move the logit-lens distribution downstream
    D = baseline.dim
    big = np.full(D, 5.0, dtype=np.float32)
    branch = intervene(tiny, PROMPT, [Perturb(layer=1, delta=big, token=-1)],
                       tokenizer=DummyTokenizer())
    assert not np.allclose(branch.logits[-1, -1], baseline.logits[-1, -1], atol=1e-3)


# --------------------------------------------------------------- other edits
def test_set_state_all_tokens(tiny, baseline):
    D = baseline.dim
    value = np.arange(D, dtype=np.float32)
    branch = intervene(tiny, PROMPT, [SetState(layer=2, value=value, token=None)],
                       tokenizer=DummyTokenizer())
    for t in range(baseline.n_tokens):
        assert np.allclose(branch.hidden[2, t], value, atol=1e-4)


def test_freeze_layer_is_identity(tiny, baseline):
    b = 1
    branch = intervene(tiny, PROMPT, [FreezeLayer(block=b)], tokenizer=DummyTokenizer())
    # skipping block b makes the residual pass through unchanged
    assert np.allclose(branch.hidden[b + 1], branch.hidden[b], atol=1e-6)
    # and that genuinely differs from the baseline, which did apply the block
    assert not np.allclose(branch.hidden[b + 1], baseline.hidden[b + 1], atol=1e-4)


def test_inject_noise_is_seed_reproducible(tiny, baseline):
    a = intervene(tiny, PROMPT, [InjectNoise(layer=2, scale=0.3, seed=7)], tokenizer=DummyTokenizer())
    b = intervene(tiny, PROMPT, [InjectNoise(layer=2, scale=0.3, seed=7)], tokenizer=DummyTokenizer())
    c = intervene(tiny, PROMPT, [InjectNoise(layer=2, scale=0.3, seed=8)], tokenizer=DummyTokenizer())
    assert np.array_equal(a.hidden, b.hidden)                     # same seed -> identical
    assert not np.allclose(a.hidden[2], c.hidden[2], atol=1e-6)   # different seed -> different
    assert np.allclose(a.hidden[:2], baseline.hidden[:2], atol=1e-5)  # earlier layers intact


def test_multiple_interventions_compose(tiny, baseline):
    D = baseline.dim
    branch = intervene(tiny, PROMPT, [
        Perturb(layer=1, delta=np.ones(D, np.float32), token=-1),
        FreezeLayer(block=2),
    ], tokenizer=DummyTokenizer())
    assert np.allclose(branch.hidden[1, -1], baseline.hidden[1, -1] + 1.0, atol=1e-4)
    assert np.allclose(branch.hidden[3], branch.hidden[2], atol=1e-6)  # block 2 frozen


# ----------------------------------------------------------------- metadata
def test_meta_records_counterfactual(tiny):
    branch = intervene(tiny, PROMPT, [Perturb(layer=1, delta=np.zeros(32, np.float32))],
                       tokenizer=DummyTokenizer())
    assert branch.meta["counterfactual"] is True
    assert "perturb@layer1" in branch.meta["interventions"][0]


def test_guards():
    with pytest.raises(ValueError, match="synthetic"):
        intervene("synthetic", PROMPT, [Perturb(0, np.zeros(4, np.float32))])
    # empty interventions is a mistake -> use capture()
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                   num_hidden_layers=2, num_attention_heads=2,
                                   num_key_value_heads=1, max_position_embeddings=16)
    m = transformers.LlamaForCausalLM(cfg).eval()
    with pytest.raises(ValueError, match="no interventions"):
        intervene(m, PROMPT, [], tokenizer=DummyTokenizer())


# --------------------------------------------------------------- divergence
def test_divergence_measures_separation(tiny, baseline):
    k = 2
    D = baseline.dim
    branch = intervene(tiny, PROMPT, [Perturb(layer=k, delta=np.full(D, 2.0, np.float32), token=-1)],
                       tokenizer=DummyTokenizer())
    dv = divergence(baseline, branch, token=-1)
    assert dv.profile.shape == (baseline.n_layers,)
    assert np.allclose(dv.profile[:k], 0.0, atol=1e-4)   # identical before the edit
    assert dv.profile[k] > 0                              # separated at the edit
    assert dv.onset >= k                                  # onset no earlier than the cause
    assert dv.profile[-1] >= dv.profile[k] - 1e-4         # separation persists/grows


def test_divergence_shape_mismatch_rejected(baseline):
    from trajectory import StateTrajectory

    other = StateTrajectory(hidden=np.zeros((baseline.n_layers, baseline.n_tokens + 1, baseline.dim)),
                            tokens=["x"] * (baseline.n_tokens + 1))
    with pytest.raises(ValueError):
        divergence(baseline, other)


# -------------------------------------------------- semantic control (GPT-2)
@pytest.mark.network
def test_perturbation_flips_prediction_gpt2():
    """The payoff: pushing a late state along a token's embedding direction
    changes what the model predicts — causal control, end to end."""
    from capture import load_model

    model, tok = load_model("gpt2")
    prompt = "The capital of France is"
    base = capture(model, prompt, tokenizer=tok, top_k=5)
    base_top = base.topk[-1][-1][0][0]

    # GPT-2 ties input/output embeddings, so the input embedding row is also
    # the unembedding direction: push the final state toward " Berlin".
    target_id = int(tok(" Berlin")["input_ids"][0])
    direction = base.embedding_matrix[target_id].astype(np.float32)
    L = base.n_layers
    branch = intervene(model, prompt,
                       [Perturb(layer=L - 1, delta=60.0 * direction, token=-1)],
                       tokenizer=tok, top_k=5)
    branch_top = branch.topk[-1][-1][0][0]

    assert branch_top != base_top                # the prediction changed
    dv = divergence(base, branch, token=-1)
    assert dv.readout_changed is not None        # and we can locate where
