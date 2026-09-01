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


@pytest.mark.network
def test_persistence_profile_gpt2_early_injection_is_legible():
    """Criterion 2: injecting the ' Berlin' direction early into the
    France->Berlin sample and profiling downstream yields a real curve —
    finite, layer-dependent structure, not flat noise."""
    from capture import load_model
    from intervene import direction_from_token, persistence_profile

    model, tok = load_model("gpt2")
    prompt = "The capital of France is"
    base = capture(model, prompt, tokenizer=tok, top_k=5, capture_components=True)

    target_id = int(tok(" Berlin")["input_ids"][0])
    direction = direction_from_token(base, target_id)
    prof = persistence_profile(model, prompt, direction, 2, target_id,
                               tokenizer=tok, scale=30.0, baseline=base)

    effects = prof.effect_sizes
    assert list(prof.layers) == list(range(2, base.n_layers))
    assert np.isfinite(effects).all()
    assert float(np.ptp(effects)) > 0.5          # the curve moves, layer to layer
    assert float(np.abs(effects).max()) > 1.0    # and the steer is not a no-op
    # shares are a real decomposition at every profiled layer
    assert np.isfinite(prof.mlp_shares).all()


# ------------------------------------------------ data-derived steering (1b)
def test_direction_from_contrast_is_normalized_diff_of_means():
    """Backend-agnostic: a diff-of-means direction from synthetic runs."""
    from intervene import direction_from_contrast
    from models import synthetic

    a = synthetic.capture("the capital of france is")
    b = synthetic.capture("the capital of germany is")

    d = direction_from_contrast(a, b, layer=-1, token=-1)
    assert d.shape == (a.dim,)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5)  # unit direction

    manual = a.hidden[-1, -1] - b.hidden[-1, -1]
    manual = manual / np.linalg.norm(manual)
    assert np.allclose(d, manual, atol=1e-4)

    # groups average, so symmetric groups cancel to (nearly) nothing
    sym = direction_from_contrast([a, b], [b, a], layer=-1, token=-1)
    assert np.allclose(sym, 0.0, atol=1e-5)

    # un-normalized keeps magnitude (the caller may want the raw contrast)
    raw = direction_from_contrast(a, b, layer=-1, token=-1, normalize=False)
    assert np.allclose(raw, a.hidden[-1, -1] - b.hidden[-1, -1], atol=1e-4)


# ------------------------------------------------------ faithfulness (1c)
def test_faithfulness_prefers_the_targeted_direction():
    """A steer along a token's own (tied) embedding direction moves the logit
    lens toward that token far more than a random delta of equal norm — the
    effect is the *direction*, not merely the perturbation's size."""
    from intervene import direction_from_token, faithfulness

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    tok = DummyTokenizer()
    base = capture(model, PROMPT, tokenizer=tok, top_k=3)

    target = 7
    direction = direction_from_token(base, target)      # unit embedding axis
    assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-5)

    fth = faithfulness(model, PROMPT, layer=base.n_layers - 1,
                       direction=direction, target=target, scale=50.0,
                       tokenizer=tok, baseline=base)

    assert fth.steer_shift > fth.control_shift   # direction-specific gain
    assert fth.effect > 0
    assert fth.steer_hits_target                 # steering made target top-1
    assert fth.target == target


# ------------------------------------------------------- persistence profile
def _tied_tiny():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def test_persistence_profile_generalizes_faithfulness():
    """The success criterion: the profile's record at the layer where
    `faithfulness()` reads out reproduces `faithfulness()`'s effect exactly —
    the new function strictly generalizes the old one, it does not
    reimplement it. Injecting at the final layer (the canonical faithfulness
    setup) makes that record `profile[inject_layer]` itself."""
    from intervene import direction_from_token, faithfulness, persistence_profile

    model = _tied_tiny()
    tok = DummyTokenizer()
    base = capture(model, PROMPT, tokenizer=tok, top_k=3, capture_components=True)
    target = 7
    direction = direction_from_token(base, target)
    last = base.n_layers - 1

    fth = faithfulness(model, PROMPT, layer=last, direction=direction,
                       target=target, scale=50.0, tokenizer=tok, baseline=base)
    prof = persistence_profile(model, PROMPT, direction, last, target,
                               tokenizer=tok, scale=50.0, baseline=base)
    assert np.isclose(prof[last].effect_size, fth.effect, atol=1e-6)
    assert np.isclose(prof[last].steer_shift, fth.steer_shift, atol=1e-6)
    assert np.isclose(prof[last].control_shift, fth.control_shift, atol=1e-6)

    # injecting earlier, the final-layer record is still faithfulness exactly
    fth1 = faithfulness(model, PROMPT, layer=1, direction=direction,
                        target=target, scale=50.0, tokenizer=tok, baseline=base)
    prof1 = persistence_profile(model, PROMPT, direction, 1, target,
                                tokenizer=tok, scale=50.0, baseline=base)
    assert np.isclose(prof1[-1].effect_size, fth1.effect, atol=1e-6)


def test_persistence_profile_records_and_shares():
    from intervene import direction_from_token, persistence_profile

    model = _tied_tiny()
    tok = DummyTokenizer()
    base = capture(model, PROMPT, tokenizer=tok, top_k=3, capture_components=True)
    target = 7
    direction = direction_from_token(base, target)

    prof = persistence_profile(model, PROMPT, direction, 1, target,
                               tokenizer=tok, scale=50.0, baseline=base)
    # one record per layer from the injection to the end, indexed absolutely
    assert list(prof.layers) == list(range(1, base.n_layers))
    assert len(prof) == base.n_layers - 1
    assert prof[1] is prof.records[0] and prof[-1] is prof.records[-1]
    with pytest.raises(IndexError):
        prof[0]                                    # precedes the injection

    # write shares come from the baseline decomposition: valid, complementary
    from metrics import component_shares
    shares = component_shares(base, token=-1)
    for r in prof.records:
        assert np.isclose(r.attn_share + r.mlp_share, 1.0, atol=1e-6)
        assert np.isclose(r.mlp_share, shares[r.layer - 1, 1], atol=1e-6)
    assert prof.target == target and prof.inject_layer == 1
    assert np.isclose(prof.scale, 50.0, atol=1e-3)

    # injecting at layer 0: the embedding state has no block writing into it
    prof0 = persistence_profile(model, PROMPT, direction, 0, target,
                                tokenizer=tok, scale=50.0, baseline=base)
    assert np.isnan(prof0[0].mlp_share) and np.isnan(prof0[0].attn_share)
    assert np.isfinite(prof0.effect_sizes).all()


def test_persistence_profile_requires_components():
    from intervene import direction_from_token, persistence_profile

    model = _tied_tiny()
    tok = DummyTokenizer()
    base = capture(model, PROMPT, tokenizer=tok, top_k=3)   # no decomposition
    direction = direction_from_token(base, 7)
    with pytest.raises(ValueError, match="capture_components"):
        persistence_profile(model, PROMPT, direction, 1, 7,
                            tokenizer=tok, baseline=base)


def test_run_intervention_attaches_persistence():
    """The UI pipeline surfaces the profile for a directional steer when the
    baseline carries the residual decomposition, and the panel's figure
    renders from it."""
    from config import MarbleConfig
    from intervene import Perturb, direction_from_token
    from render import render_persistence
    from ui import run_intervention

    model = _tied_tiny()
    tok = DummyTokenizer()
    base = capture(model, PROMPT, tokenizer=tok, top_k=3)
    target = 7
    direction = direction_from_token(base, target)

    cfg = MarbleConfig(model="tiny", use_cache=False,
                       capture_components=True, capture_attention=False)
    edits = [Perturb(base.n_layers - 1, 50.0 * direction, token=-1)]
    result = run_intervention(cfg, PROMPT, edits, model, tok, target_id=target)

    prof = result["persistence"]
    assert prof.inject_layer == base.n_layers - 1
    # the profile's final record and the faithfulness readout are the same
    # measurement (deltas differ only by float renormalization)
    assert np.isclose(prof[-1].effect_size, result["faithfulness"].effect, atol=1e-3)

    fig = render_persistence(prof)
    names = [tr.name for tr in fig.data]
    assert names == ["effect size", "mlp share"]
    assert fig.data[1].yaxis == "y2"                # mlp share on its own axis


def test_run_intervention_attaches_faithfulness():
    """The UI pipeline surfaces a faithfulness readout for a directional steer."""
    from config import MarbleConfig
    from intervene import Perturb, direction_from_token
    from ui import run_intervention

    torch.manual_seed(0)
    mcfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    model = transformers.LlamaForCausalLM(mcfg).eval()
    tok = DummyTokenizer()
    base = capture(model, PROMPT, tokenizer=tok, top_k=3)
    target = 7
    direction = direction_from_token(base, target)

    cfg = MarbleConfig(model="tiny", use_cache=False,
                       capture_components=False, capture_attention=False)
    edits = [Perturb(base.n_layers - 1, 50.0 * direction, token=-1)]
    result = run_intervention(cfg, PROMPT, edits, model, tok, target_id=target)

    assert "divergence" in result
    fth = result["faithfulness"]
    assert fth.target == target
    assert fth.effect > 0                         # direction beat the control
