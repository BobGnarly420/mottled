"""Cross-model comparison (crossmodel.py): the shared space and its limits.

Two models share no hidden space and often no tokenizer. These tests pin the
two things that make comparing them honest: readout space is a *real* shared
coordinate system (mass is conserved, the unshared part stays visible), and
CKA reports whether its own answer is identified instead of always returning
an argmax.
"""
import numpy as np
import pytest

import crossmodel as X
from models import synthetic
from trajectory import StateTrajectory

PROMPT = "the capital of france is"


def _traj(prompt=PROMPT, **kw):
    return synthetic.capture(prompt, **kw)


def _relabel(traj: StateTrajectory, vocab: list[str]) -> StateTrajectory:
    """Same trajectory, different vocabulary strings — a stand-in for a model
    with another tokenizer."""
    return StateTrajectory(
        hidden=traj.hidden, tokens=traj.tokens, logits=traj.logits,
        entropy=traj.entropy, topk=traj.topk, vocab=vocab,
        embedding_matrix=traj.embedding_matrix, meta=dict(traj.meta))


# ------------------------------------------------------------ shared vocab
def test_shared_vocab_is_the_intersection_of_strings():
    a = _traj()
    b = _relabel(a, [t if i % 2 else f"only_b_{i}" for i, t in enumerate(a.vocab)])
    shared = X.shared_vocab([a, b])
    assert shared == sorted(t for i, t in enumerate(a.vocab) if i % 2)
    assert all(not s.startswith("only_b") for s in shared)


def test_disjoint_vocabularies_refuse():
    a = _traj()
    b = _relabel(a, [f"other_{i}" for i in range(len(a.vocab))])
    with pytest.raises(ValueError, match="share no vocabulary"):
        X.shared_vocab([a, b])


# ---------------------------------------------------------- readout space
def test_readout_space_gives_models_one_coordinate_system():
    a, b = _traj(), _traj("the capital of germany is")
    (ra, rb), vocab = X.readout_space([a, b])
    assert ra.dim == rb.dim == len(vocab) + 1     # + the unshared bucket
    assert ra.vocab[-1] == X.UNSHARED_TOKEN
    ra.validate(); rb.validate()
    # a real distribution at every state
    np.testing.assert_allclose(ra.hidden.sum(axis=-1), 1.0, rtol=1e-4)
    assert (ra.hidden >= 0).all()


def test_unshared_mass_is_kept_visible_not_renormalised():
    """A model can look identical on shared tokens while spending its mass
    elsewhere; the picture has to be able to show that."""
    a = _traj()
    half = [t if i % 2 else f"only_a_{i}" for i, t in enumerate(a.vocab)]
    b = _relabel(a, half)
    (ra, _), vocab = X.readout_space([a, b])
    unshared = ra.hidden[..., -1]
    assert (unshared > 0).any(), "dropping half the vocab must leave visible mass"
    np.testing.assert_allclose(ra.hidden.sum(axis=-1), 1.0, rtol=1e-4)
    # and it is reported per layer in the metadata
    assert len(ra.meta["unshared_mass"]) == ra.n_layers
    assert ra.meta["shared_vocab"] == len(vocab)


def test_readout_space_needs_logits():
    a = synthetic.capture(PROMPT, keep_logits=False)
    with pytest.raises(ValueError, match="logit-lens"):
        X.readout_space([a, _traj()])


def test_readout_trajectory_flows_through_the_pipeline():
    from density import compute_density
    from projection import project_joint
    from terrain import mesh

    (ra, rb), _ = X.readout_space([_traj(), _traj("the capital of germany is")])
    coords, _ = project_joint([ra.hidden, rb.hidden])   # accepts a shared D
    assert coords[0].shape == (ra.n_layers, ra.n_tokens, 2)
    surface = mesh(compute_density(coords[0].reshape(-1, 2), grid_size=16))
    assert surface.z.shape == (16, 16)


# -------------------------------------------------------- readout compare
def test_compare_models_measures_divergence_per_layer():
    a, b = _traj(), _traj("the capital of germany is")
    cmp = X.compare_models(a, b)
    assert cmp.divergence.shape == (a.n_layers,)
    assert (cmp.divergence >= 0).all()
    assert cmp.shared_vocab > 0
    # identical models diverge nowhere
    same = X.compare_models(a, _traj())
    np.testing.assert_allclose(same.divergence, 0.0, atol=1e-6)
    assert same.agree_from == 0 and same.top_a == same.top_b


def test_compare_models_normalises_different_depths():
    """Layer 6 of a 12-layer model is not layer 6 of a 32-layer model, so the
    comparison runs on a resampled common depth."""
    deep = synthetic.capture(PROMPT, n_layers=13)
    shallow = synthetic.capture(PROMPT, n_layers=7)
    cmp = X.compare_models(deep, shallow)
    assert cmp.divergence.shape == (7,)
    assert cmp.unshared_a.shape == cmp.unshared_b.shape == (7,)


# ------------------------------------------------------------------- CKA
def test_cka_is_one_for_a_layer_against_itself():
    a = _traj()
    assert X.cka(a.hidden[5], a.hidden[5]) == pytest.approx(1.0, abs=1e-9)


def test_raw_cka_is_exactly_invariant_to_rotation_and_scale():
    """The invariances that let CKA compare models at all — exact in the raw
    form, which is what makes it a legitimate cross-model measure."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))       # orthogonal
    assert X.cka(x, 7.5 * (x @ q), standardize=False) == pytest.approx(1.0, abs=1e-9)
    # width-agnostic: the same structure embedded in more dimensions
    wide = np.concatenate([x, np.zeros((40, 16))], axis=1)
    assert X.cka(x, wide, standardize=False) == pytest.approx(1.0, abs=1e-9)


def test_standardizing_trades_rotation_invariance_for_resolution():
    """The default is a trade, and the cost is measured, not assumed:
    z-scoring keeps isotropic-scale invariance exactly but only approximates
    rotation invariance, because rotating mixes dimensions whose scales are
    then equalised differently."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert X.cka(x, 7.5 * x, standardize=True) == pytest.approx(1.0, abs=1e-9)
    rotated = X.cka(x, x @ q, standardize=True)
    assert 0.9 < rotated < 1.0, f"expected near-but-not-exact invariance, got {rotated}"


def test_cka_rejects_unpaired_or_degenerate_input():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="paired"):
        X.cka(rng.normal(size=(10, 4)), rng.normal(size=(9, 4)))
    with pytest.raises(ValueError, match="at least 2"):
        X.cka(rng.normal(size=(1, 4)), rng.normal(size=(1, 4)))


def test_layer_similarity_reports_whether_the_match_is_identified():
    """The value that keeps this honest: a flat row's argmax is noise, and
    the alignment says so rather than presenting it as a correspondence."""
    a = _traj("the capital of france is paris and the capital of germany is berlin")
    b = _traj("the capital of france is paris and the capital of germany is berlin")
    align = X.layer_similarity(a, b)
    assert align.matrix.shape == (a.n_layers, b.n_layers)
    assert align.contrast.shape == (a.n_layers,)
    assert align.n_states == a.n_tokens
    # identical runs: every layer matches itself, strongly and in order
    np.testing.assert_array_equal(align.best, np.arange(a.n_layers))
    assert align.monotone
    assert align.identified().all()
    assert "identified" in align.summary()


def test_layer_similarity_refuses_states_that_are_not_counterparts():
    a = _traj()
    b = _traj("a much longer prompt with rather more tokens in it than that")
    with pytest.raises(ValueError, match="paired states"):
        X.layer_similarity(a, b)

    # same token count, different segmentation -> still not counterparts
    c = _traj("the capital of spain is")
    c.tokens = ["the", "capital", "of", "SPAIN", "is"]
    with pytest.raises(ValueError, match="split the text differently"):
        X.layer_similarity(a, c)


def test_standardizing_is_what_makes_the_alignment_resolvable():
    """Residual streams carry a few very-high-variance dimensions shared by
    every layer; without standardizing they dominate and every layer looks
    similar to every other. Pinned because it drives the default."""
    rng = np.random.default_rng(2)
    n, D, L = 60, 24, 6
    # a few outlier dimensions with enormous variance, identical in every
    # layer — the shape of a real residual stream's high-norm features
    outlier = rng.normal(size=(n, 3)) * 50.0
    layers = []
    for _ in range(L):
        own = rng.normal(size=(n, D - 3)) * 0.5        # each layer's own structure
        layers.append(np.concatenate([outlier, own], axis=1))
    stack = np.stack(layers)

    raw = np.array([[X.cka(stack[i], stack[j], standardize=False)
                     for j in range(L)] for i in range(L)])
    std = np.array([[X.cka(stack[i], stack[j], standardize=True)
                     for j in range(L)] for i in range(L)])
    off = ~np.eye(L, dtype=bool)
    # raw: everything looks alike; standardized: unrelated layers separate
    assert raw[off].mean() > 0.9
    assert std[off].mean() < raw[off].mean() - 0.3


# --------------------------------------------------------------- pipeline
def test_run_model_scene_puts_models_on_one_terrain():
    from config import MarbleConfig
    from pipeline import run_model_scene

    cfg = MarbleConfig(use_cache=False, density_bootstrap=0)
    result = run_model_scene(cfg, PROMPT, ["synthetic", "synthetic"])
    assert result["model_names"] == ["synthetic", "synthetic"]
    assert result["space"] == "readout"
    assert len(result["trajs"]) == 2
    # one shared terrain, one shared projection
    assert result["mesh"].z.shape == (cfg.grid_size, cfg.grid_size)
    assert result["coords_list"][0].shape[-1] == cfg.n_components
    assert len(result["model_comparisons"]) == 1
    assert result["shared_vocab"] > 0
    # the untouched source captures come along for analyses that need them
    assert len(result["source_trajs"]) == 2


def test_run_model_scene_needs_models():
    from config import MarbleConfig
    from pipeline import run_model_scene

    with pytest.raises(ValueError, match="at least one model"):
        run_model_scene(MarbleConfig(use_cache=False), PROMPT, [])


# ------------------------------------------------------- real models (net)
@pytest.mark.network
def test_gpt2_and_distilgpt2_align_proportionally():
    """The finding this module is built around, on real models: GPT-2 and its
    distillation pass through comparable stages in order, and standardizing
    is what makes that visible."""
    from capture import capture

    # a few dozen states: CKA cannot resolve a correspondence from a handful
    text = ("The capital of France is Paris, and the capital of Germany is "
            "Berlin. Water freezes at zero degrees Celsius. The Roman Empire "
            "fell in the fifth century, and the printing press was invented "
            "much later in Mainz by Gutenberg.")
    big = capture("gpt2", text)
    small = capture("distilgpt2", text)
    assert big.n_layers > small.n_layers

    align = X.layer_similarity(big, small)
    assert align.matrix.shape == (big.n_layers, small.n_layers)
    assert align.monotone, f"expected an ordered path, got {align.best.tolist()}"
    assert align.identified().all(), f"contrast too flat: {align.contrast.tolist()}"
    # the path spans the shallower model rather than collapsing onto one layer
    assert align.best[0] == 0 and align.best[-1] >= small.n_layers - 2

    raw = X.layer_similarity(big, small, standardize=False)
    assert raw.identified().sum() < align.identified().sum(), (
        "raw CKA should resolve fewer layers than standardized")


@pytest.mark.network
def test_models_with_different_tokenizers_meet_in_readout_space():
    """Different family, different width, different depth, different
    vocabulary — and still comparable, because they read out into text."""
    from capture import capture

    prompt = "The capital of France is"
    gpt2 = capture("gpt2", prompt)
    pythia = capture("EleutherAI/pythia-70m", prompt)
    assert gpt2.dim != pythia.dim and gpt2.n_layers != pythia.n_layers

    cmp = X.compare_models(gpt2, pythia)
    assert 0 < cmp.shared_vocab < min(len(gpt2.vocab), len(pythia.vocab))
    assert cmp.divergence.shape == (min(gpt2.n_layers, pythia.n_layers),)
    # both are competent enough on this prompt to converge by the last layer
    assert cmp.final_divergence < cmp.divergence[0]
    # and the mass each spends outside the shared vocabulary is visible
    assert (cmp.unshared_a >= 0).all() and (cmp.unshared_b >= 0).all()


def test_cli_export_models(tmp_path, capsys):
    """`mottled export --models a,b` writes a cross-model scene."""
    import statefile
    from cli import main

    out = tmp_path / "models.mtj"
    assert main(["export", PROMPT, "-o", str(out),
                 "--models", "synthetic,synthetic"]) == 0
    assert "shared vocabulary" in capsys.readouterr().out
    scene = statefile.load_scene(out)
    assert len(scene["runs"]) == 2
