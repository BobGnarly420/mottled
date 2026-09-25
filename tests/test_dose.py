"""Dose–response sweeps (dose.py).

Offline on the suite's tiny Llama: these pin that dose 0 is the plain
capture, that the injected δ has the sign and size the record claims, that
controls are what they say they are, and that the record round-trips. One
GPT-2 test (network) checks a real steer produces a curve that clears its
controls.
"""
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tiny  # noqa: E402
from capture import capture  # noqa: E402
from dose import (  # noqa: E402
    DEFAULT_GRID,
    DoseSweep,
    _edits,
    _pass,
    dose_sweep,
    push,
    random_control,
    resolve_target,
    tl_hook_name,
)
from metrics import kl_divergence, logit_lens_rank  # noqa: E402
from trajectory import StateTrajectory  # noqa: E402

PROMPT = "the capital of france is"
LAYER = 2


@pytest.fixture(scope="module")
def mt():
    return tiny.model(n_layers=4), tiny.tokenizer()


@pytest.fixture(scope="module")
def v():
    return np.random.default_rng(1).normal(size=32).astype(np.float32)


@pytest.fixture(scope="module")
def sweep(mt, v):
    model, tok = mt
    return dose_sweep(model, [PROMPT], v, LAYER, "paris", tokenizer=tok,
                      n_random=3, source="token")


def _points(sweep, kind="steer", seed=None):
    return {p["dose"]: p for p in sweep.points
            if p["kind"] == kind and p["seed"] == seed}


# ------------------------------------------------------------------ dose 0
def test_dose_zero_reproduces_a_plain_capture(mt, sweep):
    model, tok = mt
    plain = capture(model, PROMPT, tokenizer=tok)
    swept = _pass(model, tok, PROMPT, 5, "auto", "float32")
    assert np.array_equal(swept.hidden, plain.hidden)
    # same logits; only the storage precision differs
    assert np.array_equal(swept.logits.astype(np.float16), plain.logits)

    zero = _points(sweep)[0.0]
    assert zero["delta_norm"] == 0.0 and zero["sign"] == 0
    assert all(d == 0.0 for d in zero["state_distance"])
    assert zero["kl"] == 0.0
    assert zero["cos_to_baseline"] == pytest.approx(1.0)


def test_default_run_output_is_unchanged(mt):
    model, tok = mt
    assert capture(model, PROMPT, tokenizer=tok).logits.dtype == np.float16


# --------------------------------------------------------------- the push
@pytest.mark.parametrize("positions", ["last", "all"])
def test_delta_has_the_recorded_sign_and_size(mt, v, positions):
    model, tok = mt
    unit = v / np.linalg.norm(v)
    base = _pass(model, tok, PROMPT, 5, "auto", "float32")
    token = -1 if positions == "last" else None
    for dose in (-0.5, 0.5):
        delta = push(dose, 0.2, unit)
        assert np.linalg.norm(delta) == pytest.approx(abs(dose) * 0.2, rel=1e-6)
        assert np.sign(delta @ unit) == np.sign(dose)
        out = _pass(model, tok, PROMPT, 5, "auto", "float32",
                    state_edits=_edits(LAYER, delta, token))
        moved = out.hidden[LAYER] - base.hidden[LAYER]
        assert np.allclose(moved[-1], delta, atol=1e-5)
        if positions == "last":
            assert np.allclose(moved[:-1], 0.0, atol=1e-6)
        else:
            assert np.allclose(moved, delta[None, :], atol=1e-5)


@pytest.mark.parametrize("positions", ["last", "all"])
def test_injection_layer_distance_is_delta_over_baseline_norm(mt, v, positions):
    model, tok = mt
    s = dose_sweep(model, [PROMPT], v, LAYER, "paris", tokenizer=tok,
                   n_random=1, positions=positions, grid=(0.0, -0.25, 0.25))
    h0 = capture(model, PROMPT, tokenizer=tok).hidden[LAYER, -1]
    for p in s.points:
        if p["dose"] == 0:
            continue
        expected = p["delta_norm"] / np.linalg.norm(h0)
        assert p["state_distance"][0] == pytest.approx(expected, rel=1e-4)
        assert p["delta_norm"] == pytest.approx(0.25 * s.spec["reference_norm"]["value"])
    if positions == "all":
        assert len(s.points[1]["position_dose"]) == len(tiny.tokenizer()(PROMPT)["input_ids"])
    else:
        assert len(s.points[1]["position_dose"]) == 1


def test_negative_dose_is_a_negative_perturbation(mt, v, sweep):
    """A −dose point is exactly the same push with the sign flipped, run
    through the ordinary intervention path."""
    from intervene import Perturb, intervene

    model, tok = mt
    unit = v / np.linalg.norm(v)
    r = sweep.spec["reference_norm"]["value"]
    branch = intervene(model, PROMPT, [Perturb(LAYER, -0.5 * r * unit, token=-1)],
                       tokenizer=tok)
    base = capture(model, PROMPT, tokenizer=tok)
    h0 = np.linalg.norm(base.hidden[:, -1], axis=-1)
    manual = np.linalg.norm(branch.hidden[:, -1] - base.hidden[:, -1], axis=-1) / h0
    point = _points(sweep)[-0.5]
    assert point["sign"] == -1
    assert np.allclose(point["state_distance"], manual[LAYER:], rtol=1e-4)


# --------------------------------------------------------------- controls
def test_random_controls_are_orthogonal_seeded_and_signed(v, sweep):
    unit = v / np.linalg.norm(v)
    a, b, c = random_control(unit, 3), random_control(unit, 3), random_control(unit, 4)
    assert abs(float(a @ unit)) < 1e-6
    assert np.linalg.norm(a) == pytest.approx(1.0, rel=1e-6)
    assert np.array_equal(a, b)                 # reproducible from its seed
    assert not np.allclose(a, c)                # a different seed is a different direction
    assert np.allclose(push(-0.5, 1.0, a), -push(0.5, 1.0, a))

    steer = _points(sweep)
    seeds = sweep.spec["controls"]["random_seeds"]
    assert len(seeds) == 3
    for seed in seeds:
        for dose, p in _points(sweep, "random", seed).items():
            assert p["sign"] == int(np.sign(dose))
            assert p["delta_norm"] == pytest.approx(steer[dose]["delta_norm"])


def test_direct_path_reference_only_for_token_directions(mt, v, sweep):
    direct = _points(sweep, "direct_path")
    assert direct and all(p["inject_layer"] == sweep.spec["n_layers"] - 1
                          for p in direct.values())
    assert sweep.spec["controls"]["direct_path_layer"] == sweep.spec["n_layers"] - 1

    model, tok = mt
    other = dose_sweep(model, [PROMPT], v, LAYER, "paris", tokenizer=tok,
                       n_random=0, grid=(0.0, 0.5))
    assert not _points(other, "direct_path")
    assert other.spec["controls"]["direct_path_layer"] is None


def test_shuffled_label_control_and_derivation_overlap(mt):
    from intervene import direction_from_contrast

    model, tok = mt
    pos = [capture(model, p, tokenizer=tok) for p in
           ("the capital of france is", "the capital of italy is", "the capital of spain is")]
    neg = [capture(model, p, tokenizer=tok) for p in
           ("the cat sat on the mat", "the quick brown fox jumps", "hello world one two")]
    v = direction_from_contrast(pos, neg, layer=LAYER, normalize=False)

    with pytest.raises(ValueError, match="derived the direction"):
        dose_sweep(model, ["the capital of spain is"], v, LAYER, "paris",
                   tokenizer=tok, contrast=(pos, neg), n_random=0)

    s = dose_sweep(model, ["the capital of germany is"], v, LAYER, "berlin",
                   tokenizer=tok, contrast=(pos, neg), n_random=0, n_shuffled=3,
                   grid=(0.0, 0.5))
    assert s.spec["direction"]["source"] == "contrast"
    assert s.spec["direction"]["raw_norm"] == pytest.approx(float(np.linalg.norm(v)), rel=1e-5)
    assert len(s.spec["controls"]["shuffled_seeds"]) == 3
    assert len({p["seed"] for p in s.points if p["kind"] == "shuffled"}) == 3
    assert len(s.spec["derivation_prompts"]) == 6


# ------------------------------------------------------------------- rank
def _logits_traj(row):
    row = np.asarray(row, dtype=np.float32)
    logits = np.stack([row * 0.0, row])[:, None, :]          # (L=2, T=1, V)
    return StateTrajectory(hidden=np.zeros((2, 1, 4), np.float32), tokens=["x"],
                           logits=logits)


def test_rank_ties_resolve_in_the_targets_favour():
    traj = _logits_traj([1.0, 3.0, 3.0, 0.5])
    assert logit_lens_rank(traj, 1)[0] == 1
    assert logit_lens_rank(traj, 2)[0] == 1        # tied with id 1: not ranked 2
    assert logit_lens_rank(traj, 0)[0] == 3
    rank, logprob = logit_lens_rank(traj, 3)
    assert rank == 4
    row = np.array([1.0, 3.0, 3.0, 0.5])
    assert logprob == pytest.approx(0.5 - np.log(np.exp(row).sum()))


def test_rank_layer_indexing_and_guards():
    traj = _logits_traj([1.0, 3.0, 2.0, 0.5])
    assert logit_lens_rank(traj, 2, layer=-1) == logit_lens_rank(traj, 2, layer=1)
    assert logit_lens_rank(traj, 2, layer=0)[0] == 1     # all zeros: all tied
    with pytest.raises(ValueError, match="outside the vocabulary"):
        logit_lens_rank(traj, 4)
    with pytest.raises(ValueError, match="no logits"):
        logit_lens_rank(StateTrajectory(hidden=np.zeros((2, 1, 4)), tokens=["x"]), 0)


def test_multi_token_target_is_refused(mt, v):
    model, tok = mt
    assert resolve_target(tok, "paris") == tok("paris")["input_ids"][0]
    with pytest.raises(ValueError, match="exactly one"):
        resolve_target(tok, "paris berlin")
    with pytest.raises(ValueError, match="exactly one"):
        dose_sweep(model, [PROMPT], v, LAYER, "paris berlin", tokenizer=tok)


# --------------------------------------------------------------- precision
def test_float32_metrics_match_a_float32_reference_at_the_smallest_dose(mt, v, sweep):
    """An independent forward pass (a plain torch hook, no Mottled capture)
    gives the same KL and log-prob as the sweep at its smallest dose — the
    regime float16 storage would have rounded into steps."""
    model, tok = mt
    unit = v / np.linalg.norm(v)
    dose = min(d for d in sweep.spec["grid"]["dose"] if d > 0)
    delta = torch.tensor(push(dose, sweep.spec["reference_norm"]["value"], unit))
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]

    def add(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        out = out.clone()
        out[:, -1] += delta
        return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

    with torch.no_grad():
        ref0 = model(ids).logits[0, -1].double().numpy()
        handle = model.model.layers[LAYER - 1].register_forward_hook(add)
        try:
            ref = model(ids).logits[0, -1].double().numpy()
        finally:
            handle.remove()

    point = _points(sweep)[dose]
    assert point["kl"] > 0
    assert point["kl"] == pytest.approx(float(kl_divergence(ref0, ref)), rel=1e-3, abs=1e-9)
    target = sweep.spec["target"]["id"]
    ref_logprob = ref[target] - ref.max() - np.log(np.exp(ref - ref.max()).sum())
    assert point["target_logprob"] == pytest.approx(ref_logprob, abs=1e-5)


# ------------------------------------------------------------------ record
def test_record_describes_the_sweep(sweep):
    spec = sweep.spec
    assert spec["inject_layer"] == LAYER
    assert spec["inject_hook_tl"] == f"blocks.{LAYER}.hook_resid_pre"
    assert tl_hook_name(4, 5) == "blocks.3.hook_resid_post"
    assert spec["readout_layers"] == list(range(LAYER, spec["n_layers"]))
    assert spec["grid"]["dose"] == sorted(DEFAULT_GRID)
    assert len(spec["direction"]["sha256"]) == 64
    assert sweep.analysis["schema"] == "mottled-analysis/1"
    flagged = {p["dose"] for p in sweep.points if p["replacement_regime"]}
    assert flagged == {-2.0, 2.0}


def test_json_and_csv_round_trip(sweep, tmp_path):
    path = tmp_path / "sweep.json"
    sweep.to_json(path)
    back = DoseSweep.from_json(path)
    assert back.to_dict() == json.loads(json.dumps(sweep.to_dict()))
    assert not any(isinstance(x, str) and x.startswith("[")
                   for p in back.points for x in p.values())

    sweep.to_csv(tmp_path / "sweep.csv")
    with (tmp_path / "sweep.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    layers = sweep.spec["readout_layers"]
    assert len(rows) == len(sweep.points) * len(layers)
    for p, first in zip(sweep.points, rows[::len(layers)]):
        assert float(first["dose"]) == p["dose"]
        assert first["kind"] == p["kind"]
        assert float(first["kl"]) == p["kl"]
        assert float(first["state_distance"]) == p["state_distance"][0]
        assert json.loads(first["position_dose"]) == p["position_dose"]


def test_per_dose_files_carry_their_dose(mt, v, tmp_path):
    import statefile

    model, tok = mt
    dose_sweep(model, [PROMPT], v, LAYER, "paris", tokenizer=tok, n_random=1,
               grid=(0.0, -0.25, 0.25), mtj_dir=tmp_path)
    files = sorted(tmp_path.glob("*.mtj"))
    assert len(files) == 3                               # steer only
    doses = sorted(statefile.load(f).meta["dose"]["dose"] for f in files)
    assert doses == [-0.25, 0.0, 0.25]
    assert statefile.load(files[0]).embedding_matrix is None


# ---------------------------------------------------------------- GPT-2
@pytest.mark.network
def test_gpt2_berlin_steer_clears_its_controls():
    from capture import load_model
    from intervene import direction_from_token

    model, tok = load_model("gpt2")
    prompt = "The capital of France is"
    target = resolve_target(tok, " Berlin")
    base = capture(model, prompt, tokenizer=tok)
    v = direction_from_token(base, target)
    s = dose_sweep(model, [prompt], v, 6, target, tokenizer=tok, n_random=8,
                   source="token", grid=(0.0, 0.25, 0.5, 1.0))
    steer = _points(s)
    assert steer[1.0]["target_logprob"] > steer[0.0]["target_logprob"]
    control_best = max(p["target_logprob"] for p in s.points
                       if p["kind"] == "random" and p["dose"] == 1.0)
    assert steer[1.0]["target_logprob"] > control_best
