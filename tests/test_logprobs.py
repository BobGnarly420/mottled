"""Closed-model producer: API logprobs -> StateTrajectory (models/logprobs.py).

The contract is the point: a degraded producer must still emit a valid
StateTrajectory that flows through the whole stack, and must say — in the
trajectory itself — what it could not provide.
"""
import math

import numpy as np
import pytest

from models.logprobs import RESIDUAL_TOKEN, from_logprobs, from_openai_logprobs

STEPS = [
    {"token": " Paris", "logprobs": {" Paris": math.log(0.7), " the": math.log(0.2)}},
    {"token": ",", "logprobs": {",": math.log(0.5), ".": math.log(0.3)}},
    {"token": " France", "logprobs": {" France": math.log(0.9), " Europe": math.log(0.05)}},
]


def test_builds_a_valid_trajectory_over_the_decode_axis():
    traj = from_logprobs(STEPS, model="gpt-4o-mini", prompt="The capital of France is")
    traj.validate()
    assert (traj.n_layers, traj.n_tokens) == (3, 1)      # steps × one path
    assert traj.meta["axis"] == "decode"
    assert traj.meta["backend"] == "logprobs"
    # every step's distribution sums to 1 including the unreported bucket
    np.testing.assert_allclose(traj.hidden.sum(axis=-1), 1.0, rtol=1e-5)


def test_unreported_mass_is_explicit_not_hidden():
    traj = from_logprobs(STEPS)
    resid = traj.vocab.index(RESIDUAL_TOKEN)
    # step 0 reported 0.9 of the mass -> 0.1 lands in the bucket, visibly
    assert traj.hidden[0, 0, resid] == pytest.approx(0.1, abs=1e-6)
    assert traj.meta["reported_mass"][0] == pytest.approx(0.9, abs=1e-6)
    assert RESIDUAL_TOKEN in traj.vocab


def test_declares_what_it_cannot_provide():
    traj = from_logprobs(STEPS)
    assert traj.meta["degraded"] is True
    assert "residual stream" in traj.meta["absent"]
    assert "attention" in traj.meta["absent"]
    # truncated top-k can only under-count entropy; the trajectory says so
    assert traj.meta["entropy_is_lower_bound"] is True
    assert traj.components is None and traj.attention is None


def test_probabilities_survive_the_logit_lens_path():
    """logits = log p, so the shared softmax/top-k machinery returns p."""
    traj = from_logprobs(STEPS, top_k=2)
    top_token, top_p = traj.topk[0][0][0]
    assert top_token == " Paris"
    assert top_p == pytest.approx(0.7, abs=1e-3)
    assert traj.entropy.shape == (3, 1)
    assert (traj.entropy >= 0).all()


def test_decode_record_matches_the_generation_schema():
    """The decode axis is recorded exactly like generate_and_capture's, so
    viewers built for one work for the other."""
    traj = from_logprobs(STEPS)
    gen = traj.meta["generation"]
    assert gen["new_tokens"] == 3 and gen["prompt_tokens"] == 0
    assert [s["token"] for s in gen["steps"]] == [" Paris", ",", " France"]
    assert gen["steps"][0]["p"] == pytest.approx(0.7, abs=1e-3)


def test_flows_through_the_full_pipeline():
    """The whole point of the interchange: a degraded producer still renders."""
    from density import compute_density
    from projection import project
    from terrain import mesh
    from trajectory import extract

    traj = from_logprobs(STEPS * 3)     # 9 steps, enough for a real path
    coords, _ = project(traj.hidden, method="pca")
    assert coords.shape == (traj.n_layers, 1, 2)
    surface = mesh(compute_density(coords, method="kde", grid_size=16))
    paths = extract(coords, traj.tokens, mode="all_tokens")
    assert len(paths) == 1 and len(paths[0]) == traj.n_layers
    assert surface.z.shape == (16, 16)


def test_rejects_unusable_input():
    with pytest.raises(ValueError, match="at least 2"):
        from_logprobs(STEPS[:1])
    with pytest.raises(ValueError, match="logprobs"):
        from_logprobs([{"token": "a", "logprobs": {}},
                       {"token": "b", "logprobs": {"b": -0.1}}])


def test_renormalises_overfull_reported_mass():
    """Provider rounding can report slightly more than 1.0 — renormalise
    rather than emit a negative residual bucket."""
    steps = [{"token": "a", "logprobs": {"a": math.log(0.8), "b": math.log(0.4)}},
             {"token": "b", "logprobs": {"b": math.log(0.6)}}]
    traj = from_logprobs(steps)
    resid = traj.vocab.index(RESIDUAL_TOKEN)
    np.testing.assert_allclose(traj.hidden.sum(axis=-1), 1.0, rtol=1e-5)
    assert traj.hidden[0, 0, resid] == pytest.approx(0.0, abs=1e-6)
    assert traj.meta["reported_mass"][0] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------ openai adapter
class _Obj:
    """Stand-in for the SDK's attribute-style response objects."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _openai_response(as_objects: bool):
    content = [
        {"token": " Paris", "logprob": math.log(0.7),
         "top_logprobs": [{"token": " Paris", "logprob": math.log(0.7)},
                          {"token": " Lyon", "logprob": math.log(0.1)}]},
        {"token": ".", "logprob": math.log(0.6),
         "top_logprobs": [{"token": ".", "logprob": math.log(0.6)},
                          {"token": ",", "logprob": math.log(0.2)}]},
    ]
    if not as_objects:
        return {"choices": [{"logprobs": {"content": content}}]}
    wrap = lambda d: _Obj(**{k: ([_Obj(**e) for e in v] if k == "top_logprobs" else v)
                             for k, v in d.items()})
    return _Obj(choices=[_Obj(logprobs=_Obj(content=[wrap(c) for c in content]))])


@pytest.mark.parametrize("as_objects", [False, True])
def test_openai_adapter_reads_dicts_and_objects(as_objects):
    traj = from_openai_logprobs(_openai_response(as_objects), model="gpt-4o")
    traj.validate()
    assert traj.n_layers == 2
    assert [s["token"] for s in traj.meta["generation"]["steps"]] == [" Paris", "."]
    assert traj.hidden[0, 0, traj.vocab.index(" Paris")] == pytest.approx(0.7, abs=1e-3)
    assert traj.meta["model"] == "gpt-4o"


def test_openai_adapter_rejects_a_response_without_logprobs():
    with pytest.raises(ValueError, match="logprobs"):
        from_openai_logprobs({"choices": [{"logprobs": {"content": []}}]})
    with pytest.raises(ValueError, match="choices"):
        from_openai_logprobs({"choices": []})
