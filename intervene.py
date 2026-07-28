"""Causal intervention: perturb-and-replay over a resumable forward pass.

Observation shows what a system *did*; intervention shows what it *would have
done*.  An intervention edits the process that generates a Trace and produces a
new **counterfactual StateTrajectory** — real DATA (the model genuinely ran and
produced it) that flows through the same projection / measurement / renderer
stack as the original.  The four layers never collapse: the counterfactual
trace is data, its distance from the baseline is a measurement, and any reading
of that ("commitment point") is a separate, defeasible narrative.

Supported edits (transformer backend):

    Perturb(layer, delta)    push a state by a vector  — the grab gesture
    SetState(layer, value)   overwrite a state
    InjectNoise(layer, scale) add Gaussian noise (seeded, reproducible)
    FreezeLayer(block)       skip a block's update (residual passes through)

All are applied by write-hooks during a single forward pass (`capture._run`),
so the model resumes from the edited residual and every downstream layer sees
the consequence.  Interventions require a real (torch) model; the synthetic
backend is analytic and not resumable.

Fork semantics: `intervene(...)` returns a branch; `baseline` (an unperturbed
capture) plus one or more branches are what the UI overlays and diffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from capture import _run
from trajectory import StateTrajectory


# --------------------------------------------------------------------------
# Intervention algebra
# --------------------------------------------------------------------------
@dataclass
class Intervention:
    """A single edit to the process at one point of the trajectory.

    `layer` indexes the state for state edits (0 = embeddings, k = output of
    block k-1) and the *block* for freezes.  `token=None` edits all positions.
    """

    layer: int
    kind: str                                   # perturb | set | noise | freeze
    token: int | None = None
    vector: np.ndarray | None = None            # delta (perturb) or value (set)
    scale: float = 0.0                          # noise magnitude
    seed: int | None = None

    def describe(self) -> str:
        where = "all tokens" if self.token is None else f"token {self.token}"
        return f"{self.kind}@layer{self.layer}[{where}]"


def Perturb(layer: int, delta, token: int | None = None) -> Intervention:
    """Add `delta` (a vector in state space) to the state at `layer`."""
    return Intervention(layer, "perturb", token=token, vector=np.asarray(delta, dtype=np.float32))


def SetState(layer: int, value, token: int | None = None) -> Intervention:
    """Replace the state at `layer` with `value`."""
    return Intervention(layer, "set", token=token, vector=np.asarray(value, dtype=np.float32))


def InjectNoise(layer: int, scale: float, token: int | None = None, seed: int | None = 0) -> Intervention:
    """Add zero-mean Gaussian noise of std `scale` to the state at `layer`."""
    return Intervention(layer, "noise", token=token, scale=float(scale), seed=seed)


def FreezeLayer(block: int) -> Intervention:
    """Skip block `block`'s update: hidden[block+1] := hidden[block]."""
    return Intervention(block, "freeze")


# --------------------------------------------------------------------------
# Torch edit closures (built lazily so numpy-only imports stay clean)
# --------------------------------------------------------------------------
def _make_edit(interventions: list[Intervention]):
    """Compose the state edits targeting one layer into a single fn(hidden)."""
    import torch

    def fn(hidden):
        h = hidden.clone()
        tok = slice(None)
        for iv in interventions:
            t = slice(None) if iv.token is None else iv.token
            if iv.kind == "perturb":
                d = torch.as_tensor(iv.vector, dtype=h.dtype, device=h.device)
                h[:, t] = h[:, t] + d
            elif iv.kind == "set":
                v = torch.as_tensor(iv.vector, dtype=h.dtype, device=h.device)
                h[:, t] = v
            elif iv.kind == "noise":
                g = torch.Generator(device=h.device)
                if iv.seed is not None:
                    g.manual_seed(int(iv.seed))
                shape = h[:, t].shape
                noise = torch.randn(shape, generator=g, dtype=h.dtype, device=h.device)
                h[:, t] = h[:, t] + iv.scale * noise
        return h

    return fn


def _compile(interventions: list[Intervention]) -> tuple[dict, set]:
    """Split interventions into per-layer state edits and frozen block ids."""
    frozen = {iv.layer for iv in interventions if iv.kind == "freeze"}
    by_layer: dict[int, list[Intervention]] = {}
    for iv in interventions:
        if iv.kind != "freeze":
            by_layer.setdefault(iv.layer, []).append(iv)
    state_edits = {layer: _make_edit(ivs) for layer, ivs in by_layer.items()}
    return state_edits, frozen


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def intervene(
    model,
    prompt: str,
    interventions: list[Intervention],
    tokenizer=None,
    top_k: int = 5,
    device: str = "auto",
    dtype: str = "float32",
    keep_logits: bool = True,
) -> StateTrajectory:
    """Run a counterfactual forward pass under `interventions`.

    Returns a StateTrajectory whose `meta["interventions"]` records the edits
    and `meta["counterfactual"]` is True.  Requires a torch model (a HF
    instance with `tokenizer`, or a hub name); "synthetic" is not resumable.
    """
    if isinstance(model, str) and model == "synthetic":
        raise ValueError("the synthetic backend is analytic and not resumable; "
                         "interventions require a torch model")
    if not interventions:
        raise ValueError("no interventions given; use capture() for a baseline pass")

    state_edits, frozen = _compile(interventions)
    return _run(
        model, prompt, tokenizer=tokenizer, top_k=top_k, device=device, dtype=dtype,
        keep_logits=keep_logits, state_edits=state_edits, frozen_blocks=frozen,
        extra_meta={
            "counterfactual": True,
            "interventions": [iv.describe() for iv in interventions],
        },
    )


# --------------------------------------------------------------------------
# Branch comparison
# --------------------------------------------------------------------------
@dataclass
class Divergence:
    """How and where a branch separates from its baseline."""

    profile: np.ndarray          # (L,) state-space distance per layer
    onset: int                   # first layer whose distance exceeds the band
    readout_changed: int | None  # first layer whose top-1 prediction differs, or None
    token: int


def divergence(baseline: StateTrajectory, branch: StateTrajectory,
               token: int = -1, rel: float = 0.05) -> Divergence:
    """Layerwise separation of `branch` from `baseline` for one unit.

    `profile[l]` is the state-space (L2) distance between the two trajectories
    at layer l.  `onset` is the first layer whose distance exceeds `rel` times
    the maximum separation — a rough "when did the intervention start to
    matter" that is *deliberately* just a measurement, not a claimed cause.
    `readout_changed` is the first layer where the top-1 logit-lens token
    differs, when logits are available.
    """
    if baseline.hidden.shape != branch.hidden.shape:
        raise ValueError("baseline and branch must share shape (same prompt/model)")
    t = int(token) % baseline.n_tokens
    profile = np.linalg.norm(baseline.hidden[:, t] - branch.hidden[:, t], axis=-1)

    peak = float(profile.max())
    if peak <= 0:
        onset = len(profile) - 1
    else:
        exceed = np.flatnonzero(profile > rel * peak)
        onset = int(exceed[0]) if exceed.size else len(profile) - 1

    readout_changed = None
    if baseline.logits is not None and branch.logits is not None:
        ba = baseline.logits[:, t].astype(np.float32).argmax(axis=-1)
        br = branch.logits[:, t].astype(np.float32).argmax(axis=-1)
        diff = np.flatnonzero(ba != br)
        readout_changed = int(diff[0]) if diff.size else None

    return Divergence(profile=profile.astype(np.float32), onset=onset,
                      readout_changed=readout_changed, token=t)


# --------------------------------------------------------------------------
# Steering directions — deltas from data, not hand-picked numbers
# --------------------------------------------------------------------------
# A `Perturb` takes a raw delta vector; where that vector comes from is what
# separates a principled steer from a magic number. These helpers derive the
# direction from the model's own representations: a token's embedding
# direction, or a diff-of-means contrast between two sets of runs. The caller
# still chooses the magnitude (how far to push), but the *direction* is
# measured, so the intervention is reproducible and interpretable.

def _as_trajectories(x) -> list[StateTrajectory]:
    if isinstance(x, StateTrajectory):
        return [x]
    xs = list(x)
    if not xs:
        raise ValueError("expected at least one StateTrajectory")
    return xs


def direction_from_token(traj: StateTrajectory, token_id: int,
                         normalize: bool = True) -> np.ndarray:
    """Unit direction of a vocabulary token's embedding (the readout axis).

    For a model with tied embeddings (GPT-2) this row is also the unembedding
    direction, so pushing a late state along it moves the logit lens toward
    that token — the mechanism behind the README's " Berlin" example, here as
    a named, reusable helper instead of an inline expression.
    """
    if traj.embedding_matrix is None:
        raise ValueError("trajectory has no embedding_matrix; capture with a "
                         "torch model (keep_logits=True) to get one")
    d = np.asarray(traj.embedding_matrix[int(token_id)], dtype=np.float32)
    if normalize:
        d = d / max(float(np.linalg.norm(d)), 1e-12)
    return d


def direction_from_contrast(pos, neg, layer: int, token: int = -1,
                            normalize: bool = True) -> np.ndarray:
    """Diff-of-means steering direction between two groups of runs.

    `pos` and `neg` are each a StateTrajectory or a list of them (e.g. runs
    that do vs. don't exhibit a behavior). The direction is the difference of
    the groups' mean hidden state at `(layer, token)` — the classic
    activation-steering construction. Backend-agnostic: it reads only
    `hidden`, so it works on synthetic runs too.
    """
    pos_t, neg_t = _as_trajectories(pos), _as_trajectories(neg)

    def _mean(group: list[StateTrajectory]) -> np.ndarray:
        vecs = []
        for t in group:
            tok = int(token) % t.n_tokens
            vecs.append(np.asarray(t.hidden[int(layer), tok], dtype=np.float64))
        return np.mean(vecs, axis=0)

    d = (_mean(pos_t) - _mean(neg_t)).astype(np.float32)
    if normalize:
        d = d / max(float(np.linalg.norm(d)), 1e-12)
    return d


# --------------------------------------------------------------------------
# Faithfulness — how much of a steer's effect is the *direction*, not the norm
# --------------------------------------------------------------------------
@dataclass
class Faithfulness:
    """Effect-size of a targeted steer against a norm-matched random control.

    A perturbation of any direction, if large enough, disturbs the readout;
    the question that matters is whether *this* direction specifically drives
    the intended `target`. We answer it by comparing the steer to a random
    delta of identical norm:

        steer_shift   — logit-lens shift toward `target` under the steer
        control_shift — the same shift under a random delta of equal norm
        effect        — steer_shift - control_shift (direction-specific gain)

    A large positive `effect` means the direction, not merely the magnitude,
    is what moved the model. `steer_hits_target` reports whether the steered
    branch's final top-1 actually became the target. This is a measurement,
    not a mechanistic proof — it quantifies a counterfactual, it does not
    isolate a circuit.
    """

    target: int
    target_token: str | None
    steer_shift: float
    control_shift: float
    effect: float
    steer_hits_target: bool
    scale: float
    token: int


def target_logit_shift(baseline: StateTrajectory, branch: StateTrajectory,
                       target: int, token: int = -1) -> float:
    """Final-layer logit-lens shift toward `target` from baseline to branch."""
    if baseline.logits is None or branch.logits is None:
        raise ValueError("both trajectories need logits (keep_logits=True)")
    t = int(token) % baseline.n_tokens
    base = float(baseline.logits[-1, t, int(target)])
    new = float(branch.logits[-1, t, int(target)])
    return new - base


def faithfulness(model, prompt: str, layer: int, direction: np.ndarray,
                 target: int, *, token: int = -1, scale: float = 1.0,
                 tokenizer=None, baseline: StateTrajectory | None = None,
                 seed: int = 0, device: str = "auto", dtype: str = "float32",
                 top_k: int = 5) -> Faithfulness:
    """Score a targeted steer against a norm-matched random control.

    Normalizes `direction`, applies `scale * direction` as a `Perturb` at
    `(layer, token)`, then applies a random delta of the *same* L2 norm, and
    compares how far each moves the logit lens toward `target`. Requires a
    torch model. `baseline` is captured if not supplied.
    """
    from capture import capture

    direction = np.asarray(direction, dtype=np.float32)
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    delta = (scale * unit).astype(np.float32)

    if baseline is None:
        baseline = capture(model, prompt, tokenizer=tokenizer, top_k=top_k,
                           device=device, dtype=dtype, keep_logits=True)

    branch = intervene(model, prompt, [Perturb(layer, delta, token=token)],
                       tokenizer=tokenizer, top_k=top_k, device=device,
                       dtype=dtype, keep_logits=True)

    rng = np.random.default_rng(seed)
    r = rng.normal(size=delta.shape).astype(np.float32)
    r *= float(np.linalg.norm(delta)) / max(float(np.linalg.norm(r)), 1e-12)
    control = intervene(model, prompt, [Perturb(layer, r, token=token)],
                        tokenizer=tokenizer, top_k=top_k, device=device,
                        dtype=dtype, keep_logits=True)

    steer_shift = target_logit_shift(baseline, branch, target, token)
    control_shift = target_logit_shift(baseline, control, target, token)
    t = int(token) % baseline.n_tokens
    hits = bool(int(np.asarray(branch.logits[-1, t]).argmax()) == int(target))
    target_token = None
    if baseline.vocab is not None and 0 <= int(target) < len(baseline.vocab):
        target_token = baseline.vocab[int(target)]

    return Faithfulness(
        target=int(target), target_token=target_token,
        steer_shift=steer_shift, control_shift=control_shift,
        effect=steer_shift - control_shift, steer_hits_target=hits,
        scale=float(scale), token=t,
    )
