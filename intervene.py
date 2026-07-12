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
