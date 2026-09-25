"""Dose–response: one steering direction, a signed grid of doses, and matched controls.

`intervene.faithfulness` scores a steer at one magnitude. A single successful
magnitude says little: a large enough push along almost any direction moves
the readout, and a push that replaces the state rather than steering it can
force an output the model would never compute. docs/validity.md lists
dose-response curves among the checks a steer needs; this module runs them.

    sweep = dose_sweep(model, ["The capital of France is"], v, layer=6,
                       target=" Berlin", tokenizer=tok)
    sweep.to_json("sweep.json"); sweep.to_csv("sweep.csv")

A sweep injects dose · r_ℓ · v̂ at one layer, one forward pass per dose, and
reads the last token. Doses are relative to r_ℓ, the median residual norm at
the injection layer over positions t >= 1. Position 0 is left out because it
carries norms far above the rest in GPT-2- and Llama-family models, and a
reference dominated by one token would make every dose mean something else.
r_ℓ is measured once from the dose-0 captures and stored. The absolute push
‖δ‖ is the primary field of every point, and the relative dose is derived
from it. That matches `Faithfulness.scale`, which records ‖δ‖ too.

Above |dose| = 1 a push orthogonal to a typical-norm state rotates what the
next block reads by more than 45°. Those points are flagged
`replacement_regime`: they measure what happens when the state is largely
replaced, not steered.

Every number here is a measurement of one counterfactual. A curve that rises
with dose and clears its controls shows the push is *sufficient* to shift the
readout under these conditions. It does not identify the mechanism that
normally produces the behavior.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCHEMA = "mottled-dose-sweep/1"
POSITIONS = ("last", "all")
DEFAULT_GRID = (0.0,) + tuple(s * d for d in (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0)
                              for s in (-1.0, 1.0))
# Above this relative dose the push outweighs a typical state (see module doc).
REPLACEMENT_THRESHOLD = 1.0

# Per-point scalar fields, in CSV column order. `state_distance` (one value per
# readout layer) and `position_dose` (one per injected position) are not
# scalars and are laid out separately in the CSV.
_SCALARS = ("prompt_index", "kind", "seed", "sign", "dose", "delta_norm",
            "inject_layer", "replacement_regime", "state_distance_final", "kl",
            "target_logprob", "target_rank", "entropy", "incumbent_logprob",
            "cos_to_baseline", "norm_ratio", "v_component", "extrapolation")


# --------------------------------------------------------------- container
@dataclass
class DoseSweep:
    """The record of one sweep: what was done (`spec`), how (`analysis`,
    the `provenance.record` block), and one entry per measured point.

    A point is one (prompt, kind, seed, dose). `kind` is "steer", "random"
    (a random direction orthogonal to v, `seed` set), "shuffled" (a
    diff-of-means over shuffled group labels, `seed` set) or "direct_path"
    (the steer injected at the final layer). Dose 0 is recorded once per
    prompt, under "steer": every kind shares that capture.
    """

    spec: dict
    points: list[dict]
    analysis: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _plain({"schema": SCHEMA, "spec": self.spec, "analysis": self.analysis,
                       "notes": self.notes, "points": self.points})

    def to_json(self, path=None) -> str:
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n")
        return text

    @classmethod
    def from_dict(cls, data: dict) -> "DoseSweep":
        if data.get("schema") != SCHEMA:
            raise ValueError(f"expected schema {SCHEMA!r}, got {data.get('schema')!r}")
        return cls(spec=data["spec"], points=data["points"],
                   analysis=data.get("analysis"), notes=data.get("notes", []))

    @classmethod
    def from_json(cls, path) -> "DoseSweep":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_csv(self, path) -> None:
        """Tidy long form: one row per point per readout layer.

        The per-point scalars repeat on each of a point's rows, so the file
        filters and plots without a join. `position_dose` is a JSON list in
        one cell, since its length depends on the positions mode.
        """
        layers = self.spec["readout_layers"]
        with Path(path).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(_SCALARS) + ["layer", "state_distance", "position_dose"])
            for p in self.points:
                row = ["" if p[k] is None else p[k] for k in _SCALARS]
                pos = json.dumps(_plain(p["position_dose"]))
                for layer, dist in zip(layers, p["state_distance"]):
                    w.writerow(row + [layer, dist, pos])


def _plain(obj):
    """JSON-safe copy that refuses what it cannot convert.

    Not `statefile._jsonable`: that stringifies anything json rejects, so a
    numpy array would arrive as numpy's truncated repr, silently.
    """
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    raise TypeError(f"cannot serialize {type(obj).__name__} in a dose sweep")


# ------------------------------------------------------------------ inputs
def resolve_target(tokenizer, text: str) -> int:
    """The single token id `text` encodes to, or ValueError.

    A multi-token target is refused rather than cut to its first piece: the
    first piece's probability is not the word's, and a sweep that silently
    measured it would answer a different question. Pass the form the model
    would emit — GPT-2 predicts " Berlin" (leading space), not "Berlin".
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"target {text!r} is {len(ids)} tokens {list(ids)}; "
                         "a dose sweep needs exactly one")
    return int(ids[0])


def reference_norm(hiddens: list[np.ndarray], layer: int) -> tuple[float, int]:
    """Median ‖hidden[layer, t]‖ over t >= 1, pooled across captures.

    Returns (value, number of states). Position 0 is excluded (see the
    module docstring).
    """
    norms = np.concatenate([np.linalg.norm(np.asarray(h[layer, 1:], np.float64), axis=-1)
                            for h in hiddens])
    if norms.size == 0:
        raise ValueError("no positions after 0 to measure a reference norm; "
                         "prompts need at least two tokens")
    return float(np.median(norms)), int(norms.size)


def tl_hook_name(layer: int, n_layers: int) -> str:
    """TransformerLens name for Mottled's `hidden[layer]`.

    `hidden[l]` enters block l, so it is `blocks.l.hook_resid_pre`; the last
    captured state has no block after it and is the final block's output.
    """
    if layer == n_layers - 1:
        return f"blocks.{layer - 1}.hook_resid_post"
    return f"blocks.{layer}.hook_resid_pre"


def random_control(unit: np.ndarray, seed: int) -> np.ndarray:
    """A seeded random unit direction, orthogonal to `unit`.

    Orthogonal exactly, not just nearly: a random direction in high
    dimension is almost orthogonal anyway, and removing the rest makes the
    control's claim ("a push of equal size, not along v") literal.
    """
    r = np.random.default_rng(int(seed)).normal(size=unit.shape)
    r = r - (r @ unit) * unit
    return (r / np.linalg.norm(r)).astype(np.float32)


def push(dose: float, r_ref: float, direction: np.ndarray) -> np.ndarray:
    """The δ a point injects: dose · r_ℓ · direction, sign included.

    Every kind goes through this, controls too. `intervene._norm_matched_random`
    draws the same direction for +δ and −δ, so a negative dose there would
    get a positive control; here the sign is applied to the control as well.
    """
    return (float(dose) * float(r_ref) * np.asarray(direction, np.float64)).astype(np.float32)


def _shuffled_directions(pos: list, neg: list, layer: int, token: int,
                         n: int, seed: int) -> list[tuple[int, np.ndarray]]:
    """Diff-of-means directions over shuffled group labels.

    Label splits equal to the real one (either way round) are skipped, so
    each control is a contrast the data could have produced by chance. A
    2-vs-2 contrast has only 3 distinct splits in total; small groups yield
    few controls, and the caller records how many.
    """
    from intervene import direction_from_contrast

    pool, k = list(pos) + list(neg), len(pos)
    real = {frozenset(range(k)), frozenset(range(k, len(pool)))}
    out, seen = [], set()
    for attempt in range(50 * max(n, 1)):
        if len(out) >= n:
            break
        s = seed + attempt
        perm = np.random.default_rng(s).permutation(len(pool))
        split = frozenset(perm[:k].tolist())
        if split in real or split in seen:
            continue
        seen.add(split)
        d = direction_from_contrast([pool[i] for i in sorted(split)],
                                    [pool[i] for i in range(len(pool)) if i not in split],
                                    layer=layer, token=token, normalize=False)
        norm = float(np.linalg.norm(d))
        if norm > 1e-12:
            out.append((s, (d / norm).astype(np.float32)))
    return out


# ----------------------------------------------------------------- sweep
def _pass(model, tokenizer, prompt: str, top_k: int, device: str, dtype: str,
          state_edits: dict | None = None):
    """One forward pass. Dose 0 and every dose come through here, with the
    same flags, so the dose-0 reference is the same computation as the
    doses minus the edit. (`pipeline.run_intervention` captures its baseline
    and its branch through different paths, so it cannot serve here.)

    Logits are kept as float32: at the smallest doses float16 rounding
    would make KL move in steps and rank tie.
    """
    from capture import _run

    return _run(model, prompt, tokenizer=tokenizer, top_k=top_k, device=device,
                dtype=dtype, keep_logits=True, state_edits=state_edits,
                logits_dtype="float32")


def _edits(layer: int, delta: np.ndarray, token: int | None) -> dict:
    from intervene import Perturb, _compile

    state_edits, _ = _compile([Perturb(layer, delta, token=token)])
    return state_edits


def dose_sweep(model, prompts, direction, layer: int, target, *, tokenizer=None,
               grid=DEFAULT_GRID, positions: str = "last", n_random: int = 8,
               seed: int = 0, source: str = "other", source_detail: dict | None = None,
               contrast: tuple | None = None, contrast_token: int = -1,
               n_shuffled: int = 4, derivation_prompts=(), cfg=None,
               top_k: int = 5, device: str = "auto", dtype: str = "float32",
               mtj_dir=None) -> DoseSweep:
    """Sweep `direction` over `grid` at `layer`, with controls. Torch only.

    direction : the steering vector v, normalized here. When it arrives
        unnormalized (e.g. `direction_from_contrast(..., normalize=False)`),
        its norm is recorded as `raw_norm`: for a diff-of-means that is the
        group-mean separation at the layer, a yardstick for the doses.
    target : a token id, or text resolved by `resolve_target` (one token).
    grid : signed relative doses (‖δ‖ = |dose| · r_ℓ); 0 is always included.
    positions : "last" pushes the final token only; "all" pushes every
        position by the same δ, which is a different relative push at each
        (recorded per point as `position_dose`).
    source : "token" (from `direction_from_token`) adds a direct-path
        reference: the same δ injected at the final layer. With tied
        embeddings, pushing along a token's row raises its logit through the
        skip path largely by construction, so only the steer's excess over
        this reference is what the downstream blocks did with the push.
    contrast : (pos, neg) groups of StateTrajectory that derived v. Sets
        source to "contrast", adds their prompts to `derivation_prompts`
        and, with >= 2 runs per group, adds `n_shuffled` label-shuffled
        controls.
    derivation_prompts : prompts used to derive v. An evaluation prompt that
        appears here is refused: a steer tested on its own training prompt
        tests nothing beyond it.
    mtj_dir : if set, each steer point's trajectory is written there as a
        kind "trajectory" .mtj with its dose in `meta["dose"]`.

    Each forward pass is reduced to its metrics as soon as it returns: a
    capture carries the embedding matrix, and holding hundreds of them does
    not fit in memory.
    """
    import provenance
    from config import MarbleConfig
    from intervene import divergence
    from metrics import entropy, kl_divergence, logit_lens_rank
    from trajectory import StateTrajectory

    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompts:
        raise ValueError("dose_sweep needs at least one prompt")
    if positions not in POSITIONS:
        raise ValueError(f"positions must be one of {POSITIONS}, got {positions!r}")
    doses = sorted({float(d) for d in grid} | {0.0})
    if not all(np.isfinite(doses)):
        raise ValueError("grid doses must be finite")
    if isinstance(model, str):
        from capture import load_model

        model, tokenizer = load_model(model, device=device, dtype=dtype)
    if tokenizer is None:
        raise ValueError("a tokenizer is required when passing a model instance")
    if isinstance(target, str):
        target = resolve_target(tokenizer, target)
    target = int(target)

    v = np.asarray(direction, dtype=np.float64).ravel()
    raw_norm = float(np.linalg.norm(v))
    if raw_norm <= 1e-12:
        raise ValueError("direction has zero norm")
    unit = (v / raw_norm).astype(np.float32)

    derivation = list(derivation_prompts)
    notes: list[str] = []
    shuffled: list[tuple[int, np.ndarray]] = []
    if contrast is not None:
        pos, neg = (list(g) for g in contrast)
        source = "contrast"
        derivation += [t.meta.get("prompt") for t in pos + neg if t.meta.get("prompt")]
        if len(pos) >= 2 and len(neg) >= 2:
            shuffled = _shuffled_directions(pos, neg, layer, contrast_token, n_shuffled, seed)
            if len(shuffled) < n_shuffled:
                notes.append(f"only {len(shuffled)} distinct shuffled-label splits exist "
                             f"for groups of {len(pos)} and {len(neg)}; asked for {n_shuffled}")
        else:
            notes.append("no shuffled-label control: it needs >= 2 runs per group")
    overlap = sorted(set(prompts) & set(derivation))
    if overlap:
        raise ValueError(f"evaluation prompts also derived the direction: {overlap}")

    # Dose-0 captures, one per prompt, kept in reduced form only.
    base = []
    for prompt in prompts:
        traj = _pass(model, tokenizer, prompt, top_k, device, dtype)
        if traj.dim != unit.shape[0]:
            raise ValueError(f"direction has {unit.shape[0]} dims, model states have {traj.dim}")
        final_logits = np.asarray(traj.logits[-1, -1], dtype=np.float64)
        base.append(SimpleNamespace(
            hidden=traj.hidden, tokens=traj.tokens, meta=traj.meta,
            final_logits=final_logits, incumbent=int(final_logits.argmax()),
            vocab_token=(traj.vocab[target] if traj.vocab and target < len(traj.vocab) else None),
            incumbent_token=(traj.vocab[int(final_logits.argmax())] if traj.vocab else None)))
        if not 0 <= target < final_logits.shape[-1]:
            raise ValueError(f"target {target} outside the vocabulary")
        del traj
    n_layers = base[0].hidden.shape[0]
    if any(b.hidden.shape[0] != n_layers for b in base):
        raise ValueError("captures disagree on depth")
    layer = int(layer) % n_layers
    final = n_layers - 1
    readout = list(range(layer, n_layers))
    r_ref, n_ref = reference_norm([b.hidden for b in base], layer)

    # The observed range of v's component, per layer, over the reference
    # states (dose-0 positions t >= 1, plus the contrast runs that derived v).
    observed = [b.hidden for b in base]
    if contrast is not None:
        observed += [t.hidden for t in list(contrast[0]) + list(contrast[1])
                     if t.hidden.shape[0] == n_layers]

    def v_range(at: int) -> tuple[float, float]:
        comp = np.concatenate([np.asarray(h[at, 1:], np.float64) @ unit for h in observed])
        return float(comp.min()), float(comp.max())

    ranges = {layer: v_range(layer)}

    arms = [("steer", None, unit, layer)]
    arms += [("random", seed + k, random_control(unit, seed + k), layer) for k in range(n_random)]
    arms += [("shuffled", s, d, layer) for s, d in shuffled]
    if source == "token":
        if layer == final:
            notes.append("no direct-path reference: the injection layer is already the final layer")
        else:
            arms.append(("direct_path", None, unit, final))
            ranges[final] = v_range(final)
    token = -1 if positions == "last" else None
    if mtj_dir is not None:
        Path(mtj_dir).mkdir(parents=True, exist_ok=True)

    def measure(i: int, b, traj, kind, arm_seed, dose, delta_norm, inject) -> dict:
        ref = StateTrajectory(hidden=b.hidden, tokens=b.tokens)
        cur = StateTrajectory(hidden=traj.hidden, tokens=traj.tokens)
        profile = divergence(ref, cur, token=-1).profile.astype(np.float64)
        h0 = np.linalg.norm(np.asarray(b.hidden[:, -1], np.float64), axis=-1)
        dist = [float(profile[l] / max(h0[l], 1e-12)) for l in readout]
        logits = np.asarray(traj.logits[-1, -1], dtype=np.float64)
        rank, logprob = logit_lens_rank(traj, target)
        h_a = np.asarray(traj.hidden[inject, -1], np.float64)
        h_0 = np.asarray(b.hidden[inject, -1], np.float64)
        lo, hi = ranges[inject]
        vc = float(h_a @ unit)
        at = np.linalg.norm(np.asarray(b.hidden[inject], np.float64), axis=-1)
        pos_dose = (delta_norm / np.maximum(at, 1e-12) if token is None
                    else np.array([delta_norm / max(at[-1], 1e-12)]))
        return {
            "prompt_index": i, "kind": kind, "seed": arm_seed,
            "sign": int(np.sign(dose)), "dose": float(dose), "delta_norm": float(delta_norm),
            "inject_layer": int(inject),
            "replacement_regime": bool(abs(dose) > REPLACEMENT_THRESHOLD),
            "state_distance": dist, "state_distance_final": dist[-1],
            "kl": float(kl_divergence(b.final_logits, logits)),
            "target_logprob": logprob, "target_rank": rank,
            "entropy": float(entropy(logits)),
            "incumbent_logprob": logit_lens_rank(traj, b.incumbent)[1],
            "cos_to_baseline": float(h_a @ h_0 / max(np.linalg.norm(h_a) * np.linalg.norm(h_0), 1e-12)),
            "norm_ratio": float(np.linalg.norm(h_a) / max(np.linalg.norm(h_0), 1e-12)),
            "v_component": vc,
            "extrapolation": bool(vc < lo or vc > hi),
            "position_dose": pos_dose.astype(np.float64).tolist(),
        }

    def save(i, traj, dose, delta_norm):
        import statefile

        traj.meta["dose"] = {"dose": float(dose), "delta_norm": float(delta_norm),
                             "sign": int(np.sign(dose)), "inject_layer": layer,
                             "positions": positions, "direction_sha256": digest,
                             "reference_norm": r_ref, "prompt_index": i}
        name = f"p{i:02d}_dose{dose:+.5f}.mtj"
        statefile.save(traj, Path(mtj_dir) / name, include_embeddings=False)

    digest = direction_digest(unit)
    points = []
    for i, (prompt, b) in enumerate(zip(prompts, base)):
        # Dose 0 once per prompt, shared by every kind. It is a second pass
        # rather than the stored capture, so a nonzero distance here would
        # expose a forward pass that does not repeat itself.
        traj = _pass(model, tokenizer, prompt, top_k, device, dtype)
        points.append(measure(i, b, traj, "steer", None, 0.0, 0.0, layer))
        if mtj_dir is not None:
            save(i, traj, 0.0, 0.0)
        del traj
        for kind, arm_seed, d_unit, inject in arms:
            for dose in doses:
                if dose == 0.0:
                    continue
                delta = push(dose, r_ref, d_unit)
                delta_norm = abs(dose) * r_ref
                traj = _pass(model, tokenizer, prompt, top_k, device, dtype,
                             state_edits=_edits(inject, delta, token))
                points.append(measure(i, b, traj, kind, arm_seed, dose, delta_norm, inject))
                if kind == "steer" and mtj_dir is not None:
                    save(i, traj, dose, delta_norm)
                del traj

    spec = {
        "direction": {"source": source, "detail": source_detail or {}, "sha256": digest,
                      "raw_norm": raw_norm, "dim": int(unit.shape[0])},
        "derivation_prompts": derivation,
        "prompts": prompts,
        "inject_layer": layer,
        "inject_hook_tl": tl_hook_name(layer, n_layers),
        "n_layers": n_layers,
        "readout_layers": readout,
        "readout_token": "last",
        "positions": positions,
        "grid": {"dose": doses, "delta_norm": [abs(d) * r_ref for d in doses]},
        "reference_norm": {"value": r_ref, "layer": layer, "n_states": n_ref,
                           "statistic": "median ||hidden[layer, t]|| over t >= 1 "
                                        "of the dose-0 captures"},
        "replacement_threshold": REPLACEMENT_THRESHOLD,
        "target": {"id": target, "token": base[0].vocab_token},
        "incumbent": [{"id": b.incumbent, "token": b.incumbent_token} for b in base],
        "controls": {"random_seeds": [a[1] for a in arms if a[0] == "random"],
                     "shuffled_seeds": [s for s, _ in shuffled],
                     "direct_path_layer": final if any(a[0] == "direct_path" for a in arms)
                     else None},
        "v_range": {str(k): list(v) for k, v in ranges.items()},
        "logits_dtype": "float32",
    }
    if cfg is None:
        cfg = MarbleConfig(model=base[0].meta.get("model") or "unknown", device=device,
                           dtype=dtype, top_k=top_k, keep_logits=True,
                           capture_components=False, capture_attention=False,
                           use_cache=False)
    analysis = provenance.record(cfg, prompts=prompts, trajs=base)
    return DoseSweep(spec=spec, points=points, analysis=analysis, notes=notes)


def direction_digest(unit: np.ndarray) -> str:
    """sha256 of a direction's float32 little-endian bytes and shape, so a
    record names the exact vector it steered with (the pattern
    `provenance.sae_digest` uses for dictionaries)."""
    arr = np.ascontiguousarray(unit, dtype="<f4")
    h = hashlib.sha256()
    h.update(repr(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()
