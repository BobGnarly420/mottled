"""The Mottled pipeline: capture -> project -> density -> terrain -> paths.

Pure functions over a `MarbleConfig`, with no Streamlit and no Plotly, so a
scene can be built from a notebook, a script, the CLI, or the capture server
exactly as the explorer builds it. `run_pipeline` is the single-prompt path;
`run_scene` joint-projects several prompts into one shared space so their
trajectories are comparable; `run_intervention` runs the counterfactual pair.

Split out of `ui.py` (which re-exports everything here, so
`from ui import run_pipeline` keeps working) — the module had grown to carry
the pipeline, two renderers and the app shell at once.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import cache as cache_mod
import compare as compare_mod
import density as density_mod
import projection as projection_mod
import sae as sae_mod
import terrain as terrain_mod
import trajectory as trajectory_mod
from capture import capture, generate_and_capture
from config import MarbleConfig
from trajectory import StateTrajectory

# Pipeline: capture -> project -> compute_density -> mesh -> trajectory
# --------------------------------------------------------------------------
def run_pipeline(cfg: MarbleConfig, prompt: str, model=None, tokenizer=None) -> dict:
    """Execute the full Mottled pipeline and return every artifact.

    `model`/`tokenizer` may be pre-loaded objects (the UI caches them); when
    omitted, `cfg.model` is loaded by name ("synthetic" needs no loading).
    """
    disk = cache_mod.DiskCache(cfg.cache_dir) if cfg.use_cache else None
    key = cache_mod.make_key("pipeline-v6", prompt, cfg.model, cfg.projection,
                             cfg.density, cfg.top_k, cfg.n_components, cfg.seed,
                             cfg.grid_size, cfg.smooth_sigma, cfg.height_scale,
                             cfg.invert_terrain, cfg.trajectory_mode,
                             cfg.trajectory_token, cfg.frames_per_layer,
                             cfg.capture_components, cfg.capture_attention,
                             cfg.density_bootstrap, cfg.generate_tokens,
                             cfg.generate_temperature)
    if disk is not None and (hit := disk.get(key)) is not None:
        return hit

    traj = _capture_with(cfg, prompt, model=model, tokenizer=tokenizer)

    coords, projector = projection_mod.project(
        traj.hidden, method=cfg.projection, n_components=cfg.n_components, seed=cfg.seed
    )
    quality = projection_mod.projection_quality(traj.hidden, coords, projector)
    landscape = density_mod.compute_density(
        coords, method=cfg.density, grid_size=cfg.grid_size, padding=cfg.grid_padding,
        bootstrap=cfg.density_bootstrap, seed=cfg.seed,
    )
    surface = terrain_mod.mesh(
        landscape, smooth_sigma=cfg.smooth_sigma,
        height_scale=cfg.height_scale, invert=cfg.invert_terrain,
    )

    flat = trajectory_mod.extract(coords, traj.tokens, mode=cfg.trajectory_mode,
                                  token=cfg.trajectory_token)
    trajectories = [
        replace(t, points=terrain_mod.drape(surface, t.points, lift=cfg.marble_lift))
        for t in flat
    ]
    fine_paths = [trajectory_mod.densify(t.points, cfg.frames_per_layer) for t in trajectories]

    result = {
        "prompt": prompt,
        "traj": traj,
        "coords": coords,
        "projector": projector,
        "quality": quality,
        "landscape": landscape,
        "mesh": surface,
        "trajectories": trajectories,
        "fine_paths": fine_paths,
    }
    if disk is not None:
        disk.put(key, result)
    return result


def _capture_with(cfg: MarbleConfig, prompt: str, model=None, tokenizer=None) -> StateTrajectory:
    """One validated capture under the config's capture knobs.

    With `cfg.generate_tokens > 0` the capture decodes that many tokens
    first (greedy, or seeded sampling at `cfg.generate_temperature`) and
    returns the trajectory of prompt + continuation, decode record in
    `meta["generation"]`."""
    kwargs = dict(
        tokenizer=tokenizer,
        top_k=cfg.top_k,
        device=cfg.device,
        dtype=cfg.dtype,
        keep_logits=cfg.keep_logits,
        capture_components=cfg.capture_components,
        capture_attention=cfg.capture_attention,
    )
    target = model if model is not None else cfg.model
    if cfg.generate_tokens > 0:
        traj = generate_and_capture(target, prompt,
                                    max_new_tokens=cfg.generate_tokens,
                                    temperature=cfg.generate_temperature,
                                    seed=cfg.seed, **kwargs)
    else:
        traj = capture(target, prompt, **kwargs)
    traj.validate()
    return traj


def _assemble_scene(cfg: MarbleConfig, trajs: list[StateTrajectory]) -> dict:
    """Shared multi-run assembly: joint projection, one terrain from the
    union of all runs' states, draped trajectories, comparisons vs run 0."""
    coords_list, projector = projection_mod.project_joint(
        [t.hidden for t in trajs],
        method=cfg.projection, n_components=cfg.n_components, seed=cfg.seed,
    )
    quality_list = [
        projection_mod.projection_quality(t.hidden, c, projector)
        for t, c in zip(trajs, coords_list)
    ]
    union = np.concatenate([c.reshape(-1, cfg.n_components) for c in coords_list])
    landscape = density_mod.compute_density(
        union, method=cfg.density, grid_size=cfg.grid_size, padding=cfg.grid_padding,
        bootstrap=cfg.density_bootstrap, seed=cfg.seed,
    )
    surface = terrain_mod.mesh(
        landscape, smooth_sigma=cfg.smooth_sigma,
        height_scale=cfg.height_scale, invert=cfg.invert_terrain,
    )

    trajectories_list, fine_paths_list = [], []
    for traj, coords in zip(trajs, coords_list):
        flat = trajectory_mod.extract(coords, traj.tokens, mode=cfg.trajectory_mode,
                                      token=cfg.trajectory_token)
        trajectories = [
            replace(t, points=terrain_mod.drape(surface, t.points, lift=cfg.marble_lift))
            for t in flat
        ]
        trajectories_list.append(trajectories)
        fine_paths_list.append(
            [trajectory_mod.densify(t.points, cfg.frames_per_layer) for t in trajectories])

    # `compare` is a layer-for-layer measurement, so it is only defined when
    # the runs have the same depth. Prompts through one model always do;
    # different *models* generally do not (GPT-2's 13 layers vs
    # DistilGPT-2's 7), and there the depth-normalised
    # `crossmodel.compare_models` is the right instrument instead — so the
    # pairwise table is simply absent rather than fabricated.
    comparisons = []
    for t, c in zip(trajs[1:], coords_list[1:]):
        if t.n_layers == trajs[0].n_layers and t.dim == trajs[0].dim:
            comparisons.append(compare_mod.compare(trajs[0], t, coords_list[0], c))

    result = {
        "trajs": trajs,
        "coords_list": coords_list,
        "projector": projector,
        "quality_list": quality_list,
        "landscape": landscape,
        "mesh": surface,
        "trajectories_list": trajectories_list,
        "fine_paths_list": fine_paths_list,
        "comparisons": comparisons,
        # run-0 view (the run_pipeline keys)
        "traj": trajs[0],
        "coords": coords_list[0],
        "quality": quality_list[0],
        "trajectories": trajectories_list[0],
        "fine_paths": fine_paths_list[0],
    }
    if len(trajs) == 2:  # the run_compare aliases
        result.update({
            "traj_b": trajs[1],
            "coords_b": coords_list[1],
            "trajectories_b": trajectories_list[1],
            "fine_paths_b": fine_paths_list[1],
        })
        if comparisons:  # absent when the two runs' depths differ (see above)
            result["comparison"] = comparisons[0]
    return result


def run_scene(cfg: MarbleConfig, prompts: list[str], model=None, tokenizer=None) -> dict:
    """Multi-prompt scene: capture every prompt, joint-project into ONE
    shared space, build one terrain from the union of all states, and
    compare each run against the first.

    Returns per-run lists ("trajs", "coords_list", "trajectories_list",
    "fine_paths_list", "comparisons") plus the `run_pipeline` keys for run 0
    and, for exactly two prompts, the `run_compare` aliases.
    """
    if not prompts:
        raise ValueError("run_scene needs at least one prompt")
    disk = cache_mod.DiskCache(cfg.cache_dir) if cfg.use_cache else None
    key = cache_mod.make_key("scene-v4", prompts, cfg.model, cfg.projection,
                             cfg.density, cfg.top_k, cfg.n_components, cfg.seed,
                             cfg.grid_size, cfg.smooth_sigma, cfg.height_scale,
                             cfg.invert_terrain, cfg.trajectory_mode,
                             cfg.trajectory_token, cfg.frames_per_layer,
                             cfg.capture_components, cfg.capture_attention,
                             cfg.density_bootstrap, cfg.generate_tokens,
                             cfg.generate_temperature)
    if disk is not None and (hit := disk.get(key)) is not None:
        return hit

    trajs = [_capture_with(cfg, p, model=model, tokenizer=tokenizer) for p in prompts]
    result = {"prompts": list(prompts), "prompt": prompts[0], **_assemble_scene(cfg, trajs)}
    if len(prompts) == 2:
        result["prompt_b"] = prompts[1]
    if disk is not None:
        disk.put(key, result)
    return result


def degraded_note(meta: dict) -> str | None:
    """The banner a producer that could not see inside the model must carry.

    Returns None for a full capture. For a degraded one (e.g. the API-logprob
    producer) it names the backend, lists what that backend cannot observe,
    says which axis is actually animated, and flags a truncated — therefore
    under-counted — entropy. Pure, so the wording is testable without
    driving the app.
    """
    if not meta.get("degraded"):
        return None
    absent = ", ".join(meta.get("absent") or []) or "internal state"
    note = (f"**Degraded capture** — this run came from `{meta.get('backend')}`, "
            f"which cannot observe: {absent}. The animated axis is "
            f"**{meta.get('axis', 'decode')}**, not depth, and the geometry is "
            f"the model's *output distribution*, not its residual stream.")
    if meta.get("entropy_is_lower_bound"):
        note += (" Reported entropy is a **lower bound** — the API truncates "
                 "its distribution, so the true spread is at least this wide.")
    return note


def attach_features(result: dict, sae, source: str | None = None,
                    hook: str | None = None) -> dict:
    """Attach an SAE feature layer to a pipeline/scene result, for export.

    Per run: the dominant feature per state (top-1 id + activation) and the
    dictionary's **measured fit** (`sae.fit_report`), so a viewer never shows
    a feature without its calibration attached. `source`/`hook` say where the
    dictionary came from. Mutates and returns `result` (adds
    "features_list", aligned with the runs); `statefile.save_scene` carries
    it into the `.mtj` additively.
    """
    trajs = result.get("trajs") or [result["traj"]]
    feats = []
    for traj in trajs:
        acts = sae.encode(traj.hidden)                      # (L, T, F)
        fit = sae_mod.fit_report(sae, traj)
        feats.append({
            "source": source, "hook": hook,
            "best_layer": fit.best_layer,
            "recon_error": fit.recon_error,
            "top_id": acts.argmax(axis=-1).astype(np.int32),
            "top_act": acts.max(axis=-1).astype(np.float32),
        })
    result["features_list"] = feats
    return result


def attach_inspector(result: dict, n_neighbors: int = 5) -> dict:
    """Precompute the inspector layers a scene file cannot recover on its own.

    The explorer can show semantic neighbors and the attention/MLP split
    because it still holds the embedding matrix (V x D) and the residual
    components (2 x (L-1) x T x D) in memory. A scene file carries neither —
    they are orders of magnitude larger than everything else in it — so the
    web viewer has been the lesser surface. Both reduce to something tiny if
    they are resolved *before* export:

    * neighbors  -> the k nearest vocabulary tokens per state, as indices into
                    a compact table of just the strings that actually appear
    * components -> the attention/MLP share per state, which is two numbers,
                    not two D-dimensional vectors

    Mutates and returns `result` (adds "inspector_list", aligned with the
    runs); `statefile.save_scene` carries it into the `.mtj` additively.
    Runs lacking the source data are skipped individually rather than
    failing the export.
    """
    from neighbors import TokenNeighbors

    trajs = result.get("trajs") or [result["traj"]]
    out = []
    for traj in trajs:
        entry: dict = {}

        if traj.embedding_matrix is not None and traj.vocab:
            tn = TokenNeighbors(traj.embedding_matrix, traj.vocab)
            flat = traj.hidden.reshape(-1, traj.dim)
            table: dict[str, int] = {}
            idx = np.zeros((len(flat), n_neighbors), dtype=np.int32)
            sim = np.zeros((len(flat), n_neighbors), dtype=np.float32)
            for i, vec in enumerate(flat):
                for j, (tok, s) in enumerate(tn.nearest(vec, k=n_neighbors)):
                    idx[i, j] = table.setdefault(tok, len(table))
                    sim[i, j] = s
            shape = (traj.n_layers, traj.n_tokens, n_neighbors)
            entry["neighbor_tokens"] = list(table)
            entry["neighbor_idx"] = idx.reshape(shape)
            entry["neighbor_sim"] = sim.reshape(shape)

        comps = traj.components
        if comps is not None and {"attn", "mlp"} <= set(comps):
            a = np.linalg.norm(comps["attn"].astype(np.float64), axis=-1)
            m = np.linalg.norm(comps["mlp"].astype(np.float64), axis=-1)
            total = np.maximum(a + m, 1e-12)
            entry["component_shares"] = np.stack(
                [a / total, m / total], axis=-1).astype(np.float32)

        out.append(entry)
    result["inspector_list"] = out
    return result


def run_model_scene(cfg: MarbleConfig, prompt: str, models: list,
                    loaded: dict | None = None) -> dict:
    """One prompt, several **models**, on one terrain.

    Different models share no hidden space — different widths, different
    depths, different tokenizers — so the scene is assembled in *readout
    space* (`crossmodel.readout_space`): every state becomes the next-token
    distribution it predicts over the vocabulary all the models share, plus a
    visible bucket for the mass spent outside it. That is a real shared
    coordinate system, so joint projection, the terrain, and every viewer
    work unchanged.

    `models` are names (or "synthetic"); `loaded` optionally maps a name to a
    preloaded (model, tokenizer) pair. The result carries the usual scene
    keys plus `model_names`, `shared_vocab`, and `model_comparisons` (each
    model measured against the first).

    Readout space says what the models *do*. For how they *represent* —
    which layer of B matches layer l of A — use `crossmodel.layer_similarity`,
    which needs no shared space but does need matching tokenization.
    """
    import crossmodel

    if not models:
        raise ValueError("run_model_scene needs at least one model")
    loaded = loaded or {}
    raw = []
    for name in models:
        model, tokenizer = loaded.get(name, (None, None))
        raw.append(_capture_with(replace(cfg, model=name), prompt,
                                 model=model, tokenizer=tokenizer))

    trajs, vocab = crossmodel.readout_space(raw)
    for traj, name in zip(trajs, models):
        traj.meta["model"] = name
    result = {"prompts": [f"{m}: {prompt}" for m in models], "prompt": prompt,
              "model_names": list(models), "shared_vocab": len(vocab),
              "source_trajs": raw, "space": "readout",
              **_assemble_scene(cfg, trajs)}
    result["model_comparisons"] = [crossmodel.compare_models(raw[0], other)
                                   for other in raw[1:]]
    return result


def run_compare(cfg: MarbleConfig, prompt_a: str, prompt_b: str,
                model=None, tokenizer=None) -> dict:
    """A/B pipeline: a two-prompt `run_scene` (kept as the pairwise API)."""
    return run_scene(cfg, [prompt_a, prompt_b], model=model, tokenizer=tokenizer)


def run_intervention(cfg: MarbleConfig, prompt: str, interventions: list,
                     model, tokenizer, target_id: int | None = None) -> dict:
    """Interactive patching: baseline capture vs perturb-and-replay branch.

    Runs the same prompt twice — once untouched, once under `interventions`
    (intervene.Intervention edits) — then assembles the two counterfactual
    runs as a scene: shared projection, one terrain, comparison, plus an
    `intervene.divergence` readout.  Requires a torch model; results are not
    cached (the edit space is unbounded).

    When `target_id` is given for a single directional steer, also attaches a
    `Faithfulness` readout: the run measures a norm-matched *random* control so
    the UI can say how much of the effect is the steering direction rather than
    the perturbation's raw size. When the baseline additionally carries the
    residual decomposition (`cfg.capture_components`), a `PersistenceProfile`
    is attached too — the same faithfulness effect read at every layer from
    the edit down, paired with the attention/MLP write shares.
    """
    from intervene import (divergence, intervene, persistence_profile,
                           score_against_control)

    baseline = _capture_with(cfg, prompt, model=model, tokenizer=tokenizer)
    branch = intervene(model, prompt, interventions, tokenizer=tokenizer,
                       top_k=cfg.top_k, device=cfg.device, dtype=cfg.dtype,
                       keep_logits=cfg.keep_logits)
    branch.validate()

    result = {"prompts": [prompt, prompt], "prompt": prompt,
              "prompt_b": "patched: " + ", ".join(iv.describe() for iv in interventions),
              **_assemble_scene(cfg, [baseline, branch])}
    result["divergence"] = divergence(baseline, branch)

    if (target_id is not None and baseline.logits is not None
            and len(interventions) == 1 and interventions[0].kind == "perturb"
            and interventions[0].vector is not None):
        iv = interventions[0]
        tok = iv.token if iv.token is not None else -1
        result["faithfulness"] = score_against_control(
            model, prompt, baseline, branch, iv.vector, iv.layer, int(target_id),
            token=tok, tokenizer=tokenizer, seed=cfg.seed, device=cfg.device,
            dtype=cfg.dtype, top_k=cfg.top_k)
        if (baseline.components is not None
                and {"attn", "mlp"} <= set(baseline.components)):
            result["persistence"] = persistence_profile(
                model, prompt, iv.vector, iv.layer, int(target_id),
                tokenizer=tokenizer, token=tok,
                scale=float(np.linalg.norm(iv.vector)),
                baseline=baseline, branch=branch, seed=cfg.seed,
                device=cfg.device, dtype=cfg.dtype, top_k=cfg.top_k)
    return result

