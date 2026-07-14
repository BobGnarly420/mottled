"""Mottled UI: interactive latent trajectory explorer.

Run with:  streamlit run ui.py

The module separates a pure pipeline (`run_pipeline`) and a pure Plotly
renderer (`render`) from the Streamlit shell, so both are importable and
testable without a browser.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import plotly.graph_objects as go

import cache as cache_mod
import compare as compare_mod
import density as density_mod
import metrics as metrics_mod
import sae as sae_mod
import statefile as statefile_mod
import projection as projection_mod
import terrain as terrain_mod
import trajectory as trajectory_mod
from capture import capture
from config import (
    DEFAULT_PROMPT,
    DENSITY_CHOICES,
    MODEL_CHOICES,
    PROJECTION_CHOICES,
    TRAJECTORY_MODES,
    MarbleConfig,
)
from neighbors import TokenNeighbors
from trajectory import StateTrajectory


# --------------------------------------------------------------------------
# Pipeline: capture -> project -> compute_density -> mesh -> trajectory
# --------------------------------------------------------------------------
def run_pipeline(cfg: MarbleConfig, prompt: str, model=None, tokenizer=None) -> dict:
    """Execute the full Mottled pipeline and return every artifact.

    `model`/`tokenizer` may be pre-loaded objects (the UI caches them); when
    omitted, `cfg.model` is loaded by name ("synthetic" needs no loading).
    """
    disk = cache_mod.DiskCache(cfg.cache_dir) if cfg.use_cache else None
    key = cache_mod.make_key("pipeline-v3", prompt, cfg.model, cfg.projection,
                             cfg.density, cfg.top_k, cfg.n_components, cfg.seed,
                             cfg.grid_size, cfg.smooth_sigma, cfg.height_scale,
                             cfg.invert_terrain, cfg.trajectory_mode,
                             cfg.trajectory_token, cfg.frames_per_layer,
                             cfg.capture_components, cfg.capture_attention)
    if disk is not None and (hit := disk.get(key)) is not None:
        return hit

    traj = _capture_with(cfg, prompt, model=model, tokenizer=tokenizer)

    coords, _ = projection_mod.project(
        traj.hidden, method=cfg.projection, n_components=cfg.n_components, seed=cfg.seed
    )
    landscape = density_mod.compute_density(
        coords, method=cfg.density, grid_size=cfg.grid_size, padding=cfg.grid_padding
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
        "landscape": landscape,
        "mesh": surface,
        "trajectories": trajectories,
        "fine_paths": fine_paths,
    }
    if disk is not None:
        disk.put(key, result)
    return result


def _capture_with(cfg: MarbleConfig, prompt: str, model=None, tokenizer=None) -> StateTrajectory:
    """One validated capture under the config's capture knobs."""
    traj = capture(
        model if model is not None else cfg.model,
        prompt,
        tokenizer=tokenizer,
        top_k=cfg.top_k,
        device=cfg.device,
        dtype=cfg.dtype,
        keep_logits=cfg.keep_logits,
        capture_components=cfg.capture_components,
        capture_attention=cfg.capture_attention,
    )
    traj.validate()
    return traj


def _assemble_scene(cfg: MarbleConfig, trajs: list[StateTrajectory]) -> dict:
    """Shared multi-run assembly: joint projection, one terrain from the
    union of all runs' states, draped trajectories, comparisons vs run 0."""
    coords_list, _ = projection_mod.project_joint(
        [t.hidden for t in trajs],
        method=cfg.projection, n_components=cfg.n_components, seed=cfg.seed,
    )
    union = np.concatenate([c.reshape(-1, cfg.n_components) for c in coords_list])
    landscape = density_mod.compute_density(
        union, method=cfg.density, grid_size=cfg.grid_size, padding=cfg.grid_padding
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

    comparisons = [
        compare_mod.compare(trajs[0], t, coords_list[0], c)
        for t, c in zip(trajs[1:], coords_list[1:])
    ]

    result = {
        "trajs": trajs,
        "coords_list": coords_list,
        "landscape": landscape,
        "mesh": surface,
        "trajectories_list": trajectories_list,
        "fine_paths_list": fine_paths_list,
        "comparisons": comparisons,
        # run-0 view (the run_pipeline keys)
        "traj": trajs[0],
        "coords": coords_list[0],
        "trajectories": trajectories_list[0],
        "fine_paths": fine_paths_list[0],
    }
    if len(trajs) == 2:  # the run_compare aliases
        result.update({
            "traj_b": trajs[1],
            "coords_b": coords_list[1],
            "trajectories_b": trajectories_list[1],
            "fine_paths_b": fine_paths_list[1],
            "comparison": comparisons[0],
        })
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
    key = cache_mod.make_key("scene-v1", prompts, cfg.model, cfg.projection,
                             cfg.density, cfg.top_k, cfg.n_components, cfg.seed,
                             cfg.grid_size, cfg.smooth_sigma, cfg.height_scale,
                             cfg.invert_terrain, cfg.trajectory_mode,
                             cfg.trajectory_token, cfg.frames_per_layer,
                             cfg.capture_components, cfg.capture_attention)
    if disk is not None and (hit := disk.get(key)) is not None:
        return hit

    trajs = [_capture_with(cfg, p, model=model, tokenizer=tokenizer) for p in prompts]
    result = {"prompts": list(prompts), "prompt": prompts[0], **_assemble_scene(cfg, trajs)}
    if len(prompts) == 2:
        result["prompt_b"] = prompts[1]
    if disk is not None:
        disk.put(key, result)
    return result


def run_compare(cfg: MarbleConfig, prompt_a: str, prompt_b: str,
                model=None, tokenizer=None) -> dict:
    """A/B pipeline: a two-prompt `run_scene` (kept as the pairwise API)."""
    return run_scene(cfg, [prompt_a, prompt_b], model=model, tokenizer=tokenizer)


def run_intervention(cfg: MarbleConfig, prompt: str, interventions: list,
                     model, tokenizer) -> dict:
    """Interactive patching: baseline capture vs perturb-and-replay branch.

    Runs the same prompt twice — once untouched, once under `interventions`
    (intervene.Intervention edits) — then assembles the two counterfactual
    runs as a scene: shared projection, one terrain, comparison, plus an
    `intervene.divergence` readout.  Requires a torch model; results are not
    cached (the edit space is unbounded).
    """
    from intervene import divergence, intervene

    baseline = _capture_with(cfg, prompt, model=model, tokenizer=tokenizer)
    branch = intervene(model, prompt, interventions, tokenizer=tokenizer,
                       top_k=cfg.top_k, device=cfg.device, dtype=cfg.dtype,
                       keep_logits=cfg.keep_logits)
    branch.validate()

    result = {"prompts": [prompt, prompt], "prompt": prompt,
              "prompt_b": "patched: " + ", ".join(iv.describe() for iv in interventions),
              **_assemble_scene(cfg, [baseline, branch])}
    result["divergence"] = divergence(baseline, branch)
    return result


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------
_MARBLE_COLORS = ["#f94144", "#f8961e", "#f9c74f", "#90be6d", "#43aa8b",
                  "#4d908e", "#577590", "#277da1", "#9b5de5", "#f15bb5"]


def _hover_text(traj: StateTrajectory, t: trajectory_mod.Trajectory) -> list[str]:
    texts = []
    for layer in range(len(t.points)):
        token = t.token if isinstance(t.token, int) else traj.n_tokens - 1
        state = traj.state(layer, token)
        top = state.topk[0] if state.topk else ("?", 0.0)
        texts.append(
            f"<b>{t.label or t.token}</b><br>layer {layer}"
            f"<br>entropy {state.entropy:.2f} nats"
            f"<br>top: '{top[0]}' ({top[1]:.0%})"
            f"<br>|h| = {state.norm:.1f}"
        )
    return texts


_DASH_CYCLE = [None, "dash", "dot", "longdash", "dashdot"]


def _attention_trace(
    trajectories: list[trajectory_mod.Trajectory],
    attention: np.ndarray,
    layer: int,
    threshold: float = 0.1,
    top_k: int = 3,
) -> go.Scatter3d | None:
    """Attention-flow segments at one layer: destination states connected to
    the source states they read from (head-averaged weight ≥ threshold,
    self-attention omitted — it would be a zero-length segment)."""
    if layer < 1 or layer - 1 >= len(attention):
        return None
    weights = attention[layer - 1]
    xs, ys, zs = [], [], []
    for dst in range(weights.shape[0]):
        for src in np.argsort(-weights[dst])[:top_k]:
            if src == dst or weights[dst, src] < threshold:
                continue
            p, q = trajectories[src].points[layer], trajectories[dst].points[layer]
            xs += [p[0], q[0], None]
            ys += [p[1], q[1], None]
            zs += [p[2], q[2], None]
    if not xs:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line={"color": "rgba(255,255,255,0.5)", "width": 2},
        name="attention", hoverinfo="skip",
    )


def render(
    traj: StateTrajectory,
    surface: terrain_mod.TerrainMesh,
    trajectories: list[trajectory_mod.Trajectory],
    fine_paths: list[np.ndarray],
    current_layer: int = 0,
    frames_per_layer: int = 4,
    frame_ms: int = 120,
    traj_b: StateTrajectory | None = None,
    trajectories_b: list[trajectory_mod.Trajectory] | None = None,
    fine_paths_b: list[np.ndarray] | None = None,
    extra_runs: list[tuple] | None = None,
    overlay: list[np.ndarray] | None = None,
    overlay_label: str = "feature",
    show_attention: bool = False,
) -> go.Figure:
    """Build the 3-D scene: terrain surface, token trajectories, animated marbles.

    Passing `traj_b` / `trajectories_b` / `fine_paths_b` (from `run_compare`)
    overlays a second run on the same terrain; `extra_runs` — a list of
    (traj, trajectories, fine_paths) tuples from `run_scene` — overlays runs
    beyond the second.  Every overlaid run gets its own dash style and an
    A/B/C… label prefix.

    `overlay` colors the primary run's trajectory markers by a per-layer
    scalar (one (L,) array per trajectory, e.g. an SAE feature's activation),
    on a shared color scale with a colorbar.  `show_attention` draws the
    primary run's attention flow at `current_layer` (per-token trajectories
    with a captured attention pattern only).
    """
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=surface.x, y=surface.y, z=surface.z,
        colorscale="Viridis", opacity=0.92, showscale=False,
        contours={"z": {"show": True, "usecolormap": True, "width": 2,
                        "highlightcolor": "white", "project": {"z": False}}},
        name="manifold", hoverinfo="skip",
    ))

    all_runs = [(traj, trajectories, fine_paths)]
    if trajectories_b:
        all_runs.append((traj_b, trajectories_b, fine_paths_b))
    if extra_runs:
        all_runs.extend(extra_runs)
    multi = len(all_runs) > 1
    runs = [
        (src, trajs, f"{chr(65 + r)} · " if multi else "",
         _DASH_CYCLE[r % len(_DASH_CYCLE)] if r else None)
        for r, (src, trajs, _) in enumerate(all_runs)
    ]
    vmin = vmax = 0.0
    if overlay is not None:
        vmin = float(min(v.min() for v in overlay))
        vmax = float(max(max(v.max() for v in overlay), vmin + 1e-6))
    i = 0
    for run_idx, (src, trajs, prefix, dash) in enumerate(runs):
        for j, t in enumerate(trajs):
            color = _MARBLE_COLORS[i % len(_MARBLE_COLORS)]
            i += 1
            line = {"color": color, "width": 5}
            if dash:
                line["dash"] = dash
            marker = {"size": 3, "color": color}
            if overlay is not None and run_idx == 0 and j < len(overlay):
                marker = {"size": 5, "color": overlay[j], "colorscale": "Turbo",
                          "cmin": vmin, "cmax": vmax, "showscale": j == 0,
                          "colorbar": {"title": overlay_label, "len": 0.5,
                                       "x": 0.02, "thickness": 12}}
            pts = t.points
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines+markers",
                line=line,
                marker=marker,
                name=prefix + str(t.label or t.token),
                text=_hover_text(src, t),
                hoverinfo="text",
            ))

    if show_attention and traj.attention is not None and len(trajectories) == traj.n_tokens:
        att = _attention_trace(trajectories, traj.attention, current_layer)
        if att is not None:
            fig.add_trace(att)

    # Marbles: one animated marker per trajectory, positioned along the
    # densified path; fine index f corresponds to layer f / frames_per_layer.
    fine_paths = [p for _, _, fp in all_runs for p in fp]
    n_frames = min(len(p) for p in fine_paths) if fine_paths else 0
    start = min(current_layer * frames_per_layer, max(n_frames - 1, 0))

    def marble_trace(f: int) -> go.Scatter3d:
        xyz = np.array([p[min(f, len(p) - 1)] for p in fine_paths])
        return go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2] + 0.02,
            mode="markers",
            marker={"size": 9, "color": [_MARBLE_COLORS[i % len(_MARBLE_COLORS)]
                                         for i in range(len(fine_paths))],
                    "symbol": "circle", "line": {"color": "white", "width": 2}},
            name="marble", hoverinfo="skip", showlegend=False,
        )

    if n_frames:
        marble_idx = len(fig.data)
        fig.add_trace(marble_trace(start))
        fig.frames = [
            go.Frame(data=[marble_trace(f)], traces=[marble_idx], name=str(f))
            for f in range(n_frames)
        ]
        slider_steps = [
            {
                "args": [[str(layer * frames_per_layer)],
                         {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": str(layer),
                "method": "animate",
            }
            for layer in range(traj.n_layers)
            if layer * frames_per_layer < n_frames
        ]
        fig.update_layout(
            updatemenus=[{
                "type": "buttons", "direction": "left",
                "x": 0.05, "y": 0.02, "xanchor": "left", "yanchor": "bottom",
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}]},
                ],
            }],
            sliders=[{
                "steps": slider_steps, "active": min(current_layer, len(slider_steps) - 1),
                "x": 0.05, "y": 0.08, "len": 0.9,
                "currentvalue": {"prefix": "layer ", "font": {"size": 13}},
            }],
        )

    fig.update_layout(
        scene={
            "xaxis": {"title": "manifold x", "showbackground": False},
            "yaxis": {"title": "manifold y", "showbackground": False},
            "zaxis": {"title": "potential", "showbackground": False},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        height=680,
        legend={"x": 0.99, "y": 0.99, "xanchor": "right"},
        template="plotly_dark",
        title={"text": "Mottled — latent trajectory explorer", "font": {"size": 14}},
    )
    return fig


# --------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Mottled", page_icon="🔮", layout="wide")

    @st.cache_resource(show_spinner="Loading model…")
    def load_model_cached(name: str):
        from capture import load_model

        return load_model(name)

    # ------------------------------------------------------------ left panel
    with st.sidebar:
        st.title("🔮 Mottled")
        prompt = st.text_area("Prompt", DEFAULT_PROMPT, key="prompt")
        prompt_b = st.text_area("Overlay prompts (one per line, optional)", "",
                                key="prompt_b",
                                help="Each line becomes another run drawn on the "
                                     "same terrain and compared against the prompt above.")
        model_name = st.selectbox("Model", MODEL_CHOICES, index=0, key="model")
        proj_name = st.selectbox("Projection", PROJECTION_CHOICES, key="projection")
        dens_name = st.selectbox("Density estimator", DENSITY_CHOICES, key="density")
        top_k = st.slider("Top-k", 1, 10, 5, key="top_k")
        mode = st.selectbox("Trajectory", TRAJECTORY_MODES, key="trajectory_mode")
        invert = st.checkbox("Dense regions as valleys", value=False, key="invert")
        sae_on = st.checkbox("SAE feature overlay", value=False, key="sae_overlay",
                             help="Color trajectory markers by one feature's activation. "
                                  "Uses an untrained demo dictionary; load real weights "
                                  "with sae.load_npz for interpretable features.")
        attention_on = st.checkbox("Show attention flow", value=False, key="attention_flow",
                                   help="Draw head-averaged attention edges between token "
                                        "states at the selected layer.")
        run = st.button("▶ Run capture", type="primary", use_container_width=True, key="run")
        st.caption("Play / Pause / scrub inside the figure animate the marble; "
                   "the layer slider below drives the inspector.")

    if run and prompt.strip():
        cfg = MarbleConfig(model=model_name, projection=proj_name, density=dens_name,
                           top_k=top_k, trajectory_mode=mode, invert_terrain=invert)
        model = tokenizer = None
        if model_name != "synthetic":
            model, tokenizer = load_model_cached(model_name)
        overlays = [p.strip() for p in prompt_b.splitlines() if p.strip()]
        with st.spinner("Capturing forward pass…"):
            if overlays:
                st.session_state["result"] = run_scene(cfg, [prompt] + overlays,
                                                       model=model, tokenizer=tokenizer)
            else:
                st.session_state["result"] = run_pipeline(cfg, prompt,
                                                          model=model, tokenizer=tokenizer)
            st.session_state["cfg"] = cfg

    result = st.session_state.get("result")
    if result is None:
        st.info("Enter a prompt and press **Run capture** to explore the latent manifold.")
        return

    cfg: MarbleConfig = st.session_state["cfg"]
    traj: StateTrajectory = result["traj"]

    @st.cache_resource(show_spinner=False)
    def _demo_sae(dim: int, n_features: int):
        return sae_mod.demo_sae(dim, n_features)

    acts = overlay = None
    overlay_label = "feature"
    with st.sidebar:
        layer = st.slider("Layer", 0, traj.n_layers - 1, traj.n_layers - 1, key="layer")
        if sae_on:
            acts = sae_mod.feature_trajectory(traj, _demo_sae(traj.dim, cfg.sae_features))
            choices = [int(f) for f in sae_mod.active_features(acts, k=25)]
            if choices and result.get("traj_b") is None:
                feat = st.selectbox(
                    "SAE feature", choices,
                    format_func=lambda f: f"f{f} · peak {acts[..., f].max():.2f}",
                    key="feature",
                )
                overlay_label = f"f{feat}"
                overlay = [
                    acts[:, t.token if isinstance(t.token, int) else traj.n_tokens - 1, feat]
                    for t in result["trajectories"]
                ]

        import io as _io

        _buf = _io.BytesIO()
        statefile_mod.save_scene(result, _buf)
        st.download_button("⬇ Export scene (.mtj)", data=_buf.getvalue(),
                           file_name="scene.mtj", mime="application/octet-stream",
                           use_container_width=True, key="export_scene",
                           help="Portable scene for the web viewer (viewer/) "
                                "or any .mtj consumer — see docs/mtj-format.md.")

        with st.expander("Intervention (perturb & replay)", expanded=False):
            if cfg.model == "synthetic":
                st.caption("The synthetic backend is analytic and not resumable; "
                           "interventions need a torch model.")
            else:
                iv_layer = st.slider("Edit layer", 0, traj.n_layers - 1,
                                     traj.n_layers - 1, key="iv_layer")
                iv_kind = st.selectbox("Edit", ["push toward token", "inject noise",
                                                "freeze block"], key="iv_kind")
                iv_target = ""
                if iv_kind == "push toward token":
                    iv_target = st.text_input("Target token", "Berlin", key="iv_target")
                iv_scale = st.slider("Strength", 0.0, 100.0, 30.0, key="iv_scale")
                if st.button("Run intervention", key="iv_run"):
                    from intervene import FreezeLayer, InjectNoise, Perturb

                    model, tokenizer = load_model_cached(cfg.model)
                    if iv_kind == "push toward token":
                        ids = tokenizer(iv_target, add_special_tokens=False)["input_ids"]
                        if not ids:
                            st.warning("target token is empty")
                            st.stop()
                        direction = traj.embedding_matrix[ids[0]]
                        edits = [Perturb(iv_layer, iv_scale * direction, token=-1)]
                    elif iv_kind == "inject noise":
                        edits = [InjectNoise(iv_layer, iv_scale, token=-1)]
                    else:
                        edits = [FreezeLayer(min(iv_layer, traj.n_layers - 2))]
                    with st.spinner("Replaying under intervention…"):
                        st.session_state["result"] = run_intervention(
                            cfg, result["prompt"], edits, model, tokenizer)
                    st.rerun()

    @st.cache_resource(show_spinner=False)
    def _neighbor_cache(key: str):  # one TokenNeighbors per capture
        return {}

    def _token_neighbors(t: StateTrajectory) -> TokenNeighbors:
        holder = _neighbor_cache(cache_mod.make_key(t.meta.get("model"), t.meta.get("prompt")))
        if "tn" not in holder:
            holder["tn"] = TokenNeighbors(t.embedding_matrix, t.vocab)
        return holder["tn"]

    col_viz, col_info = st.columns([3, 1])

    extra_runs = None
    if result.get("trajs") is not None and len(result["trajs"]) > 2:
        extra_runs = [
            (result["trajs"][i], result["trajectories_list"][i], result["fine_paths_list"][i])
            for i in range(2, len(result["trajs"]))
        ]

    with col_viz:
        fig = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                     current_layer=layer, frames_per_layer=cfg.frames_per_layer,
                     frame_ms=cfg.frame_ms,
                     traj_b=result.get("traj_b"),
                     trajectories_b=result.get("trajectories_b"),
                     fine_paths_b=result.get("fine_paths_b"),
                     extra_runs=extra_runs,
                     overlay=overlay, overlay_label=overlay_label,
                     show_attention=attention_on)
        st.plotly_chart(fig, use_container_width=True, key="scene")

    # ----------------------------------------------------------- right panel
    with col_info:
        st.subheader("Inspector")
        token = st.selectbox(
            "Token", range(traj.n_tokens),
            index=traj.n_tokens - 1,
            format_func=lambda i: f"{i}: {traj.tokens[i]!r}",
            key="token",
        )
        state = traj.state(layer, token)
        a, b = st.columns(2)
        a.metric("Layer", layer)
        b.metric("Token", f"{state.text!r}")
        a.metric("Entropy", f"{state.entropy:.2f} nats")
        b.metric("Vector norm", f"{state.norm:.1f}")

        st.markdown("**Top predictions**")
        for tok, p in state.topk[: cfg.top_k]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{tok!r} — {p:.1%}")

        if traj.embedding_matrix is not None and traj.vocab is not None:
            st.markdown("**Nearest semantic neighbors**")
            tn = _token_neighbors(traj)
            for tok, sim in tn.nearest(state.vector, k=cfg.n_neighbors):
                st.write(f"`{tok}`  ·  cos {sim:.3f}")

        if acts is not None:
            st.markdown("**Top SAE features** *(demo dictionary)*")
            fired = sae_mod.top_features(acts, layer, token, k=cfg.top_k)
            for fid, act in fired:
                st.write(f"`f{fid}`  ·  act {act:.2f}")
            if not fired:
                st.caption("no features fire at this state")

        if traj.attention is not None and layer >= 1:
            st.markdown("**Attention** *(head-averaged, into this layer)*")
            weights = traj.attention[layer - 1, token]
            for src in np.argsort(-weights)[:5]:
                if weights[src] <= 0.01:
                    continue
                st.write(f"`{traj.tokens[src]}` ({src})  ·  {weights[src]:.2f}")

        if traj.components is not None:
            with st.expander("Residual decomposition", expanded=False):
                shares = metrics_mod.component_shares(traj, token=token)
                st.markdown("Share of each block's residual write")
                st.line_chart({"attention": shares[:, 0], "mlp": shares[:, 1]})

        with st.expander("Research metrics", expanded=False):
            summary = metrics_mod.summarize(traj, result["coords"], token=token)
            for name, value in summary.items():
                st.write(f"{name.replace('_', ' ')}: **{value:.3f}**")

        if result.get("traj_b") is not None:
            with st.expander("A/B comparison", expanded=True):
                st.caption(f"B = {result.get('prompt_b', '')}")
                cmp = compare_mod.compare(traj, result["traj_b"],
                                          result["coords"], result["coords_b"],
                                          token=token)
                st.write(f"shared prefix: **{cmp.shared_tokens} tokens**")
                st.write(f"Hausdorff distance: **{cmp.hausdorff:.3f}**")
                st.write(f"DTW distance (normalized): **{cmp.dtw.normalized:.3f}**")
                if cmp.onset_token is not None:
                    st.write(f"states separate from token **{cmp.onset_token}**, "
                             f"layer **{cmp.onset_layer}**")
                if cmp.readout_changed is not None:
                    st.write(f"top-1 prediction differs from layer **{cmp.readout_changed}**")
                st.markdown("**A–B distance per layer**")
                st.line_chart(cmp.profile)

        if (div := result.get("divergence")) is not None:
            with st.expander("Intervention divergence", expanded=True):
                st.write(f"separation onset: layer **{div.onset}**")
                if div.readout_changed is not None:
                    st.write(f"prediction flips at layer **{div.readout_changed}**")
                else:
                    st.write("top-1 prediction unchanged")
                st.line_chart(div.profile)

        if result.get("comparisons") and len(result["comparisons"]) > 1:
            with st.expander("Scene comparisons (vs A)", expanded=True):
                for i, cmp in enumerate(result["comparisons"], start=1):
                    st.write(f"**{chr(65 + i)}** · Hausdorff {cmp.hausdorff:.3f} "
                             f"· DTW {cmp.dtw.normalized:.3f} "
                             f"· shared prefix {cmp.shared_tokens}")


if __name__ == "__main__":
    main()
