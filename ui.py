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

import attractor as attractor_mod
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
    key = cache_mod.make_key("pipeline-v5", prompt, cfg.model, cfg.projection,
                             cfg.density, cfg.top_k, cfg.n_components, cfg.seed,
                             cfg.grid_size, cfg.smooth_sigma, cfg.height_scale,
                             cfg.invert_terrain, cfg.trajectory_mode,
                             cfg.trajectory_token, cfg.frames_per_layer,
                             cfg.capture_components, cfg.capture_attention,
                             cfg.density_bootstrap)
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

    comparisons = [
        compare_mod.compare(trajs[0], t, coords_list[0], c)
        for t, c in zip(trajs[1:], coords_list[1:])
    ]

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
    key = cache_mod.make_key("scene-v3", prompts, cfg.model, cfg.projection,
                             cfg.density, cfg.top_k, cfg.n_components, cfg.seed,
                             cfg.grid_size, cfg.smooth_sigma, cfg.height_scale,
                             cfg.invert_terrain, cfg.trajectory_mode,
                             cfg.trajectory_token, cfg.frames_per_layer,
                             cfg.capture_components, cfg.capture_attention,
                             cfg.density_bootstrap)
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
# Renderer — styled to the Incision design language: dark navy void, one
# precision-blue accent, semantic data colors, mono for data values.
# --------------------------------------------------------------------------
_MARBLE_COLORS = ["#4B7CF3", "#00CCA8", "#D4934A", "#E05050", "#38B07A",
                  "#8FA7F7", "#5CE0C6", "#E6B884", "#F08A8A", "#7FD0AC"]

# Terrain potential ramp: void -> surfaces -> precision blue -> light blue.
_TERRAIN_COLORSCALE = [[0.0, "#04060E"], [0.22, "#0C1020"], [0.45, "#1C2A55"],
                       [0.68, "#2F55B8"], [0.86, "#4B7CF3"], [1.0, "#C8D4FB"]]

_FONT_SANS = "DM Sans, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
_FONT_MONO = "JetBrains Mono, SF Mono, Menlo, Consolas, monospace"


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
        line={"color": "rgba(129,143,184,0.55)", "width": 2},
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
    basin: attractor_mod.BasinReport | None = None,
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

    `basin` (an `attractor.analyze` report) marks the density peak and pins
    a measured explanation to it — membership count, layer range, settle
    layer, stabilized readout — so the scene says *why* the basin is there,
    not just that it is.
    """
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=surface.x, y=surface.y, z=surface.z,
        colorscale=_TERRAIN_COLORSCALE, opacity=0.94, showscale=False,
        contours={"z": {"show": True, "usecolormap": True, "width": 2,
                        "highlightcolor": "#8FA7F7", "project": {"z": False}}},
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

    basin_annotations = []
    if basin is not None:
        cx, cy = float(basin.center[0]), float(basin.center[1])
        cz = float(terrain_mod.surface_height(surface, np.array([[cx, cy]]))[0])
        span = (f"layers {basin.layer_range[0]}–{basin.layer_range[1]}"
                if basin.layer_range is not None else "no states above threshold")
        lines = [f"attractor basin · {basin.n_members}/{basin.n_states} states · {span}"]
        if basin.settle_layer is not None:
            lines.append(f"settles at layer {basin.settle_layer}")
        if basin.top_token is not None and basin.readout_stable_from is not None:
            lines.append(f"top-1 {basin.top_token!r} from layer {basin.readout_stable_from}")
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz + 0.03], mode="markers",
            marker={"size": 9, "symbol": "diamond-open", "color": "#C8D4FB",
                    "line": {"color": "#4B7CF3", "width": 3}},
            name="attractor", hoverinfo="text", text=["<br>".join(lines)],
        ))
        basin_annotations = [{
            "x": cx, "y": cy, "z": cz + 0.05,
            "text": "<br>".join(lines),
            "font": {"family": _FONT_MONO, "size": 10, "color": "#C8D4FB"},
            "bgcolor": "rgba(12,16,32,0.88)", "bordercolor": "#1E2540",
            "borderwidth": 1, "borderpad": 6,
            "arrowcolor": "#4B7CF3", "arrowwidth": 1.5,
            "ax": 60, "ay": -70, "xanchor": "left", "align": "left",
        }]

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
                    "symbol": "circle", "line": {"color": "#EDF0FA", "width": 2}},
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
                "font": {"color": "#EDF0FA"}, "bgcolor": "#11162A",
                "bordercolor": "#1E2540",
                "buttons": [
                    {"label": "Play", "method": "animate",
                     "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}]},
                ],
            }],
            sliders=[{
                "steps": slider_steps, "active": min(current_layer, len(slider_steps) - 1),
                "x": 0.05, "y": 0.08, "len": 0.9,
                "font": {"color": "#818FB8", "family": _FONT_MONO, "size": 11},
                "bgcolor": "#1E2540", "activebgcolor": "#4B7CF3",
                "bordercolor": "#1E2540", "tickcolor": "#283050",
                "currentvalue": {"prefix": "layer ", "font": {"size": 13}},
            }],
        )

    axis = {"showbackground": False, "gridcolor": "#1E2540",
            "zerolinecolor": "#283050", "color": "#818FB8"}
    # Real-model scenes span hundreds of units in x/y while terrain height is
    # normalized 0..1; pure "data" aspect flattens the relief into
    # invisibility. Keep data proportions but never let relief drop below
    # ~12% of the box (the web viewer applies the same rule).
    sx = float(surface.x.max() - surface.x.min()) or 1.0
    sy = float(surface.y.max() - surface.y.min()) or 1.0
    sz = float(surface.z.max() - surface.z.min()) or 1.0
    m = max(sx, sy)
    fig.update_layout(
        scene={
            "xaxis": {"title": "manifold x", **axis},
            "yaxis": {"title": "manifold y", **axis},
            "zaxis": {"title": "potential", **axis},
            "aspectmode": "manual",
            "aspectratio": {"x": sx / m, "y": sy / m, "z": max(sz / m, 0.12)},
            "annotations": basin_annotations,
        },
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        height=680,
        paper_bgcolor="#080B18",
        font={"family": _FONT_SANS, "color": "#EDF0FA", "size": 12},
        legend={"x": 0.99, "y": 0.99, "xanchor": "right",
                "bgcolor": "rgba(12,16,32,0.88)", "bordercolor": "#1E2540",
                "borderwidth": 1, "font": {"color": "#818FB8", "size": 11}},
        hoverlabel={"bgcolor": "#161C33", "bordercolor": "#283050",
                    "font": {"family": _FONT_MONO, "color": "#EDF0FA", "size": 11}},
        title={"text": "Mottled — latent trajectory explorer",
               "font": {"size": 13, "color": "#818FB8"}},
    )
    return fig


# --------------------------------------------------------------------------
# SAE feature field — domain coloring of the projection plane.
# --------------------------------------------------------------------------
_GOLDEN = 0.6180339887498949  # golden-ratio conjugate: well-spread hues


def _feature_hue(ids: np.ndarray) -> np.ndarray:
    """Stable, well-separated hue in [0, 1) per feature id."""
    return (np.asarray(ids, dtype=np.float64) * _GOLDEN) % 1.0


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV -> RGB, all components in [0, 1]; returns (..., 3)."""
    h6 = (np.asarray(h, dtype=np.float64) % 1.0) * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    s, v = np.broadcast_to(s, i.shape), np.broadcast_to(v, i.shape)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def field_rgb(field: sae_mod.FeatureField) -> np.ndarray:
    """Domain-color a feature field: (H, W, 3) uint8.

    The classical complex-plane recipe, transplanted: hue encodes the
    dominant feature (the "phase" — which dictionary direction this point
    of the manifold decodes into), brightness encodes its activation (the
    "modulus"), and the sawtooth banding on log2(magnitude) draws the
    contour rings at magnitude octaves.  Points where no feature fires
    render as the void.
    """
    fires = field.dominant >= 0
    m = field.magnitude / max(float(field.magnitude.max()), 1e-12)
    hue = _feature_hue(np.maximum(field.dominant, 0))
    sat = np.where(fires, 0.62, 0.0)
    with np.errstate(divide="ignore"):
        octave = np.log2(np.maximum(m, 1e-9))
    band = 0.82 + 0.18 * (octave - np.floor(octave))
    val = np.where(fires, (0.18 + 0.82 * np.sqrt(m)) * band, 0.035)
    return (np.clip(_hsv_to_rgb(hue, sat, val), 0.0, 1.0) * 255).astype(np.uint8)


def render_feature_field(
    field: sae_mod.FeatureField,
    sae: sae_mod.SAE | None = None,
    path: np.ndarray | None = None,
    relief: bool = False,
) -> go.Figure:
    """The SAE feature field as a domain-colored figure.

    `path` (L, 2) overlays a projected trajectory so the run's route through
    feature territory is visible.  `relief=False` is the flat complex-plane
    view; `relief=True` lifts activation magnitude into z, leaving holes
    where no feature fires — the manifold sheets of the dictionary.
    """
    fig = go.Figure()
    if relief:
        m = field.magnitude / max(float(field.magnitude.max()), 1e-12)
        z = np.where(field.dominant >= 0, m, np.nan)
        hue_scale = [[i / 12, "rgb({:.0f},{:.0f},{:.0f})".format(
            *(255 * _hsv_to_rgb(np.array(i / 12), 0.62, 0.9)))] for i in range(13)]
        fig.add_trace(go.Surface(
            x=field.grid_x, y=field.grid_y, z=z,
            surfacecolor=_feature_hue(np.maximum(field.dominant, 0)),
            colorscale=hue_scale, cmin=0.0, cmax=1.0, showscale=False,
            opacity=0.96, name="feature field", hoverinfo="skip",
        ))
        if path is not None:
            p = np.asarray(path, dtype=np.float64)[:, :2]
            top = float(np.nanmax(z)) if np.isfinite(z).any() else 1.0
            fig.add_trace(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=np.full(len(p), top + 0.06),
                mode="lines+markers", name="trajectory",
                line={"color": "#EDF0FA", "width": 4}, marker={"size": 3, "color": "#EDF0FA"},
            ))
        axis = {"showbackground": False, "gridcolor": "#1E2540",
                "zerolinecolor": "#283050", "color": "#818FB8"}
        fig.update_layout(scene={
            "xaxis": {"title": "manifold x", **axis},
            "yaxis": {"title": "manifold y", **axis},
            "zaxis": {"title": "activation", **axis},
        })
    else:
        dx = float(field.grid_x[1] - field.grid_x[0]) if len(field.grid_x) > 1 else 1.0
        dy = float(field.grid_y[1] - field.grid_y[0]) if len(field.grid_y) > 1 else 1.0
        fig.add_trace(go.Image(
            z=field_rgb(field),
            x0=float(field.grid_x[0]), dx=dx,
            y0=float(field.grid_y[0]), dy=dy,
            hoverinfo="skip", name="feature field",
        ))
        if path is not None:
            p = np.asarray(path, dtype=np.float64)[:, :2]
            fig.add_trace(go.Scatter(
                x=p[:, 0], y=p[:, 1], mode="lines+markers", name="trajectory",
                line={"color": "#EDF0FA", "width": 2},
                marker={"size": 5, "color": "#EDF0FA",
                        "line": {"color": "#080B18", "width": 1}},
                text=[f"layer {l}" for l in range(len(p))], hoverinfo="text",
            ))
        fig.update_xaxes(title="manifold x", gridcolor="#1E2540", color="#818FB8",
                         zeroline=False, constrain="domain")
        fig.update_yaxes(title="manifold y", gridcolor="#1E2540", color="#818FB8",
                         zeroline=False, autorange=True, scaleanchor="x")
    n_domains = len(field.features)
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        height=560,
        paper_bgcolor="#080B18", plot_bgcolor="#080B18",
        font={"family": _FONT_SANS, "color": "#EDF0FA", "size": 12},
        showlegend=False,
        hoverlabel={"bgcolor": "#161C33", "bordercolor": "#283050",
                    "font": {"family": _FONT_MONO, "color": "#EDF0FA", "size": 11}},
        title={"text": f"SAE feature field — {n_domains} feature domains · "
                       "hue = dominant feature, brightness = activation, "
                       "rings = magnitude octaves",
               "font": {"size": 12, "color": "#818FB8"}},
    )
    return fig


# --------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Mottled", page_icon="🔮", layout="wide")

    # Incision design language on top of the theme in .streamlit/config.toml:
    # 1px borders, near-sharp corners, mono for data values, tracked overlines.
    st.markdown("""<style>
      [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
      }
      [data-testid="stMetricLabel"] p {
        text-transform: uppercase; letter-spacing: 0.08em;
        font-size: 11px; color: #818FB8;
      }
      [data-testid="stSidebar"] { border-right: 1px solid #1E2540; }
      div[data-testid="stExpander"] details {
        border: 1px solid #1E2540; border-radius: 2px;
      }
      .stButton button, .stDownloadButton button, div[data-baseweb="select"] > div,
      .stTextArea textarea, .stTextInput input {
        border-radius: 3px !important;
      }
      .stTextArea textarea, .stTextInput input { border: 1px solid #1E2540; }
      code { font-family: "JetBrains Mono", ui-monospace, monospace; }
      h1, h2, h3 { letter-spacing: -0.01em; }
    </style>""", unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Loading model…")
    def load_model_cached(name: str):
        from capture import load_model

        return load_model(name)

    # ------------------------------------------------------------ left panel
    with st.sidebar:
        st.title("Mottled")
        st.caption("Latent trajectory explorer")
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
        field_on = st.checkbox("SAE feature field (domain coloring)", value=False,
                               key="sae_field",
                               help="Wolfram-style domain coloring of the projection "
                                    "plane: every point is inverse-projected to hidden "
                                    "space and run through the SAE — hue = dominant "
                                    "feature, brightness = activation, rings = "
                                    "magnitude octaves. Exact for pca; umap's inverse "
                                    "is approximate and gets unreliable away from the "
                                    "captured states (the field's grid always pads "
                                    "20% past them), so treat those regions as "
                                    "illustrative, not measured.")
        attention_on = st.checkbox("Show attention flow", value=False, key="attention_flow",
                                   help="Draw head-averaged attention edges between token "
                                        "states at the selected layer.")
        run = st.button("Run capture", type="primary", use_container_width=True, key="run")
        st.caption("Play, pause and scrub inside the figure animate the marble; "
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
        st.download_button("Export scene (.mtj)", data=_buf.getvalue(),
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

    basin = attractor_mod.analyze(traj, result["coords"], result["landscape"])

    with col_viz:
        fig = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                     current_layer=layer, frames_per_layer=cfg.frames_per_layer,
                     frame_ms=cfg.frame_ms,
                     traj_b=result.get("traj_b"),
                     trajectories_b=result.get("trajectories_b"),
                     fine_paths_b=result.get("fine_paths_b"),
                     extra_runs=extra_runs,
                     overlay=overlay, overlay_label=overlay_label,
                     show_attention=attention_on,
                     basin=basin)
        st.plotly_chart(fig, use_container_width=True, key="scene")
        st.caption('The terrain is a density field over the projected states '
                   'themselves — height means "many states landed here", not an '
                   'external landscape. The pinned callout marks the basin; open '
                   '**Why this attractor** in the inspector for this run\'s numbers.')

        if field_on:
            projector = result.get("projector")
            if projector is None or not hasattr(projector, "inverse_transform"):
                st.caption("The feature field needs an invertible projection — "
                           "re-run with projection = pca.")
            else:
                if cfg.projection != "pca":
                    st.caption("Projection is umap: its inverse is approximate, so "
                               "away from the actual captured states (the grid pads "
                               "20% past them) the field can look confident while "
                               "showing extrapolation artifacts, not measurement.")
                field_sae = _demo_sae(traj.dim, cfg.sae_features)
                landscape = result["landscape"]
                try:
                    fld = sae_mod.feature_field(field_sae, projector,
                                                landscape.grid_x, landscape.grid_y)
                except Exception as err:  # e.g. umap's approximate inverse failing
                    st.caption(f"feature field unavailable: {err}")
                else:
                    relief = st.radio("Field view", ["plane", "relief"],
                                      horizontal=True, key="field_mode") == "relief"
                    tok_idx = cfg.trajectory_token % traj.n_tokens
                    st.plotly_chart(
                        render_feature_field(fld, field_sae,
                                             path=result["coords"][:, tok_idx, :2],
                                             relief=relief),
                        use_container_width=True, key="field")
                    st.caption('Domain coloring of the projection plane — the '
                               'complex-plane analogue for the latent manifold: every '
                               'point of the plane is inverse-projected to hidden '
                               'space and run through the SAE. Hue = which feature '
                               'decodes strongest there (the "phase"), brightness = '
                               'its activation (the "modulus"), rings = magnitude '
                               'octaves; the white path is this run\'s tracked-token '
                               'trajectory crossing feature domains. Demo dictionary — '
                               'load real weights with sae.load_npz for interpretable '
                               'domains.')

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

        with st.expander("Why this attractor", expanded=False):
            report = (basin if basin.token == token % traj.n_tokens else
                      attractor_mod.analyze(traj, result["coords"],
                                            result["landscape"], token=token))
            st.markdown(attractor_mod.explain(report, traj))
            if len(report.step):
                st.markdown("**Per-layer step of this token** *(hidden space)*")
                st.line_chart({"step": report.step})
            if report.entropy is not None:
                st.markdown("**Predictive entropy per layer** *(nats)*")
                st.line_chart({"entropy": report.entropy})

        with st.expander("Research metrics", expanded=False):
            summary = metrics_mod.summarize(traj, result["coords"], token=token)
            for name, value in summary.items():
                st.write(f"{name.replace('_', ' ')}: **{value:.3f}**")

        with st.expander("Uncertainty", expanded=False):
            q = result.get("quality")
            if q is not None:
                if q.explained_variance is not None:
                    st.write(f"projection keeps **{q.explained_variance:.0%}** "
                             "of hidden-space variance")
                st.write(f"neighborhood preservation (k={q.k}): "
                         f"mean **{q.preservation.mean():.2f}** · "
                         f"this state **{q.preservation[layer, token]:.2f}**")
                if q.residual is not None:
                    st.write(f"this state lies **{q.residual[layer, token]:.0%}** "
                             "off the projection plane")
                st.markdown("**Neighborhood preservation per layer** *(this token)*")
                st.line_chart({"preservation": q.preservation[:, token]})
            se = result["landscape"].density_se
            if se is not None:
                st.write(f"density bootstrap SE: mean **{se.mean():.3f}** · "
                         f"max **{se.max():.3f}** (normalized density units)")
            st.caption("The 2-D picture is a lossy view of hidden space and the "
                       "terrain is an estimate from finitely many states — treat "
                       "low-preservation states and high-SE regions as suggestive, "
                       "not measured. The web viewer's uncertainty toggle shows "
                       "the SE field on the terrain itself.")

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
