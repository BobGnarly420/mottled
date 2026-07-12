"""MARBLE UI: interactive latent trajectory explorer.

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
import density as density_mod
import metrics as metrics_mod
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
    """Execute the full MARBLE pipeline and return every artifact.

    `model`/`tokenizer` may be pre-loaded objects (the UI caches them); when
    omitted, `cfg.model` is loaded by name ("synthetic" needs no loading).
    """
    disk = cache_mod.DiskCache(cfg.cache_dir) if cfg.use_cache else None
    key = cache_mod.make_key("pipeline-v1", prompt, cfg.model, cfg.projection,
                             cfg.density, cfg.top_k, cfg.n_components, cfg.seed,
                             cfg.grid_size, cfg.smooth_sigma, cfg.height_scale,
                             cfg.invert_terrain, cfg.trajectory_mode,
                             cfg.trajectory_token, cfg.frames_per_layer)
    if disk is not None and (hit := disk.get(key)) is not None:
        return hit

    traj = capture(
        model if model is not None else cfg.model,
        prompt,
        tokenizer=tokenizer,
        top_k=cfg.top_k,
        device=cfg.device,
        dtype=cfg.dtype,
        keep_logits=cfg.keep_logits,
    )
    traj.validate()

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


def render(
    traj: StateTrajectory,
    surface: terrain_mod.TerrainMesh,
    trajectories: list[trajectory_mod.Trajectory],
    fine_paths: list[np.ndarray],
    current_layer: int = 0,
    frames_per_layer: int = 4,
    frame_ms: int = 120,
) -> go.Figure:
    """Build the 3-D scene: terrain surface, token trajectories, animated marbles."""
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=surface.x, y=surface.y, z=surface.z,
        colorscale="Viridis", opacity=0.92, showscale=False,
        contours={"z": {"show": True, "usecolormap": True, "width": 2,
                        "highlightcolor": "white", "project": {"z": False}}},
        name="manifold", hoverinfo="skip",
    ))

    for i, t in enumerate(trajectories):
        color = _MARBLE_COLORS[i % len(_MARBLE_COLORS)]
        pts = t.points
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines+markers",
            line={"color": color, "width": 5},
            marker={"size": 3, "color": color},
            name=str(t.label or t.token),
            text=_hover_text(traj, t),
            hoverinfo="text",
        ))

    # Marbles: one animated marker per trajectory, positioned along the
    # densified path; fine index f corresponds to layer f / frames_per_layer.
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
        title={"text": "MARBLE — latent trajectory explorer", "font": {"size": 14}},
    )
    return fig


# --------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="MARBLE", page_icon="🔮", layout="wide")

    @st.cache_resource(show_spinner="Loading model…")
    def load_model_cached(name: str):
        from capture import load_model

        return load_model(name)

    # ------------------------------------------------------------ left panel
    with st.sidebar:
        st.title("🔮 MARBLE")
        prompt = st.text_area("Prompt", DEFAULT_PROMPT, key="prompt")
        model_name = st.selectbox("Model", MODEL_CHOICES, index=0, key="model")
        proj_name = st.selectbox("Projection", PROJECTION_CHOICES, key="projection")
        dens_name = st.selectbox("Density estimator", DENSITY_CHOICES, key="density")
        top_k = st.slider("Top-k", 1, 10, 5, key="top_k")
        mode = st.selectbox("Trajectory", TRAJECTORY_MODES, key="trajectory_mode")
        invert = st.checkbox("Dense regions as valleys", value=False, key="invert")
        run = st.button("▶ Run capture", type="primary", use_container_width=True, key="run")
        st.caption("Play / Pause / scrub inside the figure animate the marble; "
                   "the layer slider below drives the inspector.")

    if run and prompt.strip():
        cfg = MarbleConfig(model=model_name, projection=proj_name, density=dens_name,
                           top_k=top_k, trajectory_mode=mode, invert_terrain=invert)
        model = tokenizer = None
        if model_name != "synthetic":
            model, tokenizer = load_model_cached(model_name)
        with st.spinner("Capturing forward pass…"):
            st.session_state["result"] = run_pipeline(cfg, prompt, model=model, tokenizer=tokenizer)
            st.session_state["cfg"] = cfg

    result = st.session_state.get("result")
    if result is None:
        st.info("Enter a prompt and press **Run capture** to explore the latent manifold.")
        return

    cfg: MarbleConfig = st.session_state["cfg"]
    traj: StateTrajectory = result["traj"]

    with st.sidebar:
        layer = st.slider("Layer", 0, traj.n_layers - 1, traj.n_layers - 1, key="layer")

    @st.cache_resource(show_spinner=False)
    def _neighbor_cache(key: str):  # one TokenNeighbors per capture
        return {}

    def _token_neighbors(t: StateTrajectory) -> TokenNeighbors:
        holder = _neighbor_cache(cache_mod.make_key(t.meta.get("model"), t.meta.get("prompt")))
        if "tn" not in holder:
            holder["tn"] = TokenNeighbors(t.embedding_matrix, t.vocab)
        return holder["tn"]

    col_viz, col_info = st.columns([3, 1])

    with col_viz:
        fig = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                     current_layer=layer, frames_per_layer=cfg.frames_per_layer,
                     frame_ms=cfg.frame_ms)
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

        with st.expander("Research metrics", expanded=False):
            summary = metrics_mod.summarize(traj, result["coords"], token=token)
            for name, value in summary.items():
                st.write(f"{name.replace('_', ' ')}: **{value:.3f}**")


if __name__ == "__main__":
    main()
