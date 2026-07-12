"""Animation continuous + UI interactive (spec tests 7-8)."""

import numpy as np
import pytest

from config import MarbleConfig
from ui import render, run_pipeline


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return MarbleConfig(model="synthetic", use_cache=True,
                        cache_dir=str(tmp_path_factory.mktemp("cache")))


@pytest.fixture(scope="module")
def result(cfg):
    return run_pipeline(cfg, "The capital of France is")


def test_pipeline_artifacts(cfg, result):
    traj = result["traj"]
    L, T = traj.n_layers, traj.n_tokens
    assert result["coords"].shape == (L, T, 2)
    assert result["mesh"].z.shape == (cfg.grid_size, cfg.grid_size)
    assert len(result["trajectories"]) == T  # all_tokens mode
    assert all(t.points.shape == (L, 3) for t in result["trajectories"])


def test_pipeline_cache_roundtrip(cfg, result):
    again = run_pipeline(cfg, "The capital of France is")
    assert np.array_equal(again["coords"], result["coords"])


def test_render_figure(cfg, result):
    fig = render(result["traj"], result["mesh"], result["trajectories"],
                 result["fine_paths"], frames_per_layer=cfg.frames_per_layer)
    types = [tr.type for tr in fig.data]
    assert types[0] == "surface"
    assert types.count("scatter3d") == result["traj"].n_tokens + 1  # lines + marbles
    assert len(fig.frames) == min(len(p) for p in result["fine_paths"])
    assert fig.layout.sliders and fig.layout.updatemenus  # scrubber + play/pause


def test_marble_animation_is_continuous(result):
    """Marble positions across frames form a gap-free path on the terrain."""
    for path in result["fine_paths"]:
        assert np.isfinite(path).all()
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        span = np.linalg.norm(path.max(axis=0) - path.min(axis=0))
        assert steps.max() < 0.5 * span  # no teleporting between frames
        z = result["mesh"].z
        assert path[:, 2].min() >= z.min() - 0.1  # stays on/above the terrain


def test_streamlit_app_interactive():
    """Drive the real Streamlit app headlessly: run capture, scrub, inspect."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    at = AppTest.from_file("ui.py", default_timeout=120)
    at.run()
    assert not at.exception

    at.text_area(key="prompt").set_value("The capital of France is")
    at.selectbox(key="model").select("synthetic")
    at.button(key="run").click()
    at.run()
    assert not at.exception
    assert at.session_state["result"] is not None

    traj = at.session_state["result"]["traj"]
    at.slider(key="layer").set_value(0)
    at.selectbox(key="token").select(traj.n_tokens - 1)
    at.run()
    assert not at.exception
