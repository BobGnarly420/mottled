"""One source of truth for the design language (design_tokens.py).

The Streamlit config and the WebGL viewer mirror the same values; this guard
fails if any of them drifts, so retheming is a one-line edit instead of the
three-file hunt the previous layout invited.
"""
import re
import tomllib
from pathlib import Path

import design_tokens as T

ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_config_mirrors_tokens():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    theme = cfg["theme"]
    assert theme["primaryColor"].lower() == T.ACCENT.lower()
    assert theme["backgroundColor"].lower() == T.BASE.lower()
    assert theme["secondaryBackgroundColor"].lower() == T.SURFACE_0.lower()
    assert theme["textColor"].lower() == T.FG_1.lower()


def test_viewer_css_root_mirrors_tokens():
    css = (ROOT / "viewer" / "style.css").read_text()

    def var(name: str) -> str:
        m = re.search(rf"--{name}:\s*(#[0-9A-Fa-f]{{6}})", css)
        assert m, f"--{name} not found in viewer/style.css"
        return m.group(1).lower()

    assert var("color-base") == T.BASE.lower()
    assert var("color-surface-0") == T.SURFACE_0.lower()
    assert var("color-accent") == T.ACCENT.lower()
    assert var("color-fg-1") == T.FG_1.lower()


def test_ui_consumes_the_token_source():
    import ui

    assert ui._MARBLE_COLORS is T.MARBLE_COLORS
    assert ui._TERRAIN_COLORSCALE is T.TERRAIN_COLORSCALE
