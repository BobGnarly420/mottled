"""`mottled smoke` — does this install actually work?

A release is a claim that someone can install Mottled and use it. The test
suite does not check that claim: it runs in a clone, where every file is
present because git put it there. A wheel is a different object, and the
failure it produces is not an ImportError at install time — it is `mottled
serve` answering 404 for the URL it just printed, or a module that was never
added to `py-modules`, discovered by the first person who pip-installed.

So this runs against the installed package, not the repository: it imports the
documented flat API, resolves the viewer next to the server that serves it,
round-trips a `.mtj`, reads a bundled sample, and builds an analysis record.
Seconds, no network, no model weights — a core install with none of the extras
must pass it, because that is what most people will have.

    mottled smoke          # exits non-zero if any required check fails

Optional backends are reported, never required: a missing `torch` is a fact
about this install, not a broken one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The flat public API CLAUDE.md documents and every caller imports. Named here
# rather than probed, so dropping one is a smoke failure and not a surprise in
# somebody's notebook.
FLAT_API = ["run_pipeline", "run_scene", "run_compare", "run_intervention",
            "run_model_scene", "attach_inspector", "attach_features",
            "attach_manifest", "render", "degraded_note"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


def run_checks() -> list[Check]:
    return [
        _flat_api(),
        _viewer_assets(),
        _bundled_sample(),
        _trajectory_roundtrip(),
        _analysis_record(),
        _projection_stack(),
        _capture_backend(),
    ]


def _flat_api() -> Check:
    try:
        import ui
    except Exception as exc:
        return Check("flat API", False, f"import ui failed: {exc}")
    missing = [name for name in FLAT_API if not hasattr(ui, name)]
    return Check("flat API", not missing,
                 f"missing from ui: {', '.join(missing)}" if missing
                 else f"{len(FLAT_API)} names importable from `ui`")


def _viewer_assets() -> Check:
    """The check that a wheel fails and a clone cannot.

    `serve.py` resolves its static root next to itself, so the viewer has to
    be installed beside it. Every script the page asks for is resolved, not
    just the directory: a new `.js` that no package-data pattern matches would
    otherwise ship a page that loads and then does nothing.
    """
    try:
        import serve
    except Exception as exc:
        return Check("viewer assets", False, f"import serve failed: {exc}")

    index = Path(serve.ROOT) / "viewer" / "index.html"
    if not index.is_file():
        return Check("viewer assets", False,
                     f"no viewer/index.html under {serve.ROOT} — `mottled "
                     "serve` would 404 the page it prints")
    html = index.read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"?#:]+\.(?:js|css))"', html)
    missing = [r for r in refs if not (index.parent / r).is_file()]
    return Check("viewer assets", not missing,
                 f"index.html references missing files: {', '.join(missing)}"
                 if missing else f"index.html + {len(refs)} assets resolve")


def _bundled_sample() -> Check:
    try:
        import serve
        import statefile
    except Exception as exc:
        return Check("bundled sample", False, f"import failed: {exc}")

    samples = sorted((Path(serve.ROOT) / "viewer" / "samples").glob("*.mtj"))
    if not samples:
        return Check("bundled sample", False,
                     "no viewer/samples/*.mtj — the viewer's default scenes "
                     "did not ship")
    try:
        scene = statefile.load_scene(samples[0])
    except Exception as exc:
        return Check("bundled sample", False, f"{samples[0].name}: {exc}")
    if not scene.get("runs"):
        return Check("bundled sample", False, f"{samples[0].name} has no runs")
    return Check("bundled sample", True,
                 f"{len(samples)} scenes; {samples[0].name} loads "
                 f"({len(scene['runs'])} run(s))")


def _trajectory_roundtrip() -> Check:
    """The interchange format, end to end, with no model in sight."""
    import io

    try:
        import statefile
        from trajectory import StateTrajectory

        rng = np.random.default_rng(0)
        traj = StateTrajectory(
            hidden=rng.normal(size=(3, 4, 8)).astype(np.float32),
            tokens=["a", "b", "c", "d"], meta={"backend": "smoke"})
        traj.validate()
        buf = io.BytesIO()
        statefile.save(traj, buf)
        buf.seek(0)
        back = statefile.load(buf)
    except Exception as exc:
        return Check(".mtj round-trip", False, f"{type(exc).__name__}: {exc}")
    same = np.array_equal(back.hidden, traj.hidden) and back.tokens == traj.tokens
    return Check(".mtj round-trip", same,
                 "states returned changed" if not same else "(3, 4, 8) survives")


def _analysis_record() -> Check:
    try:
        import provenance
        from config import MarbleConfig

        record = provenance.record(MarbleConfig())
        json.dumps(record)
    except Exception as exc:
        return Check("analysis record", False, f"{type(exc).__name__}: {exc}")
    return Check("analysis record", record.get("schema") == provenance.SCHEMA,
                 f"{record.get('schema')} · mottled {record.get('mottled')}")


def _projection_stack() -> Check:
    """Projection → density → terrain on random states: the analysis half of
    the tool, which a core install has to be able to run on its own."""
    try:
        import density
        import projection
        import terrain

        rng = np.random.default_rng(1)
        hidden = rng.normal(size=(4, 5, 12)).astype(np.float32)
        coords, projector = projection.project(hidden)
        landscape = density.compute_density(coords, grid_size=16, bootstrap=0)
        mesh = terrain.mesh(landscape)
    except Exception as exc:
        return Check("projection stack", False, f"{type(exc).__name__}: {exc}")
    ok = coords.shape == (4, 5, 2) and np.isfinite(mesh.z).all()
    return Check("projection stack", ok,
                 "non-finite terrain" if not ok else
                 f"pca → kde → mesh {mesh.z.shape}")


def _capture_backend() -> Check:
    """Informational: capture needs the models extra, analysis does not."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return Check("capture backend", True, required=False,
                     detail=f'absent ({exc.name}) — analysis and the viewers '
                            'work; capture needs `pip install "mottled[models]"`')
    return Check("capture backend", True, required=False,
                 detail=f"torch {torch.__version__}, "
                        f"transformers {transformers.__version__}")


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "ok  " if c.ok else "FAIL"
        if not c.required and c.ok:
            mark = "note"
        lines.append(f"  [{mark}] {c.name:<18} {c.detail}")
    failed = [c for c in checks if c.required and not c.ok]
    lines.append("")
    lines.append(f"{len(checks) - len(failed)}/{len(checks)} checks passed"
                 if failed else "all checks passed")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    checks = run_checks()
    print("mottled smoke")
    print(format_checks(checks))
    return 1 if any(c.required and not c.ok for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
