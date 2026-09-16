"""What a wheel contains, checked in the clone where the wheel is not.

The suite runs against a git checkout, where every file is present because git
put it there. A wheel is a different object, and its failures are invisible
here: `attractor` and `mweights` were absent from `py-modules` for two
releases, which left a pip-installed Mottled unable to `import ui` — the
explorer, the documented flat API and `mottled serve` all broken on an install
that passed every test.

So these assert on the packaging *configuration*, which is the thing that was
wrong. `mottled smoke` is the other half: it runs against the installed
package and fails the same way a user would.
"""
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text())
SETUPTOOLS = CONFIG["tool"]["setuptools"]


def test_every_top_level_module_ships():
    """A module the tree has and the wheel does not is an ImportError for
    everyone who installed rather than cloned."""
    declared = set(SETUPTOOLS["py-modules"])
    # conftest is pytest's, not the package's
    on_disk = {p.stem for p in ROOT.glob("*.py")} - {"conftest"}
    assert on_disk - declared == set(), "missing from py-modules"
    assert declared - on_disk == set(), "declared in py-modules but not present"


def test_declared_modules_import():
    """Listing a module is not the same as it being importable."""
    import importlib

    for name in SETUPTOOLS["py-modules"]:
        importlib.import_module(name)


def test_the_viewer_ships_beside_the_server_that_serves_it():
    """serve.py resolves its static root next to itself, so the viewer has to
    be installed there — not merely present in the repository."""
    assert "viewer" in SETUPTOOLS["packages"]
    # setuptools recognises viewer/samples as its own package; a glob-only
    # include is ambiguous and warns, which is how samples quietly stop shipping
    assert "viewer.samples" in SETUPTOOLS["packages"]


def test_package_data_covers_every_asset_the_page_asks_for():
    """A new .js that no pattern matches ships a page that loads and then does
    nothing — the failure is a blank scene, not an error."""
    import fnmatch

    patterns = SETUPTOOLS["package-data"]["viewer"]
    html = (ROOT / "viewer" / "index.html").read_text()
    refs = re.findall(r'(?:src|href)="([^"?#:]+\.(?:js|css))"', html)
    assert refs, "index.html references no local assets — parser drifted?"
    for ref in refs:
        assert any(fnmatch.fnmatch(ref, pat) for pat in patterns), \
            f"viewer/{ref} matches no package-data pattern {patterns}"


def test_bundled_samples_are_declared():
    patterns = SETUPTOOLS["package-data"]["viewer.samples"]
    samples = list((ROOT / "viewer" / "samples").glob("*.mtj"))
    assert samples, "no bundled scenes to ship"
    assert any(p.endswith(".mtj") or p == "*" for p in patterns)


def test_console_scripts_point_at_modules_that_ship():
    for entry in CONFIG["project"]["scripts"].values():
        module = entry.split(":")[0]
        assert module in SETUPTOOLS["py-modules"], \
            f"console script {entry!r} needs {module} in py-modules"


@pytest.mark.parametrize("extra", ["models", "umap", "faiss", "tlens", "sae",
                                   "nnsight", "dev"])
def test_optional_extras_are_declared(extra):
    """The README and the error messages tell people to install these by name."""
    assert extra in CONFIG["project"]["optional-dependencies"]
