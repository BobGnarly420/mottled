# Releasing Mottled

A release is a claim that someone can install this and use it. Most of what
follows exists because the test suite cannot check that claim: it runs in a
git checkout, where every file is present because git put it there. A wheel is
a different object.

## Before tagging

Run these in order. Each one has failed for real at least once.

```bash
pytest -m "not network" -q          # what CI runs
pytest -m network -q                # the HF-hub tests CI skips — needs the hub
node --test viewer/tests/           # the viewer's parsers and ports
```

```bash
# 1. The package, as a user gets it — not as the repo has it.
python -m build                     # must emit no setuptools warnings
python -m venv /tmp/rel && /tmp/rel/bin/pip install dist/mottled-*.whl
cd /tmp && /tmp/rel/bin/mottled smoke        # run OUTSIDE the checkout
```

`mottled smoke` is the gate: run from inside the repo it will pass on files
the wheel never shipped. `attractor` and `mweights` were missing from
`py-modules` for two releases — a pip-installed Mottled could not `import ui`
at all — and every test passed the whole time.

```bash
# 2. The parity matrix: does Mottled still see what the references see?
mottled parity -o parity.json --markdown parity.md
```

Non-zero exit means a comparison exceeded tolerance. **Do not release on an
unrun matrix**: the README advertises parity with HuggingFace, TransformerLens
and NNsight, and advertising a harness nobody has run is the overclaiming
`docs/validity.md` exists to prevent. Paste the table into the release notes.

```bash
# 3. The WebGPU kernels, which CI cannot verify (no GPU on the runner).
python -m http.server   # then open /viewer/tests/parity.html in a real browser
```

```bash
# 4. A dry run of the publish workflow, which builds but does not publish
#    unless the ref is a tag.
gh workflow run publish.yml         # or the Actions tab → Publish → Run workflow
```

## Tagging

1. Rename the CHANGELOG's `## Unreleased` heading to `## X.Y.Z — YYYY-MM-DD`
   and add an `## Unreleased` above it.
2. Check `version` in `pyproject.toml` matches. `provenance.record` reports it
   into every scene, so a stale version mislabels artifacts that outlive it.
3. Commit, then `git tag vX.Y.Z && git push origin vX.Y.Z`.

The tag push is what publishes: `publish.yml` runs on `refs/tags/v*` and uploads
to PyPI through Trusted Publishing (OIDC, no stored token). It needs the repo
registered as a trusted publisher for the `mottled` project and a GitHub
environment named `pypi`; until both exist the build job still runs and uploads
artifacts, so tagging is safe but does not publish.

## What a version number promises

Mottled is pre-1.0. Minor versions may break things; patch versions never do.
Four surfaces carry a stronger promise than that, because other people's work
depends on them:

**The `.mtj` format** (`docs/mtj-format.md`) has its own version, independent
of the package's. It bumps *only* on a breaking layout change. New manifest
fields and new arrays are additive and do not bump it, because every reader is
required to ignore what it does not know — which is how a scene written today
still opens in a viewer from a year ago, and why `analysis` could be added to
both file kinds without a version change.

**The flat API** — `from ui import run_pipeline, run_scene, run_compare,
run_intervention, render` and the `attach_*` helpers — is what the README,
the tests and outside notebooks import. It keeps working across minor
versions; `ui.py` re-exports from `pipeline.py` and `render.py` precisely so
that the split did not break callers.

**The analysis record** is versioned by its own `schema` string
(`mottled-analysis/1`). Fields are added, never repurposed; a consumer that
reads a field it knows keeps reading it.

**`MarbleConfig` fields** are additive with defaults. A config pickled by an
older version still loads, which is what `cache.py` depends on.

Anything else — module layout, private helpers, the viewer's internals — may
move in a minor version. If you depend on one of those, pin exactly.

## After tagging

- Confirm the PyPI page shows the new version and that `pip install mottled`
  in a clean venv passes `mottled smoke`.
- Confirm GitHub Pages redeployed (`pages.yml` runs on pushes to `main`) and
  that the hosted viewer still loads a sample scene.
