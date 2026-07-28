# Changelog

## Unreleased

### Real SAEs, measured (roadmap M2, first slice)
- `sae.fetch_from_hub`: download a trained SAE from any SAELens-format HF
  repo (default: `jbloom/GPT2-Small-SAEs-Reformatted`, layer-8 resid_pre)
  with no `sae-lens` dependency — same fail-closed gates (standard ReLU
  only, no activation normalization, exact `apply_b_dec_to_input` fold,
  now shared via `_require_standard`). `mottled-convert-sae fetch` drives
  it from the CLI.
- `sae.fit_report`: measured calibration of a dictionary against a capture
  (per-layer median reconstruction error + firing density, best layer).
  Provenance is not calibration — public GPT-2 SAEs are trained on
  TransformerLens-processed residuals (folding/centering changes residual
  values while preserving the function), so the same SAE reads ~24% error
  on a TL capture and ~340% on raw HF states. The explorer prints the
  measured fit wherever features are shown and calls a bad fit
  extrapolation, pointing at the calibrated TL pairing.
- The explorer defaults GPT-2-width captures to the trained dictionary
  (fetched once, cached; untick for the demo), and feature labels flow
  through `SAE.feature_label` everywhere.

### Generation axis (roadmap M1)
- `ROADMAP.md`: the end goal (Mottled 1.0 — the honest atlas of latent
  dynamics) decomposed into milestones M1–M6.
- `capture.generate_and_capture`: autoregressive decode (greedy or seeded
  sampling, EOS-aware) followed by a single capture of the completed
  sequence — exact for causal models, so the decode axis costs no new
  interchange type. Per-step decode records (token, id, probability,
  entropy of the actual sampling distribution) travel in
  `meta["generation"]`; tests pin causal exactness and step fidelity
  against the model's own stepwise forward passes.
- The synthetic backend generates with its own logit lens
  (`models.synthetic.generate_and_capture`), so the decode axis works
  without torch.
- The decode axis is wired through every surface: `MarbleConfig`
  (`generate_tokens`, `generate_temperature`; cache keys bumped), scene
  `.mtj` files (additive per-run `generation` record), the capture API
  (`POST /api/scene` accepts `generate` / `temperature`, validated),
  `mottled export --generate N [--temperature T]`, and the explorer
  (decode knobs, inline decode header, `+`-prefixed open-diamond
  generated trajectories, per-step decode inspector).

### Substance: real analysis, not demo
- `sae.from_sae_lens` / `sae.from_state_dict` load a **real, trained** SAE
  (SAELens' standard SAE is a direct array copy of Mottled's ReLU forward);
  non-standard architectures (gated/JumpReLU/top-k) are rejected loudly, and an
  `apply_b_dec_to_input=False` SAE is folded into `b_enc` so it still converts
  exactly. The `mottled-convert-sae` CLI (`convert_sae.py`) drives it (`sae-lens`
  optional).
- `intervene.direction_from_token` / `direction_from_contrast`: steering deltas
  derived from data (an embedding axis, a diff-of-means) instead of hand-picked
  numbers. `intervene.faithfulness` + `target_logit_shift` score a steer against
  a **norm-matched random control**, so a flip's cause can be attributed to the
  direction rather than the perturbation size. Surfaced in `ui.run_intervention`.
- `models.hooked.from_hooked_transformer`: optional **TransformerLens**
  producer — any `HookedTransformer` becomes a `StateTrajectory` (no hard dep).

### Trust: fidelity made unavoidable
- The explorer prints a **projection-fidelity header** above every scene and
  flags low-preservation states with an amber ✕ on the terrain; the collapsed
  Uncertainty panel is no longer the only place fidelity is shown.
- `attractor.explain(..., quality=)` folds the basin's own neighborhood
  preservation into the prose ("suggestive, not established" when low).
- New **"What this is — and is not"** panel (UI) + README section; the web
  viewer's uncertainty overlay now defaults **on** when a scene carries SE.

### Adoption & sustainability
- **Relicensed to Apache-2.0** (from GPL-2.0) to match the interpretability
  ecosystem; added `NOTICE`.
- `design_tokens.py` is now the **single source of truth** for the design
  language; `.streamlit/config.toml` and `viewer/style.css` mirror it and
  `tests/test_tokens.py` fails on drift.
- Dependency **floors** in `pyproject.toml`/`requirements.txt`, a pinned
  `requirements.lock`, optional `tlens`/`sae` extras, and a **PyPI publish**
  workflow (Trusted Publishing).
- A cross-language **`.mtj` conformance test** (Python writer ↔ `viewer/mtj.js`
  reader + little-endian header). `bvh.py` is marked experimental (not yet
  wired into a live surface).

### Uncertainty visualization
- `projection.projection_quality`: measures how much a fitted projection
  distorts, per state — k-NN neighborhood preservation for any projection,
  plus reconstruction residual and explained variance for linear ones. The
  2-D picture is lossy and now says where.
- `density.compute_density(..., bootstrap=B)`: resamples the points `B`
  times and records the per-cell standard error of the density
  (`Landscape.density_se`) — a confidence field over the terrain. New
  `MarbleConfig.density_bootstrap` (default 24; cache keys bumped).
- The explorer gains an **Uncertainty** inspector panel (explained variance,
  neighborhood preservation for the selected state and per layer, density
  bootstrap SE). Scene `.mtj` files now carry `terrain.density`,
  `terrain.se`, and per-run `quality` arrays (all optional, additive).
- The web viewer gains an **uncertainty** toggle that recolors the terrain by
  its bootstrap SE, and shows per-state neighborhood fidelity on hover. The
  bundled sample scenes were regenerated to carry these layers.

### Explanatory layer
- `attractor.py`: measures why the density basin forms (per-layer step
  deceleration, settle layer), what it is made of (membership roster above
  a density threshold), and what it means (readout stabilization, entropy
  collapse, attn/MLP share of the settled writes). `explain` turns a
  report into prose generated entirely from the measurements.
- The explorer pins a measured callout to the density peak, captions the
  terrain as a density field over the states themselves, and adds a
  "Why this attractor" inspector panel with step and entropy profiles.

### SAE feature field (domain coloring)
- `projection`: PCA (exact) and UMAP (approximate) gain
  `inverse_transform` — plane coordinates back to hidden space.
- `sae.feature_field`: the SAE evaluated over the projection plane — the
  complex-plane domain-coloring analogue, with the dominant feature as the
  phase and its activation as the modulus.
- `ui.render_feature_field`: flat domain-coloring view (golden-angle hues,
  magnitude-octave rings, trajectory overlay) and a relief view lifting
  activation into z; new "SAE feature field" toggle in the explorer.
- Pipeline results now carry the fitted `projector` (cache keys bumped).

## 0.1.0 — 2026-07-14

First versioned release. Everything below landed since the MVP.

### Core
- `StateTrajectory` established as the project's interchange format:
  producers (transformers capture, Mamba, synthetic) emit one; analyses and
  viewers consume one.
- `.mtj` binary format v1 (`statefile.py`, spec in `docs/mtj-format.md`):
  full-fidelity trajectory files and compact viewer-ready scene bundles,
  with explicit forward-compatibility rules tested from Python and JS.
- Residual-stream capture with logit lens, resumable forward pass with
  causal interventions (perturb / set / noise / freeze), head-averaged
  attention capture, and exact attn/MLP residual decomposition (pinned
  against HF reference outputs on locally-built Llama and GPT-2).
- Mamba (state-space) producer via structural layout resolution — the
  abstraction is not transformer-shaped.

### Analysis
- Trajectory comparison: symmetric Hausdorff, dynamic time warping,
  shared-prefix alignment, layerwise divergence profiles.
- SAE features (applied, never trained): npz interchange, demo dictionary,
  per-state activations and top-features.
- Research metrics: path length, curvature, semantic drift, entropy
  collapse, neighbor stability, component shares.

### Viewers
- Streamlit explorer: A/B and N-prompt scenes on one shared terrain,
  animated marbles with a layer scrubber, token inspector (predictions,
  neighbors, SAE features, residual decomposition, attention), interactive
  patching panel, scene export.
- Dependency-free WebGL web viewer for `.mtj` scenes: terrain, densified
  trajectories, marbles, orbit camera, hover inspector, attention flow,
  run toggles, comparison table, drag-and-drop loading, and a capture form
  that appears when the backend is present.
- Everything styled to one design language (Incision): dark navy void,
  precision-blue accent, semantic data palette, mono-for-data typography.

### Distribution
- Pip-installable package (`pip install mottled`) with a `mottled` CLI:
  explorer (default), `serve` (viewer + capture API), `export` (prompts →
  `.mtj`).
- `serve.py`: standard-library capture backend the viewer discovers at
  runtime, so the browser can generate trajectories directly.
- GitHub Pages deployment: landing page + viewer + sample scenes
  (synthetic and real GPT-2 captures).
- CI: pytest (offline, including torch mechanism tests on locally-built
  models) + Node tests for the viewer's `.mtj` parser.
