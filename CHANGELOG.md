# Changelog

## Unreleased

### The inferential contract (docs/validity.md)
The tool's central research-validity risk — an attractive within-run
visualization mistaken for evidence of a model mechanism — now has a
dedicated answer instead of scattered caveats.
- **`docs/validity.md`**: what each Mottled artifact licenses you to claim,
  from the projection robustness envelope through the SAE claim gates to
  researcher degrees of freedom — including the tool's own known limits
  (the i.i.d. density bootstrap understates uncertainty on dependent
  states; the honest upgrades are named and marked unimplemented).
- **Framing tightened to match**: a basin is a *state concentration region*
  under the chosen projection and estimator; "semantic manifold" is gone
  from README and site; neighbors are labeled *representation-space*
  neighbors everywhere; the explorer and README's "What this is — and is
  not" open with the one-sentence boundary and link the contract.

### Features with names (roadmap M2)
An SAE's features are indices until something explains them, and an unnamed
feature overlay is a colour with no meaning.
- `sae.fetch_labels` retrieves Neuronpedia's auto-interp explanations for the
  features that actually fire — lazy (there is no bulk endpoint without a
  key), disk-cached so a second look is free, and it never raises for network
  reasons: an unreachable source just leaves the bare indices.
  `sae.apply_labels` writes them onto a dictionary, leaving unexplained
  features as `fN`.
- **The provenance travels with the text.** These explanations are written by
  a language model reading a feature's top activations, so `FeatureLabel`
  carries `explained_by` and `method`, and `ui._label_provenance` states in
  the explorer that they are auto-generated descriptions of what a feature
  *correlates with*, not of what it computes — leads, not labels.
- Checked against the live source: the strongest feature on "The capital of
  France is" is published as *"locations or cities specifically denoted as
  'capital' in the text"*.

### Real models, and a memory bomb they exposed
Mottled had been *demonstrated* on GPT-2 throughout, which invited the fair
question of whether it handles anything modern. It does, and that is now
verified rather than asserted — but proving it found a bug.
- **Qwen2.5-1.5B-Instruct** (29 x 1536, GQA / RoPE / SwiGLU / RMSNorm)
  captures end to end with attention and an *exactly* reconciling residual
  decomposition (`max |h[l+1] - (h[l] + attn + mlp)| = 0.0000`). New samples
  `qwen-capitals.mtj` and `models-qwen-gpt2.mtj`; `MODEL_CHOICES` gains
  Qwen2.5-1.5B and marks which entries are licence-gated.
- **Fixed an O(V^2) allocation in readout space.** `readout_trajectory` gave
  its trajectories `np.eye(V_shared)` as an embedding matrix — conceptually
  tidy, since each axis *is* a vocabulary entry, and a **7 GB** array for two
  models sharing 42k tokens. It killed the process outright on the
  Qwen/GPT-2 pair. The identity bought nothing either: the nearest vocabulary
  token to a readout state is its largest component, which `topk` already
  reports. Now `None`, which every consumer already had to handle.
- The viewer now reads the per-run `model` field (added writer-side in the
  previous change but never surfaced) and shows it in the runs panel — the
  point of a cross-model scene.

### Viewer inspector parity (roadmap M3)
The web viewer is the *shareable* surface — it is what someone sees when you
send them a link — but its inspector showed less than the explorer's, because
the two readouts it lacked depend on data far too large to ship in a scene:
the `(V, D)` embedding matrix (semantic neighbors) and the
`2 × (L-1) × T × D` residual components (the attention/MLP split).
- `pipeline.attach_inspector` resolves both *before* export, where they
  collapse to almost nothing: the k nearest vocabulary tokens per state as
  indices into a compact table of only the strings that appear, and the
  attn/MLP share per state, which is two numbers rather than two
  D-dimensional vectors. A 15-token GPT-2 scene gains ~0.9 s and stays 90 KB.
- Carried additively in `.mtj` as a per-run `inspector` record (documented in
  `docs/mtj-format.md`), attached by the explorer's export button and by
  `mottled export`. Runs missing either source contribute what they have
  rather than failing the export.
- The bundled `scene-abc`, `single` and `gpt2-capitals` samples were
  regenerated to carry it.

### Fixed: Streamlit app tests broke on Streamlit 1.61
`AppTest.from_file` resolves a *relative* path against the file that calls
it as of 1.61 (older releases resolved against the working directory), so
`"ui.py"` became `tests/ui.py` and six app tests failed — on `main` as much
as anywhere, the moment CI installed the new release. The path is now
absolute, in one place (`tests/apptest.py`), so the tests no longer depend on
which behaviour the installed Streamlit has. Verified against 1.61 itself.

### Audit: within-model assumptions applied across models
Until three changes ago, "two runs" always meant "two prompts through one
model". Cross-model scenes made that false, and two bugs of the same shape
had already surfaced (`_assemble_scene`, the explorer's A/B panel), so the
rest of the codebase was swept for the same assumption. Three more found:

- **The layer scrubber silently truncated to the shallowest run.** `render`
  sized the animation with `min` over the runs' path lengths, so on a
  13-layer vs 7-layer scene the slider stopped at layer 6 and the deeper
  model's entire second half was unreachable — with nothing saying so. It now
  spans the deepest run; a shorter run's marble rests at its final state.
- **Scene files identified the model only globally.** `meta` describes run 0,
  which is simply wrong for a scene whose runs are different models. Each run
  now carries its own `model` (additive).
- **The trained SAE was offered by hidden width alone.** Width identifies a
  model's residual stream only when the states *are* a residual stream; a
  readout-space trajectory's axes are vocabulary entries, so a width
  collision would have offered a residual SAE for probability vectors. The
  guard now keys on the space, not the number.

### The atlas, reachable
The cross-model work was API- and CLI-only; the explorer can now drive it.
- A **"Compare models"** field in the sidebar: name other models and they are
  drawn on the same terrain for the current prompt (via
  `pipeline.run_model_scene`, in readout space).
- A **Model comparison** panel (`ui.render_model_comparison`, pure — it takes
  the Streamlit module as an argument, so its wording is tested without a
  browser) reporting readout divergence per normalized layer, the CKA layer
  alignment *with which rows are too flat to be believed*, and the
  step-by-step generation split when both runs carry a decode record.
- Fixed: the A/B comparison panel called the layer-for-layer `compare` on
  whatever two runs were present, which raises on runs of differing depth.
  It is now shown only where it is defined — the same class of bug the
  cross-model scene hit in `_assemble_scene`.

### The decode axis crossed with the model axis
The two axes Mottled added this cycle — generation (M1) and models (M6) —
now meet: *where do two models' generations part company?*
- `crossmodel.compare_generations`: two free-running generations of the same
  prompt, step by step — who chose what, with what probability and what
  spread. It reports `comparable_steps` and stops there, because **after the
  models choose differently they are continuing different texts**, and a
  step-by-step divergence past that point compares answers to different
  questions. The boundary is measured, not glossed.
- `crossmodel.forced_divergence`: both models scored on one fixed text
  (teacher forcing), so every position stays like-for-like all the way down —
  "would B have said what A said, here?" Needs identical tokenization, for
  the same reason `layer_similarity` does, and refuses otherwise.
- On GPT-2 vs DistilGPT-2 given *"The residual stream moves, turns, and
  settles"*, both complete it with `" into"` and part company on the very
  next token: GPT-2 continues `" the ground."`, DistilGPT-2 `" a new state
  of equilibrium."`

### The cross-model atlas (roadmap M6)
`compare.py` could only compare runs inside one model — same depth, same
width. `crossmodel.py` compares *models*, which share neither, and often not
a tokenizer either. It builds on the one thing they do share: the text they
read out into.

- **Readout space** (`crossmodel.readout_space`): every state becomes the
  next-token distribution it predicts over the vocabulary all the models
  share, with the mass spent on tokens only one model knows kept in a visible
  `⟨unshared⟩` bucket rather than renormalised away. It is a real shared
  coordinate system, so `project_joint`, the terrain and both viewers accept
  it unchanged — different architectures on one manifold.
- `crossmodel.compare_models` measures where two models' readouts diverge,
  layer by layer (Jensen-Shannon on the shared vocabulary), resampling onto a
  common depth first, because layer 6 of a 12-layer model is not layer 6 of a
  32-layer one.
- `crossmodel.layer_similarity` answers the other question — which layer of B
  matches layer *l* of A — with CKA, which needs no alignment because it is
  invariant to width, rotation and scale. It requires *paired* states and
  decides that on the token **strings**, not their count: two tokenizers can
  produce the same number of pieces while cutting the text in different
  places, and pairing those would compare non-counterparts.
- **The measurement reports whether it is identified.** CKA on a raw residual
  stream saturates: a few very-high-variance dimensions shared by every layer
  dominate, so every layer looks ~1.0 similar to every other. On GPT-2 vs
  DistilGPT-2 over 45 tokens, raw CKA leaves the middle rows flat to within
  0.001 — an argmax that is pure noise — while z-scoring each dimension
  recovers a **monotone, proportional** correspondence (13 layers onto 7).
  Standardizing is therefore the default, `LayerAlignment.contrast` reports
  how far each row's winner beats its field, and the docstring states the
  cost of the trade: exact isotropic-scale invariance is kept, exact rotation
  invariance is not (~0.96 for a rotated copy).
- `pipeline.run_model_scene` builds a scene from several models on one
  prompt; `mottled export PROMPT --models a,b` writes it. New sample
  `viewer/samples/models-gpt2-distilgpt2.mtj`.
- Fixed along the way: `_assemble_scene` crashed when runs had different
  depths (it always ran the layer-for-layer `compare`). The pairwise table is
  now simply absent where it is undefined, instead of fabricated or fatal.

## 0.2.0 — 2026-07-29

Two new axes (generation, closed models), real SAEs with their calibration
measured rather than assumed, honest picking shared between both viewers, and
a smaller `ui.py`.

### A smaller core (roadmap M5)
- `ui.py` (1262 lines, carrying the pipeline, two renderers and the app at
  once) is split into **`pipeline.py`** (capture → project → density →
  terrain → paths) and **`render.py`** (the Plotly scene and the SAE feature
  field), leaving `ui.py` the Streamlit shell. Both new modules are pure —
  no Streamlit, no browser — so a scene builds and draws identically from a
  notebook, a script, the CLI or the server.
- **The flat public API is unchanged**: `ui` re-exports everything, so
  `from ui import run_pipeline, render, run_scene, …` keeps working exactly
  as the README documents. New code should import from `pipeline` / `render`.

### Picking you can trust (roadmap M3)
- `viewer/bvh.js` ports `bvh.py`'s BVH over trajectory segments to the web
  viewer, and **`tests/test_bvh_conformance.py` pins the two together** —
  identical segments and rays through both implementations, comparing the
  picked index, ray parameter, distance and point. The same ray must choose
  the same segment in both languages, which is what makes the explorer and
  the viewer one tool. This retires `bvh.py`'s "not wired into a live
  surface" caveat.
- The viewer now picks by casting a real camera ray at that index instead of
  scanning stored layer points in screen space: the cursor grabs **anywhere
  along a trajectory**, reads the **fractional layer** it landed at
  ("layer 8.4"), and costs a BVH descent per run rather than O(N·L) per
  frame. Pick tolerance tightened 14px → 6px — a continuous line needs no
  slack, and the tolerance is also the worst-case error when lines bundle.
- **Click-to-pin**: a click freezes the inspector on that reading (Escape or
  a click on empty space clears it); orbit and pan never pin.

### Closed-model producer (roadmap M4)
- `models/logprobs.py`: per-step API top-k logprobs → `StateTrajectory`.
  A hosted API exposes no residual stream, so depth is unavailable: the
  animated axis becomes **decode time** and the moving point is the model's
  own output distribution over the observed tokens. The mass the API did
  *not* report gets its own visible `⟨unreported⟩` bucket rather than being
  renormalised away, so each step's vector sums to 1 honestly.
- The trajectory states its own ceiling — `meta.degraded`, `meta.absent`
  (residual stream, per-layer readout, attention, decomposition), and
  `entropy_is_lower_bound` (top-k truncation can only under-count entropy) —
  and carries the same decode-record schema as `generate_and_capture`, so
  surfaces built for one work for the other.
- `from_openai_logprobs` adapts OpenAI-style chat-completion responses (SDK
  objects or plain dicts); any provider can be mapped into the neutral shape.
- `ui.degraded_note` (pure, tested) renders that ceiling as a banner above
  the scene in the explorer.

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
- Scenes carry real features: `ui.attach_features` computes the dominant
  feature per state plus the dictionary's measured fit, `.mtj` scene runs
  carry it additively (`features`: top_id/top_act/recon_error/best_layer/
  source/hook), and the explorer's export attaches it whenever a trained
  dictionary is active. New sample `viewer/samples/gpt2-features.mtj`: the
  calibrated TL pairing (capitals A/B), where the measured fit finds the
  training hook on its own — best layer 8, ~21% error, 0.15% firing.

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
  reader + little-endian header).

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
