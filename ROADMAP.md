# Roadmap

## End goal — Mottled 1.0: the honest atlas of latent dynamics

One tool that can show **any model** (open weights, TransformerLens hooks,
state-space models, logprobs-only APIs), on **any timescale** (layers within a
forward pass, tokens within a generation), against **any counterpart** (A/B
prompts, A/B interventions, A/B *models*) — and that states, inline and
unavoidably, how much of what you see survived the projection.

The invariants that get us there are already in place and are not negotiable:

1. **`StateTrajectory` is the only interchange.** Every milestone below is a
   new producer or a new consumer; none may couple the two.
2. **Honesty is load-bearing.** Every new surface carries the fidelity /
   uncertainty machinery from day one — a lossy picture must say where it lies.
3. **The synthetic backend keeps pace.** Every capability works without torch,
   so the whole pipeline stays instantly explorable and offline-testable.

## Milestones

### M1 — The generation axis *(complete)*
Today Mottled has one time axis: layers. Autoregressive decode is the second.
- [x] `capture.generate_and_capture`: decode then capture the completed
      sequence in one pass — exact for causal models, no approximation —
      with per-step decode records (chosen token, probability, entropy) in
      `meta["generation"]`.
- [x] Synthetic backend generates with its own logit lens, so the decode axis
      works dependency-free.
- [x] Scene + explorer wiring: decode knobs, inline decode header, generated
      trajectories visually distinct, per-step decode inspector.
- [x] Web viewer: decode axis rendered (faded lines, rimmed dots, `+` labels,
      decode summary, per-step hover); `.mtj` carries the generation record;
      real GPT-2 decode sample (`gpt2-decode.mtj`).
- [x] CLI: `mottled export --generate N [--temperature T]`; the capture API
      accepts `generate` / `temperature`.

### M2 — Real SAEs by default
`from_sae_lens` exists; nobody sees a real feature without work.
- [x] Fetch-on-demand trained GPT-2 SAE (`sae.fetch_from_hub`, HF-hub cached,
      no sae-lens dep; `mottled-convert-sae fetch`); explorer defaults to it
      for GPT-2-width captures.
- [x] Measured calibration (`sae.fit_report`) printed wherever features are
      shown — provenance is not calibration (TL-processed vs raw HF
      residuals differ), so the fit is measured, and a bad fit is labeled
      extrapolation.
- [x] Scenes carry real features (`ui.attach_features` → additive `.mtj`
      `features` layer with measured fit); bundled `gpt2-features.mtj`
      sample from the calibrated TL pairing.
- [ ] Feature labels (from SAELens/Neuronpedia metadata when present) in
      both viewers; feature-field domain coloring with real semantics on
      first contact.

### M3 — A tangible viewer
- [x] `bvh.py` ported to `viewer/bvh.js` and pinned by a cross-language
      conformance test; the viewer picks by real camera ray against the
      segment index — grab anywhere along a trajectory, fractional layer in
      the inspector, click-to-pin.
- [x] Inspector parity with the explorer: `pipeline.attach_inspector`
      resolves the two readouts a scene cannot recompute (semantic neighbors
      need the `(V, D)` embedding matrix; the attn/MLP split needs the
      residual components) *before* export, where they collapse to a compact
      table and two numbers per state. The viewer renders both.

**M3 is complete.**

### M4 — Closed-model producers
The architecture diagram has promised "OpenAI / Anthropic logprobs" from the
start.
- [x] `models/logprobs.py`: per-step top-k logprobs as a degraded
      `StateTrajectory` — depth is unavailable, so the animated axis is decode
      time and the geometry is the output distribution. Unreported top-k mass
      gets its own visible bucket; `from_openai_logprobs` adapts OpenAI-style
      responses.
- [x] The trajectory declares its own ceiling (`meta.degraded`, `meta.absent`,
      `entropy_is_lower_bound`) and the explorer prints it as a banner
      (`ui.degraded_note`) above the scene.
- [ ] A live end-to-end path (fetch logprobs from a configured provider) and a
      bundled sample scene.

### M5 — A sustainable core
- [x] Split `ui.py` into `pipeline.py` + `render.py` + the Streamlit shell,
      with `ui` re-exporting both so the documented flat API is unchanged.
- [x] Cut `0.2.0` (tag `v0.2.0` drives the existing Trusted Publishing
      workflow).
- [ ] **Deferred deliberately — the `mottled/` package move.** The original
      plan was to land the split *inside* a `mottled/` package. But the
      README documents flat imports (`from capture import capture`) as the
      public API, so that move is a breaking change, and bundling it with a
      refactor would hide the break inside a diff that otherwise changes no
      behavior. It deserves its own release with a deprecation path: ship
      `mottled/` re-exporting the flat modules, warn on the flat imports for
      one version, then remove them. Until then the flat layout's real cost
      stands — installing top-level `config` / `cache` / `metrics` into
      site-packages is poor citizenship for a package meant to be adopted.

### M6 — The cross-model atlas
- [x] Shared space for different-dimension models: `crossmodel.readout_space`
      anchors on the vocabulary the models share (the one thing they do), with
      the unshared mass kept visible. `project_joint` and both viewers accept
      it unchanged.
- [x] `compare_models` (depth-normalised readout divergence) and
      `layer_similarity` (CKA, which reports whether its own answer is
      identified rather than always returning an argmax).
- [x] "Same prompt, two models" scenes: `pipeline.run_model_scene`,
      `mottled export --models a,b`, sample
      `viewer/samples/models-gpt2-distilgpt2.mtj`.
- [x] The decode axis crossed with the model axis (M1 × M6):
      `compare_generations` (free-running, with the comparability boundary
      marked where the contexts split) and `forced_divergence` (teacher-forced
      on one fixed text, comparable throughout).
- [x] Explorer surface: a "Compare models" field and a Model comparison panel
      (readout divergence, layer alignment with its identified/flat rows, the
      generation split), so the atlas is reachable without the API.

## Status

M1, M3 and M6 are complete; M2, M4 and M5 each have exactly one item left,
listed above. Every milestone landed green (offline test suite + viewer Node
tests) before the next started, and that stays the rule.

## Model coverage

Verified end to end: **Qwen2.5-1.5B-Instruct** (29 x 1536, GQA / RoPE /
SwiGLU / RMSNorm) and GPT-2 / DistilGPT-2 / Pythia-70m, plus the synthetic
backend. The residual decomposition reconciles exactly on all of them, so
"model-agnostic" is measured rather than claimed.

Three separate reasons the samples are not all frontier models, worth keeping
distinct:

- **Licence-gated** — `meta-llama/Llama-3.2-1B`, `google/gemma-2-2b`. They
  work; they need an accepted licence and an `HF_TOKEN`, so they cannot back
  bundled samples or offline CI.
- **Hardware** — frontier MoE is not a support question. Kimi K2's weights
  are ~1 TB.
- **Ecosystem** — GPT-2 stays in the *SAE-dependent* samples because the
  public trained-dictionary ecosystem (SAELens, Neuronpedia) is
  overwhelmingly GPT-2-small, and M2's real-feature work depends on one
  existing. Gemma Scope and Llama Scope exist but sit behind the licence gate
  above. **This is the one that actually constrains the project**, and it is
  the argument for the remaining M2 item.

## Standing hazard: within-model assumptions

For most of this project's life, "two runs" meant "two prompts through one
model", so shared depth, shared width and shared tokenization were free. The
cross-model work (M6) made all three optional, and code written under the old
assumption fails in a specific way: it does not crash, it quietly measures
the wrong thing or hides part of the picture. Five instances have been found
and fixed (`_assemble_scene`, the A/B panel, the layer scrubber, scene-level
model identity, the SAE width heuristic).

New code that touches two runs should state which of these it needs — and
guard or refuse rather than assume:

- **same depth** — anything comparing layer *l* to layer *l*,
- **same width** — anything indexing hidden dimensions,
- **same tokenization** — anything pairing state (l, t) with state (m, t),
- **hidden space at all** — readout-space trajectories have vocabulary axes,
  so heuristics keyed on width do not mean there what they mean elsewhere.
