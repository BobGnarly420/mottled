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

### M1 — The generation axis *(in progress)*
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
- [ ] Inspector parity with the explorer (neighbors, residual decomposition)
      so the shareable viewer is not the lesser surface.

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
- [ ] Split `ui.py` (53 KB and growing) into a `mottled/` package with
      focused modules; keep the flat public API via re-exports.
- [ ] Cut `0.2.0` and publish to PyPI through the existing Trusted Publishing
      workflow.

### M6 — The cross-model atlas
The comparison machinery (Hausdorff, DTW, divergence profiles) currently
compares prompts. Generalize it to compare **models**.
- [ ] Shared projection space for different-dimension models (vocab-anchored
      readout alignment; CKA/Procrustes on shared prompts).
- [ ] "Same prompt, two models" scenes: where do their dynamics diverge, and
      at which decode step.

## Sequencing

M1 → M2 → M3 ship independently and in order; M4 and M5 can interleave after
M1; M6 builds on M1 (decode axis) + M5 (package split). Each milestone lands
green (offline test suite + viewer Node tests) before the next starts.
