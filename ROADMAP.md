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
- [ ] Scene + explorer wiring: decode-boundary marker on the terrain, decode
      scrubber alongside the layer scrubber.
- [ ] Web viewer: animate the trajectory *per generated token*; `.mtj` carries
      the generation record.
- [ ] CLI: `mottled export --generate N`.

### M2 — Real SAEs by default
`from_sae_lens` exists; nobody sees a real feature without work.
- [ ] Fetch-on-demand small trained GPT-2 SAE (cached, checksummed, optional).
- [ ] Regenerate bundled sample scenes with real features; the feature-field
      domain coloring shows actual semantics on first contact.
- [ ] Feature labels (from SAELens metadata when present) in both viewers.

### M3 — A tangible viewer
- [ ] Wire `bvh.py` (built, tested, currently unwired) into the WebGL viewer:
      ray-picked click-to-inspect on any trajectory segment.
- [ ] Inspector parity with the explorer (predictions, neighbors, features,
      decomposition) so the shareable viewer is not the lesser surface.

### M4 — Closed-model producers
The architecture diagram has promised "OpenAI / Anthropic logprobs" from the
start.
- [ ] Logprob-trajectory producer: what a closed API exposes, as a degraded
      `StateTrajectory` (final-layer readout evolution across decode steps).
- [ ] The fidelity machinery states exactly which layers of the picture are
      absent — honest about the ceiling, not silent about it.

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
