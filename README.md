# 🔮 Mottled

[![CI](https://github.com/BobGnarly420/marble/actions/workflows/ci.yml/badge.svg)](https://github.com/BobGnarly420/marble/actions/workflows/ci.yml)

**Interactive latent trajectory explorer for transformer forward passes.**

Mottled (formerly MARBLE) visualizes hidden-state evolution as
trajectories over a semantic manifold. It is *not* a neuron inspector,
feature-attribution tool, or explainability dashboard — it instruments
**latent dynamics**: how the
residual stream moves, turns, and settles as a prompt flows through the
layers of a transformer.

```
Prompt → forward pass → capture residual stream after every block
       → project hidden vectors → estimate local manifold
       → animate trajectory → expose semantic neighborhoods
```

## Quickstart

```bash
pip install -r requirements.txt
streamlit run ui.py
```

Enter a prompt (e.g. `The capital of France is`), pick a model, press
**Run capture**. You get an animated hidden-state trajectory over a density
terrain, semantic neighbors, entropy evolution, and a layer scrubber.

The default `synthetic` backend needs no model download (or even torch) —
it generates deterministic, realistic trajectories so you can explore the
tool instantly. Select a HuggingFace model (Qwen / Llama / Mistral / Gemma /
GPT-2) for real captures.

## Programmatic API

```python
from capture import capture                # StateTrajectory
from projection import project             # (L, T, 2) coordinates
from density import compute_density        # Landscape
from terrain import mesh, drape            # TerrainMesh
from trajectory import extract, densify    # Trajectory list, animation path
from metrics import summarize              # research metrics
from compare import compare                # A/B trajectory comparison
from sae import demo_sae, feature_trajectory  # SAE feature activations
from ui import run_pipeline, render        # everything at once → plotly Figure
from ui import run_scene, run_intervention  # multi-prompt scenes, patching

traj = capture("gpt2", "The capital of France is")
coords, projector = project(traj.hidden, method="pca")
landscape = compute_density(coords, method="kde")
surface = mesh(landscape)
paths = extract(coords, traj.tokens, mode="all_tokens")
print(summarize(traj, coords, token=-1))
```

`capture(model, prompt)` returns `hidden[layer][token][dimension]` wrapped in
a `StateTrajectory`, with logit-lens logits, entropy, and top-k predictions
attached per state.

## Architecture

Everything operates on a single abstraction — **`StateTrajectory`**
(`trajectory.py`). All analyses are functions over trajectories; nothing
above the capture layer touches transformer internals. Transformers are one
backend (`models/families.py` resolves Qwen/Llama/Mistral/Gemma/GPT-2/NeoX
layouts structurally); the synthetic generator (`models/synthetic.py`) is
another. Future backends (RNNs, Mamba, diffusion, biological recordings)
only need to expose a `StateTrajectory` and the entire stack — projection,
density, terrain, metrics, UI — works unchanged.

| Module | Role |
|---|---|
| `capture.py` | Forward hooks on every block + logit lens → `StateTrajectory` |
| `projection.py` | PCA / UMAP plugin registry, incremental `transform` |
| `neighbors.py` | FAISS or NumPy cosine k-NN over hidden states & token embeddings |
| `density.py` | KDE / kNN-inverse-distance density → scalar potential field |
| `terrain.py` | Density → smoothed height map → mesh; drapes trajectories on it |
| `trajectory.py` | `StateTrajectory`, extraction modes (token / all / mean / CLS), spline densify |
| `metrics.py` | Entropy, KL, path length, curvature, velocity, drift, NN-stability |
| `cache.py` | Disk cache keyed by prompt + config hash |
| `config.py` | One dataclass for every pipeline knob |
| `ui.py` | Pure pipeline + pure Plotly renderer + Streamlit shell |
| `bvh.py` | Spatial index over trajectory segments (ray-pick / nearest / box / frustum) for the fly-through canvas |
| `intervene.py` | Causal interventions: perturb / set / noise / freeze a state via a resumable forward pass → counterfactual trajectory |
| `compare.py` | Trajectory comparison: Hausdorff, dynamic time warping, shared-prefix alignment, layerwise divergence profiles |
| `sae.py` | Sparse-autoencoder features: apply (never train) an SAE to every captured state; demo dictionary + npz interchange |

### Causal intervention (perturb-and-replay)

Observation shows what a system *did*; intervention shows what it *would have
done*. `intervene.py` runs a **resumable forward pass**: write-hooks rewrite
the residual stream at a chosen layer and the model continues from the edited
state, producing a **counterfactual `StateTrajectory`** — real data that flows
through the same projection / measurement / renderer stack as the baseline.

```python
from capture import capture
from intervene import Perturb, intervene, divergence

base = capture(model, "The capital of France is", tokenizer=tok)
# push the final state toward the " Berlin" embedding direction
d = base.embedding_matrix[berlin_id]
branch = intervene(model, "The capital of France is",
                   [Perturb(layer=base.n_layers - 1, delta=60 * d, token=-1)],
                   tokenizer=tok)
# baseline predicts " the"; the branch now predicts " Berlin" (p≈1.0)
print(divergence(base, branch).readout_changed)   # layer where the prediction flips
```

Edits: `Perturb` (push a state — the grab gesture), `SetState`, `InjectNoise`
(seeded), `FreezeLayer` (skip a block's update). Multiple interventions compose
in one pass. `divergence(baseline, branch)` measures where a branch separates
(state-space profile + the layer the top-1 prediction flips) — a measurement,
not a claimed cause. Interventions require a torch model; the synthetic backend
is analytic and not resumable.

### Trajectory comparison (prompt A/B)

Two forward passes become comparable once their states live in **one shared
projection** (`projection.project_joint` fits on the union of both runs).
`compare.py` then measures how the trajectories relate: symmetric **Hausdorff**
distance (how far apart the paths get), **dynamic time warping** (aligns paths
that trace the same route at different speeds), **shared-prefix** alignment,
and layerwise divergence profiles in full hidden space — including the first
token position where the runs separate and the first layer where the
logit-lens top-1 prediction differs.

```python
from capture import capture
from projection import project_joint
from compare import compare

a = capture("gpt2", "The capital of France is", tokenizer=tok)
b = capture("gpt2", "The capital of Germany is", tokenizer=tok)
(ca, cb), _ = project_joint([a.hidden, b.hidden])
cmp = compare(a, b, ca, cb)                  # geometry in the shared space
print(cmp.shared_tokens, cmp.hausdorff, cmp.dtw.normalized, cmp.readout_changed)
```

In the UI, fill in **Prompt B** and run: both trajectories are drawn on a
single terrain built from the union of states (B dashed), with the comparison
metrics and the per-layer A–B distance in the inspector.
`ui.run_compare(cfg, prompt_a, prompt_b)` is the programmatic entry point.
Everything is backend-agnostic — a synthetic run and the comparison stack work
without torch; the runs only need the same layer count and hidden dimension.

### SAE features & residual decomposition

`capture(model, prompt, capture_components=True)` additionally hooks every
block's attention and MLP submodules and records their outputs — the two
additive writes to the residual stream.  For pre-norm architectures
(Llama-style, GPT-2, NeoX) the decomposition is exact:
`hidden[l+1] = hidden[l] + attn[l] + mlp[l]` (pinned by tests).
`metrics.component_shares` turns it into a per-layer attention-vs-MLP
balance, and the UI plots it in the inspector.  The synthetic backend emits
an analogous exact decomposition, so the whole path works without torch.

`sae.py` applies sparse autoencoders to trajectories — it never trains them.
An SAE is four plain numpy arrays (`w_enc`, `b_enc`, `w_dec`, `b_dec`);
export any pretrained SAE (SAELens, dictionary-learning runs) to `.npz` and
`load_npz` it.  `demo_sae` builds an untrained random dictionary so the
feature pipeline — activations, top-features, UI overlay — runs offline
(demo activations are sparse projections, *not* interpretable features).

```python
from sae import load_npz, demo_sae, feature_trajectory, top_features

sae = demo_sae(traj.dim)          # or load_npz("gpt2-res-l8.npz")
acts = feature_trajectory(traj, sae)         # (L, T, F) activations
print(top_features(acts, layer=8, token=-1)) # strongest features at a state
```

In the UI, tick **SAE feature overlay**: trajectory markers are colored by
the selected feature's activation per layer, the inspector lists the top
features at the selected state, and a **Residual decomposition** panel shows
each block's attention/MLP share.

### Multi-prompt scenes, attention flow, interactive patching

`ui.run_scene(cfg, prompts)` generalizes the A/B overlay to N prompts: every
run is captured, joint-projected into one shared space, drawn on a single
terrain built from the union of all states, and compared against the first
run (in the UI, enter one overlay prompt per line — runs get A/B/C… labels
and distinct dash styles).

`capture(..., capture_attention=True)` records each block's head-averaged
attention pattern (`StateTrajectory.attention`, `(L-1, T, T)`; the eager
attention path is forced so the matrix actually materialises).  The renderer
can draw **attention flow** — edges from each token's state to the states it
reads from at the selected layer — and the inspector lists the top attended
tokens.  The synthetic backend generates a causal, deterministic analog.

`ui.run_intervention(cfg, prompt, edits, model, tokenizer)` is interactive
patching: the baseline and a perturb-and-replay branch (`intervene.py`)
are assembled as a two-run scene, with the full comparison plus an
`intervene.divergence` readout (separation onset, prediction-flip layer).
The UI exposes it as a sidebar panel — push a state toward a token
embedding, inject noise, or freeze a block, then watch the counterfactual
trajectory diverge on the same terrain.

### Interaction layer (in progress)

The exploratory canvas renders trajectories as curves (not voxels — projected
states occupy a vanishing fraction of any 3-D volume). `bvh.py` is the spatial
acceleration structure the interaction grammar needs: `ray_pick` powers the
"grab a state" gesture (camera ray → front-most segment within a pick radius),
`nearest` powers hover, `query_box` powers region select, and `query_frustum`
powers fly-through culling. It is backend-agnostic — it consumes projected 3-D
points (a projection output), never transformer internals — so any substrate
projected to ≤3-D is pickable. A volumetric (voxel-octree) renderer for
*fields* (density / flow) will land once we render ensembles rather than single
runs.

### Plugin points

Projections (`projection.PROJECTIONS`), density estimators
(`density.DENSITY_ESTIMATORS`), neighbor backends (faiss/numpy), and metrics
(`metrics.METRICS`) are registries — register a class and it is available by
name, including in the UI dropdowns via `config.py`.

## Research metrics

Per-token trajectory summaries: path length, integrated curvature, average
semantic drift (cosine), layerwise displacement, entropy collapse, and
nearest-neighbor stability (Jaccard overlap of the token-embedding
neighborhood across layers).

## Tests

```bash
python -m pytest tests/ -q
```

Covers: hook captures match `output_hidden_states` exactly, logit lens
reproduces the model's final logits, shape consistency, projection
determinism, valid neighbor lookups, finite density (incl. degenerate
inputs), terrain mesh consistency and smoothing, animation continuity,
comparison geometry on analytic cases (Hausdorff, DTW alignment validity),
SAE encode/decode math and npz roundtrip, exact residual decomposition and
attention-pattern capture on locally-built Llama/GPT-2 models, multi-prompt
scene assembly, the intervention pipeline, and headless runs of the actual
Streamlit app — single-prompt, A/B, N-prompt scene, and SAE overlay —
(`streamlit.testing.v1.AppTest`).

## Non-goals (MVP)

No training or finetuning (SAEs are *applied*, never trained), no circuit
discovery, distributed inference, or production auth. Single-machine
research tool.

## Roadmap

- **Phase 2** — ✅ trajectory comparison: prompt A/B overlay, Hausdorff
  distance, dynamic time warping, shared-prefix divergence (`compare.py`,
  grown from the `metrics.branch_divergence` seed).
- **Phase 3** — ✅ SAE features (`sae.py`), residual decomposition
  (`capture_components`), feature overlays in the UI.
- **Phase 4** — ✅ multi-prompt scenes (`ui.run_scene`), attention flow
  (`capture_attention` + renderer edges), interactive patching
  (`ui.run_intervention` over `intervene.py`).
- **Next** — volumetric field rendering for ensembles, SAE feature flows
  across layers, richer scene management (pin/hide runs, saved scenes).
