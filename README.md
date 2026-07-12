# 🔮 MARBLE

[![CI](https://github.com/BobGnarly420/marble/actions/workflows/ci.yml/badge.svg)](https://github.com/BobGnarly420/marble/actions/workflows/ci.yml)

**Interactive latent trajectory explorer for transformer forward passes.**

MARBLE visualizes hidden-state evolution as trajectories over a semantic
manifold. It is *not* a neuron inspector, feature-attribution tool, or
explainability dashboard — it instruments **latent dynamics**: how the
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
from ui import run_pipeline, render        # everything at once → plotly Figure

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
inputs), terrain mesh consistency and smoothing, animation continuity, and a
headless run of the actual Streamlit app (`streamlit.testing.v1.AppTest`).

## Non-goals (MVP)

No training, finetuning, SAE integration, activation patching, circuit
discovery, distributed inference, or production auth. Single-machine
research tool.

## Roadmap

- **Phase 2** — trajectory comparison: prompt A/B overlay, Hausdorff
  distance, dynamic time warping, shared-prefix divergence
  (`metrics.branch_divergence` is the seed).
- **Phase 3** — SAE features, residual decomposition, feature overlays.
- **Phase 4** — multi-prompt scenes, attention flow, interactive patching.
