# 🔮 MARBLE

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
