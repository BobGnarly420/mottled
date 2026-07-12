"""Global configuration for MARBLE.

A single dataclass carries every knob the pipeline understands so that
capture -> project -> density -> terrain -> render can be driven from one
object (and hashed for caching).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

DEFAULT_PROMPT = "The capital of France is"

# Model families the transformers backend has been exercised against share
# the Llama-style module layout (Qwen / Llama / Mistral / Gemma).  "synthetic"
# is a dependency-free backend used for tests, demos and UI development.
MODEL_CHOICES = [
    "synthetic",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-2-2b",
    "gpt2",
]

PROJECTION_CHOICES = ["pca", "umap"]
DENSITY_CHOICES = ["kde", "knn"]
TRAJECTORY_MODES = ["all_tokens", "token", "mean", "cls"]


@dataclass
class MarbleConfig:
    # model / capture
    model: str = "synthetic"
    device: str = "auto"          # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "float32"
    keep_logits: bool = True      # keep full logit-lens logits (float16)

    # projection
    projection: str = "pca"
    n_components: int = 2
    seed: int = 0

    # density / terrain
    density: str = "kde"
    grid_size: int = 64
    grid_padding: float = 0.2
    smooth_sigma: float = 1.5
    height_scale: float = 1.0
    invert_terrain: bool = False  # True: dense regions become valleys
    marble_lift: float = 0.04     # trajectory height offset above terrain

    # neighbors / predictions
    top_k: int = 5
    n_neighbors: int = 5
    neighbor_backend: str = "auto"  # "auto" | "faiss" | "numpy"

    # animation
    frames_per_layer: int = 4
    frame_ms: int = 120

    # trajectory extraction
    trajectory_mode: str = "all_tokens"
    trajectory_token: int = -1

    # caching
    cache_dir: str = ".marble_cache"
    use_cache: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULTS = MarbleConfig()
