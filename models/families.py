"""Model-family abstraction for HuggingFace causal LMs.

Qwen / Llama / Mistral / Gemma all expose the Llama-style layout
(``model.layers``, ``model.norm``, ``lm_head``); GPT-2 and GPT-NeoX layouts
are also recognised so tiny test models work.  Resolution is structural —
we probe attribute paths instead of switching on class names — so any model
that follows one of these layouts is supported without registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_BLOCK_PATHS = [
    "model.layers",            # Llama, Qwen, Mistral, Gemma
    "transformer.h",           # GPT-2
    "gpt_neox.layers",         # GPT-NeoX / Pythia
    "model.decoder.layers",    # OPT
    "transformer.blocks",      # MPT
]
_EMBED_PATHS = [
    "model.embed_tokens",
    "transformer.wte",
    "gpt_neox.embed_in",
    "model.decoder.embed_tokens",
    "transformer.wte",
]
_NORM_PATHS = [
    "model.norm",
    "transformer.ln_f",
    "gpt_neox.final_layer_norm",
    "model.decoder.final_layer_norm",
]
_HEAD_PATHS = ["lm_head", "embed_out"]


def _get_path(obj: Any, path: str) -> Any | None:
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


@dataclass
class FamilyAdapter:
    """Structural handles into one transformer model."""

    blocks: Any        # iterable of transformer blocks (residual stream units)
    embed: Any         # token embedding module
    final_norm: Any    # final layernorm applied before the LM head
    lm_head: Any       # unembedding
    name: str = "generic"

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def embedding_weight(self):
        import torch  # local import: torch is optional at module level

        w = getattr(self.embed, "weight", None)
        if w is None:
            raise AttributeError("embedding module has no .weight")
        with torch.no_grad():
            return w.detach().float().cpu()


def resolve_family(model: Any) -> FamilyAdapter:
    """Find blocks / embeddings / norm / head on a HF causal LM."""
    blocks = next((m for p in _BLOCK_PATHS if (m := _get_path(model, p)) is not None), None)
    embed = next((m for p in _EMBED_PATHS if (m := _get_path(model, p)) is not None), None)
    norm = next((m for p in _NORM_PATHS if (m := _get_path(model, p)) is not None), None)
    head = next((m for p in _HEAD_PATHS if (m := _get_path(model, p)) is not None), None)
    if blocks is None or embed is None:
        raise ValueError(
            f"Unsupported model layout: {type(model).__name__}. "
            "Expected a Llama/Qwen/Mistral/Gemma-style causal LM."
        )
    if head is None:
        head = getattr(model, "get_output_embeddings", lambda: None)()
    return FamilyAdapter(
        blocks=blocks,
        embed=embed,
        final_norm=norm,
        lm_head=head,
        name=type(model).__name__,
    )
