"""Optional TransformerLens producer: a HookedTransformer -> StateTrajectory.

TransformerLens is *not* a dependency — import this module only if you have it
installed (`pip install transformer-lens`). Rather than route a
``HookedTransformer`` through Mottled's HF forward hooks, we use TL's own
``run_with_cache``: it exposes the residual stream directly (``resid_pre`` /
``resid_post``), which is the cleanest bridge and a second proof that the
producer contract is all a new backend needs — emit a ``StateTrajectory`` and
the entire projection / density / metrics / viewer stack works unchanged.

The residual-extraction step is separated from the ``StateTrajectory`` assembly
(`_to_trajectory`) so the assembly is unit-testable without the heavy library.
"""
from __future__ import annotations

import numpy as np

from capture import _clean_token, _entropy_topk
from trajectory import StateTrajectory


def _to_trajectory(hidden: np.ndarray, tokens: list[str], lens: np.ndarray,
                   vocab: list[str], embedding_matrix: np.ndarray,
                   top_k: int = 5, keep_logits: bool = True,
                   model_name: str = "hooked", prompt: str = "") -> StateTrajectory:
    """Assemble a StateTrajectory from an extracted residual stack + logit lens.

    hidden: (L, T, D) residual stream (layer 0 = embeddings), lens: (L, T, V)
    logit-lens logits for every state. Pure numpy — this is where the entropy
    and top-k are computed, shared with the HF capture path.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    lens = np.asarray(lens, dtype=np.float32)
    entropy, topk = _entropy_topk(lens, vocab, top_k)
    traj = StateTrajectory(
        hidden=hidden,
        tokens=list(tokens),
        logits=lens.astype(np.float16) if keep_logits else None,
        entropy=entropy,
        topk=topk,
        vocab=vocab,
        embedding_matrix=np.asarray(embedding_matrix, dtype=np.float32),
        meta={"backend": "transformer_lens", "model": model_name,
              "prompt": prompt, "family": "HookedTransformer"},
    )
    traj.validate()
    return traj


def from_hooked_transformer(model, prompt: str, top_k: int = 5,
                            keep_logits: bool = True) -> StateTrajectory:
    """Capture a TransformerLens ``HookedTransformer`` as a StateTrajectory.

    Uses ``run_with_cache`` to read the residual stream after every block, then
    applies TL's final layer-norm + unembed as the logit lens over every state.
    """
    import torch

    tokens_t = model.to_tokens(prompt)
    str_tokens = [_clean_token(t) for t in model.to_str_tokens(prompt)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens_t)
        n = int(model.cfg.n_layers)
        resid = [cache["resid_pre", 0]] + [cache["resid_post", l] for l in range(n)]
        hidden = torch.stack(resid, dim=0)[:, 0]              # (L, T, D), batch 0
        lens = model.unembed(model.ln_final(hidden))          # (L, T, V) logit lens

    hidden_np = hidden.float().cpu().numpy()
    lens_np = lens.float().cpu().numpy()
    vocab = [_clean_token(t)
             for t in model.tokenizer.convert_ids_to_tokens(range(lens_np.shape[-1]))]
    emb = model.W_E.detach().float().cpu().numpy()
    return _to_trajectory(
        hidden_np, str_tokens, lens_np, vocab, emb, top_k=top_k,
        keep_logits=keep_logits,
        model_name=getattr(model.cfg, "model_name", "hooked"), prompt=prompt,
    )
