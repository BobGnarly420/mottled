"""Bring your own states: activations captured elsewhere -> StateTrajectory.

`capture.py` owns the residual stream from end to end — it loads the model,
hooks it, and reads the states back. That is the wrong shape for a researcher
who already has the states: an NNsight trace with `.save()` on each layer, a
vLLM or custom-loop hook, a run on a machine that is not this one. Re-running
the model to get a picture of a pass you already ran is the expensive half of
the work, and on a frontier-scale model it may not be possible at all.

    traj = from_hidden_states(saved, tokens, model, tokenizer)

`hidden` is whatever the capture left you with — an (L, T, D) array, a list of
per-layer (T, D) or (1, T, D) tensors, numpy or torch, NNsight proxies
included. `model` is used only for the readout: the final norm, the LM head
and the embedding table, resolved structurally by `models/families.py`, so
anything `capture.py` supports works here too.

**Layer 0 is whatever you pass.** Mottled's convention is that layer 0 is the
embedding stream and layer l+1 is block l's output, which is what `capture`
and `models/hooked.py` produce. Hand this function N block outputs and you get
an N-layer trajectory whose layer 0 is the first block's output — every
analysis still runs, but "the embedding stream" is then not in the picture and
depth indices are off by one against a native capture. The function cannot
tell the two apart, so it does not try.
"""

from __future__ import annotations

import numpy as np

from capture import _clean_token, _entropy_topk, logit_lens
from models.families import resolve_family
from trajectory import StateTrajectory


def from_hidden_states(hidden, tokens, model, tokenizer, prompt: str = "",
                       top_k: int = 5, keep_logits: bool = True,
                       backend: str = "external") -> StateTrajectory:
    """Assemble a StateTrajectory from residual states captured outside Mottled.

    `hidden` is coerced to (L, T, D) by `stack_layers`; `tokens` may be token
    strings or ids. The logit lens, entropy and top-k are computed here by the
    same code the native capture uses, so an external trajectory and a
    `capture()` one of the same pass are the same numbers, not merely the same
    shape — `tests/test_external.py` pins that.
    """
    import torch

    hidden = stack_layers(hidden)
    tokens = list(tokens)
    if tokens and isinstance(tokens[0], (int, np.integer)):
        tokens = tokenizer.convert_ids_to_tokens([int(t) for t in tokens])
    tokens = [_clean_token(t) for t in tokens]
    if len(tokens) != hidden.shape[1]:
        # the failure this function actually produces: a BOS the tokenizer adds
        # and the caller's own token list does not have, or vice versa
        raise ValueError(
            f"{len(tokens)} tokens for {hidden.shape[1]} captured positions — "
            "pass the exact ids the states were captured for")

    adapter = resolve_family(model)
    lens = logit_lens(torch.from_numpy(hidden), adapter).numpy()
    vocab = [_clean_token(t)
             for t in tokenizer.convert_ids_to_tokens(range(lens.shape[-1]))]
    entropy, topk = _entropy_topk(lens, vocab, top_k)

    config = getattr(model, "config", None)
    traj = StateTrajectory(
        hidden=hidden,
        tokens=tokens,
        logits=lens.astype(np.float16) if keep_logits else None,
        entropy=entropy,
        topk=topk,
        vocab=vocab,
        embedding_matrix=adapter.embedding_weight().numpy(),
        meta={
            "backend": backend,
            "model": getattr(config, "name_or_path", type(model).__name__),
            "prompt": prompt,
            "family": adapter.name,
            # provenance.record reads these. `dtype` is the readout model's,
            # not the supplied states': how they were computed is the caller's
            # to report, and claiming to know it would be a guess.
            "revision": getattr(config, "_commit_hash", None),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        },
    )
    traj.validate()
    return traj


def stack_layers(states) -> np.ndarray:
    """(L, T, D) float32 from whatever shape the capture left behind.

    Accepts an (L, T, D) or (L, 1, T, D) array/tensor, or a per-layer sequence
    of (T, D) / (1, T, D). The singleton batch axis is dropped because every
    tracing library hands one back and no caller means it as a layer.
    """
    if not isinstance(states, (list, tuple)):
        arr = _as_numpy(states)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 3:
            raise ValueError(
                f"expected (L, T, D) residual states, got shape {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float32)

    if not states:
        raise ValueError("no layers to stack")
    layers = []
    for i, state in enumerate(states):
        arr = _as_numpy(state)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(
                f"layer {i}: expected (T, D) states, got shape {arr.shape}")
        layers.append(arr)
    shapes = {a.shape for a in layers}
    if len(shapes) > 1:
        # a ragged stack means layers from different passes, which would put
        # unrelated states on one trajectory without numpy ever complaining
        raise ValueError(f"layers have differing shapes: {sorted(shapes)}")
    return np.ascontiguousarray(np.stack(layers), dtype=np.float32)


def _as_numpy(x) -> np.ndarray:
    # `.value` is what NNsight's .save() hands back outside the trace; a bare
    # tensor (newer nnsight, plain torch, numpy) falls through unchanged
    x = getattr(x, "value", x)
    if hasattr(x, "detach"):
        # .float() before numpy: bfloat16 has no numpy dtype, and a capture
        # taken in bf16 is the common case rather than an exotic one
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)
