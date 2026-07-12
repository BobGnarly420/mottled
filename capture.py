"""Residual-stream capture via forward hooks.

`capture(model, prompt)` runs one forward pass and records the residual
stream after every transformer block (plus the initial embedding stream as
layer 0), then applies the logit lens (final norm + LM head) to every
captured state to obtain per-layer, per-token logits / entropy / top-k.

The result is a StateTrajectory — the only thing downstream modules see.
Transformers are just one backend; `model="synthetic"` routes to the
dependency-free generator in models/synthetic.py.
"""

from __future__ import annotations

import numpy as np

from trajectory import StateTrajectory

try:  # torch/transformers are optional: the synthetic backend needs neither.
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("torch is required for transformer capture; use model='synthetic' otherwise")


def resolve_device(device: str = "auto") -> str:
    _require_torch()
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str = "auto", dtype: str = "float32"):
    """Load a HF causal LM + tokenizer in eval mode."""
    _require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype)
    model.to(device).eval()
    return model, tokenizer


class HookCapture:
    """Context manager that records the residual stream around every block.

    Layer 0 is taken from a forward *pre*-hook on the first block — the exact
    tensor entering the residual stream (this is also correct for Gemma,
    which scales embeddings after the embedding module).  Layers 1..N come
    from forward hooks on each block's output.
    """

    def __init__(self, model):
        from models.families import resolve_family

        self.adapter = resolve_family(model)
        self.states: dict[int, "torch.Tensor"] = {}
        self._handles = []

    def __enter__(self):
        blocks = list(self.adapter.blocks)

        def pre_hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            self.states[0] = hs.detach()

        self._handles.append(blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
        for i, block in enumerate(blocks):
            def hook(module, args, output, _layer=i + 1):
                out = output[0] if isinstance(output, tuple) else output
                self.states[_layer] = out.detach()

            self._handles.append(block.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stacked(self) -> "torch.Tensor":
        """(L, T, D) float32 on CPU, layer 0 first, batch dim squeezed."""
        n = len(self.adapter.blocks) + 1
        missing = [i for i in range(n) if i not in self.states]
        if missing:
            raise RuntimeError(f"missing hook captures for layers {missing}")
        return torch.stack([self.states[i][0] for i in range(n)]).float().cpu()


def logit_lens(hidden: "torch.Tensor", adapter, chunk: int = 4) -> "torch.Tensor":
    """Apply final norm + LM head to every captured state: (L, T, V) logits."""
    _require_torch()
    if adapter.lm_head is None:
        raise ValueError("model exposes no LM head; cannot apply logit lens")
    device = next(adapter.lm_head.parameters()).device
    dtype = next(adapter.lm_head.parameters()).dtype
    outs = []
    with torch.no_grad():
        for i in range(0, hidden.shape[0], chunk):
            h = hidden[i : i + chunk].to(device=device, dtype=dtype)
            if adapter.final_norm is not None:
                h = adapter.final_norm(h)
            outs.append(adapter.lm_head(h).float().cpu())
    return torch.cat(outs, dim=0)


def _entropy_topk(logits: np.ndarray, vocab: list[str], k: int):
    x = logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    ent = (-(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)).astype(np.float32)
    order = np.argsort(-p, axis=-1)[..., :k]
    L, T = logits.shape[:2]
    topk = [
        [[(vocab[j], float(p[layer, t, j])) for j in order[layer, t]] for t in range(T)]
        for layer in range(L)
    ]
    return ent, topk


def capture(
    model,
    prompt: str,
    tokenizer=None,
    top_k: int = 5,
    device: str = "auto",
    dtype: str = "float32",
    keep_logits: bool = True,
) -> StateTrajectory:
    """Run a forward pass and capture the residual stream at every layer.

    `model` may be a HF model instance (with `tokenizer` supplied), a HF hub
    name, or the string "synthetic".  Returns hidden[layer][token][dimension]
    wrapped in a StateTrajectory with logit-lens statistics attached.
    """
    if isinstance(model, str) and model == "synthetic":
        from models import synthetic

        return synthetic.capture(prompt, top_k=top_k, keep_logits=keep_logits)

    _require_torch()
    if isinstance(model, str):
        model, tokenizer = load_model(model, device=device, dtype=dtype)
    if tokenizer is None:
        raise ValueError("a tokenizer is required when passing a model instance")

    model_device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model_device)
    tokens = [_clean_token(t) for t in tokenizer.convert_ids_to_tokens(input_ids[0].tolist())]

    model.eval()
    with HookCapture(model) as cap, torch.no_grad():
        model(input_ids)
    hidden = cap.stacked()  # (L, T, D)

    logits_t = logit_lens(hidden, cap.adapter)
    vocab = [_clean_token(t) for t in tokenizer.convert_ids_to_tokens(range(logits_t.shape[-1]))]
    logits = logits_t.numpy()
    entropy, topk = _entropy_topk(logits, vocab, top_k)

    return StateTrajectory(
        hidden=hidden.numpy(),
        tokens=tokens,
        logits=logits.astype(np.float16) if keep_logits else None,
        entropy=entropy,
        topk=topk,
        vocab=vocab,
        embedding_matrix=cap.adapter.embedding_weight().numpy(),
        meta={
            "backend": "transformers",
            "model": getattr(getattr(model, "config", None), "name_or_path", type(model).__name__),
            "prompt": prompt,
            "family": cap.adapter.name,
        },
    )


def _clean_token(tok) -> str:
    """Make tokenizer pieces human-readable (SentencePiece/BPE markers)."""
    if tok is None:
        return "<unk>"
    return str(tok).replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\\n")
