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
    """Context manager that records — and optionally *edits* — the residual
    stream around every block.

    Layer 0 is taken from a forward *pre*-hook on the first block — the exact
    tensor entering the residual stream (this is also correct for Gemma,
    which scales embeddings after the embedding module).  Layers 1..N come
    from forward hooks on each block's output.

    Passing ``state_edits`` and/or ``frozen_blocks`` turns the capture into a
    *resumable/intervened* pass: the forward runs to a block, the residual is
    rewritten, and the model continues from the edited state — this is what
    makes perturb-and-replay possible.  With neither argument the behaviour is
    a pure read-only capture, identical to before.

        state_edits[k]  : callable(hidden) -> hidden applied to the state at
                          index k (0 = embeddings) *before* it is captured and
                          before it propagates downstream.
        frozen_blocks   : block indices whose update is skipped (output := input),
                          i.e. the residual stream passes through unchanged.

    Captured states always reflect the value that actually propagated, so a
    perturbation at layer k appears in ``hidden[k]`` and in every layer after.

    ``capture_components=True`` additionally hooks every block's attention and
    MLP submodules and records their outputs — the two additive writes to the
    residual stream, so for pre-norm architectures (Llama-style, GPT-2, NeoX)
    ``hidden[l+1] = hidden[l] + attn[l] + mlp[l]`` exactly.  Frozen blocks
    skip their update, so their recorded components no longer propagate.
    """

    def __init__(self, model, state_edits: dict | None = None,
                 frozen_blocks: set | None = None, capture_components: bool = False):
        from models.families import resolve_family

        self.adapter = resolve_family(model)
        self.states: dict[int, "torch.Tensor"] = {}
        self.components: dict[str, dict[int, "torch.Tensor"]] = {"attn": {}, "mlp": {}}
        self.state_edits = dict(state_edits or {})
        self.frozen_blocks = set(frozen_blocks or set())
        self.capture_components = capture_components
        self._handles = []

    def __enter__(self):
        blocks = list(self.adapter.blocks)

        def pre_hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            edit = self.state_edits.get(0)
            if edit is not None:
                hs = edit(hs)
                self.states[0] = hs.detach()
                if args:
                    return (hs,) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hs
                return args, kwargs
            self.states[0] = hs.detach()
            return None

        self._handles.append(blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
        for i, block in enumerate(blocks):
            def hook(module, args, output, _block=i, _layer=i + 1):
                changed = False
                if _block in self.frozen_blocks:            # skip the update
                    out = args[0]
                    changed = True
                else:
                    out = output[0] if isinstance(output, tuple) else output
                edit = self.state_edits.get(_layer)
                if edit is not None:
                    out = edit(out)
                    changed = True
                self.states[_layer] = out.detach()
                if not changed:
                    return None
                if isinstance(output, tuple):
                    return (out,) + tuple(output[1:])
                return out

            self._handles.append(block.register_forward_hook(hook))

        if self.capture_components:
            def component_hook(name: str, i: int):
                def hook(module, args, output):
                    out = output[0] if isinstance(output, tuple) else output
                    self.components[name][i] = out.detach()
                return hook

            for i, block in enumerate(blocks):
                attn, mlp = self.adapter.block_submodules(block)
                self._handles.append(attn.register_forward_hook(component_hook("attn", i)))
                self._handles.append(mlp.register_forward_hook(component_hook("mlp", i)))
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

    def stacked_components(self) -> dict[str, "torch.Tensor"]:
        """{"attn": (L-1, T, D), "mlp": (L-1, T, D)} float32 on CPU."""
        n = len(self.adapter.blocks)
        out = {}
        for name, per_block in self.components.items():
            missing = [i for i in range(n) if i not in per_block]
            if missing:
                raise RuntimeError(f"missing {name} captures for blocks {missing}")
            out[name] = torch.stack([per_block[i][0] for i in range(n)]).float().cpu()
        return out


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
    capture_components: bool = False,
    capture_attention: bool = False,
) -> StateTrajectory:
    """Run a forward pass and capture the residual stream at every layer.

    `model` may be a HF model instance (with `tokenizer` supplied), a HF hub
    name, or the string "synthetic".  Returns hidden[layer][token][dimension]
    wrapped in a StateTrajectory with logit-lens statistics attached.

    `capture_components=True` also records each block's attention and MLP
    outputs — the residual decomposition — in `StateTrajectory.components`.
    `capture_attention=True` records each block's head-averaged attention
    pattern (L-1, T, T) in `StateTrajectory.attention`.
    """
    if isinstance(model, str) and model == "synthetic":
        from models import synthetic

        return synthetic.capture(prompt, top_k=top_k, keep_logits=keep_logits,
                                 capture_components=capture_components,
                                 capture_attention=capture_attention)

    _require_torch()
    return _run(model, prompt, tokenizer=tokenizer, top_k=top_k, device=device,
                dtype=dtype, keep_logits=keep_logits,
                capture_components=capture_components,
                capture_attention=capture_attention)


def _run(model, prompt, tokenizer=None, top_k=5, device="auto", dtype="float32",
         keep_logits=True, state_edits: dict | None = None,
         frozen_blocks: set | None = None, extra_meta: dict | None = None,
         capture_components: bool = False,
         capture_attention: bool = False) -> StateTrajectory:
    """Forward pass (optionally intervened) -> StateTrajectory. Shared by
    capture() and intervene()."""
    if isinstance(model, str):
        model, tokenizer = load_model(model, device=device, dtype=dtype)
    if tokenizer is None:
        raise ValueError("a tokenizer is required when passing a model instance")

    model_device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model_device)
    tokens = [_clean_token(t) for t in tokenizer.convert_ids_to_tokens(input_ids[0].tolist())]

    model.eval()
    # sdpa/flash kernels never materialise the attention matrix; switch the
    # dispatch to eager for this pass so output_attentions actually returns.
    config = getattr(model, "config", None)
    prev_impl = getattr(config, "_attn_implementation", None)
    if capture_attention and prev_impl and prev_impl != "eager":
        config._attn_implementation = "eager"
    try:
        with HookCapture(model, state_edits=state_edits, frozen_blocks=frozen_blocks,
                         capture_components=capture_components) as cap, torch.no_grad():
            out = model(input_ids, output_attentions=capture_attention or None)
    finally:
        if capture_attention and prev_impl and prev_impl != "eager":
            config._attn_implementation = prev_impl
    hidden = cap.stacked()  # (L, T, D)
    components = None
    if capture_components:
        components = {k: v.numpy() for k, v in cap.stacked_components().items()}
    attention = None
    if capture_attention:
        if not getattr(out, "attentions", None):
            raise ValueError("model returned no attention patterns; "
                             "this architecture does not support capture_attention")
        # (blocks, B, H, T, T) -> head-average, squeeze batch -> (L-1, T, T)
        attention = torch.stack([a.float().mean(dim=1)[0] for a in out.attentions]).cpu().numpy()

    logits_t = logit_lens(hidden, cap.adapter)
    vocab = [_clean_token(t) for t in tokenizer.convert_ids_to_tokens(range(logits_t.shape[-1]))]
    logits = logits_t.numpy()
    entropy, topk = _entropy_topk(logits, vocab, top_k)

    meta = {
        "backend": "transformers",
        "model": getattr(getattr(model, "config", None), "name_or_path", type(model).__name__),
        "prompt": prompt,
        "family": cap.adapter.name,
    }
    if extra_meta:
        meta.update(extra_meta)
    return StateTrajectory(
        hidden=hidden.numpy(),
        tokens=tokens,
        logits=logits.astype(np.float16) if keep_logits else None,
        entropy=entropy,
        topk=topk,
        vocab=vocab,
        embedding_matrix=cap.adapter.embedding_weight().numpy(),
        components=components,
        attention=attention,
        meta=meta,
    )


def _clean_token(tok) -> str:
    """Make tokenizer pieces human-readable (SentencePiece/BPE markers)."""
    if tok is None:
        return "<unk>"
    return str(tok).replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\\n")
