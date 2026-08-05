"""Layer-by-layer capture for models too large to hold in memory.

A forward pass with no gradients only needs block *i*'s weights while block
*i* is running. So a model whose weights do not fit in RAM can still be
traced: build the module skeleton with no weights at all, then materialise
each block from disk immediately before it runs and release it immediately
after. Peak memory becomes **one block plus the activations**, not the whole
model.

That is exactly the shape of what Mottled wants — the residual stream at
every layer for a short prompt — so the expensive resource is a single
sequential read of the weights, not RAM. For a 93-layer, 1.5 TB model like
Kimi K3 the difference is between "impossible" and "slow".

    from stream import stream_capture
    traj = stream_capture("/path/to/checkpoint", "The capital of France is")

Weights are materialised by *hooks around the model's own blocks*, never by
reimplementing a forward pass. That matters: modern frontier models use
attention this module has never heard of (linear/delta variants, latent
attention, MoE routing), and hand-rolling those forwards would be both
wrong and unmaintainable. Here the architecture stays HuggingFace's problem
and only *when weights exist* is ours.

What this does not do: it does not make a 1.5 TB model fast, and it does not
avoid needing the weights on disk. It removes the RAM ceiling, nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trajectory import StateTrajectory

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


class ShardIndex:
    """Which file holds which tensor, for a HF checkpoint directory."""

    def __init__(self, path: str | Path):
        self.root = Path(path)
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self.map = {k: self.root / v for k, v in weight_map.items()}
        else:
            single = self.root / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"no model.safetensors or shard index under {self.root}")
            from safetensors import safe_open

            with safe_open(str(single), framework="pt") as f:
                self.map = {k: single for k in f.keys()}
        self._open: dict[Path, object] = {}

    def prefixed(self, prefix: str) -> dict:
        """Every tensor whose name starts with `prefix`, keyed without it."""
        from safetensors import safe_open

        out, wanted = {}, [k for k in self.map if k.startswith(prefix)]
        by_file: dict[Path, list[str]] = {}
        for k in wanted:
            by_file.setdefault(self.map[k], []).append(k)
        for file, keys in by_file.items():
            with safe_open(str(file), framework="pt") as f:
                for k in keys:
                    out[k[len(prefix):]] = f.get_tensor(k)
        return out

    def __len__(self) -> int:
        return len(self.map)


def _materialise_computed_buffers(model, config) -> None:
    """Give back the buffers that a meta-device build leaves empty.

    Some buffers are *computed* at construction rather than loaded from the
    checkpoint — rotary `inv_freq` is the usual one — so building on the meta
    device leaves them with no data and the first forward dies inside the
    rope helper. They are also tiny, so the fix is to re-instantiate just the
    module that owns them on a real device and copy the fresh values across.
    """
    for name, module in list(model.named_modules()):
        stranded = [n for n, b in module.named_buffers(recurse=False) if b.is_meta]
        if not stranded:
            continue
        fresh = None
        for build in (lambda: type(module)(config=config),
                      lambda: type(module)(config)):
            try:
                fresh = build()
                break
            except Exception:
                continue
        if fresh is None:
            raise RuntimeError(
                f"{name} has computed buffers {stranded} that a meta-device "
                f"build leaves empty, and it could not be rebuilt from the "
                f"config. Streaming this architecture needs a hand-written "
                f"buffer rule.")
        for n in stranded:
            module.register_buffer(n, getattr(fresh, n).clone(), persistent=False)


def _fuse_experts(sd: dict, block) -> dict:
    """Reshape per-expert checkpoint tensors into a fused runtime layout.

    Sparse-MoE checkpoints usually store one tensor per expert
    (`mlp.experts.3.gate_proj.weight`), while the runtime module keeps them
    stacked (`mlp.experts.gate_up_proj` of shape `(n_experts, …)`) so the
    forward can use a grouped matmul. `from_pretrained` does this conversion
    internally; a streamed load has to do it too, or the experts silently
    stay empty.

    Only the common `gate/up/down` fusion is handled. Anything else trips the
    stranded-parameter check in `_load`, which is the intended outcome — a
    loud, named failure beats a plausible wrong answer.
    """
    per_expert = {}
    for key in list(sd):
        parts = key.split(".")
        if "experts" not in parts:
            continue
        e = parts.index("experts")
        if e + 2 >= len(parts) or not parts[e + 1].isdigit():
            continue
        idx, proj = int(parts[e + 1]), parts[e + 2]
        per_expert.setdefault(".".join(parts[:e + 1]), {}).setdefault(proj, {})[idx] = sd.pop(key)
    if not per_expert:
        return sd

    want = {n for n, _ in block.named_parameters()}
    for root, projs in per_expert.items():
        order = lambda d: [d[i] for i in sorted(d)]
        if {"gate_proj", "up_proj"} <= set(projs) and f"{root}.gate_up_proj" in want:
            # (E, 2*inter, hidden) -> transposed to the runtime's (E, ..., ...)
            gate, up = order(projs["gate_proj"]), order(projs["up_proj"])
            fused = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate, up)])
            target = dict(block.named_parameters())[f"{root}.gate_up_proj"]
            sd[f"{root}.gate_up_proj"] = (fused.transpose(1, 2).contiguous()
                                          if fused.shape != target.shape else fused)
        if "down_proj" in projs and f"{root}.down_proj" in want:
            down = torch.stack(order(projs["down_proj"]))
            target = dict(block.named_parameters())[f"{root}.down_proj"]
            sd[f"{root}.down_proj"] = (down.transpose(1, 2).contiguous()
                                       if down.shape != target.shape else down)
    return sd


def _release(module) -> None:
    """Return a block's parameters to the meta device, freeing their memory.

    Replaces the entries in each owning module's `_parameters` rather than
    assigning to `.data`: after a `load_state_dict(assign=True)` the stored
    object may not be a `Parameter` of a compatible type, and `.data =` on a
    differently-shaped meta tensor is rejected.
    """
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is None:
                continue
            sub._parameters[name] = torch.nn.Parameter(
                torch.empty(0, device="meta", dtype=p.dtype), requires_grad=False)


class StreamedBlocks:
    """Materialise each block just before it runs; release it just after.

    Records the residual stream on the way through, so a streamed pass and an
    in-memory pass produce the same `hidden` — that equality is the whole
    correctness claim and is pinned by a test.
    """

    def __init__(self, model, shards: ShardIndex, blocks, prefix: str = "model.layers.",
                 keep: int = 1):
        self.model, self.shards, self.blocks = model, shards, blocks
        self.prefix, self.keep = prefix, max(1, int(keep))
        self.states: dict[int, "torch.Tensor"] = {}
        self.resident: list[int] = []
        self.loads = 0
        self._handles = []

    def _load(self, i: int) -> None:
        if i in self.resident:
            return
        block = self.blocks[i]
        sd = self.shards.prefixed(f"{self.prefix}{i}.")
        if not sd:
            raise RuntimeError(f"no weights found for block {i} "
                               f"(prefix {self.prefix}{i}.)")
        sd = _fuse_experts(sd, block)
        block.load_state_dict(sd, assign=True, strict=False)

        # Never run a block with weights still on meta. `strict=False` is
        # needed because checkpoints and runtime modules disagree about
        # layout, but a silent skip means a fused kernel quietly takes the
        # meta path and the whole capture is wrong in a way that still looks
        # like numbers. Fail here instead, naming what did not land.
        stranded = [n for n, prm in block.named_parameters() if prm.is_meta]
        if stranded:
            raise RuntimeError(
                f"block {i}: {len(stranded)} parameter(s) never loaded — "
                f"{stranded[:4]}{'…' if len(stranded) > 4 else ''}. The "
                f"checkpoint's layout does not match this module's; streaming "
                f"needs a mapping rule for it (see `_fuse_experts` for the "
                f"per-expert → fused case).")
        self.resident.append(i)
        self.loads += 1
        while len(self.resident) > self.keep:
            _release(self.blocks[self.resident.pop(0)])

    def __enter__(self):
        def pre(module, args, kwargs, _i):
            self._load(_i)
            if _i == 0:
                hs = args[0] if args else kwargs.get("hidden_states")
                self.states[0] = hs.detach().clone()
            return None

        def post(module, args, output, _i):
            out = output[0] if isinstance(output, tuple) else output
            self.states[_i + 1] = out.detach().clone()
            return None

        for i, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_pre_hook(
                lambda m, a, k, _i=i: pre(m, a, k, _i), with_kwargs=True))
            self._handles.append(block.register_forward_hook(
                lambda m, a, o, _i=i: post(m, a, o, _i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for i in list(self.resident):
            _release(self.blocks[i])
        self.resident.clear()

    def stacked(self) -> "torch.Tensor":
        n = len(self.blocks) + 1
        missing = [i for i in range(n) if i not in self.states]
        if missing:
            raise RuntimeError(f"missing streamed captures for layers {missing}")
        return torch.stack([self.states[i][0] for i in range(n)]).float().cpu()


def stream_capture(
    checkpoint: str | Path,
    prompt: str,
    tokenizer=None,
    top_k: int = 5,
    keep_logits: bool = True,
    keep_resident: int = 1,
    capture_routing: bool = False,
    dtype: str = "float32",
) -> StateTrajectory:
    """Capture the residual stream without ever holding the whole model.

    `checkpoint` is a local HuggingFace checkpoint directory (config + one or
    more `.safetensors`). Blocks are streamed from those files; the
    embeddings, final norm and LM head stay resident, since they are a single
    matrix each rather than a per-layer cost.

    `keep_resident` is how many blocks may be materialised at once — 1 is the
    minimum-memory setting. Raise it only to trade memory for fewer loads.

    The result is an ordinary `StateTrajectory`: everything downstream is
    unchanged, which is the point of having an interchange format.
    """
    if not HAS_TORCH:
        raise ImportError("stream_capture needs torch")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    root = Path(checkpoint)
    config = AutoConfig.from_pretrained(root)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(root)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    model.eval()
    _materialise_computed_buffers(model, config)

    shards = ShardIndex(root)
    inner = getattr(model, "model", model)
    blocks = list(inner.layers)

    # Everything that is not a block: one matrix each, so they stay resident.
    torch_dtype = getattr(torch, dtype)
    for name, module in (("embed", getattr(inner, "embed_tokens", None)),
                         ("norm", getattr(inner, "norm", None)),
                         ("head", getattr(model, "lm_head", None))):
        if module is None:
            continue
        prefix = {"embed": "model.embed_tokens.", "norm": "model.norm.",
                  "head": "lm_head."}[name]
        sd = shards.prefixed(prefix)
        if sd:
            module.load_state_dict(sd, assign=True, strict=False)
        elif name == "head" and getattr(config, "tie_word_embeddings", False):
            module.weight = getattr(inner, "embed_tokens").weight

    from capture import _clean_token, _entropy_topk, _routing_from

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    tokens = [_clean_token(t)
              for t in tokenizer.convert_ids_to_tokens(input_ids[0].tolist())]

    with StreamedBlocks(model, shards, blocks, keep=keep_resident) as sb, \
            torch.no_grad():
        out = model(input_ids,
                    **({"output_router_logits": True} if capture_routing else {}))
    hidden = sb.stacked()

    routing = _routing_from(out, model, capture_routing)

    # Logit lens, one chunk at a time: the head is resident, the states are not
    # large, and this mirrors `capture.logit_lens` exactly.
    norm = getattr(inner, "norm", None)
    head = getattr(model, "lm_head", None)
    if head is None:
        raise ValueError("checkpoint exposes no LM head; cannot apply the logit lens")
    outs = []
    with torch.no_grad():
        for i in range(hidden.shape[0]):
            h = hidden[i : i + 1].to(dtype=torch_dtype)
            if norm is not None:
                h = norm(h)
            outs.append(head(h).float())
    logits = torch.cat(outs, dim=0).numpy()

    vocab = [_clean_token(t)
             for t in tokenizer.convert_ids_to_tokens(range(logits.shape[-1]))]
    entropy, topk = _entropy_topk(logits, vocab, top_k)

    traj = StateTrajectory(
        hidden=hidden.numpy(),
        tokens=tokens,
        logits=logits.astype(np.float16) if keep_logits else None,
        entropy=entropy,
        topk=topk,
        vocab=vocab,
        routing=routing,
        meta={
            "backend": "streamed",
            "model": str(root),
            "prompt": prompt,
            "streamed": True,
            "n_blocks": len(blocks),
            "block_loads": sb.loads,
            "keep_resident": keep_resident,
            # what a streamed pass cannot give you, stated rather than implied
            "absent": ["residual decomposition", "attention patterns"],
        },
    )
    traj.validate()
    return traj
