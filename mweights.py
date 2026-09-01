"""`.mwt` — model weights for the browser.

The web viewer runs its own forward pass (`viewer/model.js`) so it can record
the residual stream, which means it needs weights in a form a browser can
fetch and use directly: no Python, no runtime, no dependency on how the model
was stored upstream.

`.mwt` is that container, and it is deliberately the same shape as `.mtj`
(spec: docs/mtj-format.md) — a magic, a JSON header, then raw little-endian
buffers — so one parser idea serves both and any language can read it.

Quantisation is per-output-row int8 by default: each row of a weight matrix
gets its own float32 scale, which is the cheapest scheme that survives a
transformer's dynamic range without per-tensor outliers destroying small
rows. It roughly halves an f16 download. The norms and the embedding table
that Mottled reads back as text stay f32 — they are small, and rounding them
would show up directly in the readout the inspector displays.

`export_model` writes one; `viewer/weights.js` reads it. The pair is pinned
end to end by `tests/test_weights_conformance.py`, which exports a real model
and checks the browser's forward pass still matches HuggingFace's.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"MWT1"
ALIGN = 32


# The names viewer/model.js asks for, and where they live in a HuggingFace
# Llama/Qwen3 state dict. Kept as data so a new architecture is a table edit.
_LAYER_TENSORS = [
    ("q_proj", "self_attn.q_proj"),
    ("k_proj", "self_attn.k_proj"),
    ("v_proj", "self_attn.v_proj"),
    ("o_proj", "self_attn.o_proj"),
    ("gate_proj", "mlp.gate_proj"),
    ("up_proj", "mlp.up_proj"),
    ("down_proj", "mlp.down_proj"),
    ("input_layernorm", "input_layernorm"),
    ("post_attention_layernorm", "post_attention_layernorm"),
    # Qwen3 only; absent on Llama and simply skipped when missing.
    ("q_norm", "self_attn.q_norm"),
    ("k_norm", "self_attn.k_norm"),
]

# Kept in full precision regardless of the requested quantisation: the 1-D
# norm weights. They are negligible in size (one vector per norm) and they
# scale everything written after them, so rounding them is all cost.
#
# The embedding table is NOT on this list, though the first version of this
# file had it there — the reasoning being that Mottled reads nearest tokens
# out of it, making its geometry a displayed quantity. That is true and still
# does not justify f32: at Qwen3's 151936 x 1024 the table is 622 MB on its
# own, which made the "small" model a *larger* download than the 4B one it
# was chosen over. Per-row int8 is measured to preserve the neighbour ranking
# the inspector actually shows (test_weights_conformance.py), so it is
# quantised like everything else.
_KEEP_F32 = ("norm", "layernorm")


def _quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 with one float32 scale per output row.

    scale = max|row| / 127, so the largest magnitude in each row lands on the
    end of the int8 range and a row of zeros stays zero (guarded, since a
    zero scale would otherwise produce NaN on the way back).
    """
    w = np.asarray(w, dtype=np.float32)
    peak = np.abs(w).max(axis=1)
    scale = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float32)
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def _dtype_for(name: str, quant: str) -> str:
    if any(k in name for k in _KEEP_F32):
        return "f32"
    return quant


def export_model(model, config, path, *, quant: str = "q8", tokenizer=None) -> dict:
    """Write a HuggingFace causal LM to `path` as `.mwt`.

    `quant` is "q8" (per-row int8), "f16", or "f32". Returns the header that
    was written, so a caller can report the size breakdown.
    """
    if quant not in ("q8", "f16", "f32"):
        raise ValueError(f"unknown quant {quant!r}; use q8, f16 or f32")

    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}

    wanted: dict[str, np.ndarray] = {}
    prefix = "model." if any(k.startswith("model.") for k in sd) else ""
    wanted["embed_tokens"] = sd[f"{prefix}embed_tokens.weight"]
    wanted["norm"] = sd[f"{prefix}norm.weight"]

    # A tied model still lists `lm_head.weight` in its state dict, sharing
    # storage with the embedding table — writing both would ship the same
    # matrix twice. That is not a rounding error at this scale: Qwen3-0.6B's
    # table is 151936 x 1024, so the duplicate alone would add ~156 MB to a
    # q8 download. `viewer/model.js` asks for lm_head optionally and falls
    # back to embed_tokens, so omitting it is what the reader already expects.
    tied = bool(getattr(config, "tie_word_embeddings", False))
    if "lm_head.weight" in sd and not tied:
        wanted["lm_head"] = sd["lm_head.weight"]

    for layer in range(config.num_hidden_layers):
        for dst, src in _LAYER_TENSORS:
            key = f"{prefix}layers.{layer}.{src}.weight"
            if key in sd:
                wanted[f"layers.{layer}.{dst}"] = sd[key]

    tensors: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0

    def _append(buf: bytes) -> tuple[int, int]:
        nonlocal offset
        pad = (-offset) % ALIGN
        if pad:
            blobs.append(b"\0" * pad)
            offset += pad
        start = offset
        blobs.append(buf)
        offset += len(buf)
        return start, len(buf)

    for name, arr in wanted.items():
        dtype = _dtype_for(name, quant)
        entry: dict = {"shape": list(arr.shape), "dtype": dtype}

        if dtype == "q8" and arr.ndim == 2:
            q, scale = _quantize_rows(arr)
            entry["offset"], entry["bytes"] = _append(q.tobytes())
            so, sb = _append(scale.astype("<f4").tobytes())
            entry["scaleOffset"], entry["scaleBytes"] = so, sb
        elif dtype == "f16":
            entry["offset"], entry["bytes"] = _append(
                arr.astype("<f2").tobytes())
        else:
            entry["dtype"] = "f32"
            entry["offset"], entry["bytes"] = _append(
                arr.astype("<f4").tobytes())

        tensors[name] = entry

    header = {
        "format": "mwt",
        "version": 1,
        "config": {
            "hiddenSize": int(config.hidden_size),
            "numLayers": int(config.num_hidden_layers),
            "numHeads": int(config.num_attention_heads),
            "numKvHeads": int(getattr(config, "num_key_value_heads",
                                      config.num_attention_heads)),
            "headDim": int(getattr(config, "head_dim", None)
                           or config.hidden_size // config.num_attention_heads),
            "intermediateSize": int(config.intermediate_size),
            "vocabSize": int(config.vocab_size),
            "rmsNormEps": float(getattr(config, "rms_norm_eps", 1e-6)),
            "ropeTheta": float(getattr(config, "rope_theta", 10000.0)),
            "tiedEmbeddings": bool(getattr(config, "tie_word_embeddings", False)),
        },
        "tensors": tensors,
        "quant": quant,
    }
    if tokenizer is not None:
        header["tokenizer"] = _tokenizer_record(tokenizer)

    raw = json.dumps(header).encode("utf-8")
    # Pad the header so the data section starts on an ALIGN boundary. Tensor
    # offsets are relative to it, and a reader that takes a *view* over the
    # buffer (rather than copying) needs the absolute address aligned to the
    # element width — `new Uint16Array(buffer, off, n)` throws on an odd
    # offset. Without this the file is readable or not depending on how many
    # bytes the JSON happened to occupy, which is the worst kind of bug:
    # it passes until an unrelated header change shifts the length by one.
    # Trailing spaces are insignificant whitespace, so the JSON still parses.
    pad = (-(8 + len(raw))) % ALIGN
    raw += b" " * pad

    out = Path(path)
    with out.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(raw)))
        fh.write(raw)
        for blob in blobs:
            fh.write(blob)

    return header


def _tokenizer_record(tokenizer) -> dict:
    """Everything `viewer/tokenizer.js` needs to encode a typed prompt.

    The id -> piece table alone only supports *labelling* states. A live
    viewer has to go the other way too — text to ids — so the merge ranks and
    the added tokens travel with it. Merges are the bulk of this (Qwen3 has
    ~151k), but they are short strings and compress well over the wire, and
    without them a prompt cannot be tokenized in the page at all.
    """
    vocab = tokenizer.get_vocab()
    pieces = [""] * (max(vocab.values()) + 1)
    for piece, idx in vocab.items():
        pieces[idx] = piece

    record = {"pieces": pieces, "size": len(pieces)}

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        spec = json.loads(backend.to_str())
        model = spec.get("model", {})
        if model.get("type") == "BPE":
            # Normalise both published shapes ("a b" and ["a", "b"]) to pairs;
            # the JS accepts either, but one shape in the file is simpler.
            record["merges"] = [
                m if isinstance(m, list) else m.split(" ", 1)
                for m in model.get("merges", [])
            ]
        record["addedTokens"] = [
            {"content": a["content"], "id": a["id"]}
            for a in spec.get("added_tokens", [])
        ]
    return record


def read_header(path) -> dict:
    """Parse just the header — enough to report shapes and sizes."""
    with Path(path).open("rb") as fh:
        if fh.read(4) != MAGIC:
            raise ValueError("not a .mwt file")
        (n,) = struct.unpack("<I", fh.read(4))
        return json.loads(fh.read(n).decode("utf-8"))


def load_tensor(path, name: str) -> np.ndarray:
    """Read one tensor back as float32 — the reference the JS loader must match."""
    header = read_header(path)
    entry = header["tensors"][name]
    data = Path(path).read_bytes()
    base = 8 + len(json.dumps(header).encode("utf-8"))
    # The header round-trips through json.dumps identically only if key order
    # is preserved, which it is; but read the offset from the file instead of
    # recomputing it to stay robust.
    with Path(path).open("rb") as fh:
        fh.seek(4)
        (n,) = struct.unpack("<I", fh.read(4))
        base = 8 + n

    off = base + entry["offset"]
    if entry["dtype"] == "q8":
        q = np.frombuffer(data, dtype=np.int8, count=entry["bytes"], offset=off)
        so = base + entry["scaleOffset"]
        scale = np.frombuffer(data, dtype="<f4",
                              count=entry["scaleBytes"] // 4, offset=so)
        return (q.reshape(entry["shape"]).astype(np.float32)
                * scale[:, None]).astype(np.float32)
    if entry["dtype"] == "f16":
        arr = np.frombuffer(data, dtype="<f2", count=entry["bytes"] // 2, offset=off)
        return arr.reshape(entry["shape"]).astype(np.float32)
    arr = np.frombuffer(data, dtype="<f4", count=entry["bytes"] // 4, offset=off)
    return arr.reshape(entry["shape"]).astype(np.float32)
