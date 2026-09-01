"""GGUF conformance: the browser's dequantiser is llama.cpp's dequantiser.

`viewer/gguf.js` lets the viewer load open-weight models in the format they
are actually published in — including the ternary (1.58-bit) builds, which are
the only reason a 4B model is a ~1 GB download instead of an 8 GB one.

A quantised block layout is exactly the kind of thing that is easy to get
*almost* right: a wrong unpacking produces a tensor of plausible magnitude and
a model that runs and predicts nonsense. So none of this is checked against my
reading of the format. It is checked against the `gguf` package's own
reference dequantiser, over blocks that package produced.
"""
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
GGUF_JS = ROOT / "viewer" / "gguf.js"
MODEL_JS = ROOT / "viewer" / "model.js"

gguf = pytest.importorskip("gguf")

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")

from gguf.constants import GGMLQuantizationType as QT  # noqa: E402


def _run_node(script, *args):
    proc = subprocess.run(["node", "-e", script, *[str(a) for a in args]],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"node failed: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def _write_gguf(path, tensors, meta=None, arch="qwen3"):
    """Write a real .gguf with the reference writer, quantising as asked.

    `tensors` maps name -> (float32 array, GGMLQuantizationType).
    """
    writer = gguf.GGUFWriter(str(path), arch)
    for key, value in (meta or {}).items():
        if isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)

    for name, (arr, qtype) in tensors.items():
        if qtype == QT.F32:
            writer.add_tensor(name, arr, raw_dtype=qtype)
            continue
        data = gguf.quants.quantize(arr, qtype)
        # With pre-quantised data the writer reads `raw_shape` as the *byte*
        # shape and converts it back to the logical one itself; passing the
        # logical shape makes it reject the tensor.
        writer.add_tensor(name, data, raw_dtype=qtype, raw_shape=data.shape)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


JS_DEQUANT = """
const fs = require("fs");
const G = require(process.argv[1]);
const buf = fs.readFileSync(process.argv[2]);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const parsed = G.parse(ab);
const info = parsed.tensors[process.argv[3]];
const out = G.dequantize(ab, parsed.dataStart, info);
process.stdout.write(JSON.stringify({
  values: Array.from(out),
  dims: info.dims,
  type: info.type,
}));
"""


@pytest.mark.parametrize("qtype,name", [
    (QT.F32, "F32"), (QT.F16, "F16"), (QT.Q8_0, "Q8_0"),
    (QT.TQ1_0, "TQ1_0"), (QT.TQ2_0, "TQ2_0"),
])
def test_dequant_matches_reference(tmp_path, qtype, name):
    """Every supported block type decodes to the reference's floats."""
    rng = np.random.default_rng(0)
    # 512 columns = 2 full QK_K blocks per row, so the block loop is exercised
    # rather than only its first iteration.
    arr = (rng.normal(size=(4, 512)) * 0.05).astype(np.float32)

    path = tmp_path / f"{name}.gguf"
    _write_gguf(path, {"token_embd.weight": (arr, qtype)})

    js = _run_node(JS_DEQUANT, GGUF_JS, path, "token_embd.weight")
    got = np.asarray(js["values"], dtype=np.float32)

    if qtype == QT.F32:
        want = arr.ravel()
    else:
        quantized = gguf.quants.quantize(arr, qtype)
        want = gguf.quants.dequantize(quantized, qtype).astype(np.float32).ravel()

    assert got.shape == want.shape, f"{name}: {got.shape} != {want.shape}"
    assert np.array_equal(got, want), (
        f"{name}: JS dequant differs from the reference, "
        f"max |Δ| = {np.abs(got - want).max():.3e}, "
        f"first mismatch at {int(np.argmax(got != want))}")


def test_ternary_values_really_are_ternary(tmp_path):
    """A sanity check the reference comparison alone would not catch.

    If both sides shared a misconception the equality test would still pass,
    so this asserts the defining property directly: every dequantised value is
    -d, 0, or +d for that block's scale.
    """
    rng = np.random.default_rng(1)
    arr = (rng.normal(size=(2, 256)) * 0.05).astype(np.float32)
    path = tmp_path / "tq.gguf"
    _write_gguf(path, {"token_embd.weight": (arr, QT.TQ2_0)})

    js = _run_node(JS_DEQUANT, GGUF_JS, path, "token_embd.weight")
    got = np.asarray(js["values"], dtype=np.float32).reshape(2, 256)

    for row in got:
        scale = np.abs(row).max()
        assert scale > 0
        ratios = row / scale
        assert np.allclose(np.abs(np.round(ratios)), np.abs(ratios), atol=1e-6)
        assert set(np.unique(np.round(ratios).astype(int))) <= {-1, 0, 1}


def test_unsupported_type_fails_by_name(tmp_path):
    """A type we cannot read must throw, never mis-read bytes."""
    rng = np.random.default_rng(2)
    arr = (rng.normal(size=(4, 256)) * 0.05).astype(np.float32)
    path = tmp_path / "q4.gguf"
    _write_gguf(path, {"token_embd.weight": (arr, QT.Q4_0)})

    script = JS_DEQUANT.replace(
        'process.stdout.write', 'if(0)process.stdout.write')
    proc = subprocess.run(
        ["node", "-e", script, str(GGUF_JS), str(path), "token_embd.weight"],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "does not implement" in proc.stderr


def test_metadata_and_names_map_to_the_model(tmp_path):
    """Config and tensor names arrive in the shape viewer/model.js expects."""
    rng = np.random.default_rng(3)
    small = lambda *s: (rng.normal(size=s) * 0.05).astype(np.float32)

    tensors = {
        "token_embd.weight": (small(32, 64), QT.F32),
        "output_norm.weight": (small(64), QT.F32),
        "blk.0.attn_q.weight": (small(64, 64), QT.F32),
        "blk.0.attn_k.weight": (small(32, 64), QT.F32),
        "blk.0.attn_v.weight": (small(32, 64), QT.F32),
        "blk.0.attn_output.weight": (small(64, 64), QT.F32),
        "blk.0.ffn_gate.weight": (small(128, 64), QT.F32),
        "blk.0.ffn_up.weight": (small(128, 64), QT.F32),
        "blk.0.ffn_down.weight": (small(64, 128), QT.F32),
        "blk.0.attn_norm.weight": (small(64), QT.F32),
        "blk.0.ffn_norm.weight": (small(64), QT.F32),
    }
    meta = {
        "qwen3.block_count": 1,
        "qwen3.embedding_length": 64,
        "qwen3.attention.head_count": 4,
        "qwen3.attention.head_count_kv": 2,
        "qwen3.feed_forward_length": 128,
        "qwen3.attention.key_length": 16,
        "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen3.rope.freq_base": 1000000.0,
    }
    path = tmp_path / "tiny.gguf"
    _write_gguf(path, tensors, meta)

    script = """
    const fs = require("fs");
    const G = require(process.argv[1]);
    const buf = fs.readFileSync(process.argv[2]);
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const m = G.open(ab);
    process.stdout.write(JSON.stringify({
      config: m.config,
      names: m.names.sort(),
      hasLmHead: m.get("lm_head", true) !== null,
      qLen: m.get("layers.0.q_proj").length,
    }));
    """
    js = _run_node(script, GGUF_JS, path)

    cfg = js["config"]
    assert cfg["numLayers"] == 1
    assert cfg["hiddenSize"] == 64
    assert cfg["numHeads"] == 4 and cfg["numKvHeads"] == 2
    assert cfg["headDim"] == 16
    assert cfg["intermediateSize"] == 128
    assert cfg["ropeTheta"] == 1000000.0
    assert cfg["architecture"] == "qwen3"

    # The names model.js asks for, not GGUF's own.
    assert "layers.0.q_proj" in js["names"]
    assert "layers.0.input_layernorm" in js["names"]
    assert "embed_tokens" in js["names"] and "norm" in js["names"]
    # No output.weight was written, so the head is tied and resolves to null
    # (model.js falls back to embed_tokens).
    assert js["hasLmHead"] is False
    assert js["qLen"] == 64 * 64


def test_forward_pass_runs_on_a_ternary_gguf(tmp_path):
    """End to end: a ternary GGUF drives the browser's forward pass.

    Ternary weights are far too coarse to reproduce a float model's states, so
    this does not check numerical agreement -- it checks that the whole path
    holds together on the format Bonsai ships in, and that the residual
    decomposition invariant still reconciles exactly, which it must regardless
    of how the weights were stored.
    """
    rng = np.random.default_rng(4)
    D, I, V, H, KV, HD = 256, 512, 512, 4, 2, 64
    w = lambda *s: (rng.normal(size=s) * 0.05).astype(np.float32)
    norm = lambda n: (np.abs(rng.normal(size=n)) * 0.1 + 1).astype(np.float32)

    tensors = {
        "token_embd.weight": (w(V, D), QT.TQ2_0),
        "output_norm.weight": (norm(D), QT.F32),
        "blk.0.attn_q.weight": (w(H * HD, D), QT.TQ2_0),
        "blk.0.attn_k.weight": (w(KV * HD, D), QT.TQ2_0),
        "blk.0.attn_v.weight": (w(KV * HD, D), QT.TQ2_0),
        "blk.0.attn_output.weight": (w(D, H * HD), QT.TQ2_0),
        "blk.0.ffn_gate.weight": (w(I, D), QT.TQ2_0),
        "blk.0.ffn_up.weight": (w(I, D), QT.TQ2_0),
        "blk.0.ffn_down.weight": (w(D, I), QT.TQ2_0),
        "blk.0.attn_norm.weight": (norm(D), QT.F32),
        "blk.0.ffn_norm.weight": (norm(D), QT.F32),
    }
    meta = {
        "qwen3.block_count": 1, "qwen3.embedding_length": D,
        "qwen3.attention.head_count": H, "qwen3.attention.head_count_kv": KV,
        "qwen3.feed_forward_length": I, "qwen3.attention.key_length": HD,
        "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen3.rope.freq_base": 1000000.0,
        "tokenizer.ggml.tokens": [f"t{i}" for i in range(V)],
    }
    path = tmp_path / "ternary.gguf"
    _write_gguf(path, tensors, meta)

    script = """
    const fs = require("fs");
    const G = require(process.argv[1]);
    const M = require(process.argv[2]);
    const buf = fs.readFileSync(process.argv[3]);
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const m = G.open(ab);
    const cfg = Object.assign({}, m.config, { vocabSize: m.config.vocabSize });
    M.forward(m, [3, 17, 42], cfg, { captureComponents: true }).then(out => {
      process.stdout.write(JSON.stringify({
        hidden: Array.from(out.hidden),
        attn: Array.from(out.components.attn),
        mlp: Array.from(out.components.mlp),
        logits: Array.from(out.logits),
        nLayers: out.nLayers,
        vocab: cfg.vocabSize,
      }));
    }).catch(e => { console.error(e); process.exit(1); });
    """
    js = _run_node(script, GGUF_JS, MODEL_JS, path)

    T = 3
    hidden = np.asarray(js["hidden"], dtype=np.float64).reshape(2, T, D)
    attn = np.asarray(js["attn"], dtype=np.float64).reshape(1, T, D)
    mlp = np.asarray(js["mlp"], dtype=np.float64).reshape(1, T, D)

    assert np.isfinite(hidden).all(), "ternary forward produced non-finite states"
    assert np.allclose(hidden[:-1] + attn + mlp, hidden[1:], atol=1e-5), \
        "residual decomposition broke on ternary weights"
    assert len(js["logits"]) == T * js["vocab"]
    # The states must actually move; an all-zero dequant would still satisfy
    # the decomposition above.
    assert np.abs(hidden[1] - hidden[0]).max() > 1e-6
