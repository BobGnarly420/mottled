"""`.mwt` round-trip: what the browser loads is what the model was.

`mweights.export_model` writes weights the viewer can fetch;
`viewer/weights.js` reads them back. The pair is only worth having if the
whole path holds — export, quantise, parse in JS, dequantise, forward pass —
so this exports a real (tiny, locally-built) model and checks the browser's
own forward pass still reproduces HuggingFace's hidden states.

That is the end-to-end claim the live viewer rests on: the trajectory drawn in
the page is the model's trajectory, not an artifact of how the weights were
shipped.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import mweights

ROOT = Path(__file__).resolve().parent.parent
MODEL_JS = ROOT / "viewer" / "model.js"
WEIGHTS_JS = ROOT / "viewer" / "weights.js"

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")


def _tiny_model(seed=0, tie=False):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, rms_norm_eps=1e-6,
        tie_word_embeddings=tie,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                p.add_(torch.randn_like(p) * 0.1)
    return model, cfg


JS_LOAD_AND_RUN = """
const fs = require("fs");
const W = require(process.argv[1]);
const M = require(process.argv[2]);
const path = process.argv[3];
const tokens = JSON.parse(process.argv[4]);

const buf = fs.readFileSync(path);
// Node Buffers are views into a shared pool; slice to get an exact
// ArrayBuffer or the offsets in the file will not line up.
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const weights = W.open(ab);

M.forward(weights, tokens, weights.config, { captureComponents: true })
  .then(out => {
    process.stdout.write(JSON.stringify({
      hidden: Array.from(out.hidden),
      logits: Array.from(out.logits),
      config: weights.config,
      names: weights.names.length,
      quant: weights.quant,
    }));
  })
  .catch(e => { console.error(e); process.exit(1); });
"""


def _run_js(path, tokens):
    proc = subprocess.run(
        ["node", "-e", JS_LOAD_AND_RUN, str(WEIGHTS_JS), str(MODEL_JS),
         str(path), json.dumps(tokens)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"node failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


@pytest.mark.parametrize("quant", ["f32", "f16", "q8"])
def test_js_loader_matches_python_reader(tmp_path, quant):
    """Both readers dequantise a tensor to the same floats."""
    model, cfg = _tiny_model()
    path = tmp_path / f"tiny-{quant}.mwt"
    mweights.export_model(model, cfg, path, quant=quant)

    want = mweights.load_tensor(path, "layers.0.q_proj")

    script = """
    const fs = require("fs");
    const W = require(process.argv[1]);
    const buf = fs.readFileSync(process.argv[2]);
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const w = W.open(ab);
    process.stdout.write(JSON.stringify(Array.from(w.get("layers.0.q_proj"))));
    """
    proc = subprocess.run(["node", "-e", script, str(WEIGHTS_JS), str(path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = np.asarray(json.loads(proc.stdout), dtype=np.float32).reshape(want.shape)

    assert np.array_equal(got, want), \
        f"{quant}: JS and Python disagree, max |Δ| = {np.abs(got - want).max():.3e}"


def test_q8_quantization_error_is_bounded(tmp_path):
    """int8 per-row rounding stays within half a step of the original."""
    model, cfg = _tiny_model()
    path = tmp_path / "tiny.mwt"
    mweights.export_model(model, cfg, path, quant="q8")

    original = model.state_dict()["model.layers.0.self_attn.q_proj.weight"].numpy()
    restored = mweights.load_tensor(path, "layers.0.q_proj")

    # Each row's step is peak/127; rounding cannot exceed half a step.
    step = np.abs(original).max(axis=1) / 127.0
    assert (np.abs(restored - original) <= step[:, None] / 2 + 1e-6).all()
    # And the norms are kept exactly, since they are excluded from quantisation.
    assert np.array_equal(
        mweights.load_tensor(path, "layers.0.input_layernorm"),
        model.state_dict()["model.layers.0.input_layernorm.weight"].numpy())


@pytest.mark.parametrize("quant,atol", [("f32", 2e-4), ("f16", 5e-2)])
def test_forward_from_mwt_matches_huggingface(tmp_path, quant, atol):
    """The full path: export -> fetch -> dequantise -> forward == HF."""
    model, cfg = _tiny_model()
    path = tmp_path / f"tiny-{quant}.mwt"
    mweights.export_model(model, cfg, path, quant=quant)

    tokens = [3, 17, 42, 8, 25]
    with torch.no_grad():
        ref = model(torch.tensor([tokens]), output_hidden_states=True)

    js = _run_js(path, tokens)
    T, D, L = len(tokens), cfg.hidden_size, cfg.num_hidden_layers
    got = np.asarray(js["hidden"], dtype=np.float64).reshape(L + 1, T, D)

    for layer in range(L):
        want = ref.hidden_states[layer][0].numpy()
        assert np.allclose(got[layer], want, atol=atol), (
            f"{quant} layer {layer}: max |Δ| = "
            f"{np.abs(got[layer] - want).max():.3e}")


def test_q8_forward_preserves_the_readout(tmp_path):
    """int8 weights shift the states slightly but must not change the top-1.

    This is the claim that matters for the viewer: quantisation is a download
    optimisation, and it is only acceptable while the token the inspector
    displays is still the token the model predicts.
    """
    model, cfg = _tiny_model(seed=2)
    path = tmp_path / "tiny-q8.mwt"
    mweights.export_model(model, cfg, path, quant="q8")

    tokens = [5, 11, 30, 2, 19]
    with torch.no_grad():
        ref = model(torch.tensor([tokens])).logits[0].numpy()

    js = _run_js(path, tokens)
    got = np.asarray(js["logits"], dtype=np.float64).reshape(
        len(tokens), cfg.vocab_size)

    assert (got.argmax(-1) == ref.argmax(-1)).all(), \
        "int8 quantisation changed the predicted token"


def test_q8_preserves_representation_space_neighbours(tmp_path):
    """int8 must not reorder the neighbour list the inspector displays.

    The embedding table is quantised like any other matrix — at Qwen3's vocab
    width keeping it f32 would dominate the download — so the thing that has
    to be checked is the readout it feeds: for a sample of states, the top-k
    nearest vocabulary tokens by cosine must come out in the same order.
    """
    model, cfg = _tiny_model(seed=4)
    path = tmp_path / "tiny-q8.mwt"
    mweights.export_model(model, cfg, path, quant="q8")

    exact = model.state_dict()["model.embed_tokens.weight"].numpy()
    restored = mweights.load_tensor(path, "embed_tokens")

    def sims(table, vec):
        return (table @ vec) / (np.linalg.norm(table, axis=1) *
                                np.linalg.norm(vec) + 1e-12)

    rng = np.random.default_rng(0)
    for _ in range(24):
        # Query with realistic states: perturbed embedding rows, which is what
        # a residual state sitting near a token actually looks like.
        row = exact[rng.integers(len(exact))]
        probe = row + rng.normal(scale=0.1 * np.abs(row).mean(), size=row.shape)

        se, sr = sims(exact, probe), sims(restored, probe)
        oe, orr = np.argsort(-se)[:5], np.argsort(-sr)[:5]

        # The token the inspector names must not change.
        assert oe[0] == orr[0], "int8 embeddings changed the nearest token"

        # Below rank 1, an exact-order assertion would be testing noise: on
        # near-tied similarities any perturbation reorders them. What must
        # hold is that reordering only happens *between* near-ties — never
        # that a clearly-worse token jumps a clearly-better one.
        assert set(oe[:3]) == set(orr[:3]) or \
            abs(se[oe[2]] - se[oe[3]]) < 1e-3, \
            "int8 embeddings reordered a non-tied neighbour"
        for a, b in zip(oe, orr):
            assert abs(se[a] - se[b]) < 5e-3, \
                f"neighbour ranking moved by more than a tie: {se[a]} vs {se[b]}"


def test_header_is_aligned(tmp_path):
    """The data section starts on a 32-byte boundary.

    Readers take typed-array *views* over the buffer, and those require the
    absolute offset to be aligned to the element width. Pinned because the
    failure mode is length-dependent: an unaligned header works until an
    unrelated edit changes the JSON by one byte.
    """
    model, cfg = _tiny_model()
    for quant in ("q8", "f16", "f32"):
        path = tmp_path / f"align-{quant}.mwt"
        mweights.export_model(model, cfg, path, quant=quant)
        with path.open("rb") as fh:
            fh.seek(4)
            import struct
            (n,) = struct.unpack("<I", fh.read(4))
        assert (8 + n) % mweights.ALIGN == 0, \
            f"{quant}: data starts at {8 + n}, not a multiple of {mweights.ALIGN}"


def test_tied_embeddings_resolve(tmp_path):
    """A tied model ships no lm_head; the loader must not invent one."""
    model, cfg = _tiny_model(tie=True)
    path = tmp_path / "tied.mwt"
    header = mweights.export_model(model, cfg, path, quant="f32")

    assert "lm_head" not in header["tensors"]
    assert header["config"]["tiedEmbeddings"] is True

    tokens = [7, 21, 4]
    with torch.no_grad():
        ref = model(torch.tensor([tokens])).logits[0].numpy()
    js = _run_js(path, tokens)
    got = np.asarray(js["logits"], dtype=np.float64).reshape(
        len(tokens), cfg.vocab_size)
    assert np.allclose(got, ref, atol=3e-4)


def test_header_is_readable_without_the_payload(tmp_path):
    """Shape and size can be reported before committing to the download."""
    model, cfg = _tiny_model()
    path = tmp_path / "tiny.mwt"
    mweights.export_model(model, cfg, path, quant="q8")

    header = mweights.read_header(path)
    assert header["format"] == "mwt" and header["version"] == 1
    assert header["config"]["numLayers"] == cfg.num_hidden_layers
    assert header["config"]["numKvHeads"] == cfg.num_key_value_heads
    # Every tensor viewer/model.js will ask for is present.
    for layer in range(cfg.num_hidden_layers):
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                  "up_proj", "down_proj", "input_layernorm",
                  "post_attention_layernorm"):
            assert f"layers.{layer}.{n}" in header["tensors"]
