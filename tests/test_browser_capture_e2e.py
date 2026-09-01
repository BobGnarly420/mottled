"""End to end: a typed prompt becomes a scene, entirely in browser code.

Every other conformance test in this suite pins one stage. This one runs the
whole chain the live viewer runs — export weights, load them in JS, tokenize a
prompt, forward pass, logit lens at every layer, joint projection, density,
terrain, drape — and checks the object that comes out is one the renderer can
actually draw, with the states the model really computed.

The renderer is deliberately not stubbed out here: the assertions are the
invariants `viewer/mtj.js` enforces on a loaded scene file, so a browser
capture and a `mottled export` capture are interchangeable by construction
rather than by hope.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import mweights

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "viewer"

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")


def _tiny_model_and_tokenizer(tmp_path):
    """A small Llama plus a real (tiny) byte-level BPE tokenizer.

    The tokenizer is trained here rather than downloaded so the test stays
    offline; it is a genuine BPE with merges, which is what the encoder needs
    to exercise.
    """
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = [
        "The capital of France is Paris and the capital of Germany is Berlin.",
        "The residual stream moves, turns, and settles into the ground.",
        "hidden states travel through layers and pile up in a basin.",
    ] * 20
    tok.train_from_iterator(corpus, trainer)

    hf = transformers.PreTrainedTokenizerFast(tokenizer_object=tok)

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=tok.get_vocab_size(), hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, rms_norm_eps=1e-6, tie_word_embeddings=False,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()

    path = tmp_path / "tiny.mwt"
    mweights.export_model(model, cfg, path, quant="f32", tokenizer=hf)
    return model, cfg, hf, path


JS_CAPTURE = """
const fs = require("fs");
const path = require("path");
const dir = process.argv[1];
const W = require(path.join(dir, "weights.js"));
const C = require(path.join(dir, "capture.js"));

const buf = fs.readFileSync(process.argv[2]);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const weights = W.open(ab);
const prompts = JSON.parse(process.argv[3]);

C.capture(weights, prompts, { gridSize: 24 }).then(scene => {
  process.stdout.write(JSON.stringify({
    runs: scene.runs.map(r => ({
      label: r.label, prompt: r.prompt, tokens: r.tokens,
      pointsShape: r.points.shape,
      points: Array.from(r.points.data),
      entropyShape: r.entropy.shape,
      entropy: Array.from(r.entropy.data),
      topkLayers: r.topk.length,
      topkFirst: r.topk[0][0],
      model: r.model,
    })),
    terrain: {
      xShape: scene.terrain.x.shape, yShape: scene.terrain.y.shape,
      zShape: scene.terrain.z.shape,
      z: Array.from(scene.terrain.z.data),
    },
    meta: scene.meta,
  }));
}).catch(e => { console.error(e.stack || String(e)); process.exit(1); });
"""


def _capture(path, prompts):
    proc = subprocess.run(
        ["node", "-e", JS_CAPTURE, str(VIEWER), str(path), json.dumps(prompts)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"node failed: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_prompt_becomes_a_renderable_scene(tmp_path):
    """One prompt in, one scene out, in the shape the renderer requires."""
    model, cfg, hf, path = _tiny_model_and_tokenizer(tmp_path)
    prompt = "The capital of France is"

    scene = _capture(path, [prompt])
    assert len(scene["runs"]) == 1
    run = scene["runs"][0]

    n_tokens = len(hf.encode(prompt))
    n_layers = cfg.num_hidden_layers + 1

    # The invariants viewer/mtj.js enforces on a loaded scene.
    assert run["pointsShape"] == [n_tokens, n_layers, 3], run["pointsShape"]
    assert len(run["points"]) == n_tokens * n_layers * 3
    assert np.isfinite(run["points"]).all()
    assert scene["terrain"]["zShape"] == [24, 24]
    assert scene["terrain"]["xShape"] == [24]
    assert (scene["terrain"]["zShape"][0] == scene["terrain"]["yShape"][0]
            and scene["terrain"]["zShape"][1] == scene["terrain"]["xShape"][0])
    assert run["entropyShape"] == [n_layers, n_tokens]
    assert run["topkLayers"] == n_layers
    assert run["label"] == "A" and run["prompt"] == prompt


def test_tokens_match_the_reference_tokenizer(tmp_path):
    """The prompt the browser captured is the prompt that was typed."""
    model, cfg, hf, path = _tiny_model_and_tokenizer(tmp_path)
    prompt = "The residual stream moves, turns, and settles"

    run = _capture(path, [prompt])["runs"][0]
    want = [hf.decode([i]) for i in hf.encode(prompt)]
    assert run["tokens"] == want, f"{run['tokens']} != {want}"


def test_captured_states_are_the_models_states(tmp_path):
    """The trajectory drawn is the model's trajectory.

    The scene only carries *projected* points, so this checks the readout the
    inspector shows instead: the top-1 at the final layer must be the token
    the model actually predicts.
    """
    model, cfg, hf, path = _tiny_model_and_tokenizer(tmp_path)
    prompt = "The capital of France is"
    ids = hf.encode(prompt)

    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0, -1]
    want = hf.decode([int(ref.argmax())])

    run = _capture(path, [prompt])["runs"][0]
    # topk[layer][token] -> [[text, p], ...]; the final layer's final token.
    scene = _capture(path, [prompt])
    got = run["topkFirst"]
    assert isinstance(got, list) and len(got) >= 1

    # Re-run asking for the final layer explicitly.
    script = JS_CAPTURE.replace(
        "topkFirst: r.topk[0][0],",
        "topkFirst: r.topk[r.topk.length - 1][r.tokens.length - 1],")
    proc = subprocess.run(
        ["node", "-e", script, str(VIEWER), str(path), json.dumps([prompt])],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    final = json.loads(proc.stdout)["runs"][0]["topkFirst"]

    assert final[0][0] == want, \
        f"browser top-1 {final[0][0]!r} != model's {want!r}"


def test_multiple_prompts_share_one_terrain(tmp_path):
    """A/B runs land in one space, as pipeline.run_scene does in Python."""
    model, cfg, hf, path = _tiny_model_and_tokenizer(tmp_path)
    prompts = ["The capital of France is", "The capital of Germany is"]

    scene = _capture(path, prompts)
    assert [r["label"] for r in scene["runs"]] == ["A", "B"]

    # One terrain for the scene, and both runs' points lie on it.
    z = np.asarray(scene["terrain"]["z"])
    assert np.isfinite(z).all()
    for r in scene["runs"]:
        pts = np.asarray(r["points"]).reshape(-1, 3)
        assert np.isfinite(pts).all()
        # Draped heights sit within the terrain's range plus the lift.
        assert pts[:, 2].min() >= z.min() - 1e-3
        assert pts[:, 2].max() <= z.max() + 0.05 + 1e-3

    # A sharper claim than "the runs differ": the prompts share the prefix
    # "The capital of", and attention is causal, so those states are the same
    # computation in both runs. In one shared projection they must therefore
    # land on exactly the same points -- and the states after the prompts
    # diverge must not. A projection fitted per-run, or a transpose bug in the
    # reshape, would break the first half; a collapsed scene the second.
    ra, rb = scene["runs"]
    shared = 0
    while (shared < min(len(ra["tokens"]), len(rb["tokens"]))
           and ra["tokens"][shared] == rb["tokens"][shared]):
        shared += 1
    assert shared >= 3, f"expected a shared prefix, got {ra['tokens']} vs {rb['tokens']}"

    la, lb = ra["pointsShape"][1], rb["pointsShape"][1]
    assert la == lb
    a = np.asarray(ra["points"]).reshape(ra["pointsShape"])
    b = np.asarray(rb["points"]).reshape(rb["pointsShape"])

    assert np.allclose(a[:shared], b[:shared], atol=1e-5), (
        "shared-prefix states differ between runs; causality or the shared "
        "projection is broken")
    assert not np.allclose(a[shared], b[shared], atol=1e-5), \
        "the diverging token produced identical geometry in both runs"


def test_empty_prompt_is_refused(tmp_path):
    """An empty prompt fails with a reason, not an unhandled exception."""
    model, cfg, hf, path = _tiny_model_and_tokenizer(tmp_path)
    proc = subprocess.run(
        ["node", "-e", JS_CAPTURE, str(VIEWER), str(path), json.dumps([""])],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "empty after tokenizing" in proc.stderr
