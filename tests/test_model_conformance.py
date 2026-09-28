"""Model conformance: the in-browser forward pass is the model's forward pass.

`viewer/model.js` reimplements a Llama/Qwen3-family forward pass so the browser
can produce a residual stream without a server. A reimplementation is only
worth having if it is *the same computation*, so this runs a real (tiny,
locally-built) HuggingFace model and the JS port over identical weights and
compares: per-layer residual states, the exact attn/MLP decomposition, and the
final logits.

Built locally rather than downloaded — the point is the arithmetic, and a
2-layer model exercises every path a 36-layer one does (GQA grouping, rotary
pairing, SwiGLU, tied vs untied heads) while running offline in CI.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_JS = ROOT / "viewer" / "model.js"

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")


def _tiny_model(n_kv_heads=2, tie=False, seed=0):
    """A small Llama with grouped-query attention (4 query heads, 2 kv)."""
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=48, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=n_kv_heads, max_position_embeddings=64,
        rms_norm_eps=1e-6, tie_word_embeddings=tie,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    # Random init gives near-identical norm weights; perturb so a bug in the
    # norm path cannot hide behind weights that are all ~1.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                p.add_(torch.randn_like(p) * 0.1)
    return model, cfg


def _tiny_qwen2(seed=0):
    """Qwen2 puts a bias on q/k/v. Llama and Qwen3 do not.

    This is the architecture that catches a forward pass which ignores biases:
    it runs, and every state is wrong by exactly the bias.
    """
    torch.manual_seed(seed)
    cfg = transformers.Qwen2Config(
        vocab_size=48, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p_ in model.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                p_.add_(torch.randn_like(p_) * 0.1)
            # A zero bias would let a forward pass that ignores biases pass.
            if name.endswith(".bias"):
                p_.add_(torch.randn_like(p_) * 0.5)
    return model, cfg


def _export_qwen2(model, cfg):
    sd = model.state_dict()
    w = {"embed_tokens": sd["model.embed_tokens.weight"],
         "norm": sd["model.norm.weight"],
         "lm_head": sd["lm_head.weight"]}
    for l in range(cfg.num_hidden_layers):
        src, dst = f"model.layers.{l}.", f"layers.{l}."
        for a, b in [("self_attn.q_proj", "q_proj"), ("self_attn.k_proj", "k_proj"),
                     ("self_attn.v_proj", "v_proj"), ("self_attn.o_proj", "o_proj"),
                     ("mlp.gate_proj", "gate_proj"), ("mlp.up_proj", "up_proj"),
                     ("mlp.down_proj", "down_proj"),
                     ("input_layernorm", "input_layernorm"),
                     ("post_attention_layernorm", "post_attention_layernorm")]:
            w[dst + b] = sd[src + a + ".weight"]
        for a, b in [("self_attn.q_proj", "q_bias"), ("self_attn.k_proj", "k_bias"),
                     ("self_attn.v_proj", "v_bias")]:
            key = src + a + ".bias"
            if key in sd:
                w[dst + b] = sd[key]
    return {k: v.detach().float().numpy().ravel().tolist() for k, v in w.items()}


def test_attention_bias_is_applied():
    """Qwen2-style attention biases must reach the states.

    Without addBias this test fails on every layer -- which is the point:
    ignoring a bias that exists is silent, and only a reference comparison
    catches it.
    """
    model, cfg = _tiny_qwen2()
    tokens = [3, 17, 42, 8]

    with torch.no_grad():
        ref = model(torch.tensor([tokens]), output_hidden_states=True)

    js = _run_js({"weights": _export_qwen2(model, cfg), "tokens": tokens,
                  "config": _js_config(cfg)})

    T, D, L = len(tokens), cfg.hidden_size, cfg.num_hidden_layers
    got = np.asarray(js["hidden"], dtype=np.float64).reshape(L + 1, T, D)
    for layer in range(L):
        want = ref.hidden_states[layer][0].numpy()
        assert np.allclose(got[layer], want, atol=2e-4), (
            f"layer {layer} max |Δ| = {np.abs(got[layer] - want).max():.3e} "
            "(attention bias not applied?)")


def _export(model, cfg):
    """HF parameter names -> the flat names viewer/model.js asks for."""
    sd = model.state_dict()
    w = {
        "embed_tokens": sd["model.embed_tokens.weight"],
        "norm": sd["model.norm.weight"],
    }
    if not cfg.tie_word_embeddings:
        w["lm_head"] = sd["lm_head.weight"]
    for l in range(cfg.num_hidden_layers):
        src = f"model.layers.{l}."
        dst = f"layers.{l}."
        for a, b in [("self_attn.q_proj", "q_proj"), ("self_attn.k_proj", "k_proj"),
                     ("self_attn.v_proj", "v_proj"), ("self_attn.o_proj", "o_proj"),
                     ("mlp.gate_proj", "gate_proj"), ("mlp.up_proj", "up_proj"),
                     ("mlp.down_proj", "down_proj"),
                     ("input_layernorm", "input_layernorm"),
                     ("post_attention_layernorm", "post_attention_layernorm")]:
            w[dst + b] = sd[src + a + ".weight"]

    return {k: v.detach().float().numpy().ravel().tolist() for k, v in w.items()}


JS_FORWARD = """
const M = require(process.argv[1]);
let raw = "";
process.stdin.on("data", d => raw += d).on("end", () => {
  const p = JSON.parse(raw);
  const store = new Map(Object.entries(p.weights).map(
    ([k, v]) => [k, Float32Array.from(v)]));
  const weights = { get: (n, optional) => {
    if (store.has(n)) return store.get(n);
    if (optional) return null;
    throw new Error("missing weight " + n);
  }};
  M.forward(weights, p.tokens, p.config, { captureComponents: true }).then(out => {
  process.stdout.write(JSON.stringify({
    hidden: Array.from(out.hidden),
    logits: Array.from(out.logits),
    attn: Array.from(out.components.attn),
    mlp: Array.from(out.components.mlp),
    nLayers: out.nLayers,
  }));
  }).catch(e => { console.error(e); process.exit(1); });
});
"""


def _run_js(payload):
    proc = subprocess.run(["node", "-e", JS_FORWARD, str(MODEL_JS)],
                          input=json.dumps(payload), capture_output=True,
                          text=True)
    if proc.returncode != 0:
        raise AssertionError(f"node failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def _js_config(cfg):
    return {
        "hiddenSize": cfg.hidden_size,
        "numLayers": cfg.num_hidden_layers,
        "numHeads": cfg.num_attention_heads,
        "numKvHeads": cfg.num_key_value_heads,
        "headDim": cfg.hidden_size // cfg.num_attention_heads,
        "intermediateSize": cfg.intermediate_size,
        "vocabSize": cfg.vocab_size,
        "rmsNormEps": cfg.rms_norm_eps,
        "ropeTheta": float(getattr(cfg, "rope_theta", 10000.0)),
    }


@pytest.mark.parametrize("n_kv_heads,tie", [(2, False), (4, False), (2, True)])
def test_hidden_states_match_huggingface(n_kv_heads, tie):
    """Every residual state the browser records is the state HF computes."""
    model, cfg = _tiny_model(n_kv_heads=n_kv_heads, tie=tie)
    tokens = [3, 17, 42, 8, 25]

    with torch.no_grad():
        ref = model(torch.tensor([tokens]), output_hidden_states=True)

    js = _run_js({"weights": _export(model, cfg), "tokens": tokens,
                  "config": _js_config(cfg)})

    T, D, L = len(tokens), cfg.hidden_size, cfg.num_hidden_layers
    got = np.asarray(js["hidden"], dtype=np.float64).reshape(L + 1, T, D)

    # HF applies the final norm to hidden_states[-1] before returning it, so
    # only 0..L-1 are directly comparable as raw residual states; the final
    # state is covered by the logits check below (which runs through the norm).
    for layer in range(L):
        want = ref.hidden_states[layer][0].numpy()
        assert np.allclose(got[layer], want, atol=2e-4), (
            f"layer {layer} max |Δ| = {np.abs(got[layer] - want).max():.3e}")


@pytest.mark.parametrize("n_kv_heads", [2, 4])
def test_logits_match_huggingface(n_kv_heads):
    """The readout agrees, which exercises the final norm and the head."""
    model, cfg = _tiny_model(n_kv_heads=n_kv_heads)
    tokens = [5, 11, 30, 2]

    with torch.no_grad():
        ref = model(torch.tensor([tokens])).logits[0].numpy()

    js = _run_js({"weights": _export(model, cfg), "tokens": tokens,
                  "config": _js_config(cfg)})
    got = np.asarray(js["logits"], dtype=np.float64).reshape(len(tokens), cfg.vocab_size)

    assert np.allclose(got, ref, atol=3e-4), \
        f"logits max |Δ| = {np.abs(got - ref).max():.3e}"
    # The argmax is what the viewer actually displays; a tolerance that passes
    # while the top-1 differs would be a false pass.
    assert (got.argmax(-1) == ref.argmax(-1)).all()


def test_residual_decomposition_is_exact():
    """hidden[l+1] == hidden[l] + attn[l] + mlp[l], the invariant the README pins."""
    model, cfg = _tiny_model()
    tokens = [7, 21, 4]

    js = _run_js({"weights": _export(model, cfg), "tokens": tokens,
                  "config": _js_config(cfg)})

    T, D, L = len(tokens), cfg.hidden_size, cfg.num_hidden_layers
    hidden = np.asarray(js["hidden"], dtype=np.float64).reshape(L + 1, T, D)
    attn = np.asarray(js["attn"], dtype=np.float64).reshape(L, T, D)
    mlp = np.asarray(js["mlp"], dtype=np.float64).reshape(L, T, D)

    recon = hidden[:-1] + attn + mlp
    assert np.allclose(recon, hidden[1:], atol=1e-6), \
        f"decomposition max |Δ| = {np.abs(recon - hidden[1:]).max():.3e}"


def test_causality_holds():
    """A later token cannot change an earlier token's states."""
    model, cfg = _tiny_model()
    base = [9, 14, 33, 6]
    changed = base[:-1] + [41]

    cfg_js = _js_config(cfg)
    weights = _export(model, cfg)
    a = _run_js({"weights": weights, "tokens": base, "config": cfg_js})
    b = _run_js({"weights": weights, "tokens": changed, "config": cfg_js})

    T, D, L = len(base), cfg.hidden_size, cfg.num_hidden_layers
    ha = np.asarray(a["hidden"], dtype=np.float64).reshape(L + 1, T, D)
    hb = np.asarray(b["hidden"], dtype=np.float64).reshape(L + 1, T, D)

    assert np.allclose(ha[:, :-1], hb[:, :-1], atol=1e-9)
    assert not np.allclose(ha[:, -1], hb[:, -1])
