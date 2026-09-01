"""Generation-axis capture: decode records, determinism, causal exactness."""

import zlib

import numpy as np
import pytest

from capture import generate_and_capture
import tiny as synthetic

PROMPT = "the capital of france is"


# ---------------------------------------------------------------- synthetic

def test_tiny_greedy_decode_shapes_and_meta():
    traj = synthetic.generate_and_capture(PROMPT, max_new_tokens=4)
    n_prompt = len(PROMPT.split())
    assert traj.n_tokens == n_prompt + 4
    gen = traj.meta["generation"]
    assert gen["prompt_tokens"] == n_prompt
    assert gen["new_tokens"] == 4
    assert gen["mode"] == "greedy"
    assert [s["token"] for s in gen["steps"]] == traj.tokens[n_prompt:]
    for s in gen["steps"]:
        assert 0.0 < s["p"] <= 1.0
        assert s["entropy"] >= 0.0
    assert traj.meta["prompt"] == PROMPT
    traj.validate()


def test_tiny_greedy_is_deterministic():
    a = synthetic.generate_and_capture(PROMPT, max_new_tokens=3)
    b = synthetic.generate_and_capture(PROMPT, max_new_tokens=3)
    assert a.tokens == b.tokens
    np.testing.assert_array_equal(a.hidden, b.hidden)


def test_tiny_sampling_reproducible_per_seed():
    a = synthetic.generate_and_capture(PROMPT, max_new_tokens=5,
                                       temperature=2.0, seed=7)
    b = synthetic.generate_and_capture(PROMPT, max_new_tokens=5,
                                       temperature=2.0, seed=7)
    assert a.tokens == b.tokens
    gen = a.meta["generation"]
    assert gen["mode"] == "sample"
    assert gen["temperature"] == 2.0
    assert gen["seed"] == 7


def test_tiny_generated_capture_flags_pass_through():
    traj = synthetic.generate_and_capture(PROMPT, max_new_tokens=2,
                                capture_components=True, capture_attention=True)
    assert traj.components is not None
    assert traj.attention is not None
    traj.validate()


# --------------------------------------------------------------- transformers

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from capture import capture  # noqa: E402

VOCAB_SIZE = 128


class DummyTokenizer:
    """Deterministic word-level tokenizer (mirrors tests/test_capture.py)."""

    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._names = {}

    def _id(self, word):
        i = zlib.crc32(word.encode()) % (self.vocab_size - 1) + 1
        self._names[i] = word
        return i

    def __call__(self, text, return_tensors="pt"):
        ids = [self._id(w) for w in text.split()]
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids):
        return [self._names.get(int(i), f"<{int(i)}>") for i in ids]


@pytest.fixture(scope="module")
def tiny_llama():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def test_generation_extends_sequence_and_records_steps(tiny_llama):
    tok = DummyTokenizer()
    traj = generate_and_capture(tiny_llama, PROMPT, max_new_tokens=4,
                                tokenizer=tok)
    n_prompt = len(PROMPT.split())
    gen = traj.meta["generation"]
    assert gen["prompt_tokens"] == n_prompt
    assert traj.n_tokens == n_prompt + gen["new_tokens"]
    assert 1 <= gen["new_tokens"] <= 4  # may stop early at EOS
    assert [s["token"] for s in gen["steps"]] == traj.tokens[n_prompt:]
    traj.validate()


def test_generation_greedy_is_deterministic(tiny_llama):
    tok = DummyTokenizer()
    a = generate_and_capture(tiny_llama, PROMPT, max_new_tokens=3, tokenizer=tok)
    b = generate_and_capture(tiny_llama, PROMPT, max_new_tokens=3, tokenizer=tok)
    assert a.tokens == b.tokens
    np.testing.assert_allclose(a.hidden, b.hidden)


def test_generation_sampling_reproducible_per_seed(tiny_llama):
    tok = DummyTokenizer()
    a = generate_and_capture(tiny_llama, PROMPT, max_new_tokens=4,
                             tokenizer=tok, temperature=1.5, seed=11)
    b = generate_and_capture(tiny_llama, PROMPT, max_new_tokens=4,
                             tokenizer=tok, temperature=1.5, seed=11)
    assert a.tokens == b.tokens
    assert a.meta["generation"] == b.meta["generation"]


def test_generation_capture_is_causally_exact(tiny_llama):
    """The claim the whole design rests on: one pass over the completed
    sequence reproduces the prompt-prefix states of a plain capture."""
    tok = DummyTokenizer()
    full = generate_and_capture(tiny_llama, PROMPT, max_new_tokens=3,
                                tokenizer=tok)
    base = capture(tiny_llama, PROMPT, tokenizer=tok)
    n_prompt = base.n_tokens
    np.testing.assert_allclose(full.hidden[:, :n_prompt], base.hidden,
                               rtol=1e-4, atol=1e-5)


def test_generation_decode_matches_stepwise_forward(tiny_llama):
    """Each recorded step is the argmax of the model's own next-token
    distribution at that prefix."""
    tok = DummyTokenizer()
    traj = generate_and_capture(tiny_llama, PROMPT, max_new_tokens=3,
                                tokenizer=tok)
    ids = [int(i) for i in tok(PROMPT)["input_ids"][0]]
    with torch.no_grad():
        for step in traj.meta["generation"]["steps"]:
            logits = tiny_llama(torch.tensor([ids])).logits[0, -1]
            assert int(torch.argmax(logits)) == step["id"]
            ids.append(step["id"])


# ------------------------------------------------------------ pipeline wiring

def test_pipeline_generates_when_configured():
    from config import MarbleConfig
    from ui import run_pipeline

    cfg = MarbleConfig(model="tiny", use_cache=False, generate_tokens=3,
                       density_bootstrap=0)
    traj = run_pipeline(cfg, PROMPT, **synthetic.mt())["traj"]
    assert traj.n_tokens == len(PROMPT.split()) + 3
    assert traj.meta["generation"]["new_tokens"] == 3


def test_render_marks_generated_tokens():
    from config import MarbleConfig
    from ui import render, run_pipeline

    cfg = MarbleConfig(model="tiny", use_cache=False, generate_tokens=2,
                       density_bootstrap=0)
    r = run_pipeline(cfg, PROMPT, **synthetic.mt())
    fig = render(r["traj"], r["mesh"], r["trajectories"], r["fine_paths"])
    plus = [t.name for t in fig.data if t.name and t.name.startswith("+")]
    assert len(plus) == 2  # one per generated token


def test_scene_mtj_carries_generation(tmp_path):
    import statefile
    from config import MarbleConfig
    from ui import run_scene

    cfg = MarbleConfig(model="tiny", use_cache=False, generate_tokens=2,
                       density_bootstrap=0)
    path = tmp_path / "scene.mtj"
    statefile.save_scene(run_scene(cfg, [PROMPT], **synthetic.mt()), path)
    run = statefile.load_scene(path)["runs"][0]
    gen = run["generation"]
    assert gen["prompt_tokens"] == len(PROMPT.split())
    assert gen["new_tokens"] == 2
    assert [s["token"] for s in gen["steps"]] == run["tokens"][-2:]


def test_cli_export_generate(tmp_path):
    import statefile
    from cli import main

    out = tmp_path / "scene.mtj"
    assert main(["export", PROMPT, "-o", str(out), "--generate", "2"]) == 0
    assert statefile.load_scene(out)["runs"][0]["generation"]["new_tokens"] == 2
