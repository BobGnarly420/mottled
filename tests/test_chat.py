"""Chat: the forward pass captured is the one the model was sent.

The explorer's chat mode captures, each turn, the conversation as the model
was sent it plus the reply it produced. The claim that makes the scene worth
reading is token identity: the ids captured are the ids the model's own chat
format produces. That is pinned here against two real instruct templates —
Qwen2.5's ChatML, which injects a default system prompt, and Mistral v0.3's,
which writes BOS itself while its tokenizer adds one too — copied into
`chat_templates.json` from the published tokenizer configs. Their vocabulary
here is tiny; what the tests pin is which ids the chat format produces and
the capture receives, not the words.
"""
import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import tiny  # noqa: E402
from pipeline import chat_input, chat_reply, run_pipeline  # noqa: E402

TEMPLATES = json.loads((Path(__file__).parent / "chat_templates.json").read_text())

MESSAGES = [{"role": "user", "content": "the capital of france is"},
            {"role": "assistant", "content": "paris"},
            {"role": "user", "content": "and germany"}]


def _tokenizer(name, add_eos=False):
    """A word-level tokenizer carrying a real model's chat template and
    special tokens, adding BOS itself where that model's tokenizer does (and
    EOS, as an `add_eos_token` config would, when asked)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    spec = TEMPLATES[name]
    vocab = {"<unk>": 0}
    for word in tiny.CORPUS_WORDS:
        vocab.setdefault(word, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(spec["special_tokens"])
    parts = (([spec["bos_token"]] if spec["add_bos_token"] else []) + ["$A"]
             + ([spec["eos_token"]] if add_eos else []))
    if parts != ["$A"]:
        tok.post_processor = processors.TemplateProcessing(
            single=" ".join(parts),
            special_tokens=[(t, tok.token_to_id(t)) for t in parts if t != "$A"])
    hf = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token=spec["bos_token"],
        eos_token=spec["eos_token"], pad_token=spec.get("pad_token") or "<unk>")
    hf.chat_template = spec["chat_template"]
    return hf


def _model(tok):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256)
    return transformers.LlamaForCausalLM(cfg).eval()


def _own(tok, messages):
    """The tokenizer's own chat tokenization, as a flat list of ids."""
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


@pytest.fixture
def seen_ids(monkeypatch):
    """The ids every capture actually ran on, as `_run` received them."""
    import capture

    seen, run = [], capture._run

    def spy(*args, **kwargs):
        seen.append(kwargs["input_ids"][0].tolist())
        return run(*args, **kwargs)

    monkeypatch.setattr(capture, "_run", spy)
    return seen


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_the_capture_runs_on_the_chat_format_s_own_ids(name, seen_ids):
    tok = _tokenizer(name)
    text, ids = chat_input(tok, MESSAGES)
    assert ids == _own(tok, MESSAGES)

    result = run_pipeline(tiny.config(generate_tokens=3), text, model=_model(tok),
                          tokenizer=tok, input_ids=ids)
    gen = result["traj"].meta["generation"]
    assert gen["prompt_tokens"] == len(ids)
    assert seen_ids[-1][:len(ids)] == ids
    assert result["prompt"] == text


def test_a_capture_without_decoding_runs_on_the_given_ids_too(seen_ids):
    tok = _tokenizer("mistral-7b-instruct-v0.3")
    text, ids = chat_input(tok, MESSAGES)
    result = run_pipeline(tiny.config(), text, model=_model(tok), tokenizer=tok,
                          input_ids=ids)
    assert seen_ids[-1] == ids and result["traj"].n_tokens == len(ids)


def test_the_same_text_with_other_ids_is_not_a_cache_hit(tmp_path):
    """A cached capture is keyed by its prompt text; two id sequences for
    one text are two different questions."""
    tok = _tokenizer("mistral-7b-instruct-v0.3")
    text, ids = chat_input(tok, MESSAGES)
    cfg = tiny.config(use_cache=True, cache_dir=str(tmp_path))
    model = _model(tok)
    one = run_pipeline(cfg, text, model=model, tokenizer=tok, input_ids=ids)
    two = run_pipeline(cfg, text, model=model, tokenizer=tok,
                       input_ids=[tok.bos_token_id] + ids)
    assert two["traj"].n_tokens == one["traj"].n_tokens + 1


def test_the_real_templates_are_the_ones_rendered():
    qwen, _ = chat_input(_tokenizer("qwen2.5-instruct"), MESSAGES)
    assert qwen.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba "
                           "Cloud. You are a helpful assistant.<|im_end|>\n")
    assert qwen.endswith("<|im_start|>assistant\n")
    mistral, _ = chat_input(_tokenizer("mistral-7b-instruct-v0.3"), MESSAGES)
    assert mistral == ("<s>[INST] the capital of france is[/INST] paris</s>"
                       "[INST] and germany[/INST]")


def test_tokenizing_the_rendered_text_again_would_change_the_question():
    """Why the ids are passed rather than re-derived from the text: Mistral's
    template writes BOS and its tokenizer adds another, and a tokenizer set
    to add EOS would end the prompt before the reply could begin."""
    tok = _tokenizer("mistral-7b-instruct-v0.3", add_eos=True)
    text, ids = chat_input(tok, MESSAGES)
    bos, eos = tok.bos_token_id, tok.eos_token_id
    again = tok(text)["input_ids"]
    assert again[:2] == [bos, bos] and again[-1] == eos
    assert ids == _own(tok, MESSAGES)
    assert ids[0] == bos and ids[1] != bos and ids[-1] != eos


def test_a_second_turn_runs_on_the_whole_conversation_as_the_template_renders_it(
        seen_ids):
    """Earlier replies are re-sent as the template renders their text, as any
    chat re-sends its history, so the second turn's ids are the template's
    rendering of the whole conversation."""
    tok = _tokenizer("qwen2.5-instruct")
    model, cfg = _model(tok), tiny.config(generate_tokens=3)
    first = [MESSAGES[0]]
    text, ids = chat_input(tok, first)
    reply = chat_reply(tok, run_pipeline(cfg, text, model=model, tokenizer=tok,
                                         input_ids=ids)["traj"])

    second = first + [{"role": "assistant", "content": reply}, MESSAGES[2]]
    text, ids = chat_input(tok, second)
    run_pipeline(cfg, text, model=model, tokenizer=tok, input_ids=ids)
    assert seen_ids[-1][:len(ids)] == _own(tok, second)


def test_a_model_without_a_chat_template_gets_a_plain_transcript():
    tok = tiny.tokenizer()
    messages = [{"role": "system", "content": "the model"}] + MESSAGES
    text, ids = chat_input(tok, messages)
    assert text == ("System: the model\nUser: the capital of france is\n"
                    "Assistant: paris\nUser: and germany\nAssistant:")
    assert ids == tok(text)["input_ids"]


@pytest.fixture
def chat_app(monkeypatch):
    """The explorer, offline: every model name loads the tiny Llama."""
    pytest.importorskip("streamlit.testing.v1")
    import capture
    import streamlit as st
    from tests.apptest import app_test

    monkeypatch.setattr(capture, "load_model",
                        lambda name, **kw: (tiny.model(n_layers=2), tiny.tokenizer()))
    st.cache_resource.clear()       # a model cached by another test skips the patch
    at = app_test(default_timeout=120)
    at.run()
    at.toggle(key="chat_mode").set_value(True).run()
    yield at
    st.cache_resource.clear()


def test_chat_mode_puts_the_conversation_beside_its_scene(chat_app):
    """Chat on the left, the scene of that conversation on the right: the
    capture is the transcript plus the reply the decode produced."""
    at = chat_app
    assert not at.exception
    at.chat_input(key="chat_input").set_value("the capital of france is").run()
    assert not at.exception

    chat, scene = at.columns[0], at.columns[1]
    assert [m.name for m in chat.chat_message] == ["user", "assistant"]
    assert len(scene.get("plotly_chart")) >= 1

    result = at.session_state["result"]
    gen = result["traj"].meta["generation"]
    assert result["prompt"] == "User: the capital of france is\nAssistant:"
    assert gen["new_tokens"] >= 1
    assert result["traj"].n_tokens == gen["prompt_tokens"] + gen["new_tokens"]
    assert at.session_state["chat"][-1] == {
        "role": "assistant", "content": chat_reply(tiny.tokenizer(), result["traj"])}


def test_a_second_turn_captures_the_whole_conversation(chat_app):
    at = chat_app
    at.chat_input(key="chat_input").set_value("the capital of france is").run()
    reply = at.session_state["chat"][-1]["content"]
    at.chat_input(key="chat_input").set_value("and germany").run()
    assert not at.exception
    assert at.session_state["result"]["prompt"] == (
        f"User: the capital of france is\nAssistant: {reply}\n"
        "User: and germany\nAssistant:")
    assert len(at.session_state["chat"]) == 4


def test_clearing_the_conversation_clears_its_scene(chat_app):
    at = chat_app
    at.chat_input(key="chat_input").set_value("the capital of france is").run()
    at.button(key="chat_clear").click().run()
    assert not at.exception
    assert at.session_state["chat"] == []
    assert any("Say something" in i.value for i in at.info)


def test_changing_the_model_starts_a_new_conversation(chat_app):
    """The replies so far were another model's; carrying them over would
    send the new model a history it did not write."""
    from config import MODEL_CHOICES

    at = chat_app
    at.chat_input(key="chat_input").set_value("the capital of france is").run()
    at.selectbox(key="model").select(MODEL_CHOICES[1]).run()
    assert not at.exception
    assert at.session_state["chat"] == []
    assert any("started over" in c.value for c in at.caption)


def test_a_failed_capture_leaves_the_conversation_as_it_was(chat_app, monkeypatch):
    import pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("capture failed")

    at = chat_app
    monkeypatch.setattr(pipeline, "run_pipeline", fail)
    at.chat_input(key="chat_input").set_value("the capital of france is").run()
    assert at.exception
    assert at.session_state["chat"] == []


def test_an_empty_reply_says_so(chat_app, monkeypatch):
    import pipeline

    monkeypatch.setattr(pipeline, "chat_reply", lambda tokenizer, traj: "")
    at = chat_app
    at.chat_input(key="chat_input").set_value("the capital of france is").run()
    assert not at.exception
    assert "no text" in at.chat_message[1].markdown[0].value


def test_the_reply_is_decoded_from_the_ids_the_decode_chose():
    """Joined display strings are not text (they are cleaned for the scene);
    the reply comes from the ids, with special tokens dropped."""
    tok = tiny.tokenizer()
    ids = [tok.convert_tokens_to_ids(w) for w in ("paris", "is")] + [tok.unk_token_id]
    traj = types.SimpleNamespace(
        meta={"generation": {"steps": [{"id": i} for i in ids]}})
    assert chat_reply(tok, traj) == "paris is"
