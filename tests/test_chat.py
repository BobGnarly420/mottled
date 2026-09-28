"""Chat: the conversation a reply came from is the prompt that was captured.

The explorer's chat mode re-captures the whole conversation plus the reply
on every turn. What matters is that the tokens captured are the tokens the
model's chat format means, and that the reply shown is what the decode chose.
"""
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import tiny  # noqa: E402
from pipeline import chat_prompt, chat_reply  # noqa: E402

MESSAGES = [{"role": "user", "content": "the capital of france is"},
            {"role": "assistant", "content": "paris"},
            {"role": "user", "content": "and germany"}]


def _bos_tokenizer():
    """A tokenizer that adds BOS itself, with a template that also writes it —
    the Gemma/Llama shape, where rendering then re-tokenizing doubles it."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"<unk>": 0, "<s>": 1}
    for word in tiny.CORPUS_WORDS + ["user", "assistant", ":"]:
        vocab.setdefault(word, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)])
    hf = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", pad_token="<unk>")
    hf.chat_template = (
        "{{ bos_token }}{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    return hf


def test_the_captured_ids_are_what_the_chat_format_tokenizes_to():
    """Rendering the template and tokenizing that text again would put BOS in
    twice. The ids a chat capture sees must be exactly what the tokenizer's
    own chat tokenization produces."""
    tok = _bos_tokenizer()
    own = tok.apply_chat_template(MESSAGES, tokenize=True, add_generation_prompt=True)
    own = list(own["input_ids"] if hasattr(own, "keys") else own)

    got = tok(chat_prompt(tok, MESSAGES))["input_ids"]
    assert got == own
    assert got[:2] == [tok.bos_token_id, own[1]] and own[1] != tok.bos_token_id


def test_a_model_without_a_chat_template_gets_a_plain_transcript():
    assert chat_prompt(tiny.tokenizer(), MESSAGES) == (
        "User: the capital of france is\nAssistant: paris\n"
        "User: and germany\nAssistant:")


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


def test_the_reply_is_decoded_from_the_ids_the_decode_chose():
    """Joined display strings are not text (they are cleaned for the scene);
    the reply comes from the ids, with special tokens dropped."""
    tok = tiny.tokenizer()
    ids = [tok.convert_tokens_to_ids(w) for w in ("paris", "is")] + [tok.unk_token_id]
    traj = types.SimpleNamespace(
        meta={"generation": {"steps": [{"id": i} for i in ids]}})
    assert chat_reply(tok, traj) == "paris is"
