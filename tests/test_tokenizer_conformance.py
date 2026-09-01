"""Tokenizer conformance: the browser encodes what the model was trained on.

`viewer/tokenizer.js` turns a typed prompt into ids so the page can capture
without a server. A near-miss here is the worst kind of bug in this project:
it does not crash, it silently changes what the model was asked, and every
state, basin and readout downstream is a picture of a different question.

So the JS is checked against the real tokenizer, over strings chosen to hit
the places byte-level BPE actually goes wrong: contractions (the case-folded
alternation JavaScript cannot express inline), digit runs (Qwen3 splits them
one at a time, unlike the GPT-2 pattern usually copied), leading and repeated
whitespace, newlines, non-Latin scripts, emoji outside the BMP, and
combining marks that NFC changes.

Marked `network` because it needs the published tokenizer; a local
`tokenizers`-built fixture would only prove the JS agrees with my own
assumptions, which is what this test exists to avoid.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_JS = ROOT / "viewer" / "tokenizer.js"

transformers = pytest.importorskip("transformers")

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(shutil.which("node") is None, reason="node not available"),
]

MODEL = "Qwen/Qwen3-0.6B"

CORPUS = [
    "The capital of France is",
    "The residual stream moves, turns, and settles",
    # Contractions: the (?i:'s|'t|...) branch, both cases.
    "it's Bob's, they're here, I'VE seen it, don'T",
    # Digits: Qwen3 splits every digit separately.
    "1 12 123 1234 in 2026, 3.14159 and 007",
    # Whitespace shapes: leading, doubled, trailing, tabs, newlines.
    "  leading", "double  space", "trailing   ", "\ttab\tseparated",
    "line one\nline two\n\n\nline three", "\n", "   \n  ",
    # Punctuation runs and symbol clusters.
    "!!! ??? ---> <<>> {}[]() ***", "a,b;c:d/e\\f|g",
    # Non-Latin and mixed scripts.
    "日本語のテキスト", "Привет, мир", "مرحبا بالعالم", "한국어 텍스트",
    "mixed 日本語 and English 123",
    # Astral-plane characters (surrogate pairs in JS).
    "emoji 🔮 and 🧠🌍 together", "👨‍👩‍👧‍👦 family",
    # NFC: decomposed vs composed must agree after normalisation.
    "café vs café", "Å vs Å",
    # Empty-ish and edge inputs.
    " ", "a",
    # Code-like text, which mixes every branch at once.
    "def f(x):\n    return x['k'] + 1  # comment",
    "hidden[l+1] = hidden[l] + attn[l] + mlp[l]",
]


@pytest.fixture(scope="module")
def tok():
    return transformers.AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def js_encode(tok, tmp_path_factory):
    """Ship the real vocabulary/merges to Node once, then encode many strings."""
    tj = json.loads(tok.backend_tokenizer.to_str())
    payload = {
        "vocab": tj["model"]["vocab"],
        "merges": tj["model"]["merges"],
        "addedTokens": [{"content": a["content"], "id": a["id"]}
                        for a in tj.get("added_tokens", [])],
    }
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    path.write_text(json.dumps(payload))

    script = """
    const fs = require("fs");
    const T = require(process.argv[1]);
    const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
    const texts = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
    const enc = T.create(spec);
    process.stdout.write(JSON.stringify({
      ids: texts.map(t => enc.encode(t)),
      decoded: texts.map(t => enc.decode(enc.encode(t))),
    }));
    """

    def run(texts):
        tp = path.parent / "texts.json"
        tp.write_text(json.dumps(texts))
        proc = subprocess.run(
            ["node", "-e", script, str(TOKENIZER_JS), str(path), str(tp)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f"node failed: {proc.stderr[-3000:]}")
        return json.loads(proc.stdout)

    return run


def test_encode_matches_reference(tok, js_encode):
    """Every corpus string encodes to exactly the reference's ids."""
    result = js_encode(CORPUS)
    failures = []
    for text, got in zip(CORPUS, result["ids"]):
        want = tok.encode(text, add_special_tokens=False)
        if got != want:
            failures.append(
                f"  {text!r}\n    js  = {got}\n    ref = {want}\n"
                f"    js pieces  = {[tok.convert_ids_to_tokens(i) for i in got]}\n"
                f"    ref pieces = {[tok.convert_ids_to_tokens(i) for i in want]}")
    assert not failures, "tokenization differs from the reference:\n" + "\n".join(failures)


def test_roundtrip_decodes_back(tok, js_encode):
    """encode -> decode returns the (NFC-normalised) input."""
    result = js_encode(CORPUS)
    for text, decoded in zip(CORPUS, result["decoded"]):
        import unicodedata
        assert decoded == unicodedata.normalize("NFC", text), \
            f"roundtrip changed {text!r} -> {decoded!r}"


def test_special_tokens_stay_whole(tok, js_encode):
    """Chat-template control tokens encode as one id, not as byte pieces."""
    texts = ["<|im_start|>user\nhi<|im_end|>", "<|endoftext|>"]
    result = js_encode(texts)
    for text, got in zip(texts, result["ids"]):
        want = tok.encode(text, add_special_tokens=False)
        assert got == want, f"{text!r}: {got} != {want}"


def test_longer_prose_matches(tok, js_encode):
    """A realistic paragraph, where a rank mistake would surface as drift."""
    text = (
        "Mottled visualizes and quantitatively summarizes representation-space "
        "behavior under declared analysis choices; it generates mechanistic "
        "hypotheses that require full-dimensional, controlled, and causally "
        "targeted validation. A basin shows states accumulating, not a circuit "
        "computing -- 1, 2, 3 and then 42."
    )
    got = js_encode([text])["ids"][0]
    want = tok.encode(text, add_special_tokens=False)
    assert got == want
