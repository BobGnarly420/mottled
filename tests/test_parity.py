"""The parity harness (parity.py, `mottled parity`).

The matrix itself downloads real weights and is not a CI job. What is pinned
here is the machinery a reader's verdict depends on: that the deviation
measure says zero when two stacks agree and says the truth when they don't,
that a skipped comparison is never counted as a passed one, and that the
HuggingFace comparison — the one that needs no optional library — really runs.
"""
import json

import numpy as np
import pytest

import parity as P
import tiny as synthetic

torch = pytest.importorskip("torch")

PROMPT = "the capital of france is"


@pytest.fixture(scope="module")
def loaded():
    return synthetic.model(n_layers=3), synthetic.tokenizer()


# ------------------------------------------------------------------ measures
def test_deviation_is_zero_for_identical_stacks():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 5, 8))
    assert P._deviation(a, a) == (0.0, 0.0)


def test_deviation_is_relative_to_each_state_not_the_whole_stack():
    """Residual norms grow with depth; one global scale would hide a
    late-layer disagreement behind an early-layer magnitude."""
    want = np.array([[[1.0, 0.0]], [[100.0, 0.0]]])       # norms 1 and 100
    got = want + np.array([[[0.0, 0.0]], [[0.0, 1.0]]])   # abs error 1, deep
    max_abs, max_rel = P._deviation(got, want)
    assert max_abs == pytest.approx(1.0)
    assert max_rel == pytest.approx(0.01)                  # 1/100, not 1/1


def test_deviation_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        P._deviation(np.zeros((2, 3, 4)), np.zeros((2, 3, 5)))


# ------------------------------------------------------------------- verdict
def test_a_skip_is_not_a_pass():
    skipped = P.Comparison(model="m", reference="nnsight", status="skipped",
                           detail="not installed")
    assert not skipped.passed
    # and a report of nothing but skips has not demonstrated anything
    report = P.Report(created="", prompt="", tolerance=P.TOLERANCE,
                      environment={}, comparisons=[skipped])
    assert not report.passed


def test_over_tolerance_fails_even_when_it_ran():
    over = P.Comparison(model="m", reference="transformers",
                        state_max_rel=P.TOLERANCE * 10)
    assert not over.passed
    assert P.Comparison(model="m", reference="transformers",
                        state_max_rel=0.0, logit_max_abs=0.0).passed


def test_an_error_row_does_not_lose_the_rest_of_the_table(monkeypatch):
    """One gated model or one missing wheel must not cost the reader the
    other rows — each cell is isolated."""
    def explode(*a, **k):
        raise RuntimeError("gated repo")

    monkeypatch.setattr(P, "against_transformer_lens", explode)
    cell = P._cell("some/model", "transformer_lens", PROMPT)
    assert cell.status == "error"
    assert "gated repo" in cell.detail


# --------------------------------------------------------------- the real run
def test_capture_matches_huggingfaces_own_hidden_states(loaded):
    """The claim the whole harness exists to support, on a model that needs
    no network: Mottled's residual stream is the one the model computed."""
    model, tokenizer = loaded
    result = P.against_transformers(model, tokenizer, PROMPT, name="tiny")

    assert result.status == "ok"
    assert result.n_layers == 4 and result.n_tokens == 5
    assert result.state_max_rel < P.TOLERANCE
    assert result.passed


def test_the_deepest_lens_reproduces_the_models_own_logits(loaded):
    """Not circular: if the lens at the last layer disagrees with what the
    model actually predicts, every shallower readout is measuring something
    other than what the explorer says it is."""
    model, tokenizer = loaded
    result = P.against_transformers(model, tokenizer, PROMPT)
    assert result.logit_max_abs < P.TOLERANCE
    assert result.top1_agreement == 1.0


def test_transformer_lens_skips_cleanly_when_absent():
    if _installed("transformer_lens"):
        pytest.skip("transformer-lens is installed; the skip path is moot here")
    out = P.against_transformer_lens("gpt2")
    assert out.status == "skipped" and "transformer-lens" in out.detail
    assert not out.passed          # and it never touched the network to say so


def test_nnsight_states_are_the_states_capture_reads(loaded):
    """Verified against a locally-built model, so it needs no hub — but it
    does need nnsight, which is an extra rather than a CI dependency."""
    pytest.importorskip("nnsight")
    model, tokenizer = loaded
    native = synthetic.capture(PROMPT, n_layers=4)

    traj = P.trace_with_nnsight(model, tokenizer, PROMPT, tokens=native.tokens)
    assert traj.hidden.shape == native.hidden.shape
    assert traj.meta["backend"] == "nnsight"
    # layer 0 is block 0's *input*, which is what HookCapture records; the
    # embedding module's output would differ on GPT-2, which adds positions
    assert np.abs(traj.hidden - native.hidden).max() == 0.0


# -------------------------------------------------------------------- report
def _report() -> P.Report:
    return P.Report(
        created="2026-09-16T00:00:00Z", prompt=PROMPT, tolerance=P.TOLERANCE,
        environment={"python": "3.11.0", "platform": "test"},
        comparisons=[
            P.Comparison(model="gpt2", reference="transformers", n_layers=13,
                         state_max_abs=1e-7, state_max_rel=1e-9,
                         logit_max_abs=1e-6, top1_agreement=1.0),
            P.Comparison(model="gpt2", reference="nnsight", status="skipped",
                         detail="nnsight is not installed"),
            P.Comparison(model="bad", reference="transformers",
                         state_max_rel=0.5),
        ])


def test_report_json_is_machine_readable():
    parsed = json.loads(_report().to_json())
    assert parsed["tolerance"] == P.TOLERANCE
    assert [c["reference"] for c in parsed["comparisons"]] == [
        "transformers", "nnsight", "transformers"]
    assert parsed["comparisons"][1]["status"] == "skipped"


def test_report_table_shows_numbers_and_names_the_failure():
    table = P.format_report(_report())
    assert "| `gpt2` | transformers |" in table
    assert "1.00e-07" in table and "100.0%" in table
    assert "skipped — nnsight is not installed" in table
    assert "**OVER TOLERANCE**" in table
    assert "A skipped row is not a passed row." in table


def test_cli_exits_nonzero_when_a_comparison_is_over_tolerance(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    import cli

    monkeypatch.setattr(P, "run", lambda *a, **k: _report())
    out, md = tmp_path / "parity.json", tmp_path / "parity.md"
    assert cli.main(["parity", "-o", str(out), "--markdown", str(md)]) == 1

    assert "OVER TOLERANCE" in capsys.readouterr().out
    assert json.loads(out.read_text())["prompt"] == PROMPT
    assert md.read_text().startswith("# Mottled parity report")


def _installed(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None
