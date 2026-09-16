"""`mottled smoke` — the check that runs where the test suite cannot.

These pin the checker's own behaviour: that it passes on a working tree, that
each check fails for the reason it exists rather than by accident, and that a
missing optional backend is reported without failing the run — a core install
is a supported install, not a broken one.
"""
import smoke


def test_a_working_tree_passes():
    checks = smoke.run_checks()
    failed = [c for c in checks if c.required and not c.ok]
    assert not failed, [f"{c.name}: {c.detail}" for c in failed]
    assert smoke.main() == 0


def test_the_flat_api_check_names_what_went_missing(monkeypatch):
    """Dropping a documented name is a smoke failure, not a surprise in
    somebody's notebook."""
    import ui

    monkeypatch.delattr(ui, "run_pipeline")
    check = smoke._flat_api()
    assert not check.ok
    assert "run_pipeline" in check.detail


def test_the_viewer_check_fails_when_the_assets_did_not_ship(monkeypatch,
                                                             tmp_path):
    """The defect this exists for: a wheel with no viewer/, where `mottled
    serve` answers 404 for the URL it just printed."""
    import serve

    monkeypatch.setattr(serve, "ROOT", tmp_path)
    check = smoke._viewer_assets()
    assert not check.ok
    assert "404" in check.detail


def test_the_viewer_check_fails_on_a_referenced_file_that_is_absent(monkeypatch,
                                                                    tmp_path):
    """A page that loads and then does nothing is worse than one that 404s."""
    import serve

    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text('<script src="ghost.js"></script>')
    monkeypatch.setattr(serve, "ROOT", tmp_path)
    check = smoke._viewer_assets()
    assert not check.ok and "ghost.js" in check.detail


def test_the_sample_check_fails_when_no_scene_shipped(monkeypatch, tmp_path):
    import serve

    (tmp_path / "viewer" / "samples").mkdir(parents=True)
    monkeypatch.setattr(serve, "ROOT", tmp_path)
    check = smoke._bundled_sample()
    assert not check.ok and "did not ship" in check.detail


def test_a_missing_capture_backend_is_reported_not_failed():
    """Analysis and both viewers work without torch; capture does not. That is
    a fact about the install, not a broken one."""
    check = smoke._capture_backend()
    assert check.ok and not check.required


def test_the_report_marks_failures_and_counts_them():
    checks = [smoke.Check("good", True, "fine"),
              smoke.Check("bad", False, "why"),
              smoke.Check("optional", True, "absent", required=False)]
    text = smoke.format_checks(checks)
    assert "[FAIL] bad" in text
    assert "[note] optional" in text
    assert "2/3 checks passed" in text


def test_cli_exposes_smoke(capsys):
    import cli

    assert cli.main(["smoke"]) == 0
    assert "all checks passed" in capsys.readouterr().out
