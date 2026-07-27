"""`.mtj` interchange format: container layout, round-trips, scene export."""

import io
import json
import struct

import numpy as np
import pytest

import statefile as F
from config import MarbleConfig
from models import synthetic
from ui import run_scene

PROMPT = "the capital of france is"


@pytest.fixture(scope="module")
def traj():
    return synthetic.capture(PROMPT, capture_components=True, capture_attention=True)


# ----------------------------------------------------------------- container
def test_container_layout(tmp_path, traj):
    path = tmp_path / "run.mtj"
    F.save(traj, path)
    raw = path.read_bytes()
    assert raw[:4] == b"MTRJ"
    version, mlen = struct.unpack_from("<II", raw, 4)
    assert version == 1
    assert (12 + mlen) % 16 == 0                      # blob starts aligned
    manifest = json.loads(raw[12 : 12 + mlen].decode("utf-8"))
    assert manifest["format"] == "mottled-trajectory"
    assert manifest["kind"] == "trajectory"
    for ref in manifest["arrays"].values():
        assert ref["offset"] % 16 == 0                # every array aligned
        size = np.dtype(ref["dtype"]).itemsize * int(np.prod(ref["shape"]))
        assert ref["length"] == size


def test_reader_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.mtj"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(ValueError, match="magic"):
        F.read_container(bad)
    versioned = tmp_path / "v9.mtj"
    versioned.write_bytes(b"MTRJ" + struct.pack("<II", 9, 0))
    with pytest.raises(ValueError, match="version"):
        F.read_container(versioned)


def test_kind_mismatch(tmp_path, traj):
    path = tmp_path / "run.mtj"
    F.save(traj, path)
    with pytest.raises(ValueError, match="scene"):
        F.load_scene(path)


# ---------------------------------------------------------------- round-trip
def test_trajectory_roundtrip(tmp_path, traj):
    path = tmp_path / "run.mtj"
    F.save(traj, path)
    back = F.load(path)
    back.validate()
    assert np.array_equal(back.hidden, traj.hidden)
    assert np.array_equal(back.entropy, traj.entropy)
    assert np.array_equal(back.logits, traj.logits)
    assert np.array_equal(back.attention, traj.attention)
    assert np.array_equal(back.components["attn"], traj.components["attn"])
    assert np.array_equal(back.embedding_matrix, traj.embedding_matrix)
    assert back.tokens == traj.tokens
    assert back.vocab == traj.vocab
    assert back.topk[0][0][0][0] == traj.topk[0][0][0][0]
    assert back.meta["prompt"] == PROMPT


def test_roundtrip_minimal_and_file_objects():
    lean = synthetic.capture(PROMPT, keep_logits=False)
    lean.entropy = None
    lean.topk = None
    buf = io.BytesIO()
    F.save(lean, buf, include_embeddings=False)
    buf.seek(0)
    back = F.load(buf)
    assert back.logits is None and back.entropy is None
    assert back.embedding_matrix is None and back.components is None
    assert np.array_equal(back.hidden, lean.hidden)


def test_reader_ignores_unknown_fields(tmp_path, traj):
    """Additive evolution: unknown manifest fields must not break readers."""
    path = tmp_path / "run.mtj"
    F.save(traj, path)
    raw = bytearray(path.read_bytes())
    _, mlen = struct.unpack_from("<II", raw, 4)
    manifest = json.loads(raw[12 : 12 + mlen].decode("utf-8"))
    manifest["future_field"] = {"anything": [1, 2, 3]}
    blob = json.dumps(manifest).encode("utf-8")
    blob += b" " * ((-(12 + len(blob))) % 16)
    patched = raw[:8] + struct.pack("<I", len(blob)) + blob + raw[12 + mlen:]
    out = tmp_path / "future.mtj"
    out.write_bytes(patched)
    assert np.array_equal(F.load(out).hidden, traj.hidden)


# -------------------------------------------------------------------- scenes
@pytest.fixture(scope="module")
def scene_result():
    cfg = MarbleConfig(model="synthetic", use_cache=False, density_bootstrap=8)
    return run_scene(cfg, [PROMPT, "the capital of germany is"])


def test_scene_roundtrip(tmp_path, scene_result):
    path = tmp_path / "scene.mtj"
    F.save_scene(scene_result, path)
    scene = F.load_scene(path)
    assert scene["kind"] == "scene"
    assert len(scene["runs"]) == 2
    mesh = scene_result["mesh"]
    assert np.array_equal(scene["terrain"]["z"], mesh.z)
    for run, traj, trajectories in zip(scene["runs"], scene_result["trajs"],
                                       scene_result["trajectories_list"]):
        assert run["points"].shape == (len(trajectories), traj.n_layers, 3)
        assert np.array_equal(run["points"][0], trajectories[0].points)
        assert np.array_equal(run["entropy"], traj.entropy)
        assert run["attention"].shape == (traj.n_layers - 1, traj.n_tokens, traj.n_tokens)
        assert run["tokens"] == traj.tokens
        assert len(run["topk"]) == traj.n_layers
    cmp = scene["comparisons"][0]
    assert cmp["label"] == "B"
    assert cmp["hausdorff"] == pytest.approx(scene_result["comparisons"][0].hausdorff)


def test_scene_carries_uncertainty_layers(tmp_path, scene_result):
    """Bootstrap SE, density and per-run projection quality survive export."""
    path = tmp_path / "scene.mtj"
    F.save_scene(scene_result, path)
    scene = F.load_scene(path)
    assert np.array_equal(scene["terrain"]["se"], scene_result["landscape"].density_se)
    assert np.array_equal(scene["terrain"]["density"], scene_result["landscape"].density)
    for run, q in zip(scene["runs"], scene_result["quality_list"]):
        assert np.array_equal(run["quality"], q.preservation)


def test_scene_without_bootstrap_omits_se(tmp_path):
    cfg = MarbleConfig(model="synthetic", use_cache=False, density_bootstrap=0)
    result = run_scene(cfg, [PROMPT])
    buf = io.BytesIO()
    F.save_scene(result, buf)
    buf.seek(0)
    scene = F.load_scene(buf)
    assert "se" not in scene["terrain"]
    # quality is always present; density too
    assert "density" in scene["terrain"]
    assert "quality" in scene["runs"][0]


def test_scene_contains_no_hidden_states(tmp_path, scene_result):
    """Scene files are viewer bundles: heavy capture arrays must stay out."""
    path = tmp_path / "scene.mtj"
    F.save_scene(scene_result, path)
    manifest, arrays = F.read_container(path)
    assert not any("hidden" in name or "logits" in name or "embedding" in name
                   for name in arrays)


def test_scene_from_single_run_pipeline(tmp_path):
    from ui import run_pipeline

    cfg = MarbleConfig(model="synthetic", use_cache=False)
    result = run_pipeline(cfg, PROMPT)
    buf = io.BytesIO()
    F.save_scene(result, buf)
    buf.seek(0)
    scene = F.load_scene(buf)
    assert len(scene["runs"]) == 1
    assert "comparisons" not in scene
