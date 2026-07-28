"""Phase 3 — SAE features and residual decomposition."""

import numpy as np
import pytest

import metrics as M
import sae as S
from models import synthetic

PROMPT = "the capital of france is"


# ------------------------------------------------------------------ SAE math
def _identity_sae(dim: int) -> S.SAE:
    eye = np.eye(dim, dtype=np.float32)
    return S.SAE(w_enc=eye.copy(), b_enc=np.zeros(dim, np.float32),
                 w_dec=eye.copy(), b_dec=np.zeros(dim, np.float32))


def test_encode_is_relu_of_affine():
    sae = S.SAE(
        w_enc=np.array([[1.0, -1.0]], dtype=np.float32),   # D=1, F=2
        b_enc=np.array([0.5, 0.5], dtype=np.float32),
        w_dec=np.zeros((2, 1), dtype=np.float32),
        b_dec=np.array([2.0], dtype=np.float32),
    )
    sae.validate()
    # h=3: pre-bias 3-2=1 -> [1*1+0.5, 1*-1+0.5] = [1.5, -0.5] -> ReLU
    acts = sae.encode(np.array([3.0]))
    assert np.allclose(acts, [1.5, 0.0])


def test_identity_sae_reconstructs_nonnegative_inputs():
    sae = _identity_sae(4)
    h = np.abs(np.random.default_rng(0).normal(size=(5, 4))).astype(np.float32)
    assert np.allclose(sae.reconstruct(h), h, atol=1e-6)
    assert np.allclose(sae.reconstruction_error(h), 0.0, atol=1e-5)


def test_validate_rejects_bad_shapes():
    sae = _identity_sae(4)
    sae.b_enc = np.zeros(3, np.float32)
    with pytest.raises(ValueError):
        sae.validate()


def test_demo_sae_deterministic_sparse_unit_dictionary():
    a = S.demo_sae(16, n_features=64, seed=3)
    b = S.demo_sae(16, n_features=64, seed=3)
    assert np.array_equal(a.w_enc, b.w_enc)
    assert (a.dim, a.n_features) == (16, 64)
    assert np.allclose(np.linalg.norm(a.w_dec, axis=1), 1.0, atol=1e-6)
    acts = a.encode(np.random.default_rng(1).normal(size=(32, 16)))
    assert (acts >= 0).all()
    assert 0 < (acts > 0).mean() < 0.6  # the negative bias keeps it sparse


def test_npz_roundtrip(tmp_path):
    sae = S.demo_sae(8, n_features=12, seed=1)
    sae.labels = [f"feat-{i}" for i in range(12)]
    path = tmp_path / "sae.npz"
    S.save_npz(sae, path)
    back = S.load_npz(path)
    assert np.allclose(back.w_enc, sae.w_enc)
    assert np.allclose(back.b_dec, sae.b_dec)
    assert back.labels == sae.labels
    assert back.feature_label(3) == "feat-3"
    assert S.demo_sae(8, 12, 1).feature_label(3) == "f3"  # unlabeled fallback


# -------------------------------------------------- real-weights loading (1a)
def test_from_state_dict_maps_and_orients():
    """The standard four-array convention maps directly; either weight
    orientation is accepted and normalized to Mottled's (D, F) / (F, D)."""
    rng = np.random.default_rng(0)
    D, F = 6, 10
    w_enc = rng.normal(size=(D, F)).astype(np.float32)
    w_dec = rng.normal(size=(F, D)).astype(np.float32)
    b_enc = rng.normal(size=F).astype(np.float32)
    b_dec = rng.normal(size=D).astype(np.float32)

    sae = S.from_state_dict({"W_enc": w_enc, "b_enc": b_enc,
                             "W_dec": w_dec, "b_dec": b_dec})
    assert (sae.dim, sae.n_features) == (D, F)
    h = rng.normal(size=(4, D)).astype(np.float32)
    expected = np.maximum((h - b_dec) @ w_enc + b_enc, 0.0)
    assert np.allclose(sae.encode(h), expected, atol=1e-5)

    # a transposed encoder (F, D) is auto-oriented back to (D, F)
    flipped = S.from_state_dict({"W_enc": w_enc.T, "b_enc": b_enc,
                                 "W_dec": w_dec, "b_dec": b_dec})
    assert np.allclose(flipped.w_enc, w_enc, atol=1e-6)

    with pytest.raises(KeyError):
        S.from_state_dict({"b_enc": b_enc, "b_dec": b_dec})  # missing weights


def test_from_sae_lens_stub_and_rejects_nonstandard():
    rng = np.random.default_rng(1)
    D, F = 5, 8

    class _Cfg:
        architecture = "standard"

    class _StubSAE:
        cfg = _Cfg()
        W_enc = rng.normal(size=(D, F)).astype(np.float32)
        b_enc = rng.normal(size=F).astype(np.float32)
        W_dec = rng.normal(size=(F, D)).astype(np.float32)
        b_dec = rng.normal(size=D).astype(np.float32)

    stub = _StubSAE()
    sae = S.from_sae_lens(stub)
    assert np.allclose(sae.w_enc, stub.W_enc) and np.allclose(sae.b_dec, stub.b_dec)

    stub.cfg.architecture = "jumprelu"   # different nonlinearity → refuse
    with pytest.raises(ValueError, match="architecture"):
        S.from_sae_lens(stub)


def _fitted_sae(hidden_flat: np.ndarray) -> S.SAE:
    """A data-aligned dictionary built (in-test) from the states' own SVD.

    Antipodal singular directions + a ReLU encoder reconstruct signed hidden
    states in their own subspace — this is *not* training shipped in the
    library (a non-goal); it stands in for a real trained SAE so we can show
    the reconstruction metric separates a meaningful dictionary from
    `demo_sae`'s random one."""
    H = np.asarray(hidden_flat, dtype=np.float64)
    mean = H.mean(axis=0)
    _, _, vt = np.linalg.svd(H - mean, full_matrices=False)
    dirs = np.vstack([vt, -vt]).astype(np.float32)      # (2k, D) antipodal
    return S.SAE(w_enc=dirs.T.copy(), b_enc=np.zeros(len(dirs), np.float32),
                 w_dec=dirs, b_dec=mean.astype(np.float32))


def test_fitted_sae_beats_demo_reconstruction():
    """The acceptance check: a real (data-aligned) dictionary reconstructs
    captured states far better than the untrained demo dictionary."""
    traj = synthetic.capture(PROMPT)
    H = traj.flat_hidden()

    real = _fitted_sae(H)
    demo = S.demo_sae(traj.dim, n_features=256)

    real_err = float(real.reconstruction_error(H).mean())
    demo_err = float(demo.reconstruction_error(H).mean())
    assert real_err < 0.05                     # near-exact on the real states
    assert real_err < 0.3 * demo_err           # and materially below the demo


# ------------------------------------------------------------- features
def test_feature_trajectory_shapes():
    traj = synthetic.capture(PROMPT)
    sae = S.demo_sae(traj.dim, n_features=32)
    acts = S.feature_trajectory(traj, sae)
    assert acts.shape == (traj.n_layers, traj.n_tokens, 32)
    assert np.isfinite(acts).all() and (acts >= 0).all()
    with pytest.raises(ValueError):
        S.feature_trajectory(traj, S.demo_sae(traj.dim + 1, 32))


def test_top_and_active_features():
    acts = np.zeros((2, 2, 5))
    acts[1, 0] = [0.0, 3.0, 1.0, 0.0, 2.0]
    top = S.top_features(acts, layer=1, token=0, k=3)
    assert top == [(1, 3.0), (4, 2.0), (2, 1.0)]
    assert S.top_features(acts, layer=0, token=0, k=3) == []  # nothing fires
    ranked = S.active_features(acts, k=10)
    assert list(ranked[:2]) == [1, 4]
    assert 0 not in ranked and 3 not in ranked  # silent features excluded


# ---------------------------------------------------------- feature field
def test_pca_inverse_transform_roundtrip():
    from projection import project

    traj = synthetic.capture(PROMPT)
    coords, proj = project(traj.hidden)
    flat = coords.reshape(-1, 2)
    assert np.allclose(proj.transform(proj.inverse_transform(flat)), flat, atol=1e-4)


def test_feature_field_shapes_and_determinism():
    from projection import project

    traj = synthetic.capture(PROMPT)
    coords, proj = project(traj.hidden)
    sae = S.demo_sae(traj.dim, n_features=64)
    gx = np.linspace(coords[..., 0].min(), coords[..., 0].max(), 24)
    gy = np.linspace(coords[..., 1].min(), coords[..., 1].max(), 16)
    fld = S.feature_field(sae, proj, gx, gy)
    assert fld.magnitude.shape == fld.dominant.shape == fld.n_active.shape == (16, 24)
    assert np.isfinite(fld.magnitude).all() and (fld.magnitude >= 0).all()
    assert fld.dominant.min() >= -1 and fld.dominant.max() < 64
    assert ((fld.dominant == -1) == (fld.magnitude <= 0)).all()
    assert (fld.n_active >= (fld.magnitude > 0)).all()
    assert set(fld.features) == set(np.unique(fld.dominant[fld.dominant >= 0]))
    assert len(fld.features) > 1  # a field, not a single domain
    again = S.feature_field(sae, proj, gx, gy)
    assert np.array_equal(fld.dominant, again.dominant)
    assert np.array_equal(fld.magnitude, again.magnitude)


@pytest.mark.skipif("umap" not in __import__("projection").PROJECTIONS, reason="umap not registered")
def test_feature_field_with_umap_inverse_is_approximate_but_shape_safe():
    """umap's inverse_transform is an optimization-based approximation, not
    an exact map — the field must still come back finite and correctly
    shaped, even though the values away from the fitted data are not to be
    trusted as measurement (see the UI caption)."""
    pytest.importorskip("umap")
    from projection import project

    traj = synthetic.capture(PROMPT)
    coords, proj = project(traj.hidden, method="umap")
    sae = S.demo_sae(traj.dim, n_features=32)
    gx = np.linspace(coords[..., 0].min(), coords[..., 0].max(), 12)
    gy = np.linspace(coords[..., 1].min(), coords[..., 1].max(), 10)
    fld = S.feature_field(sae, proj, gx, gy)
    assert fld.magnitude.shape == (10, 12)
    assert np.isfinite(fld.magnitude).all()
    assert fld.dominant.min() >= -1


def test_feature_field_requires_an_invertible_projection():
    traj = synthetic.capture(PROMPT)
    sae = S.demo_sae(traj.dim, 16)

    class NoInverse:
        pass

    with pytest.raises(ValueError, match="inverse"):
        S.feature_field(sae, NoInverse(), np.zeros(4), np.zeros(4))

    class WrongDim:
        def inverse_transform(self, Y):
            return np.zeros((len(Y), traj.dim + 1), np.float32)

    with pytest.raises(ValueError, match="dim"):
        S.feature_field(sae, WrongDim(), np.zeros(4), np.zeros(4))


def test_field_rgb_domain_coloring():
    from ui import field_rgb

    fld = S.FeatureField(
        grid_x=np.array([0.0, 1.0], np.float32),
        grid_y=np.array([0.0, 1.0], np.float32),
        magnitude=np.array([[0.0, 1.0], [0.5, 2.0]], np.float32),
        dominant=np.array([[-1, 3], [5, 3]], np.int32),
        n_active=np.array([[0, 2], [1, 3]], np.int32),
    )
    rgb = field_rgb(fld)
    assert rgb.dtype == np.uint8 and rgb.shape == (2, 2, 3)
    dead = rgb[0, 0]
    assert dead[0] == dead[1] == dead[2] and dead[0] < 20  # achromatic void
    assert not np.array_equal(rgb[0, 1], rgb[1, 0])  # different features, different hue
    assert rgb[1, 1].max() > dead.max()  # firing cells outshine the void


def test_render_feature_field_modes():
    from config import MarbleConfig
    from ui import render_feature_field, run_pipeline

    cfg = MarbleConfig(model="synthetic", use_cache=False)
    result = run_pipeline(cfg, PROMPT)
    traj = result["traj"]
    assert hasattr(result["projector"], "inverse_transform")
    sae = S.demo_sae(traj.dim, 64)
    land = result["landscape"]
    fld = S.feature_field(sae, result["projector"], land.grid_x, land.grid_y)
    path = result["coords"][:, -1, :2]

    flat = render_feature_field(fld, sae, path=path)
    assert flat.data[0].type == "image"
    assert np.asarray(flat.data[0].z).shape == (len(land.grid_y), len(land.grid_x), 3)
    assert flat.data[1].type == "scatter" and len(flat.data[1].x) == traj.n_layers

    relief = render_feature_field(fld, sae, path=path, relief=True)
    assert relief.data[0].type == "surface"
    assert relief.data[1].type == "scatter3d"


def test_streamlit_app_feature_field():
    """Drive the real app with the domain-coloring field enabled."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    at = AppTest.from_file("ui.py", default_timeout=120)
    at.run()
    at.text_area(key="prompt").set_value(PROMPT)
    at.selectbox(key="model").select("synthetic")
    at.checkbox(key="sae_field").check()
    at.button(key="run").click()
    at.run()
    assert not at.exception
    at.radio(key="field_mode").set_value("relief")
    at.run()
    assert not at.exception


# ----------------------------------------------- residual decomposition
def test_synthetic_components_sum_to_updates():
    traj = synthetic.capture(PROMPT, capture_components=True)
    traj.validate()
    comps = traj.components
    L, T, D = traj.hidden.shape
    assert set(comps) == {"attn", "mlp"}
    assert comps["attn"].shape == comps["mlp"].shape == (L - 1, T, D)
    updates = np.diff(traj.hidden, axis=0)
    assert np.allclose(comps["attn"] + comps["mlp"], updates, atol=1e-5)
    # off by default, and deterministic when on
    assert synthetic.capture(PROMPT).components is None
    again = synthetic.capture(PROMPT, capture_components=True)
    assert np.array_equal(comps["attn"], again.components["attn"])


def test_component_shares():
    traj = synthetic.capture(PROMPT, capture_components=True)
    shares = M.component_shares(traj, token=-1)
    assert shares.shape == (traj.n_layers - 1, 2)
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert (shares >= 0).all() and (shares <= 1).all()
    with pytest.raises(ValueError):
        M.component_shares(synthetic.capture(PROMPT))


def test_summary_includes_attn_share():
    from projection import project

    traj = synthetic.capture(PROMPT, capture_components=True)
    coords, _ = project(traj.hidden)
    out = M.summarize(traj, coords, token=-1)
    assert 0.0 <= out["avg_attn_share"] <= 1.0


# ------------------------------------------------- torch component capture
def test_torch_component_capture_exact():
    """attn + mlp writes reproduce the residual stream exactly (pre-norm)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tests.test_capture import VOCAB_SIZE, DummyTokenizer

    from capture import capture

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_components=True)
    traj.validate()
    updates = np.diff(traj.hidden, axis=0)
    total = traj.components["attn"] + traj.components["mlp"]
    assert np.allclose(total, updates, atol=1e-4)


def test_torch_component_capture_gpt2_layout():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tests.test_capture import VOCAB_SIZE, DummyTokenizer

    from capture import capture

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=VOCAB_SIZE, n_embd=32,
                                  n_layer=2, n_head=4, n_positions=64)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_components=True)
    updates = np.diff(traj.hidden, axis=0)
    total = traj.components["attn"] + traj.components["mlp"]
    assert np.allclose(total, updates, atol=1e-4)


# ----------------------------------------------------------------- UI
def test_pipeline_and_overlay_render():
    from config import MarbleConfig
    from ui import render, run_pipeline

    cfg = MarbleConfig(model="synthetic", use_cache=False)
    result = run_pipeline(cfg, PROMPT)
    traj = result["traj"]
    assert traj.components is not None  # capture_components defaults on

    sae = S.demo_sae(traj.dim, cfg.sae_features)
    acts = S.feature_trajectory(traj, sae)
    feat = int(S.active_features(acts, k=1)[0])
    overlay = [acts[:, t.token, feat] for t in result["trajectories"]]

    fig = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                 overlay=overlay, overlay_label=f"f{feat}")
    lines = [tr for tr in fig.data if tr.type == "scatter3d" and tr.mode == "lines+markers"]
    assert len(lines) == traj.n_tokens
    assert all(len(tr.marker.color) == traj.n_layers for tr in lines)
    assert lines[0].marker.showscale and not lines[1].marker.showscale


def test_streamlit_app_sae_overlay():
    """Drive the real app with the SAE overlay enabled."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    at = AppTest.from_file("ui.py", default_timeout=120)
    at.run()
    at.text_area(key="prompt").set_value(PROMPT)
    at.selectbox(key="model").select("synthetic")
    at.checkbox(key="sae_overlay").check()
    at.button(key="run").click()
    at.run()
    assert not at.exception
    at.selectbox(key="feature").select(at.selectbox(key="feature").options[1])
    at.run()
    assert not at.exception
