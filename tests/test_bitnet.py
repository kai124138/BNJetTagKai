"""
Unit tests for qkerasModel.py — all tests use random tensors or synthetic data,
no external data files required.  Run with:  pytest tests/test_bitnet.py
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import tensorflow as tf

import qkerasModel as bm
from qkerasModel import (
    build_bitnet_jet_tagger,
    focal_loss,
    pauc_loss_fn,
    pauc2way_loss_fn,
    quantize_act_int8,
    AUCReshapingCallback,
    AbsMeanQuantizer,
    BitLinear,
    BitMHSA,
    write_hls4ml_config,
    N_PART_PER_JET,
    N_FEAT,
    QAT_ENABLED,
    ACT_QAT_ENABLED,
    STOCH_ROUND,
    FP_EDGES,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_globals():
    """Restore module globals to safe defaults around each test."""
    QAT_ENABLED.assign(True)
    ACT_QAT_ENABLED.assign(False)
    STOCH_ROUND.assign(False)
    FP_EDGES.assign(True)
    yield
    QAT_ENABLED.assign(True)
    ACT_QAT_ENABLED.assign(False)
    STOCH_ROUND.assign(False)
    FP_EDGES.assign(True)


# ── Test 1: Shape ─────────────────────────────────────────────────────────────

def test_output_shape():
    """model(random_input).shape == (8, 1)."""
    model = build_bitnet_jet_tagger(fp_edges=True)
    x = np.random.randn(8, N_PART_PER_JET, N_FEAT).astype(np.float32)
    out = model(x, training=False)
    assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"


# ── Test 2: Ternary weights after one training step ───────────────────────────

def test_ternary_weights_after_training():
    """
    After one training step with QAT enabled, inner BitLinear kernels are
    in {-1, 0, +1}; FP-edge layers (input_proj, head_fc2) are not ternary.
    """
    FP_EDGES.assign(True)
    QAT_ENABLED.assign(True)
    model = build_bitnet_jet_tagger(fp_edges=True)
    model.compile(loss=focal_loss(1.0, 0.5), optimizer="adam")

    x = np.random.randn(16, N_PART_PER_JET, N_FEAT).astype(np.float32)
    y = np.random.randint(0, 2, 16).astype(np.float32)
    model.fit(x, y, epochs=1, batch_size=16, verbose=0)

    fp_names = {"input_proj", "head_fc2"}
    for sub in model.submodules:
        if not isinstance(sub, BitLinear):
            continue
        is_edge = any(fp in sub.kernel.name for fp in fp_names)
        vals = np.unique(np.round(sub.kernel.numpy(), 4))
        if is_edge:
            # FP-edge layers: Dense, not BitLinear — should not reach here
            pass
        else:
            bad = [v for v in vals if v not in (-1.0, 0.0, 1.0)]
            assert not bad, f"{sub.kernel.name}: non-ternary values {bad[:5]}"

    # FP-edge layers are Dense — verify their kernels are NOT ternary
    for layer in model.layers:
        if layer.name in ("input_proj", "head_fc2"):
            k = layer.kernel.numpy()
            vals = np.unique(np.round(k, 4))
            assert not set(vals).issubset({-1.0, 0.0, 1.0}), \
                f"{layer.name} kernel should be FP32, not ternary"


# ── Test 3: STE gradient flow ─────────────────────────────────────────────────

def test_ste_gradient_flow():
    """Gradient w.r.t. a ternary kernel is non-zero (STE passes gradients through)."""
    QAT_ENABLED.assign(True)
    model = build_bitnet_jet_tagger(fp_edges=False)  # all layers ternary

    x = tf.constant(np.random.randn(4, N_PART_PER_JET, N_FEAT).astype(np.float32))
    with tf.GradientTape() as tape:
        out = tf.reduce_sum(model(x, training=True))

    # Find any ternary kernel and check its gradient
    found = False
    for sub in model.submodules:
        if isinstance(sub, BitLinear):
            g = tape.gradient(out, sub.kernel)
            assert g is not None, f"Gradient is None for {sub.kernel.name}"
            assert tf.reduce_any(g != 0.0).numpy(), \
                f"All-zero gradient for {sub.kernel.name}"
            found = True
            break
    assert found, "No BitLinear layer found to test gradient"


# ── Test 4: int8 activation quantization round-trip ──────────────────────────

def test_int8_act_quant_roundtrip():
    """quantize_act_int8(x) is bounded and recovers x within s_max/127."""
    tf.random.set_seed(0)
    x = tf.random.normal((8, N_FEAT), dtype=tf.float32)
    xq = quantize_act_int8(x)

    # Must be finite
    assert tf.reduce_all(tf.math.is_finite(xq)).numpy(), "int8 quant produced non-finite"

    # Reconstructed value must be within 1 LSB (s_max / 127) of original
    s_max = (tf.reduce_max(tf.abs(x), axis=-1, keepdims=True) / 127.0 + 1e-8)
    err   = tf.abs(xq - x)
    assert tf.reduce_all(err <= s_max + 1e-5).numpy(), \
        f"int8 round-trip error too large: max={tf.reduce_max(err).numpy():.4f}"


# ── Test 5: pAUC loss values ──────────────────────────────────────────────────

def test_pauc_loss_ordering():
    """
    Positives at logit=+2, negatives at logit=0: pauc_loss < 0.05.
    Reversed scores (positives at 0, negatives at 2): pauc_loss > 1.0.
    """
    N = 200
    y  = tf.concat([tf.ones(N//2), tf.zeros(N//2)], axis=0)
    # Good case: signal above background
    logit_good = tf.concat([tf.fill([N//2], 2.0), tf.fill([N//2], 0.0)], axis=0)
    loss_good  = pauc_loss_fn(y, logit_good, fpr_thresh=0.1).numpy()
    assert loss_good < 0.05, f"Expected low loss for easy case, got {loss_good:.4f}"

    # Bad case: signal below background
    logit_bad = tf.concat([tf.fill([N//2], 0.0), tf.fill([N//2], 2.0)], axis=0)
    loss_bad  = pauc_loss_fn(y, logit_bad, fpr_thresh=0.1).numpy()
    assert loss_bad > 1.0, f"Expected high loss for reversed case, got {loss_bad:.4f}"


# ── Test 6: AUCReshaping weight update ───────────────────────────────────────

def test_auc_reshaping_boost():
    """
    Positives below τ get exactly boost× weight after one on_epoch_end call.
    Positives above τ keep weight 1.0.
    Uses an untrained model with FPR_T=0.5 so ~half the positives score
    below the median negative score (guaranteed with random logits).
    """
    BOOST = 2.0; CAP = 8.0; FPR_T = 0.50
    np.random.seed(0)
    tf.random.set_seed(0)

    N = 400
    X_tr = np.random.randn(N, N_PART_PER_JET, N_FEAT).astype(np.float32)
    y_tr = np.concatenate([np.ones(N//2), np.zeros(N//2)]).astype(np.float32)
    X_vl = np.random.randn(100, N_PART_PER_JET, N_FEAT).astype(np.float32)
    y_vl = np.concatenate([np.ones(50), np.zeros(50)]).astype(np.float32)

    # Fresh (untrained) model: random logits, no class separation
    QAT_ENABLED.assign(True)
    model = build_bitnet_jet_tagger(fp_edges=True)

    cb = AUCReshapingCallback(model, X_tr, y_tr, X_vl, y_vl,
                               fpr_thresh=FPR_T, boost=BOOST, cap=CAP)
    cb.on_epoch_end(0)

    pos_mask = y_tr == 1.0
    pos_weights = cb.sample_weights[pos_mask]
    valid = set(np.round(np.unique(pos_weights), 6))
    assert valid.issubset({1.0, BOOST}), \
        f"Positive weights should be 1.0 or {BOOST}, got {valid}"

    # Negative weights must all remain 1.0
    neg_weights = cb.sample_weights[~pos_mask]
    assert np.all(neg_weights == 1.0), "Negative weights should stay 1.0"

    # With random logits at FPR_T=0.5, at least some positives are boosted
    assert np.any(pos_weights == BOOST), "No positives were boosted — check τ logic"


# ── Test 7: Determinism with STOCH_ROUND=False ────────────────────────────────

def test_determinism_no_stoch_round():
    """With STOCH_ROUND=False, two forward passes produce bitwise-identical outputs."""
    STOCH_ROUND.assign(False)
    QAT_ENABLED.assign(True)

    model = build_bitnet_jet_tagger(fp_edges=True)
    x = tf.constant(np.random.randn(4, N_PART_PER_JET, N_FEAT).astype(np.float32))

    out1 = model(x, training=False).numpy()
    out2 = model(x, training=False).numpy()
    np.testing.assert_array_equal(out1, out2,
        err_msg="Forward passes not deterministic with STOCH_ROUND=False")


# ── Test 8: Quantization Variation — V eps differs from Q/K eps ───────────────

def test_qv_eps_stored():
    """
    build_bitnet_jet_tagger(v_eps=1e-4) wires a larger eps into every W_v
    BitLinear while W_q/W_k keep the default 1e-6.
    Verified by inspecting sub.eps on BitLinear submodules whose name ends
    in '_Wv' vs '_Wq' / '_Wk'.
    Huang et al. (2023, arXiv:2307.00331): different eps per projection.
    """
    V_EPS = 1e-4
    model = build_bitnet_jet_tagger(fp_edges=True, v_eps=V_EPS)

    wq_eps_vals, wk_eps_vals, wv_eps_vals = [], [], []
    for sub in model.submodules:
        if not isinstance(sub, BitLinear):
            continue
        n = sub.name
        if n.endswith("_Wq"):
            wq_eps_vals.append(sub.eps)
        elif n.endswith("_Wk"):
            wk_eps_vals.append(sub.eps)
        elif n.endswith("_Wv"):
            wv_eps_vals.append(sub.eps)

    assert wv_eps_vals, "No W_v BitLinear found — check BitMHSA layer names"
    assert wq_eps_vals and wk_eps_vals, "No W_q or W_k BitLinear found"

    for eps in wv_eps_vals:
        assert abs(eps - V_EPS) < 1e-12, \
            f"W_v eps should be {V_EPS}, got {eps}"
    for eps in wq_eps_vals + wk_eps_vals:
        assert abs(eps - 1e-6) < 1e-12, \
            f"W_q/W_k eps should be 1e-6, got {eps}"


# ── Test 9: Stage-2 KD teacher forward is finite ─────────────────────────────

def test_kd_teacher_forward():
    """
    A teacher model built from Stage-1 FP32 weights produces finite logits.
    Mirrors the KD setup in main(): copy weights, freeze, run forward pass.
    Huang et al. (2023, arXiv:2307.00331).
    """
    QAT_ENABLED.assign(False)   # FP32 student first
    student = build_bitnet_jet_tagger(fp_edges=True)

    # Build FP32 teacher, copy weights
    teacher = build_bitnet_jet_tagger(fp_edges=True)
    teacher.set_weights(student.get_weights())
    teacher.trainable = False

    x = np.random.randn(8, N_PART_PER_JET, N_FEAT).astype(np.float32)
    t_logit = teacher(x, training=False).numpy()
    assert np.all(np.isfinite(t_logit)), "Teacher forward pass produced non-finite logits"

    # Ternary student forward
    QAT_ENABLED.assign(True)
    s_logit = student(x, training=False).numpy()
    assert np.all(np.isfinite(s_logit)), "Student forward pass produced non-finite logits"

    # KD MSE loss must be finite and non-negative
    kd_loss = np.mean((
        1.0 / (1.0 + np.exp(-s_logit / 2.0)) -
        1.0 / (1.0 + np.exp(-t_logit / 2.0))
    ) ** 2)
    assert np.isfinite(kd_loss) and kd_loss >= 0.0, \
        f"KD MSE loss is not valid: {kd_loss}"


# ── Test 10: HLS4ML config is written with required keys ──────────────────────

def test_hls4ml_config_written(tmp_path, monkeypatch):
    """
    write_hls4ml_config() writes a YAML file under bitnet/ with all required
    top-level keys and correct layer precision entries.
    """
    import yaml

    monkeypatch.chdir(tmp_path)   # write into a temp dir, not the repo
    os.makedirs("bitnet", exist_ok=True)

    QAT_ENABLED.assign(True)
    model = build_bitnet_jet_tagger(fp_edges=True)

    class _FakeArgs:
        qv_eps = 2e-6

    cfg_path = write_hls4ml_config(model, _FakeArgs(), tag="test_model",
                                   act_bits=8, fp_edges=True)

    assert os.path.isfile(cfg_path), f"Config file not found: {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    for key in ("backend", "project_name", "part", "hls_config",
                "model_info", "resource_estimate"):
        assert key in cfg, f"Missing key '{key}' in HLS4ML config"

    # Ternary layers use ap_int<2>; FP edge layers use ap_fixed<16,6>
    layer_prec = cfg["hls_config"]["LayerName"]
    for lname, ldata in layer_prec.items():
        prec = ldata["Precision"]["weight"]
        if lname in ("input_proj", "head_fc2"):
            assert prec == "ap_fixed<16,6>", \
                f"FP-edge layer {lname} should use ap_fixed<16,6>, got {prec}"
        else:
            assert prec == "ap_int<2>", \
                f"Ternary layer {lname} should use ap_int<2>, got {prec}"

    # Resource estimate must be positive
    assert cfg["resource_estimate"]["lut_estimate"] > 0
