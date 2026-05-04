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
