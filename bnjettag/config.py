"""
Shared constants, dataset paths, and runtime toggles for the BitNet jet tagger.

Everything in this module is a single source of truth imported by the rest of
the `bnjettag` package. In particular the four `tf.Variable` toggles below are
module-level singletons: assigning to them anywhere (e.g. inside the training
loop) mutates the same object read by `AbsMeanQuantizer`/`BitLinear`, so the
ternary/FP32/activation-quant behaviour flips without rebuilding the graph.
"""

import tensorflow as tf

# ─────────────────────────────────────────────
# Constants  (must match your data pipeline)
# ─────────────────────────────────────────────
N_FEAT          = 14
N_PART_PER_JET  = 10

# ─────────────────────────────────────────────
# Default data paths  (new dataset, May 2026)
# Override via CLI flags if needed.
# ─────────────────────────────────────────────
_DATA_ROOT = (
    "/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM"
    "/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data"
)
DEFAULT_SIG_PART = f"{_DATA_ROOT}/train_merged/merged_trainPart.h5"
DEFAULT_SIG_JET  = f"{_DATA_ROOT}/train_merged/merged_trainJet.h5"
DEFAULT_BKG_PART      = f"{_DATA_ROOT}/QCD_Pt15To3000_Flat_PU200/Bkg_train.h5"
DEFAULT_BKG_JET       = f"{_DATA_ROOT}/QCD_Pt15To3000_Flat_PU200/Bkg_trainJets.h5"
DEFAULT_BKG_TEST_PART = f"{_DATA_ROOT}/QCD_Pt15To3000_Flat_PU200/Bkg_test.h5"
DEFAULT_TEST_DIR      = f"{_DATA_ROOT}/test_merged"

# All signal test categories: (mass_label, decay_mode)
# Files are named  {mass}_{decay}_merged_test{Part,Jet}.h5
TEST_CATEGORIES = [
    ("phi15", "bbbb"), ("phi15", "cccc"), ("phi15", "uuuu"),
    ("phi30", "bbbb"), ("phi30", "cccc"), ("phi30", "uuuu"),
    ("phi60", "bbbb"), ("phi60", "cccc"), ("phi60", "uuuu"),
]

# ─────────────────────────────────────────────
# Hyperparameters  (tunable)
# ─────────────────────────────────────────────
D_MODEL    = 32   # embedding dimension  (keep small for L1 latency)
N_HEADS    = 4    # attention heads      (D_MODEL must be divisible by N_HEADS)
N_LAYERS   = 2    # transformer blocks
FFN_DIM    = 64   # feed-forward hidden dim  (typically 2–4 × D_MODEL)
DROPOUT    = 0.0  # set >0 only if overfitting is observed
L1_REG     = 1e-4 # matches your original regularisation

# Per-particle feature indices, confirmed from dataForgeScripts/dataForge.py:
#   [0..7] one-hot particle type, [8] dZ, [9] dX, [10] dY,
#   [11] pT (jet-relative), [12] Δη (jet-relative), [13] Δφ (signed delta).
IDX_PT, IDX_ETA, IDX_PHI = 11, 12, 13

# ─────────────────────────────────────────────
# Two-stage QAT warm-start toggle
# ─────────────────────────────────────────────
# Module-level switch read inside AbsMeanQuantizer.__call__.
# Stage 1 (FP32 warm-start): set False  → constraint is identity
# Stage 2 (ternary QAT)     : set True   → constraint snaps weights to {-1,0,+1}
# Using a tf.Variable lets the value flip between fit() calls without
# rebuilding the graph and without losing AdamW optimizer state.
QAT_ENABLED = tf.Variable(True, trainable=False, dtype=tf.bool, name="qat_enabled")

# FP_EDGES: keep input_proj and head_fc2 in full FP32 (not ternary).
# BitNet b1.58 (Ma et al. 2024, arXiv:2402.17764): embedding and lm_head
# are deliberately left in FP32 — <0.5% of params, large ROC-tail gain.
FP_EDGES = tf.Variable(True, trainable=False, dtype=tf.bool, name="fp_edges")

# ACT_QAT_ENABLED: per-token absmax int8 activation quantization inside BitLinear.
# BitNet a4.8 (Wang/Ma/Wei 2024, arXiv:2411.04965): W1A8 — every interconnect
# on the FPGA drops from 32-bit to 8-bit (~4× bandwidth saving).
ACT_QAT_ENABLED = tf.Variable(False, trainable=False, dtype=tf.bool,
                              name="act_qat_enabled")

# STOCH_ROUND: stochastic rounding in the ternary STE during training.
# Rounds up with probability = fractional part — strictly better convergence
# than deterministic round for ternary weights (Zhao et al. NeurIPS 2024,
# arXiv:2412.04787). Set False during eval/inference for determinism.
STOCH_ROUND = tf.Variable(True, trainable=False, dtype=tf.bool,
                          name="stoch_round")
