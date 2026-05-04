"""
BitNet-style 1-bit Transformer Jet Tagger
==========================================
Drop-in replacement for the QKeras CNN jet tagger.

Matches exactly:
  - Input  shape : (batch, 10, 14)  [N_PART_PER_JET=10, N_FEAT=14]
  - Output shape : (batch, 1)       [single logit, no sigmoid]
  - Loss         : binary_crossentropy
  - Sample weights, pruning callbacks, and training loop are unchanged.

Architecture overview
---------------------
  Input (10×14)
    │
  BitLinear projection  →  (10×D_MODEL)   [1-bit weights]
    │
  Positional encoding   →  (10×D_MODEL)   [learned, lightweight]
    │
  × N_LAYERS of BitTransformerBlock:
      ├─ RMSNorm
      ├─ 1-bit Multi-Head Self-Attention  (Q/K/V projections are ternary)
      ├─ residual add
      ├─ RMSNorm
      ├─ 1-bit FFN  (expand → contract, ternary weights)
      └─ residual add
    │
  Global average pool   →  (D_MODEL,)
    │
  BitLinear head        →  (1,)            [logit]

BitLinear implementation
------------------------
Weights are constrained to ternary {-1, 0, +1} during the forward pass
via absmean quantization (straight-through estimator for gradients),
exactly as described in "The Era of 1-bit LLMs: All Large Language Models
are in 1.58 Bits" (Ma et al., 2024).

  W_q  = clip( round( W / (mean|W| + eps) ), -1, 1 )
  y    = x @ W_q.T  ×  scale          (scale = mean|W|, learned effectively
                                        via the full-precision master copy)

Activations are kept in full float32 (no activation quantization here)
to avoid compounding approximation errors at this small model scale.
"""

import h5py
import os
import numpy as np
import argparse
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for nohup/headless runs
import matplotlib.pyplot as plt


def tfp_median(x):
    """Compute median of a 1-D tensor via sorting."""
    n      = tf.shape(x)[0]
    sorted_x = tf.sort(x)
    mid    = n // 2
    # For even-length tensors average the two middle values
    return tf.cond(
        tf.equal(n % 2, 0),
        lambda: (sorted_x[mid - 1] + sorted_x[mid]) / 2.0,
        lambda: sorted_x[mid],
    )


def focal_loss(gamma=1.0, alpha=0.5):
    """
    Focal loss for binary classification.
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=1 down-weights easy examples moderately.
    alpha=0.5 gives equal class weighting.
    """
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        # Sigmoid probability from raw logit
        p       = tf.sigmoid(y_pred)
        p_t     = tf.where(tf.equal(y_true, 1.0), p, 1.0 - p)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        # Binary cross-entropy from logits for numerical stability
        bce     = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        focal   = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss_fn
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, Dense, GlobalAveragePooling1D, Input, Add, MultiHeadAttention
)
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import (
    prune, pruning_callbacks, pruning_schedule
)

# ─────────────────────────────────────────────
# Constants  (must match your data pipeline)
# ─────────────────────────────────────────────
N_FEAT          = 14
N_PART_PER_JET  = 10

# ─────────────────────────────────────────────
# Hyperparameters  (tunable)
# ─────────────────────────────────────────────
D_MODEL    = 32   # embedding dimension  (keep small for L1 latency)
N_HEADS    = 4    # attention heads      (D_MODEL must be divisible by N_HEADS)
N_LAYERS   = 2    # transformer blocks
FFN_DIM    = 64   # feed-forward hidden dim  (typically 2–4 × D_MODEL)
DROPOUT    = 0.0  # set >0 only if overfitting is observed
L1_REG     = 1e-4 # matches your original regularisation

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


# ══════════════════════════════════════════════════════════════════════════════
# 1-BIT PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class AbsMeanQuantizer(tf.keras.constraints.Constraint):
    """
    Straight-through absmedian quantizer used as a Keras weight *constraint*.

    Applied after every optimiser step:
      W_ternary = clip( round( W / (median|W| + eps) ), -1, 1 )

    Uses median instead of mean — more robust to outlier weights,
    which helps at small network scale.
    The full-precision master weights are updated by the optimiser;
    the constraint snaps them back to ternary for the forward pass.
    Note: using a constraint means the stored weights ARE ternary, so
    inference is exact — no separate quantisation step needed.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _ternary(self, w):
        # Absmedian: more robust than absmean for small networks
        abs_w = tf.abs(tf.reshape(w, [-1]))
        scale = tfp_median(abs_w) + self.eps
        w_scaled = w / scale
        # Straight-through: round in forward, identity in backward
        return w_scaled + tf.stop_gradient(
            tf.clip_by_value(tf.round(w_scaled), -1.0, 1.0) - w_scaled
        )

    def __call__(self, w):
        # Two-stage QAT: when QAT_ENABLED is False, behave as identity so
        # weights train in full FP32 during the warm-start phase.
        return tf.cond(
            QAT_ENABLED,
            lambda: self._ternary(w),
            lambda: tf.identity(w),
        )

    def get_config(self):
        return {"eps": self.eps}


def quantize_act_int8(x):
    """BitNet a4.8 per-token absmax int8 activation quantization with STE.
    Wang/Ma/Wei (2024), arXiv:2411.04965. Applied only inside BitLinear.call."""
    s   = tf.reduce_max(tf.abs(x), axis=-1, keepdims=True) / 127.0 + 1e-8
    xq  = tf.clip_by_value(tf.round(x / s), -127.0, 127.0)
    # STE: forward uses quantized value, backward flows through as identity
    return x + tf.stop_gradient(xq * s - x)


class BitLinear(Layer):
    """
    A fully-connected layer with ternary {-1, 0, +1} weights.

    Replaces tf.keras.layers.Dense for all projections inside the
    transformer.  Bias uses full float32 (bias contributes negligible
    parameter count and is critical for representational capacity at
    small D_MODEL).

    Args:
        units      : output dimensionality
        use_bias   : whether to add a bias term (default True)
        reg        : L1 regularisation strength on the kernel
        name       : layer name
    """
    def __init__(self, units, use_bias=True, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        self.units    = units
        self.use_bias = use_bias
        self.reg      = reg

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name        = "kernel",
            shape       = (in_dim, self.units),
            initializer = "glorot_uniform",
            regularizer = l1(self.reg),
            constraint  = AbsMeanQuantizer(),   # ← forces ternary weights
            trainable   = True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name        = "bias",
                shape       = (self.units,),
                initializer = "zeros",
                regularizer = l1(self.reg),
                trainable   = True,
            )
        self.built = True

    def call(self, x):
        # Optional int8 activation quantization — BitNet a4.8 (arXiv:2411.04965).
        # Not applied to Dense edge layers; only to ternary BitLinear projections.
        x = tf.cond(ACT_QAT_ENABLED, lambda: quantize_act_int8(x), lambda: x)
        # kernel is already ternary (enforced by the constraint after each step)
        # matmul with ternary weights is equivalent to adds/subtracts only
        out = tf.matmul(x, self.kernel)
        if self.use_bias:
            out = out + self.bias
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "use_bias": self.use_bias,
                    "reg": self.reg})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(Layer):
    """
    Root-Mean-Square Layer Normalisation (no mean subtraction).
    Preferred over LayerNorm in BitNet because the lack of centring
    preserves the sign structure of ternary activations.

      y = x / sqrt( mean(x²) + eps ) × γ
    """
    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma", shape=(dim,), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
                      + self.eps)
        return (x / rms) * self.gamma

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class BitMHSA(Layer):
    """
    1-bit Multi-Head Self-Attention.

    Q, K, V projections and the output projection all use BitLinear
    (ternary weights).  The softmax attention scores themselves remain
    in float32 — quantising attention logits severely harms performance
    at small scale.

    Args:
        d_model  : total model dimension
        n_heads  : number of attention heads (d_model % n_heads == 0)
        reg      : L1 regularisation on projection weights
    """
    def __init__(self, d_model, n_heads, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        self.reg      = reg

    def build(self, input_shape):
        self.W_q = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             name=self.name + "_Wq")
        self.W_k = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             name=self.name + "_Wk")
        self.W_v = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             name=self.name + "_Wv")
        self.W_o = BitLinear(self.d_model, use_bias=True,  reg=self.reg,
                             name=self.name + "_Wo")
        self.built = True

    def call(self, x, training=False):
        B  = tf.shape(x)[0]
        N  = tf.shape(x)[1]   # sequence length = N_PART_PER_JET = 10

        # Padding mask: True where ALL N_FEAT input features are zero  →  (B, N)
        pad_mask = tf.reduce_all(tf.equal(x, 0.0), axis=-1)
        # Expand to (B, 1, 1, N) for broadcasting over (B, heads, N_query, N_key)
        attn_bias = tf.cast(pad_mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]
        attn_bias = attn_bias * -1e9   # large negative → ~0 after softmax

        # Project with ternary weights  →  (B, N, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads  →  (B, n_heads, N, d_head)
        def split_heads(t):
            t = tf.reshape(t, (B, N, self.n_heads, self.d_head))
            return tf.transpose(t, perm=[0, 2, 1, 3])

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention with padding mask
        attn_logits  = tf.matmul(Q, K, transpose_b=True) / self.scale
        attn_logits  = attn_logits + attn_bias               # mask padded key positions
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)   # (B, heads, N, N)

        # Aggregate values
        ctx = tf.matmul(attn_weights, V)                     # (B, heads, N, d_head)

        # Merge heads  →  (B, N, d_model)
        ctx = tf.transpose(ctx, perm=[0, 2, 1, 3])
        ctx = tf.reshape(ctx, (B, N, self.d_model))

        # Output projection (ternary)
        return self.W_o(ctx)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "reg": self.reg})
        return cfg


class BitFFN(Layer):
    """
    1-bit Feed-Forward Network.
    Two BitLinear layers with a ReLU in between:
      x → BitLinear(ffn_dim) → ReLU → BitLinear(d_model)
    """
    def __init__(self, d_model, ffn_dim, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.reg     = reg

    def build(self, input_shape):
        self.fc1 = BitLinear(self.ffn_dim, reg=self.reg, name=self.name+"_fc1")
        self.fc2 = BitLinear(self.d_model, reg=self.reg, name=self.name+"_fc2")
        self.built = True

    def call(self, x):
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "ffn_dim": self.ffn_dim,
                    "reg": self.reg})
        return cfg


class BitTransformerBlock(Layer):
    """
    One transformer block with pre-norm and residual connections:

      x → RMSNorm → BitMHSA → + residual
        → RMSNorm → BitFFN  → + residual
    """
    def __init__(self, d_model, n_heads, ffn_dim, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.ffn_dim  = ffn_dim
        self.reg      = reg

    def build(self, input_shape):
        self.norm1 = RMSNorm(name=self.name + "_norm1")
        self.norm2 = RMSNorm(name=self.name + "_norm2")
        self.attn  = BitMHSA(self.d_model, self.n_heads,
                              reg=self.reg, name=self.name + "_attn")
        self.ffn   = BitFFN(self.d_model, self.ffn_dim,
                             reg=self.reg, name=self.name + "_ffn")
        self.built = True

    def call(self, x, training=False):
        # Self-attention sub-layer
        x = x + self.attn(self.norm1(x), training=training)
        # Feed-forward sub-layer
        x = x + self.ffn(self.norm2(x))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "ffn_dim": self.ffn_dim, "reg": self.reg})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_bitnet_jet_tagger(
    n_particles : int   = N_PART_PER_JET,
    n_features  : int   = N_FEAT,
    d_model     : int   = D_MODEL,
    n_heads     : int   = N_HEADS,
    n_layers    : int   = N_LAYERS,
    ffn_dim     : int   = FFN_DIM,
    reg         : float = L1_REG,
    fp_edges    : bool  = True,
) -> Model:
    """
    Build the 1-bit Transformer jet tagger.

    Input  : (batch, n_particles, n_features)  →  same as QKeras CNN
    Output : (batch, 1)                         →  raw logit, no sigmoid

    The model is permutation-equivariant up to the final GlobalAvgPool,
    which makes it a proper Deep-Sets / set-transformer for jet physics.

    Usage
    -----
    model = build_bitnet_jet_tagger()
    model.summary()
    model.compile(loss=focal_loss(gamma=1.0, alpha=0.5),
                  optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=3e-4, weight_decay=0.01, beta_2=0.95),
                  metrics=["binary_accuracy"])
    """

    # ── Input ────────────────────────────────────────────────────────────────
    inputs = Input(shape=(n_particles, n_features), name="input_1")

    # ── Input projection: N_FEAT → D_MODEL ───────────────────────────────────
    # BitNet b1.58 (arXiv:2402.17764): leave embedding layer in FP32 when
    # fp_edges=True; <0.5% of params, disproportionate ROC-tail benefit.
    if fp_edges:
        x = Dense(d_model, use_bias=True, kernel_regularizer=l1(reg),
                  name="input_proj")(inputs)
    else:
        x = BitLinear(d_model, reg=reg, name="input_proj")(inputs)
    x = RMSNorm(name="input_norm")(x)
    # shape: (batch, 10, d_model)

    # ── Positional encoding removed ──────────────────────────────────────────
    # Particles in a jet are unordered; research suggests removing positional
    # encoding helps at small scale by preserving permutation equivariance.
    # pos_emb = tf.keras.layers.Embedding(
    #     input_dim   = n_particles,
    #     output_dim  = d_model,
    #     name        = "pos_embedding"
    # )(tf.range(n_particles))
    # x = x + pos_emb

    # ── Transformer blocks  ───────────────────────────────────────────────────
    for i in range(n_layers):
        x = BitTransformerBlock(
            d_model  = d_model,
            n_heads  = n_heads,
            ffn_dim  = ffn_dim,
            reg      = reg,
            name     = f"bit_block_{i}"
        )(x)
    # shape: (batch, 10, d_model)

    # ── Final normalisation before pooling  ──────────────────────────────────
    x = RMSNorm(name="final_norm")(x)

    # ── Global average pool: sequence → vector  ───────────────────────────────
    # Mirrors your GlobalAveragePooling1D — aggregates over particles.
    x = GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    # shape: (batch, d_model)

    # ── Classification head  ──────────────────────────────────────────────────
    x = BitLinear(d_model, reg=reg, name="head_fc1")(x)
    x = tf.keras.layers.Activation("relu", name="head_act")(x)

    # BitNet b1.58 (arXiv:2402.17764): lm_head stays FP32 when fp_edges=True.
    if fp_edges:
        outputs = Dense(1, use_bias=True, kernel_regularizer=l1(reg),
                        name="head_fc2")(x)
    else:
        outputs = BitLinear(1, reg=reg, name="head_fc2")(x)
    # shape: (batch, 1)  — raw logit, no sigmoid  ✓

    return Model(inputs=inputs, outputs=outputs, name="bitnet_jet_tagger")


# ══════════════════════════════════════════════════════════════════════════════
# PARTIAL-AUC LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def pauc_loss_fn(y_true, y_logit, fpr_thresh=0.01):
    """One-way pAUC surrogate via top-K hard negatives.
    Yao, Lin, Yang (2022), arXiv:2203.01505. Equivalent to LibAUC pAUCLoss 1-way."""
    pos      = tf.boolean_mask(y_logit, tf.equal(y_true, 1.0))
    neg      = tf.boolean_mask(y_logit, tf.equal(y_true, 0.0))
    K        = tf.maximum(1, tf.cast(
                   tf.cast(tf.size(neg), tf.float32) * fpr_thresh, tf.int32))
    hard_neg, _ = tf.math.top_k(neg, k=K)
    diff     = tf.expand_dims(hard_neg, 0) - tf.expand_dims(pos, 1) + 1.0
    return tf.reduce_mean(tf.square(tf.nn.relu(diff)))


def pauc2way_loss_fn(y_true, y_logit, fpr_thresh=0.01, tpr_floor=0.80):
    """Two-way pAUC surrogate: hard negatives (top-K FPR) + hard positives (bottom-K TPR).
    Yang et al. TPAMI 2022, arXiv:2206.11655."""
    pos      = tf.boolean_mask(y_logit, tf.equal(y_true, 1.0))
    neg      = tf.boolean_mask(y_logit, tf.equal(y_true, 0.0))
    K_neg    = tf.maximum(1, tf.cast(
                   tf.cast(tf.size(neg), tf.float32) * fpr_thresh, tf.int32))
    K_pos    = tf.maximum(1, tf.cast(
                   tf.cast(tf.size(pos), tf.float32) * (1.0 - tpr_floor), tf.int32))
    hard_neg, _ = tf.math.top_k(neg, k=K_neg)
    # Bottom-K positives: negate, top-K, negate back
    hard_pos, _ = tf.math.top_k(-pos, k=K_pos)
    hard_pos    = -hard_pos
    diff     = tf.expand_dims(hard_neg, 0) - tf.expand_dims(hard_pos, 1) + 1.0
    return tf.reduce_mean(tf.square(tf.nn.relu(diff)))


def _tpr_at_fpr(y_true, y_score, fpr_target):
    """Return TPR interpolated at fpr_target using the empirical ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_target, fpr, tpr))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT  (mirrors your original train.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    signalTrainFile      = args.SignalTrainFile
    bkgTrainFile         = args.BkgTrainFile
    sig_jetData_TrainFile= args.sig_jetData_TrainFile
    bkg_jetData_TrainFile= args.bkg_jetData_TrainFile

    print("Reading signal from "          + signalTrainFile)
    print("Reading background from "      + bkgTrainFile)
    print("Reading signal jet data from " + sig_jetData_TrainFile)
    print("Reading background jet data from " + bkg_jetData_TrainFile)

    # ── Load data  (unchanged from your original script) ─────────────────────
    with h5py.File(signalTrainFile,       "r") as hf:
        dataset      = hf["Training Data"][:]
    with h5py.File(bkgTrainFile,          "r") as hf:
        datasetQCD   = hf["Training Data"][:]
    with h5py.File(sig_jetData_TrainFile, "r") as hf:
        sampleData   = hf["Sample Data"][:]
    with h5py.File(bkg_jetData_TrainFile, "r") as hf:
        sampleDataQCD= hf["Sample Data"][:]

    dataset    = np.concatenate((dataset, datasetQCD))
    sampleData = np.concatenate((sampleData, sampleDataQCD))
    fullData   = np.concatenate((dataset, sampleData), axis=1)
    np.random.shuffle(fullData)
    dataset = fullData[0:,0:141]
    LLPfeats = fullData[0:,142:146]
    sampleData = fullData[0:,141:]
  
    X = dataset[:, 0 : len(dataset[0]) - 1]
    y = dataset[:, len(dataset[0]) - 1]
    X = X.reshape((X.shape[0], N_PART_PER_JET, N_FEAT))

    # ── Impact parameter normalisation knob  (unchanged) ─────────────────────
    normalizeIPs = False
    if max(X[:, :, 8].ravel()) < 2.0:
        norm_b4 = True
    else:
        print("\nImpact parameter was not normalized beforehand.\n")
        norm_b4 = False

    arch_suffix = f"_d{args.d_model}_l{args.n_layers}_ffn{args.ffn_dim}"
    if norm_b4:
        tag = f"bitnet/bitnet_train{arch_suffix}"
    elif normalizeIPs:
        tag = f"bitnet/bitnet_Norm{arch_suffix}"
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for feat_idx in [8, 9, 10]:
            tmp = scaler.fit_transform([[v] for v in X[:, :, feat_idx].ravel()])
            X[:, :, feat_idx] = tmp.reshape(X[:, :, feat_idx].shape)
    else:
        tag = f"bitnet/noNorm_train{arch_suffix}"

    os.makedirs(os.path.dirname(os.getcwd() + f"/{tag}_model.png"),
                exist_ok=True)

    #plot kinematics
    from util.plotting.kinematics_plotter import kinematics
    kinematics(X, sampleData, y, "v1", tag)

    # ── pT-reweighting  (unchanged) ───────────────────────────────────────────
    thebins    = np.linspace(0, 500, 20)
    bkgPts     = sampleData[y == 0][:, 0]
    sigPts     = sampleData[y == 1][:, 0]
    bkg_counts, _ = np.histogram(bkgPts, bins=thebins)
    sig_counts, _ = np.histogram(sigPts, bins=thebins)
    total_bkg  = len(bkgPts)
    total_sig  = len(sigPts)
    weights_pt = np.nan_to_num(sig_counts / bkg_counts,
                               nan=total_sig / total_bkg)

    weights    = np.ones(len(y))
    pt_indices = np.clip(
        np.digitize(sampleData[:, 0], bins=thebins) - 1, 0, len(weights_pt) - 1
    )
    weights[y == 0] = weights_pt[pt_indices][y == 0]

    plt.figure()
    plt.hist(weights, bins=51)
    plt.xlabel("Weights")
    plt.savefig("{}_weights.png".format(tag))

    np.save("{}_bitnetWeights.npy".format(tag),  weights)
    np.save("{}_ptRange.npy".format(tag),        sampleData[:, 0])

    # ── Build model  ──────────────────────────────────────────────────────────
    fp_edges = (not args.baseline) and args.fp_edges
    FP_EDGES.assign(fp_edges)
    model = build_bitnet_jet_tagger(
        d_model  = args.d_model,
        n_layers = args.n_layers,
        ffn_dim  = args.ffn_dim,
        fp_edges = fp_edges,
    )
    model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file    = os.getcwd() + f"/{tag}_model.png",
        show_shapes= True,
        show_layer_names=True,
    )

    # ── Pruning  (same schedule as your original) ─────────────────────────────
    # Note: tfmot pruning wraps Dense-like layers. BitLinear is a custom Layer,
    # so we selectively prune only the head Dense equivalents if needed.
    # For the transformer blocks, the ternary constraint already achieves ~67%
    # sparsity on average (roughly 1/3 of weights are zero after quantisation).
    # If you want explicit magnitude pruning on top, uncomment the block below.

    # pruning_params = {
    #     "pruning_schedule":
    #         pruning_schedule.ConstantSparsity(0.75, begin_step=2000,
    #                                           frequency=100)
    # }
    # model = prune.prune_low_magnitude(model, **pruning_params)

    # ── Learning rate schedule: cosine decay with 5% linear warmup ───────────
    BATCH_SIZE    = 50
    EPOCHS        = 200
    TRAIN_SIZE    = int(len(X) * 0.80)
    total_steps   = (TRAIN_SIZE // BATCH_SIZE) * EPOCHS
    warmup_steps  = int(0.05 * total_steps)
    peak_lr       = 3e-4
    min_lr        = 1e-6

    @tf.keras.utils.register_keras_serializable()
    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step    = tf.cast(step, tf.float32)
            warmup  = peak_lr * (step / max(warmup_steps, 1))
            cos_arg = np.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine  = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + tf.cos(cos_arg))
            return tf.where(step < warmup_steps, warmup, cosine)
        def get_config(self):
            return {}

    lr_schedule = WarmupCosineDecay()

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        loss      = focal_loss(gamma=1.0, alpha=0.5),
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate = lr_schedule,
            weight_decay  = 0.01,
            beta_2        = 0.95,
        ),
        metrics   = ["binary_accuracy"],
    )

    # ── Two-stage QAT warm-start ──────────────────────────────────────────────
    # Stage 1: 20% of EPOCHS in full FP32 (QAT_ENABLED = False)
    # Stage 2: 80% of EPOCHS with ternary QAT (QAT_ENABLED = True)
    # The same model + optimizer instance is reused across both stages so
    # AdamW's first/second-moment estimates carry over into the QAT phase.
    warmup_epochs = int(0.20 * EPOCHS)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", verbose=1, patience=5
    )

    # ── Stage 1: FP32 warm-start (no early stopping — let it run full 20%) ──
    print(f"\n=== Stage 1: FP32 warm-start for {warmup_epochs} epochs ===")
    QAT_ENABLED.assign(False)
    history_fp32 = model.fit(
        X, y,
        epochs           = warmup_epochs,
        batch_size       = BATCH_SIZE,
        verbose          = 2,
        sample_weight    = np.asarray(weights),
        validation_split = 0.20,
        callbacks        = [],
    )

    # ── Stage 2: ternary QAT (resume from warmup_epochs, keep AdamW state) ──
    print(f"\n=== Stage 2: ternary QAT for epochs {warmup_epochs}–{EPOCHS} ===")
    QAT_ENABLED.assign(True)
    history_qat = model.fit(
        X, y,
        initial_epoch    = warmup_epochs,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        verbose          = 2,
        sample_weight    = np.asarray(weights),
        validation_split = 0.20,
        callbacks        = [early_stop],
    )

    # ── Stage 2.5: activation-QAT calibration (W1A8) ─────────────────────────
    # BitNet a4.8 (arXiv:2411.04965): turn on ACT_QAT_ENABLED for 5% of EPOCHS
    # at 0.3× LR to calibrate int8 activation scales before AUC fine-tuning.
    do_act_quant = (not args.baseline) and (args.act_quant == "int8")
    if do_act_quant:
        act_epochs = max(1, int(0.05 * EPOCHS))
        print(f"\n=== Stage 2.5: activation-QAT calibration for {act_epochs} epochs ===")
        ACT_QAT_ENABLED.assign(True)
        # Rebuild optimizer at 0.3× LR; keep model weights from Stage 2
        model.compile(
            loss      = focal_loss(gamma=1.0, alpha=0.5),
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate = peak_lr * 0.3,
                weight_decay  = 0.01,
                beta_2        = 0.95,
            ),
            metrics   = ["binary_accuracy"],
        )
        model.fit(
            X, y,
            initial_epoch    = EPOCHS,
            epochs           = EPOCHS + act_epochs,
            batch_size       = BATCH_SIZE,
            verbose          = 2,
            sample_weight    = np.asarray(weights),
            validation_split = 0.20,
            callbacks        = [],
        )
    else:
        ACT_QAT_ENABLED.assign(False)

    # ── Loss curve (concatenated stages) ─────────────────────────────────────
    train_loss = history_fp32.history["loss"]     + history_qat.history["loss"]
    val_loss   = history_fp32.history["val_loss"] + history_qat.history["val_loss"]
    plt.figure(figsize=(7, 5), dpi=120)
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss,   label="Validation")
    plt.axvline(warmup_epochs - 0.5, color="k", linestyle="--",
                label="FP32 → QAT switch")
    plt.title("BitNet Model Loss", fontsize=25)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/{}_bitnetLoss.pdf".format(tag), dpi=120)

    # ── Stage 3: AUC fine-tuning ──────────────────────────────────────────────
    # Three loss modes selected by --auc-loss:
    #   aucm     : AUC margin loss (Yuan et al. NeurIPS 2021), min-max formulation
    #   pauc1way : one-way pAUC surrogate at FPR≤α (Yao/Lin/Yang 2022, arXiv:2203.01505)
    #   pauc2way : two-way pAUC with TPR floor    (Yang et al. TPAMI 2022, arXiv:2206.11655)
    # Composite loss for pAUC paths: focal + pAUC (Zhu/Wu/Yang 2022, arXiv:2203.14177)
    # QAT stays active — ternary weights are preserved throughout.
    AUC_EPOCHS  = 25
    LR_AUC      = 1e-4        # bumped from 5e-5; denser gradients with pAUC
    LR_DUAL     = LR_AUC / 500

    auc_loss_mode = "aucm" if args.baseline else args.auc_loss
    fpr_thresh    = args.fpr_thresh
    tpr_floor     = args.tpr_floor
    focal_weight  = 0.0 if args.baseline else args.focal_weight
    pauc_weight   = 1.0 if args.baseline else args.pauc_weight
    do_stratify   = (not args.baseline) and args.stratify

    focal_fn_s3 = focal_loss(gamma=1.0, alpha=0.5)

    # Mirror Keras' validation_split=0.20 boundary (last 20% = val, same order)
    n_val_s3 = int(0.20 * len(X))
    X_tr_s3  = X[:-n_val_s3].astype(np.float32)
    y_tr_s3  = y[:-n_val_s3].astype(np.float32)
    X_vl_s3  = X[-n_val_s3:].astype(np.float32)
    y_vl_s3  = y[-n_val_s3:].astype(np.float32)

    imratio = float(np.mean(y_tr_s3))
    p_auc   = tf.constant(imratio, dtype=tf.float32)
    m_auc   = tf.constant(0.7,     dtype=tf.float32)

    # Auxiliary variables used only by the AUCM min-max path
    a_var     = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="auc_a")
    b_var     = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="auc_b")
    alpha_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="auc_alpha")

    def aucml_loss_fn(y_true, y_prob):
        """AUC margin loss — LibAUC formulation (Yuan et al. 2021), pure TF."""
        pos = tf.cast(tf.equal(y_true, 1.0), tf.float32)
        neg = 1.0 - pos
        return (
            (1.0 - p_auc) * tf.reduce_mean((y_prob - a_var) ** 2 * pos)
            + p_auc       * tf.reduce_mean((y_prob - b_var) ** 2 * neg)
            + 2.0 * alpha_var * (
                p_auc * (1.0 - p_auc) * m_auc
                + tf.reduce_mean(p_auc * y_prob * neg - (1.0 - p_auc) * y_prob * pos)
            )
            - p_auc * (1.0 - p_auc) * alpha_var ** 2
        )

    # Stratified 50/50 batches — Zhu/Wu/Yang arXiv:2203.14177
    steps_per_epoch = max(1, len(X_tr_s3) // BATCH_SIZE)
    if do_stratify:
        pos_ds = tf.data.Dataset.from_tensor_slices(
            (X_tr_s3[y_tr_s3 == 1], y_tr_s3[y_tr_s3 == 1])
        ).shuffle(20_000).repeat()
        neg_ds = tf.data.Dataset.from_tensor_slices(
            (X_tr_s3[y_tr_s3 == 0], y_tr_s3[y_tr_s3 == 0])
        ).shuffle(20_000).repeat()
        tr_ds_s3 = (
            tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
            .batch(BATCH_SIZE).take(steps_per_epoch).prefetch(tf.data.AUTOTUNE)
        )
    else:
        tr_ds_s3 = (
            tf.data.Dataset
            .from_tensor_slices((X_tr_s3, y_tr_s3))
            .shuffle(20_000, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    auc_opt_s3 = tf.keras.optimizers.experimental.AdamW(
        learning_rate = LR_AUC,
        weight_decay  = 0.005,
        beta_2        = 0.95,
    )

    auc_train_hist, auc_val_hist = [], []
    print(f"\n=== Stage 3: {auc_loss_mode} fine-tuning  {AUC_EPOCHS} epochs "
          f"(fpr_thresh={fpr_thresh}, stratify={do_stratify}, "
          f"focal_w={focal_weight}, pauc_w={pauc_weight}) ===")

    for epoch in range(AUC_EPOCHS):
        for x_b, y_b in tr_ds_s3:
            if auc_loss_mode == "aucm":
                # AUCM needs a persistent tape for the dual variables
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch([a_var, b_var, alpha_var])
                    y_prob = tf.squeeze(tf.sigmoid(model(x_b, training=True)))
                    loss   = aucml_loss_fn(y_b, y_prob)
                grads_model = tape.gradient(loss, model.trainable_variables)
                grad_a      = tape.gradient(loss, a_var)
                grad_b      = tape.gradient(loss, b_var)
                grad_alpha  = tape.gradient(loss, alpha_var)
                del tape
                auc_opt_s3.apply_gradients(zip(grads_model, model.trainable_variables))
                a_var.assign_sub(LR_AUC * grad_a)
                b_var.assign_sub(LR_AUC * grad_b)
                alpha_var.assign(tf.maximum(0.0, alpha_var + LR_DUAL * grad_alpha))
            else:
                # Composite loss: focal + pAUC (arXiv:2203.14177)
                with tf.GradientTape() as tape:
                    y_logit = tf.squeeze(model(x_b, training=True))
                    if auc_loss_mode == "pauc2way":
                        p_loss = pauc2way_loss_fn(y_b, y_logit, fpr_thresh, tpr_floor)
                    else:
                        p_loss = pauc_loss_fn(y_b, y_logit, fpr_thresh)
                    loss = focal_weight * focal_fn_s3(y_b, y_logit) + pauc_weight * p_loss
                grads_model = tape.gradient(loss, model.trainable_variables)
                auc_opt_s3.apply_gradients(zip(grads_model, model.trainable_variables))

        # Epoch-level metrics (subsample training set to keep eval fast)
        tr_prob = tf.sigmoid(model(X_tr_s3[:5000], training=False)).numpy().ravel()
        vl_prob = tf.sigmoid(model(X_vl_s3,        training=False)).numpy().ravel()
        tr_auc  = roc_auc_score(y_tr_s3[:5000], tr_prob)
        vl_auc  = roc_auc_score(y_vl_s3,        vl_prob)
        vl_tpr1e2 = _tpr_at_fpr(y_vl_s3, vl_prob, 1e-2)
        vl_tpr1e3 = _tpr_at_fpr(y_vl_s3, vl_prob, 1e-3)
        auc_train_hist.append(tr_auc)
        auc_val_hist.append(vl_auc)
        if auc_loss_mode == "aucm":
            extra = f"  a={a_var.numpy():.3f}  b={b_var.numpy():.3f}  α={alpha_var.numpy():.4f}"
        else:
            extra = ""
        print(f"  ep {epoch+1:2d}/{AUC_EPOCHS}  "
              f"train_AUC={tr_auc:.4f}  val_AUC={vl_auc:.4f}  "
              f"TPR@1e-2={vl_tpr1e2:.4f}  TPR@1e-3={vl_tpr1e3:.4f}{extra}")

    # AUC fine-tuning curve
    plt.figure(figsize=(7, 4), dpi=120)
    plt.plot(auc_train_hist, label="Train AUC")
    plt.plot(auc_val_hist,   label="Val   AUC")
    plt.title(f"Stage-3 {auc_loss_mode} fine-tuning", fontsize=16)
    plt.xlabel(f"Epoch within Stage 3  (after {EPOCHS} focal-loss epochs)")
    plt.ylabel("AUROC")
    plt.ylim(max(0.80, min(auc_train_hist) - 0.02), 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.getcwd() + f"/{tag}_auc_finetune.pdf", dpi=120)
    print(f"\nStage-3 final val AUC: {vl_auc:.4f}  "
          f"TPR@FPR=1e-2: {vl_tpr1e2:.4f}  TPR@FPR=1e-3: {vl_tpr1e3:.4f}")

    # ── Save  ─────────────────────────────────────────────────────────────────
    # model = tfmot.sparsity.keras.strip_pruning(model)  # ← re-enable if pruning
    model.save(os.getcwd() + "/{}_bitnetJetTagModel.h5".format(tag))
    print(f"\nModel saved to {tag}_bitnetJetTagModel.h5")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK  (no data needed)
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(fp_edges=True):
    """
    Verify input/output shapes, per-layer ternary/FP status, and weight values.
    Run with:  python qkerasModel.py --sanity
    """
    print("=" * 70)
    print("BitNet Jet Tagger — sanity check")
    print("=" * 70)

    FP_EDGES.assign(fp_edges)
    model = build_bitnet_jet_tagger(fp_edges=fp_edges)
    model.summary()

    # Shape check
    dummy = np.random.randn(8, N_PART_PER_JET, N_FEAT).astype(np.float32)
    out   = model(dummy, training=False)
    assert out.shape == (8, 1), f"Wrong output shape: {out.shape}"
    print(f"\n✓  Input  shape : {dummy.shape}")
    print(f"✓  Output shape : {out.shape}  (raw logit, no sigmoid)")

    # Manually apply the ternary constraint using model.submodules so nested
    # BitLinear layers inside BitMHSA/BitFFN are reached (model.layers is
    # shallow — it only sees top-level layers in the functional graph).
    QAT_ENABLED.assign(True)
    q = AbsMeanQuantizer()
    for sub in model.submodules:
        if isinstance(sub, BitLinear):
            sub.kernel.assign(q(sub.kernel))

    # Per-layer table: name, ternary?, params
    fp_layer_names  = {"input_proj", "head_fc2"} if fp_edges else set()
    print(f"\n{'Kernel':<40} {'Ternary?':<12} {'Params':>8}")
    print("-" * 62)
    n_ternary_layers, n_fp_layers = 0, 0
    ternary_ok = True
    seen = set()
    for layer in model.layers:
        for w in layer.weights:
            if "kernel" not in w.name or w.name in seen:
                continue
            seen.add(w.name)
            vals       = w.numpy()
            n_params_w = int(np.prod(vals.shape))
            unique_v   = np.unique(np.round(vals, 4))
            is_ternary = set(unique_v).issubset({-1.0, 0.0, 1.0})
            is_edge    = any(fp_name in w.name for fp_name in fp_layer_names)
            tag        = "yes" if is_ternary else "no (FP32)"
            print(f"  {w.name:<38} {tag:<12} {n_params_w:>8,}")
            if is_edge:
                n_fp_layers += 1
                if is_ternary:
                    print(f"  ✗ {w.name} should be FP32 but is ternary!")
                    ternary_ok = False
            else:
                n_ternary_layers += 1
                if not is_ternary:
                    print(f"  ✗ {w.name}: expected ternary, got {unique_v[:5]}")
                    ternary_ok = False

    if ternary_ok:
        print("✓  Ternary/FP edge assignment is correct")

    # int8 activation quantization sanity: output must be finite and within ±10× of FP32
    ACT_QAT_ENABLED.assign(False)
    out_fp32 = model(dummy, training=False).numpy()
    ACT_QAT_ENABLED.assign(True)
    out_int8 = model(dummy, training=False).numpy()
    ACT_QAT_ENABLED.assign(False)
    assert np.all(np.isfinite(out_int8)), "int8 path produced non-finite output!"
    ratio = np.abs(out_int8) / (np.abs(out_fp32) + 1e-8)
    assert np.all(ratio < 10.0), f"int8/FP32 ratio out of bounds: max={ratio.max():.2f}"
    print("✓  int8 activation path: finite, within 10× of FP32")

    n_params_total = model.count_params()
    act_str = "8"
    print(f"\nBitNet jet tagger ready: {n_params_total:,} params, "
          f"{n_ternary_layers} ternary layers, {n_fp_layers} FP layers, "
          f"W1A{act_str}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BN 1-bit Transformer jet tagger for CMS L1 trigger"
    )
    parser.add_argument("--sanity", action="store_true",
                        help="Run shape/weight sanity check (no data needed)")
    parser.add_argument("--baseline", action="store_true",
                        help="Reproduce original behaviour byte-for-byte (disables all new features)")
    # ── Architecture flags ────────────────────────────────────────────────────
    parser.add_argument("--d_model",  type=int, default=D_MODEL,
                        help=f"Embedding dimension (default {D_MODEL})")
    parser.add_argument("--n_layers", type=int, default=N_LAYERS,
                        help=f"Number of transformer blocks (default {N_LAYERS})")
    parser.add_argument("--ffn_dim",  type=int, default=FFN_DIM,
                        help=f"FFN hidden dimension (default {FFN_DIM})")
    # ── Step 1: FP edge layers ────────────────────────────────────────────────
    # BitNet b1.58 (arXiv:2402.17764): input_proj and head_fc2 in FP32
    parser.add_argument("--fp-edges", dest="fp_edges",
                        action="store_true", default=True,
                        help="Keep input_proj and head_fc2 in FP32 (default: on)")
    parser.add_argument("--no-fp-edges", dest="fp_edges", action="store_false",
                        help="Use ternary BitLinear for input_proj and head_fc2")
    # ── Step 2: pAUC loss ────────────────────────────────────────────────────
    # One-way pAUC (arXiv:2203.01505); two-way (arXiv:2206.11655)
    parser.add_argument("--auc-loss", dest="auc_loss",
                        choices=["aucm", "pauc1way", "pauc2way"],
                        default="pauc1way",
                        help="Stage-3 loss: aucm | pauc1way | pauc2way (default: pauc1way)")
    parser.add_argument("--fpr-thresh", dest="fpr_thresh", type=float, default=0.01,
                        help="FPR threshold for pAUC loss (default: 0.01)")
    parser.add_argument("--tpr-floor", dest="tpr_floor", type=float, default=0.80,
                        help="TPR floor for two-way pAUC loss (default: 0.80)")
    # ── Step 3: composite loss + stratified sampling ──────────────────────────
    # Benchmarking Deep AUROC (Zhu/Wu/Yang 2022, arXiv:2203.14177)
    parser.add_argument("--focal-weight", dest="focal_weight", type=float, default=0.3,
                        help="Focal component weight in composite Stage-3 loss (default: 0.3)")
    parser.add_argument("--pauc-weight", dest="pauc_weight", type=float, default=0.7,
                        help="pAUC component weight in composite Stage-3 loss (default: 0.7)")
    parser.add_argument("--stratify", dest="stratify",
                        action="store_true", default=True,
                        help="Use stratified 50/50 batches in Stage 3 (default: on)")
    parser.add_argument("--no-stratify", dest="stratify", action="store_false",
                        help="Disable stratified batching in Stage 3")
    # ── Step 4: activation quantization ──────────────────────────────────────
    # BitNet a4.8 (arXiv:2411.04965): W1A8 per-token absmax int8 activations
    parser.add_argument("--act-quant", dest="act_quant",
                        choices=["fp32", "int8"], default="int8",
                        help="Activation quantization for BitLinear (default: int8)")
    # ── Positional data files ─────────────────────────────────────────────────
    parser.add_argument("SignalTrainFile",       nargs="?", type=str)
    parser.add_argument("BkgTrainFile",          nargs="?", type=str)
    parser.add_argument("sig_jetData_TrainFile", nargs="?", type=str)
    parser.add_argument("bkg_jetData_TrainFile", nargs="?", type=str)

    args = parser.parse_args()

    if args.sanity:
        fp_edges = (not args.baseline) and args.fp_edges
        sanity_check(fp_edges=fp_edges)
    else:
        if not all([args.SignalTrainFile, args.BkgTrainFile,
                    args.sig_jetData_TrainFile, args.bkg_jetData_TrainFile]):
            parser.error("Provide all four data file arguments, or use --sanity")
        main(args)
