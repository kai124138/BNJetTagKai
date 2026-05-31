"""
Custom Keras layers for the BitNet-style 1-bit Transformer jet tagger.

Contents
--------
  * tfp_median / AbsMeanQuantizer / quantize_act_int8 — ternary + int8 helpers
  * BitLinear  — Dense with ternary {-1,0,+1} weights (absmedian STE)
  * RMSNorm    — root-mean-square normalisation
  * BitMHSA / BitFFN / BitTransformerBlock — the standard ternary transformer
  * PairFeatures / PairBiasMHSA / PairBiasTransformerBlock — ParT pair-bias attn
  * ClassToken / ClassAttentionBlock — CaiT/ParT learned [CLS] readout

BitLinear implementation
------------------------
Weights are constrained to ternary {-1, 0, +1} during the forward pass
via absmean quantization (straight-through estimator for gradients),
exactly as described in "The Era of 1-bit LLMs: All Large Language Models
are in 1.58 Bits" (Ma et al., 2024).
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.regularizers import l1

from .config import (
    L1_REG, QAT_ENABLED, ACT_QAT_ENABLED, STOCH_ROUND,
    IDX_PT, IDX_ETA, IDX_PHI,
)


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
        # Stochastic rounding (Zhao et al. NeurIPS 2024, arXiv:2412.04787):
        # rounds up with probability = fractional part; deterministic otherwise.
        def _stoch():
            noise   = tf.random.uniform(tf.shape(w_scaled), -0.5, 0.5)
            return tf.clip_by_value(tf.round(w_scaled + noise), -1.0, 1.0)
        def _det():
            return tf.clip_by_value(tf.round(w_scaled), -1.0, 1.0)
        w_round = tf.cond(STOCH_ROUND, _stoch, _det)
        # STE: round in forward, identity in backward
        return w_scaled + tf.stop_gradient(w_round - w_scaled)

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
        eps        : epsilon added to absmedian scale in the ternary quantizer.
                     Larger eps → more weights quantized to zero.
                     Huang et al. (2023, arXiv:2307.00331) recommend a larger
                     eps for the V projection than for Q/K.
        name       : layer name
    """
    def __init__(self, units, use_bias=True, reg=L1_REG, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.units    = units
        self.use_bias = use_bias
        self.reg      = reg
        self.eps      = eps

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name        = "kernel",
            shape       = (in_dim, self.units),
            initializer = "glorot_uniform",
            regularizer = l1(self.reg),
            constraint  = AbsMeanQuantizer(eps=self.eps),   # ← forces ternary weights
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
                    "reg": self.reg, "eps": self.eps})
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
    def __init__(self, d_model, n_heads, reg=L1_REG, v_eps=2e-6, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        self.reg      = reg
        # Quantization Variation (Huang et al. 2023, arXiv:2307.00331):
        # V uses a larger eps than Q/K so its distribution is compressed less
        # aggressively, preserving attention value resolution.
        self.v_eps    = v_eps

    def build(self, input_shape):
        self.W_q = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             name=self.name + "_Wq")
        self.W_k = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             name=self.name + "_Wk")
        self.W_v = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             eps=self.v_eps, name=self.name + "_Wv")
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
                    "reg": self.reg, "v_eps": self.v_eps})
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
    def __init__(self, d_model, n_heads, ffn_dim, reg=L1_REG, v_eps=2e-6, **kwargs):
        super().__init__(**kwargs)
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.ffn_dim  = ffn_dim
        self.reg      = reg
        self.v_eps    = v_eps

    def build(self, input_shape):
        self.norm1 = RMSNorm(name=self.name + "_norm1")
        self.norm2 = RMSNorm(name=self.name + "_norm2")
        self.attn  = BitMHSA(self.d_model, self.n_heads,
                              reg=self.reg, v_eps=self.v_eps,
                              name=self.name + "_attn")
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
                    "ffn_dim": self.ffn_dim, "reg": self.reg,
                    "v_eps": self.v_eps})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# PARTICLE-TRANSFORMER BUILDING BLOCKS  (pair-feature bias + class-token readout)
# ──────────────────────────────────────────────────────────────────────────────
# Implements the two architectural ideas from "Particle Transformer for Jet
# Tagging" (Qu, Li, Qian 2022, arXiv:2202.03772) that survive the FPGA /
# ternary-weight budget at N=10 particles:
#   1. PairFeatures + PairBiasMHSA: an additive interaction-matrix bias on
#      the attention logits, derived from pairwise kinematic distances
#      between particles. The bias is FP32 (small), the QKV projections
#      stay ternary.
#   2. ClassToken + ClassAttentionBlock: a learned [CLS] token that
#      cross-attends to the particle sequence, replacing GlobalAveragePool
#      with a learned, data-dependent pooling op (CaiT/ParT readout).
# ══════════════════════════════════════════════════════════════════════════════


class PairFeatures(Layer):
    """Compute (B, N, N, n_pair) pairwise kinematic distances from raw inputs.

    Used as the source of the additive attention bias in PairBiasMHSA.
    Channels: |Δη|, wrapped |Δφ|, Δlog(pT), ΔR, both_padded_indicator.
    All FP32 — this tensor is added to attention logits, not multiplied by
    ternary weights, so it costs no quantized-weight budget.
    """
    N_PAIR_FEATS = 5

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        # x: (B, N, F)
        pt   = x[..., IDX_PT]                 # (B, N)
        eta  = x[..., IDX_ETA]                # (B, N)
        phi  = x[..., IDX_PHI]                # (B, N)

        # Padded particles are all-zero in every feature slot
        pad  = tf.cast(tf.reduce_all(tf.equal(x, 0.0), axis=-1), tf.float32)  # (B, N)

        # Pairwise differences  →  (B, N, N)
        d_eta = tf.abs(eta[:, :, None] - eta[:, None, :])
        d_phi = phi[:, :, None] - phi[:, None, :]
        # Wrap Δφ into (-π, π] via atan2(sin, cos) — handles the seam
        d_phi = tf.atan2(tf.sin(d_phi), tf.cos(d_phi))
        d_phi = tf.abs(d_phi)
        d_R   = tf.sqrt(d_eta * d_eta + d_phi * d_phi + self.eps)

        log_pt = tf.math.log(tf.abs(pt) + self.eps)
        d_lpt  = tf.abs(log_pt[:, :, None] - log_pt[:, None, :])

        both_pad = pad[:, :, None] * pad[:, None, :]

        return tf.stack([d_eta, d_phi, d_lpt, d_R, both_pad], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


class PairBiasMHSA(Layer):
    """BitMHSA + additive FP32 pair-feature bias on the attention logits.

    Identical to BitMHSA except `call(x, pair_feats)` takes the precomputed
    (B, N, N, n_pair) pair tensor and projects it once with a single
    FP32 Dense(n_heads) before adding to the dot-product logits. The QKV
    and output projections stay ternary BitLinear.
    """
    def __init__(self, d_model, n_heads, reg=L1_REG, v_eps=2e-6, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        self.reg     = reg
        self.v_eps   = v_eps

    def build(self, input_shape):
        self.W_q = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             name=self.name + "_Wq")
        self.W_k = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             name=self.name + "_Wk")
        self.W_v = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             eps=self.v_eps, name=self.name + "_Wv")
        self.W_o = BitLinear(self.d_model, use_bias=True,  reg=self.reg,
                             name=self.name + "_Wo")
        # FP32 pair-bias projection: (B, N, N, n_pair) → (B, N, N, n_heads)
        # Tiny (5*n_heads params); kept in FP32 because it is *added* to
        # softmax logits — quantizing it would dominate the bias magnitude.
        self.pair_proj = Dense(self.n_heads, use_bias=False,
                               kernel_regularizer=l1(self.reg),
                               name=self.name + "_pair_proj")
        self.built = True

    def call(self, x, pair_feats, training=False):
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]

        pad_mask  = tf.reduce_all(tf.equal(x, 0.0), axis=-1)
        attn_bias = tf.cast(pad_mask, tf.float32)[:, tf.newaxis, tf.newaxis, :] * -1e9

        Q = self.W_q(x); K = self.W_k(x); V = self.W_v(x)

        def split_heads(t):
            t = tf.reshape(t, (B, N, self.n_heads, self.d_head))
            return tf.transpose(t, perm=[0, 2, 1, 3])

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Pair bias: (B, N, N, n_heads) → (B, n_heads, N, N)
        bias = self.pair_proj(pair_feats)
        bias = tf.transpose(bias, perm=[0, 3, 1, 2])

        attn_logits  = tf.matmul(Q, K, transpose_b=True) / self.scale + bias + attn_bias
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        ctx = tf.matmul(attn_weights, V)
        ctx = tf.transpose(ctx, perm=[0, 2, 1, 3])
        ctx = tf.reshape(ctx, (B, N, self.d_model))
        return self.W_o(ctx)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "reg": self.reg, "v_eps": self.v_eps})
        return cfg


class PairBiasTransformerBlock(Layer):
    """RMSNorm → PairBiasMHSA → +residual → RMSNorm → BitFFN → +residual."""
    def __init__(self, d_model, n_heads, ffn_dim, reg=L1_REG, v_eps=2e-6, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.reg     = reg
        self.v_eps   = v_eps

    def build(self, input_shape):
        self.norm1 = RMSNorm(name=self.name + "_norm1")
        self.norm2 = RMSNorm(name=self.name + "_norm2")
        self.attn  = PairBiasMHSA(self.d_model, self.n_heads,
                                  reg=self.reg, v_eps=self.v_eps,
                                  name=self.name + "_attn")
        self.ffn   = BitFFN(self.d_model, self.ffn_dim,
                            reg=self.reg, name=self.name + "_ffn")
        self.built = True

    def call(self, x, pair_feats, training=False):
        x = x + self.attn(self.norm1(x), pair_feats, training=training)
        x = x + self.ffn(self.norm2(x))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "ffn_dim": self.ffn_dim, "reg": self.reg,
                    "v_eps": self.v_eps})
        return cfg


class ClassToken(Layer):
    """Prepend a learnable FP32 [CLS] vector to the sequence.

    Input:  (B, N, d_model)
    Output: (B, N+1, d_model)  — slot 0 is the broadcast CLS token.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls", shape=(1, 1, self.d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        self.built = True

    def call(self, x):
        B   = tf.shape(x)[0]
        cls = tf.tile(self.cls, (B, 1, 1))
        return tf.concat([cls, x], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


class ClassAttentionBlock(Layer):
    """CaiT/ParT-style class-attention readout.

    Query = [CLS] only (1 token), Key/Value = full sequence (N+1 tokens).
    A single cross-attention step (ternary QKV + FP32 softmax) followed by
    a BitFFN, producing the pooled (B, d_model) embedding from slot 0.

    Replaces GlobalAveragePooling1D with a learned, data-dependent pooling
    op while keeping every matmul ternary.
    """
    def __init__(self, d_model, n_heads, ffn_dim, reg=L1_REG, v_eps=2e-6, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        self.ffn_dim = ffn_dim
        self.reg     = reg
        self.v_eps   = v_eps

    def build(self, input_shape):
        self.norm1 = RMSNorm(name=self.name + "_norm1")
        self.norm2 = RMSNorm(name=self.name + "_norm2")
        self.W_q   = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                               name=self.name + "_Wq")
        self.W_k   = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                               name=self.name + "_Wk")
        self.W_v   = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                               eps=self.v_eps, name=self.name + "_Wv")
        self.W_o   = BitLinear(self.d_model, use_bias=True,  reg=self.reg,
                               name=self.name + "_Wo")
        self.ffn   = BitFFN(self.d_model, self.ffn_dim,
                            reg=self.reg, name=self.name + "_ffn")
        self.built = True

    def call(self, x, training=False):
        # x: (B, N+1, d_model) — slot 0 is CLS
        B = tf.shape(x)[0]
        S = tf.shape(x)[1]
        xn = self.norm1(x)

        q_in = xn[:, :1, :]            # (B, 1, d_model)
        Q = self.W_q(q_in)             # (B, 1, d_model)
        K = self.W_k(xn)               # (B, S, d_model)
        V = self.W_v(xn)               # (B, S, d_model)

        Q = tf.reshape(Q, (B, 1, self.n_heads, self.d_head))
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])                    # (B, h, 1, d_head)
        K = tf.reshape(K, (B, S, self.n_heads, self.d_head))
        K = tf.transpose(K, perm=[0, 2, 1, 3])                    # (B, h, S, d_head)
        V = tf.reshape(V, (B, S, self.n_heads, self.d_head))
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        logits = tf.matmul(Q, K, transpose_b=True) / self.scale   # (B, h, 1, S)
        attn   = tf.nn.softmax(logits, axis=-1)
        ctx    = tf.matmul(attn, V)                               # (B, h, 1, d_head)
        ctx    = tf.transpose(ctx, perm=[0, 2, 1, 3])
        ctx    = tf.reshape(ctx, (B, 1, self.d_model))
        ctx    = self.W_o(ctx)

        cls_in = x[:, :1, :]
        cls    = cls_in + ctx                                     # residual on CLS only
        cls    = cls + self.ffn(self.norm2(cls))                  # FFN residual
        return tf.squeeze(cls, axis=1)                            # (B, d_model)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "ffn_dim": self.ffn_dim, "reg": self.reg,
                    "v_eps": self.v_eps})
        return cfg
