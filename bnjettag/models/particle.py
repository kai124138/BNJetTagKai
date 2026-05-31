"""
Particle BitNet jet tagger  (--arch particle) — ParT-style ternary transformer
with pair-feature attention bias and a learnable [CLS] readout.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l1

from ..config import (
    N_PART_PER_JET, N_FEAT, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, L1_REG,
)
from ..layers import (
    BitLinear, RMSNorm, PairFeatures, PairBiasTransformerBlock,
    ClassToken, ClassAttentionBlock,
)


def build_particle_bitnet_tagger(
    n_particles : int   = N_PART_PER_JET,
    n_features  : int   = N_FEAT,
    d_model     : int   = D_MODEL,
    n_heads     : int   = N_HEADS,
    n_layers    : int   = N_LAYERS,
    ffn_dim     : int   = FFN_DIM,
    reg         : float = L1_REG,
    fp_edges    : bool  = True,
    v_eps       : float = 2e-6,
) -> Model:
    """
    Particle BitNet — ternary transformer with pair-feature attention bias
    and class-token readout (Qu et al. ParT, arXiv:2202.03772).

    Same `(B, n_particles, n_features) → (B, 1)` contract as the other
    builders, so the existing three-stage training pipeline, KD, and AUC
    fine-tuning all work without modification. The only architectural
    differences vs build_bitnet_jet_tagger are:

      1. Each transformer block receives an additive (B, n_heads, N, N)
         attention bias derived from pairwise (|Δη|, |Δφ|, |Δlog pT|, ΔR,
         pad-pair) features, projected by a single shared FP32 Dense.
      2. GlobalAveragePooling1D is replaced by a learnable [CLS] token
         that cross-attends to the particle sequence in one ternary
         ClassAttentionBlock, then feeds the head.

    All matmuls remain ternary BitLinear; only the small pair-bias
    projection and the [CLS] vector are FP32 (negligible param count).
    """
    inputs = Input(shape=(n_particles, n_features), name="input_1")

    # Pair-feature tensor computed once from the raw input
    pair_feats = PairFeatures(name="pair_features")(inputs)
    # shape: (batch, N, N, 5)

    # Input projection (FP-edge by convention; matches build_bitnet_jet_tagger)
    if fp_edges:
        x = Dense(d_model, use_bias=True, kernel_regularizer=l1(reg),
                  name="input_proj")(inputs)
    else:
        x = BitLinear(d_model, reg=reg, name="input_proj")(inputs)
    x = RMSNorm(name="input_norm")(x)
    # shape: (batch, N, d_model)

    # ── ParT-style transformer blocks with pair-bias attention ───────────────
    for i in range(n_layers):
        x = PairBiasTransformerBlock(
            d_model = d_model,
            n_heads = n_heads,
            ffn_dim = ffn_dim,
            reg     = reg,
            v_eps   = v_eps,
            name    = f"pair_block_{i}",
        )(x, pair_feats)

    x = RMSNorm(name="pre_class_norm")(x)

    # ── [CLS] token + class-attention readout ───────────────────────────────
    x = ClassToken(d_model, name="class_token")(x)         # (B, N+1, d_model)
    x = ClassAttentionBlock(
        d_model = d_model,
        n_heads = n_heads,
        ffn_dim = ffn_dim,
        reg     = reg,
        v_eps   = v_eps,
        name    = "class_attn",
    )(x)                                                    # (B, d_model)

    # ── Classification head ───────────────────────────────────────────────────
    x = BitLinear(d_model, reg=reg, name="head_fc1")(x)
    x = tf.keras.layers.Activation("relu", name="head_act")(x)

    if fp_edges:
        outputs = Dense(1, use_bias=True, kernel_regularizer=l1(reg),
                        name="head_fc2")(x)
    else:
        outputs = BitLinear(1, reg=reg, name="head_fc2")(x)

    return Model(inputs=inputs, outputs=outputs, name="particle_bitnet_tagger")
