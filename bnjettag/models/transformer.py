"""
BitNet transformer jet tagger  (--arch bitnet, the default).

  Input (N×F)
    → Dense/BitLinear projection → RMSNorm
    → N_LAYERS × BitTransformerBlock  (ternary MHSA + BitFFN, pre-norm)
    → RMSNorm → GlobalAveragePooling1D
    → BitLinear head → ReLU → Dense/BitLinear logit
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input
from tensorflow.keras.regularizers import l1

from ..config import (
    N_PART_PER_JET, N_FEAT, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, L1_REG,
)
from ..layers import BitLinear, RMSNorm, BitTransformerBlock


def build_bitnet_jet_tagger(
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
                  metrics=["binary_accuracy"],
                  weighted_metrics=[tf.keras.metrics.AUC(name="auc")])
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
            v_eps    = v_eps,
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
