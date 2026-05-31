"""
DeepSets jet tagger  (--arch deepsets) — attention-free, fully hls4ml-compatible.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input
from tensorflow.keras.regularizers import l1

from ..config import N_PART_PER_JET, N_FEAT, D_MODEL, N_LAYERS, FFN_DIM, L1_REG
from ..layers import BitLinear


def build_deepsets_jet_tagger(
    n_particles : int   = N_PART_PER_JET,
    n_features  : int   = N_FEAT,
    d_model     : int   = D_MODEL,
    n_layers    : int   = N_LAYERS,
    ffn_dim     : int   = FFN_DIM,
    reg         : float = L1_REG,
    fp_edges    : bool  = True,
) -> Model:
    """
    Attention-free Deep Sets jet tagger — fully hls4ml-compatible.

    All layers map directly to hls4ml primitives:
      Dense / BitLinear  →  Dense / TernaryDense
      LayerNormalization →  LayerNormalization
      Activation (ReLU)  →  Activation
      GlobalAveragePooling1D → GlobalAveragePooling1D
      Add                →  Add

    Architecture:
      Input (n_particles × n_features)
        → Dense(d_model)  [FP32 edge]
        → LayerNormalization
        → N_LAYERS × (LayerNorm → BitLinear(ffn_dim) → ReLU
                                → BitLinear(d_model)  → Add residual)
        → LayerNormalization
        → GlobalAveragePooling1D
        → BitLinear(d_model) → ReLU
        → Dense(1)  [FP32 edge]
    """
    inputs = Input(shape=(n_particles, n_features), name="input_1")

    # Input projection (FP32 edge)
    if fp_edges:
        x = Dense(d_model, use_bias=True, kernel_regularizer=l1(reg),
                  name="input_proj")(inputs)
    else:
        x = BitLinear(d_model, reg=reg, name="input_proj")(inputs)
    x = tf.keras.layers.LayerNormalization(name="input_norm")(x)

    # Per-particle BitFFN blocks with residual (no attention)
    for i in range(n_layers):
        residual = x
        x = tf.keras.layers.LayerNormalization(name=f"ds_block_{i}_norm1")(x)
        x = BitLinear(ffn_dim, reg=reg, name=f"ds_block_{i}_fc1")(x)
        x = tf.keras.layers.Activation("relu", name=f"ds_block_{i}_act")(x)
        x = BitLinear(d_model, reg=reg, name=f"ds_block_{i}_fc2")(x)
        x = tf.keras.layers.Add(name=f"ds_block_{i}_add")([x, residual])

    x = tf.keras.layers.LayerNormalization(name="final_norm")(x)
    x = GlobalAveragePooling1D(name="global_average_pooling1d")(x)

    x = BitLinear(d_model, reg=reg, name="head_fc1")(x)
    x = tf.keras.layers.Activation("relu", name="head_act")(x)

    if fp_edges:
        outputs = Dense(1, use_bias=True, kernel_regularizer=l1(reg),
                        name="head_fc2")(x)
    else:
        outputs = BitLinear(1, reg=reg, name="head_fc2")(x)

    return Model(inputs=inputs, outputs=outputs, name="deepsets_jet_tagger")
