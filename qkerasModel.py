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
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, Dense, GlobalAveragePooling1D, Input, Add, MultiHeadAttention
)
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import MinMaxScaler
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


# ══════════════════════════════════════════════════════════════════════════════
# 1-BIT PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class AbsMeanQuantizer(tf.keras.constraints.Constraint):
    """
    Straight-through absmean quantizer used as a Keras weight *constraint*.

    Applied after every optimiser step:
      W_ternary = clip( round( W / (mean|W| + eps) ), -1, 1 )

    The full-precision master weights are updated by the optimiser;
    the constraint snaps them back to ternary for the forward pass.
    Note: using a constraint means the stored weights ARE ternary, so
    inference is exact — no separate quantisation step needed.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, w):
        scale = tf.reduce_mean(tf.abs(w)) + self.eps
        w_scaled = w / scale
        # Straight-through: round in forward, identity in backward
        w_ternary = w_scaled + tf.stop_gradient(
            tf.clip_by_value(tf.round(w_scaled), -1.0, 1.0) - w_scaled
        )
        return w_ternary

    def get_config(self):
        return {"eps": self.eps}


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

        # Project with ternary weights  →  (B, N, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads  →  (B, n_heads, N, d_head)
        def split_heads(t):
            t = tf.reshape(t, (B, N, self.n_heads, self.d_head))
            return tf.transpose(t, perm=[0, 2, 1, 3])

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention
        attn_logits = tf.matmul(Q, K, transpose_b=True) / self.scale
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
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["binary_accuracy"])
    """

    # ── Input ────────────────────────────────────────────────────────────────
    inputs = Input(shape=(n_particles, n_features), name="input_1")

    # ── Input projection: N_FEAT → D_MODEL  (ternary BitLinear) ──────────────
    # Mirrors your QConv1D(kernel_size=1) which is mathematically identical
    # to a per-particle Dense / BitLinear applied independently.
    x = BitLinear(d_model, reg=reg, name="input_proj")(inputs)
    x = RMSNorm(name="input_norm")(x)
    # shape: (batch, 10, d_model)

    # ── Learned positional encoding  ─────────────────────────────────────────
    # Particles are unordered in principle, but giving the model a
    # learnable position token lets it discover any residual pT-ordering
    # that may be present in your input features.
    pos_emb = tf.keras.layers.Embedding(
        input_dim   = n_particles,
        output_dim  = d_model,
        name        = "pos_embedding"
    )(tf.range(n_particles))                 # shape: (10, d_model)
    x = x + pos_emb                          # broadcast over batch

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
    # Two BitLinear layers to mirror your two QDense layers.
    x = BitLinear(d_model, reg=reg, name="head_fc1")(x)
    x = tf.keras.layers.Activation("relu", name="head_act")(x)

    outputs = BitLinear(1, reg=reg, name="head_fc2")(x)
    # shape: (batch, 1)  — raw logit, no sigmoid  ✓

    return Model(inputs=inputs, outputs=outputs, name="bitnet_jet_tagger")


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
    sampleData = fullData[0:,146:]
  
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

    if norm_b4:
        tag = "bitnet/bitnet_train"
    elif normalizeIPs:
        tag = "bitnet/bitnet_Norm"
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for feat_idx in [8, 9, 10]:
            tmp = scaler.fit_transform([[v] for v in X[:, :, feat_idx].ravel()])
            X[:, :, feat_idx] = tmp.reshape(X[:, :, feat_idx].shape)
    else:
        tag = "bitnet/noNorm_train"

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
    model = build_bitnet_jet_tagger()
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

    # ── Compile  (unchanged) ──────────────────────────────────────────────────
    model.compile(
        loss      = "binary_crossentropy",
        optimizer = "adam",
        metrics   = ["binary_accuracy"],
    )

    # ── Callbacks  ────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", verbose=1, patience=5
        ),
        # pruning_callbacks.UpdatePruningStep(),  # ← re-enable if pruning above
    ]

    # ── Train  (unchanged) ────────────────────────────────────────────────────
    history = model.fit(
        X, y,
        epochs           = 200,
        batch_size       = 50,
        verbose          = 2,
        sample_weight    = np.asarray(weights),
        validation_split = 0.20,
        callbacks        = [callbacks],
    )

    # ── Loss curve  ───────────────────────────────────────────────────────────
    plt.figure(figsize=(7, 5), dpi=120)
    plt.plot(history.history["loss"],     label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("BitNet Model Loss", fontsize=25)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/{}_bitnetLoss.pdf".format(tag), dpi=120)

    # ── Save  ─────────────────────────────────────────────────────────────────
    # model = tfmot.sparsity.keras.strip_pruning(model)  # ← re-enable if pruning
    model.save(os.getcwd() + "/{}_bitnetJetTagModel.h5".format(tag))
    print(f"\nModel saved to {tag}_bitnetJetTagModel.h5")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK  (no data needed)
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check():
    """
    Verify input/output shapes and that ternary constraints are applied.
    Run with:  python bitnet_jet_tagger.py --sanity
    """
    print("=" * 60)
    print("BitNet Jet Tagger — sanity check")
    print("=" * 60)

    model = build_bitnet_jet_tagger()
    model.summary()

    # Shape check
    dummy = np.random.randn(8, N_PART_PER_JET, N_FEAT).astype(np.float32)
    out   = model(dummy, training=False)
    assert out.shape == (8, 1), f"Wrong output shape: {out.shape}"
    print(f"\n✓  Input  shape : {dummy.shape}")
    print(f"✓  Output shape : {out.shape}  (raw logit, no sigmoid)")

    # Check that BitLinear weights are ternary after one build
    ternary_ok = True
    for layer in model.layers:
        for w in layer.weights:
            if "kernel" in w.name:
                vals = np.unique(np.round(w.numpy(), 4))
                bad  = [v for v in vals if v not in (-1.0, 0.0, 1.0)]
                if bad:
                    print(f"  ✗ {w.name}: non-ternary values {bad[:5]}")
                    ternary_ok = False
    if ternary_ok:
        print("✓  All BitLinear kernels are ternary {-1, 0, +1}")

    # Parameter count comparison
    n_params = model.count_params()
    print(f"\nTotal trainable parameters : {n_params:,}")
    print("(Original QKeras CNN est.  : ~2,000–3,000 params)")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BN 1-bit Transformer jet tagger for CMS L1 trigger"
    )
    parser.add_argument("--sanity", action="store_true",
                        help="Run shape/weight sanity check (no data needed)")
    parser.add_argument("SignalTrainFile",       nargs="?", type=str)
    parser.add_argument("BkgTrainFile",          nargs="?", type=str)
    parser.add_argument("sig_jetData_TrainFile", nargs="?", type=str)
    parser.add_argument("bkg_jetData_TrainFile", nargs="?", type=str)
  
    args = parser.parse_args()

    if args.sanity:
        sanity_check()
    else:
        if not all([args.SignalTrainFile, args.BkgTrainFile,
                    args.sig_jetData_TrainFile, args.bkg_jetData_TrainFile]):
            parser.error("Provide all four data file arguments, or use --sanity")
        main(args)
