"""
hls4ml conversion for DeepSets jet tagger.
Fixes:
  - LayerNorm accum precision (wide enough to compute mean/variance)
  - LayerNorm table_range_power2 < 0 (covers variance up to 4096)
  - LayerNorm table precision (enough fractional bits for small 1/sqrt values)
  - Dense result precision sized for actual output ranges
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import tensorflow as tf
import hls4ml

MODEL_PATH  = "models/deepsets_d64_l3_ffn128/deepsets_clean.h5"
HLS_DIR     = "models/hls4ml_deepsets_v2"
PART        = "xcvu9p-flgb2104-2L-e"
CLOCK_NS    = 5

# ── observed Keras intermediate max variances (from profiling on noise data) ──
# ds_block_1_norm1 input: var ≤ 3861 → need range up to 4096 = 2^12
# so table_range_power2 = -12 means max_val = pow(2, -(-12)) = 4096
# Per-layer analysis (from Keras trace on X=N(0,0.1)):
#   input_norm:      input from input_proj [-1,1], per-sample var ≈ 0.046  → range [0,1] OK
#   ds_block_0_norm1: input from input_norm [-7,7], per-sample var ≈ 0.83  → range [0,1] OK
#   ds_block_1_norm1: input from block0_add [-159,121], var ≈ 2642         → range [0,4096]
#   ds_block_2_norm1: similar, var ≈ 2575                                   → range [0,4096]
#   final_norm:       input from block2_add [-200,228], var ≈ 3680          → range [0,4096]
#
# Per-layer sum_cache2 max (before /n): n * max_diff^2 = 64 * max_val^2
#   input_norm:      64 * 1   = 64    → ap_fixed<16,8>  (2^7=128 > 64)
#   ds_block_0_norm1: 64 * 81 = 5184  → ap_fixed<16,14> (2^13=8192 > 5184)
#   blocks 1,2 + final: 64 * 40000 = 2.56M → ap_fixed<32,22> (2^21=2M ~ OK, use 23 to be safe)
LN_CONFIGS = {
    # accum must hold: sum_cache2 (int range) AND k_inv=1/64=0.015625 (>=6 frac bits)
    # ap_fixed<32,10>: max=511 >= 256 (64*4), 22 frac bits → k_inv exact, small var preserved
    'input_norm':      {'table_range_power2':  0,  'accum': 'ap_fixed<32,10>',
                        'table': 'ap_fixed<16,6>'},
    # ap_fixed<32,15>: max=16383 >= 12544 (64*196), 17 frac bits → k_inv=2048/2^17 exact
    'ds_block_0_norm1':{'table_range_power2':  0,  'accum': 'ap_fixed<32,15>',
                        'table': 'ap_fixed<16,6>'},
    'ds_block_1_norm1':{'table_range_power2': -12, 'accum': 'ap_fixed<32,23>',
                        'table': 'ap_fixed<24,8>'},
    'ds_block_2_norm1':{'table_range_power2': -12, 'accum': 'ap_fixed<32,23>',
                        'table': 'ap_fixed<24,8>'},
    'final_norm':      {'table_range_power2': -12, 'accum': 'ap_fixed<32,23>',
                        'table': 'ap_fixed<24,8>'},
}

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

rng = np.random.default_rng(42)
X_noise = rng.normal(0, 0.1, size=(32, 10, 14)).astype(np.float32)
keras_logits = model.predict(X_noise, verbose=0).ravel()
print(f"Keras logit range (noise): [{keras_logits.min():.4f}, {keras_logits.max():.4f}]")

# ── Build config ──────────────────────────────────────────────────────────────
cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")

# Default: reasonable precision
cfg["Model"]["Precision"]["default"] = "ap_fixed<16,6>"

# ── LayerNorm layers ──────────────────────────────────────────────────────────
for ln, lncfg in LN_CONFIGS.items():
    cfg["LayerName"][ln].update({
        "table_range_power2": lncfg["table_range_power2"],
        "table_size": 4096,
        "Precision": {
            "result":  "ap_fixed<16,6>",
            "scale":   "ap_fixed<16,6>",
            "bias":    "ap_fixed<16,6>",
            "table":   lncfg["table"],
            "accum":   lncfg["accum"],
        },
    })

# ── Dense/PointwiseConv layers: result and accum precision ──
# Key lessons from debugging:
#   1. accum_t must hold N_inputs * max_input (not just the final result range)
#   2. *_fc2_linear and head_fc2_linear are separate activation layers with own result type
#      that default to ap_fixed<16,6> (±32) and must be set explicitly
#   3. head_fc2 bias ≈ -64 overflows ap_fixed<16,6>; use ap_fixed<16,8>
dense_result_prec = {
    "input_proj":          "ap_fixed<16,6>",
    "ds_block_0_fc1":      "ap_fixed<16,11>",
    # fc2 result: set for Dense compute layer; *_fc2_linear must also be set
    "ds_block_0_fc2":      "ap_fixed<16,9>",   # Keras [-159,121] → I=8 (256>159) ✓
    "ds_block_0_fc2_linear": "ap_fixed<16,9>",
    "ds_block_0_add":      "ap_fixed<16,9>",
    "ds_block_1_fc1":      "ap_fixed<16,8>",
    "ds_block_1_fc2":      "ap_fixed<16,7>",   # Keras [-37,23] → I=6 (64>37) ✓
    "ds_block_1_fc2_linear": "ap_fixed<16,7>",
    "ds_block_1_add":      "ap_fixed<16,9>",
    "ds_block_2_fc1":      "ap_fixed<16,8>",
    "ds_block_2_fc2":      "ap_fixed<16,8>",   # Keras [-99,114] → I=7 (128>114) ✓
    "ds_block_2_fc2_linear": "ap_fixed<16,8>",
    "ds_block_2_add":      "ap_fixed<16,9>",
    "head_fc1":            "ap_fixed<16,9>",
    "head_fc2":            "ap_fixed<16,8>",   # FP32 weights; bias ≈-64 needs I=7+
    "head_fc2_linear":     "ap_fixed<16,8>",   # model output; must hold [-40,-38]
    "global_average_pooling1d": "ap_fixed<16,9>",
}
for layer_name, prec in dense_result_prec.items():
    if layer_name not in cfg["LayerName"]:
        cfg["LayerName"][layer_name] = {}
    if "Precision" not in cfg["LayerName"][layer_name]:
        cfg["LayerName"][layer_name]["Precision"] = {}
    cfg["LayerName"][layer_name]["Precision"]["result"] = prec
    # ternary weights only for actual Dense compute layers
    if layer_name not in ("input_proj", "head_fc2", "head_fc2_linear", "global_average_pooling1d",
                          "ds_block_0_fc2_linear", "ds_block_1_fc2_linear", "ds_block_2_fc2_linear",
                          "ds_block_0_add", "ds_block_1_add", "ds_block_2_add"):
        cfg["LayerName"][layer_name]["Precision"]["weight"] = "ap_int<2>"

# head_fc2: FP32 weight/bias precision (bias ≈ -64 overflows ap_fixed<16,6>)
for key in ("weight", "bias"):
    cfg["LayerName"]["head_fc2"]["Precision"][key] = "ap_fixed<16,8>"

# Dense accumulator precision (must hold sum during MAC loop, not just final result)
dense_accum_prec = {
    "ds_block_0_fc1": "ap_fixed<24,10>",  # 64 inputs * max~6 = 384; max=512 ✓
    "ds_block_0_fc2": "ap_fixed<24,12>",  # 128 inputs * max~15 = 1920; max=2048 ✓
    "ds_block_1_fc1": "ap_fixed<24,10>",
    "ds_block_1_fc2": "ap_fixed<24,10>",
    "ds_block_2_fc1": "ap_fixed<24,10>",
    "ds_block_2_fc2": "ap_fixed<24,12>",  # 128 inputs * max~11 = 1408; max=2048 ✓
    "head_fc1":       "ap_fixed<24,10>",
    "head_fc2":       "ap_fixed<24,12>",  # 64 inputs, FP32 weights
}
for layer_name, prec in dense_accum_prec.items():
    if layer_name not in cfg["LayerName"]:
        cfg["LayerName"][layer_name] = {}
    if "Precision" not in cfg["LayerName"][layer_name]:
        cfg["LayerName"][layer_name]["Precision"] = {}
    cfg["LayerName"][layer_name]["Precision"]["accum"] = prec

print("Converting...")
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    output_dir=HLS_DIR,
    backend="Vivado",
    io_type="io_parallel",
    part=PART,
    clock_period=CLOCK_NS,
)

print("Compiling (csim)...")
hls_model.compile()

hls_logits = hls_model.predict(X_noise).ravel()
print(f"\nKeras logit range (noise): [{keras_logits.min():.4f}, {keras_logits.max():.4f}]")
print(f"HLS   logit range (noise): [{hls_logits.min():.4f}, {hls_logits.max():.4f}]")
corr = np.corrcoef(keras_logits, hls_logits)[0, 1]
mae  = np.mean(np.abs(keras_logits - hls_logits))
print(f"Correlation (noise): {corr:.6f}")
print(f"MAE (noise):         {mae:.6f}")

# ── Test on real physics data ──────────────────────────────────────────────────
import h5py
DATA_DIR = "/home/users/kayamaguchi/BNJetTag/dataForgeScripts"
try:
    with h5py.File(f"{DATA_DIR}/testingDataSigpt30.h5", "r") as hf:
        sigData = hf["Testing Data"][:, :-1].reshape(-1, 10, 14).astype(np.float32)
    with h5py.File(f"{DATA_DIR}/testingDataQCDpt30.h5", "r") as hf:
        bkgData = hf["Testing Data"][:, :-1].reshape(-1, 10, 14).astype(np.float32)
    X_phys = np.concatenate([sigData, bkgData], axis=0)
    y_phys = np.array([1]*len(sigData) + [0]*len(bkgData))

    keras_phys = model.predict(X_phys, verbose=0).ravel()
    hls_phys   = hls_model.predict(X_phys).ravel()
    corr_phys = np.corrcoef(keras_phys, hls_phys)[0, 1]
    mae_phys  = np.mean(np.abs(keras_phys - hls_phys))
    print(f"\nPhysics data ({len(X_phys)} jets: {len(sigData)} sig, {len(bkgData)} bkg):")
    print(f"  Keras: [{keras_phys.min():.3f}, {keras_phys.max():.3f}]")
    print(f"  HLS:   [{hls_phys.min():.3f}, {hls_phys.max():.3f}]")
    print(f"  Corr:  {corr_phys:.6f}")
    print(f"  MAE:   {mae_phys:.6f}")

    from sklearn.metrics import roc_auc_score
    auc_keras = roc_auc_score(y_phys, keras_phys)
    auc_hls   = roc_auc_score(y_phys, hls_phys)
    print(f"  ROC AUC (Keras): {auc_keras:.4f}")
    print(f"  ROC AUC (HLS):   {auc_hls:.4f}")

    if corr_phys > 0.99:
        print("\n✓ HLS model matches Keras within tolerance on physics data!")
    else:
        print(f"\n✗ Physics Corr={corr_phys:.4f} — below 0.99 threshold")
except Exception as e:
    print(f"\nCould not test physics data: {e}")

print("\nDone!")
