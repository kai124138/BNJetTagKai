"""hls4ml C-simulation for the DeepSets jet tagger (io_parallel).

What it does
    Loads the trained Keras DeepSets model, builds the io_parallel hls4ml config
    from bnjettag.hls_precision, runs C-sim (hls_model.compile + predict), and
    compares HLS vs Keras logits on (a) random noise and (b) real physics jets.
    This is the primary "does the HLS model still match Keras?" check.

Run (from repo root)
    python hls4ml/hls_convert_v2.py

Needs
    Patched hls4ml (bash hls4ml/setup_hls4ml.sh), TensorFlow, the model at
    models/deepsets_d64_l3_ffn128/deepsets_clean.h5. Physics comparison also
    needs the h5 files under DATA_DIR (edit it for your machine); without them
    it still runs the noise C-sim.

Outputs
    Generated project in models/hls4ml_deepsets_v2/ and printed noise/physics
    correlation + AUC. Verified 2026-05-31: physics corr 0.995, AUC matches.

Precision
    All precision lives in bnjettag/hls_precision.py (single source of truth) —
    do NOT re-inline configs here. See docs/hls4ml_precision_bugs.md for why.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import tensorflow as tf
import hls4ml

from bnjettag.hls_precision import build_hls_config  # single source of precision truth

MODEL_PATH  = "models/deepsets_d64_l3_ffn128/deepsets_clean.h5"
HLS_DIR     = "models/hls4ml_deepsets_v2"
PART        = "xcvu9p-flgb2104-2L-e"
CLOCK_NS    = 5

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

rng = np.random.default_rng(42)
X_noise = rng.normal(0, 0.1, size=(32, 10, 14)).astype(np.float32)
keras_logits = model.predict(X_noise, verbose=0).ravel()
print(f"Keras logit range (noise): [{keras_logits.min():.4f}, {keras_logits.max():.4f}]")

# ── Build config (single source of truth: bnjettag/hls_precision.py) ──
cfg = build_hls_config(model, io_type="io_parallel")

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
