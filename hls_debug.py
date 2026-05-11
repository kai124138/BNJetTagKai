"""
Debug HLS4ML conversion for DeepSets model.
Step 1: Use wide precision (ap_fixed<32,16>) to rule out overflow.
Step 2: Enable tracing to find the first diverging layer.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import tensorflow as tf
import hls4ml

MODEL_PATH = "bitnet/deepsets_clean.h5"
HLS_DIR    = "bitnet/hls4ml_deepsets_debug"

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.summary()

# ── Build test input (real-ish: values near 0, not random noise) ─────────────
# Use a small batch of zeros-ish input (all background) + small random
rng = np.random.default_rng(42)
X = rng.normal(0, 0.1, size=(32, 10, 14)).astype(np.float32)

keras_logits = model.predict(X, verbose=0).ravel()
print(f"\nKeras logit range: [{keras_logits.min():.4f}, {keras_logits.max():.4f}]")

# ── Convert with WIDE precision to rule out overflow ─────────────────────────
print("\nConverting with wide precision (ap_fixed<32,16>)...")

config = hls4ml.utils.config_from_keras_model(model, granularity="name")

# Global wide precision
config["Model"]["Precision"]["default"] = "ap_fixed<32,16>"

# Enable tracing on every layer
for layer_name in config["LayerName"]:
    config["LayerName"][layer_name]["Trace"] = True

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=HLS_DIR,
    backend="Vivado",
    io_type="io_parallel",
    part="xcvu9p-flgb2104-2L-e",
    clock_period=5,
)

print("Compiling HLS model (wide precision)...")
hls_model.compile()

hls_logits = hls_model.predict(X).ravel()
print(f"\nKeras logit range: [{keras_logits.min():.4f}, {keras_logits.max():.4f}]")
print(f"HLS   logit range: [{hls_logits.min():.4f}, {hls_logits.max():.4f}]")
corr = np.corrcoef(keras_logits, hls_logits)[0, 1]
mae  = np.mean(np.abs(keras_logits - hls_logits))
print(f"Correlation: {corr:.6f}")
print(f"MAE:         {mae:.6f}")

if corr > 0.99:
    print("\n✓ Wide precision is accurate — original overflow confirmed.")
    print("  Now trace to find minimal precision requirements.")
else:
    print("\n✗ Wide precision still fails — structural conversion issue.")
    print("  Checking intermediate layer outputs...")

    # ── Per-layer trace comparison ───────────────────────────────────────────
    keras_trace = {}
    for layer in model.layers:
        try:
            sub = tf.keras.Model(inputs=model.input, outputs=layer.output)
            out = sub.predict(X, verbose=0)
            keras_trace[layer.name] = out
        except Exception:
            pass

    try:
        hls_trace = hls4ml.model.profiling.get_ymodel_keras(hls_model, X)
    except Exception as e:
        print(f"  hls4ml trace failed: {e}")
        hls_trace = {}

    print("\nLayer-by-layer comparison (first diverging layer):")
    print(f"{'Layer':<35} {'Keras max':>12} {'HLS max':>12} {'Corr':>8}")
    print("-" * 70)
    for layer_name, k_out in keras_trace.items():
        h_out = hls_trace.get(layer_name)
        if h_out is None:
            continue
        k_flat = k_out.ravel()
        h_flat = h_out.ravel()
        k_max = np.abs(k_flat).max()
        h_max = np.abs(h_flat).max()
        c = np.corrcoef(k_flat, h_flat)[0, 1] if len(k_flat) > 1 else float('nan')
        flag = " ← DIVERGES" if c < 0.95 else ""
        print(f"  {layer_name:<33} {k_max:>12.4f} {h_max:>12.4f} {c:>8.4f}{flag}")

print("\nDone!")
