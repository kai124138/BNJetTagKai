"""Per-layer trace: HLS vs Keras intermediate outputs (io_parallel).

What it does
    Enables Trace=True on every layer, runs the HLS C-sim, and prints a table of
    each layer's Keras-vs-HLS min/max/correlation so you can pinpoint the FIRST
    diverging layer. This is the debugging tool behind the precision tuning.

Run (from repo root)
    python hls4ml/hls_trace.py

Needs
    Patched hls4ml, TensorFlow, the model h5 (see hls_convert_v2.py header).
    Writes the throwaway project to /tmp/hls_trace_v2.

Reading the output
    corr ~1.000 = layer matches; abs(corr) < 0.9 prints " <- DIVERGE". A tight
    output cluster can make corr look low even when absolute error is tiny
    (head_fc2 = 0.979 is this artifact, not a real problem — see docs/GLOSSARY.md).
    Verified 2026-05-31: 1.000 through head_fc1, Final Corr 0.9788.

Precision
    Uses bnjettag.hls_precision.build_hls_config(io_parallel) + enable_trace —
    same config as hls_convert_v2.py, so the trace reflects the real model.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf, numpy as np, hls4ml

from bnjettag.hls_precision import build_hls_config, enable_trace  # single source of truth

model = tf.keras.models.load_model('models/deepsets_d64_l3_ffn128/deepsets_clean.h5', compile=False)
rng = np.random.default_rng(42)
X = rng.normal(0, 0.1, size=(8, 10, 14)).astype(np.float32)

# Build the io_parallel config from the shared module, then turn on per-layer trace.
cfg = build_hls_config(model, io_type='io_parallel')
enable_trace(cfg)

print('Converting and compiling...')
hls_m = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir='/tmp/hls_trace_v2',
    backend='Vivado', io_type='io_parallel',
    part='xcvu9p-flgb2104-2L-e', clock_period=5)
hls_m.compile()

hls_pred, hls_trace = hls_m.trace(X)

print('%-35s %10s %10s | %10s %10s | %6s' % ('Layer', 'K_min', 'K_max', 'H_min', 'H_max', 'Corr'))
print('-' * 90)
for layer in model.layers[1:]:
    try:
        sub = tf.keras.Model(inputs=model.input, outputs=layer.output)
        k = sub.predict(X, verbose=0).ravel()
        h = hls_trace.get(layer.name)
        if h is None:
            print('%-35s  (no HLS trace)' % layer.name)
            continue
        h = np.array(h).ravel()
        c = np.corrcoef(k, h)[0, 1] if len(k) > 1 else float('nan')
        flag = ' <- DIVERGE' if abs(c) < 0.9 else ''
        print('%-35s %10.3f %10.3f | %10.3f %10.3f | %6.3f%s' % (
            layer.name, k.min(), k.max(), h.min(), h.max(), c, flag))
    except Exception as e:
        print('%-35s  error: %s' % (layer.name, e))

keras_final = model.predict(X, verbose=0).ravel()
hls_final   = np.array(hls_pred).ravel()
print('\nFinal: Keras=[%.3f, %.3f]  HLS=[%.3f, %.3f]  Corr=%.4f' % (
    keras_final.min(), keras_final.max(), hls_final.min(), hls_final.max(),
    np.corrcoef(keras_final, hls_final)[0,1]))
