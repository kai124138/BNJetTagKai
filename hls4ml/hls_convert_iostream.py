"""hls4ml C-simulation for the DeepSets jet tagger (io_stream).

What it does
    Same idea as hls_convert_v2.py but for the io_stream (streaming dataflow)
    backend: builds the io_stream config from bnjettag.hls_precision, patches in
    a missing LayerNorm stream overload, runs C-sim, and compares HLS vs Keras
    on noise + physics jets. Writes to models/hls4ml_deepsets_iostream/.

Run (from repo root)
    python hls4ml/hls_convert_iostream.py

Needs
    Patched hls4ml, TensorFlow, the model h5 (see hls_convert_v2.py header).

io_parallel vs io_stream
    io_stream uses WIDER types (more integer headroom for the streaming path) —
    this is a separate, intentional precision profile, not a mistake. Both
    profiles live in bnjettag/hls_precision.py; pick via io_type=. Do NOT
    re-inline configs here.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from pathlib import Path

import numpy as np
import tensorflow as tf
import hls4ml

from bnjettag.hls_precision import build_hls_config  # single source of precision truth

MODEL_PATH  = "models/deepsets_d64_l3_ffn128/deepsets_clean.h5"
HLS_DIR     = "models/hls4ml_deepsets_iostream"
PART        = "xcvu9p-flgb2104-2L-e"
CLOCK_NS    = 5


def patch_layernorm_stream_overload(hls_dir):
    """Add the LayerNorm stream overload missing from the local hls4ml patch."""
    header_path = Path(hls_dir) / "firmware" / "nnet_utils" / "nnet_layernorm.h"
    text = header_path.read_text()
    marker = "} // namespace nnet"
    if "void layernormalize(hls::stream<data_T> &data, hls::stream<res_T> &res" in text:
        return
    stream_overload = r'''
template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(hls::stream<data_T> &data, hls::stream<res_T> &res,
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

LAYERNORM_STREAM_SEQ_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        typename data_T::value_type in_val[dim];
        typename res_T::value_type out_val[dim];
        #pragma HLS ARRAY_PARTITION variable=in_val complete
        #pragma HLS ARRAY_PARTITION variable=out_val complete

        data_T in_pack = data.read();
        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    LAYERNORM_STREAM_LOAD:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            in_val[i] = in_pack[i];
        }

        layernorm_1d<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(in_val, out_val, scale, bias);

    LAYERNORM_STREAM_STORE:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            out_pack[i] = out_val[i];
        }
        res.write(out_pack);
    }
}

'''
    header_path.write_text(text.replace(marker, stream_overload + marker))

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

rng = np.random.default_rng(42)
X_noise = rng.normal(0, 0.1, size=(32, 10, 14)).astype(np.float32)
keras_logits = model.predict(X_noise, verbose=0).ravel()
print(f"Keras logit range (noise): [{keras_logits.min():.4f}, {keras_logits.max():.4f}]")

# Build config (single source of truth: bnjettag/hls_precision.py).
# The io_stream profile uses wider types than io_parallel (more integer
# headroom for the streaming dataflow); see the module for the values.
cfg = build_hls_config(model, io_type="io_stream")

print("Converting...")
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    output_dir=HLS_DIR,
    backend="Vivado",
    io_type="io_stream",
    part=PART,
    clock_period=CLOCK_NS,
)

print("Compiling (csim)...")
hls_model.write()
patch_layernorm_stream_overload(HLS_DIR)
hls_model._compile()

hls_logits = hls_model.predict(X_noise).ravel()
print(f"\nKeras logit range (noise): [{keras_logits.min():.4f}, {keras_logits.max():.4f}]")
print(f"HLS   logit range (noise): [{hls_logits.min():.4f}, {hls_logits.max():.4f}]")
corr = np.corrcoef(keras_logits, hls_logits)[0, 1]
mae  = np.mean(np.abs(keras_logits - hls_logits))
print(f"Correlation (noise): {corr:.6f}")
print(f"MAE (noise):         {mae:.6f}")

# Test on real physics data
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
        print("\nHLS model matches Keras within tolerance on physics data.")
    else:
        print(f"\nPhysics Corr={corr_phys:.4f} -- below 0.99 threshold")
except Exception as e:
    print(f"\nCould not test physics data: {e}")

print("\nDone!")
