# hls4ml Conversion for the DeepSets Jet Tagger

This directory holds everything that converts the trained Keras DeepSets model
(`models/deepsets_d64_l3_ffn128/deepsets_clean.h5`) into HLS C++ for FPGA
synthesis.

All scripts here are designed to be run **from the repository root**, e.g.:

```bash
cd ~/BNJetTagKai
python hls4ml/hls_convert_v2.py
# or, for the streaming interface variant:
python hls4ml/hls_convert_iostream.py
```

(They reference paths like `models/deepsets_d64_l3_ffn128/deepsets_clean.h5`
relative to the repo root.)

## Files

| Script               | What it does                                                    |
| -------------------- | --------------------------------------------------------------- |
| `hls_convert_v2.py`  | Main conversion. Produces `models/hls4ml_deepsets_v2/`.         |
| `hls_convert_iostream.py` | Streaming-interface conversion. Produces `models/hls4ml_deepsets_iostream/`. |
| `hls_trace.py`       | Layer-by-layer comparison: Keras vs HLS C-sim per layer output. |
| `hls_debug.py`       | Sanity check with wide global precision (`ap_fixed<32,16>`).    |
| `hls_build.py`       | Runs Vivado HLS synthesis (`hls_model.build()`) — long. ~30–60min on the target part. |

## Prerequisites

1. Trained model exists at `models/deepsets_d64_l3_ffn128/deepsets_clean.h5`.
2. hls4ml is installed under `software/hls4ml/` (editable install).
3. The **three source patches** in `patches/hls4ml/` have been applied to that
   local hls4ml clone — otherwise the LayerNorm output will be wrong. See
   `patches/hls4ml/README.md`.
4. For synthesis only: `vivado_hls` from Vivado 2020.1 on PATH
   (`hls_build.py` sets `/data/software/xilinx/Vivado/2020.1/bin` automatically).

## Typical workflow

```bash
# 1. Convert + C-simulate
python hls4ml/hls_convert_v2.py

# Optional: convert + C-simulate the io_stream variant
python hls4ml/hls_convert_iostream.py

# 2. If the predictions don't match Keras, run the trace to find the
#    first diverging layer:
python hls4ml/hls_trace.py

# 3. Once accuracy is good, synthesize:
python hls4ml/hls_build.py
```

## Per-layer precision

`hls_convert_v2.py` configures per-layer precision because a single global
setting can't handle both the small-variance LayerNorms (`input_norm`,
`ds_block_0_norm1`, vars 0.009–0.83) and the large-variance ones after
residual blocks (`ds_block_1_norm1` onward, vars up to ~3800).

| Layer              | `table_range_power2` | `accum_t`          | `table_t`        |
| ------------------ | -------------------- | ------------------ | ---------------- |
| `input_norm`       | 0                    | `ap_fixed<32,10>`  | `ap_fixed<16,6>` |
| `ds_block_0_norm1` | 0                    | `ap_fixed<32,15>`  | `ap_fixed<16,6>` |
| `ds_block_1_norm1` | -12                  | `ap_fixed<32,23>`  | `ap_fixed<24,8>` |
| `ds_block_2_norm1` | -12                  | `ap_fixed<32,23>`  | `ap_fixed<24,8>` |
| `final_norm`       | -12                  | `ap_fixed<32,23>`  | `ap_fixed<24,8>` |

Dense layers and the `add` residual outputs are also given individually-sized
result precisions — see `LN_CONFIGS` and `dense_result_prec` in
`hls_convert_v2.py` for the exact values and the rationale per layer in the
inline comments.

The io_stream variant starts from the same LayerNorm settings but adds a
generated-project stream overload for LayerNorm before C-sim. Its dense and
input-projection precisions are widened relative to v2 after retuning on the
same fixed noise input. Current C-sim comparison:

| Variant | Output directory | Noise correlation | Noise MAE | Physics correlation | Physics MAE |
| ------- | ---------------- | ----------------- | --------- | ------------------- | ----------- |
| `io_parallel` | `models/hls4ml_deepsets_v2/` | 0.969850 | 0.383298 | 0.997195 | 0.544165 |
| `io_stream` | `models/hls4ml_deepsets_iostream/` | 0.972328 | 0.302683 | 0.999144 | 0.322316 |

See `docs/hls4ml_precision_bugs.md` and `docs/hls4ml_layernorm_patches.md` for
the full technical write-up of what was wrong with stock hls4ml and how each
issue was diagnosed.
