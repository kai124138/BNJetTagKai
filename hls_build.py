"""
Run Vivado HLS synthesis on the DeepSets HLS project.
Re-runs conversion (fast, ~1min) then calls hls_model.build() for synthesis (~30-60min).
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# vivado_hls is in Vivado 2020.1 (2023.2 uses vitis_hls which is not available here)
VIVADO_BIN = "/data/software/xilinx/Vivado/2020.1/bin"
os.environ["PATH"] = VIVADO_BIN + ":" + os.environ.get("PATH", "")

import numpy as np
import tensorflow as tf
import hls4ml

MODEL_PATH = "bitnet/deepsets_clean.h5"
HLS_DIR    = "bitnet/hls4ml_deepsets_v2"
PART       = "xcvu9p-flgb2104-2L-e"
CLOCK_NS   = 5

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ── Re-run config (same as hls_convert_v2.py) ────────────────────────────────
cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
cfg["Model"]["Precision"]["default"] = "ap_fixed<16,6>"
cfg["Model"]["ReuseFactor"] = 64

LN_CONFIGS = {
    "input_norm":      {"table_range_power2":  0, "accum": "ap_fixed<32,10>", "table": "ap_fixed<16,6>"},
    "ds_block_0_norm1":{"table_range_power2":  0, "accum": "ap_fixed<32,15>", "table": "ap_fixed<16,6>"},
    "ds_block_1_norm1":{"table_range_power2":-12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
    "ds_block_2_norm1":{"table_range_power2":-12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
    "final_norm":      {"table_range_power2":-12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
}
for ln, lncfg in LN_CONFIGS.items():
    if ln not in cfg["LayerName"]:
        cfg["LayerName"][ln] = {}
    cfg["LayerName"][ln].update({
        "table_range_power2": lncfg["table_range_power2"],
        "Precision": {
            "result": "ap_fixed<16,6>",
            "accum":  lncfg["accum"],
        },
    })
    if "table_t" not in cfg["LayerName"][ln].get("Precision", {}):
        cfg["LayerName"][ln]["Precision"]["table_t"] = lncfg["table"]

dense_result_prec = {
    "input_proj":            "ap_fixed<16,6>",
    "ds_block_0_fc1":        "ap_fixed<16,11>",
    "ds_block_0_fc2":        "ap_fixed<16,9>",
    "ds_block_0_fc2_linear": "ap_fixed<16,9>",
    "ds_block_0_add":        "ap_fixed<16,9>",
    "ds_block_1_fc1":        "ap_fixed<16,8>",
    "ds_block_1_fc2":        "ap_fixed<16,7>",
    "ds_block_1_fc2_linear": "ap_fixed<16,7>",
    "ds_block_1_add":        "ap_fixed<16,9>",
    "ds_block_2_fc1":        "ap_fixed<16,8>",
    "ds_block_2_fc2":        "ap_fixed<16,8>",
    "ds_block_2_fc2_linear": "ap_fixed<16,8>",
    "ds_block_2_add":        "ap_fixed<16,9>",
    "head_fc1":              "ap_fixed<16,9>",
    "head_fc2":              "ap_fixed<16,8>",
    "head_fc2_linear":       "ap_fixed<16,8>",
    "global_average_pooling1d": "ap_fixed<16,9>",
}
no_ternary = {"input_proj", "head_fc2", "head_fc2_linear", "global_average_pooling1d",
              "ds_block_0_fc2_linear", "ds_block_1_fc2_linear", "ds_block_2_fc2_linear",
              "ds_block_0_add", "ds_block_1_add", "ds_block_2_add"}
for layer_name, prec in dense_result_prec.items():
    if layer_name not in cfg["LayerName"]:
        cfg["LayerName"][layer_name] = {}
    if "Precision" not in cfg["LayerName"][layer_name]:
        cfg["LayerName"][layer_name]["Precision"] = {}
    cfg["LayerName"][layer_name]["Precision"]["result"] = prec
    if layer_name not in no_ternary:
        cfg["LayerName"][layer_name]["Precision"]["weight"] = "ap_int<2>"

for key in ("weight", "bias"):
    cfg["LayerName"]["head_fc2"]["Precision"][key] = "ap_fixed<16,8>"

dense_accum_prec = {
    "ds_block_0_fc1": "ap_fixed<24,10>",
    "ds_block_0_fc2": "ap_fixed<24,12>",
    "ds_block_1_fc1": "ap_fixed<24,10>",
    "ds_block_1_fc2": "ap_fixed<24,10>",
    "ds_block_2_fc1": "ap_fixed<24,10>",
    "ds_block_2_fc2": "ap_fixed<24,12>",
    "head_fc1":       "ap_fixed<24,10>",
    "head_fc2":       "ap_fixed<24,12>",
}
for layer_name, prec in dense_accum_prec.items():
    if layer_name not in cfg["LayerName"]:
        cfg["LayerName"][layer_name] = {}
    if "Precision" not in cfg["LayerName"][layer_name]:
        cfg["LayerName"][layer_name]["Precision"] = {}
    cfg["LayerName"][layer_name]["Precision"]["accum"] = prec

# ── Convert (regenerate project files) ───────────────────────────────────────
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

# ── Patch reuse_factor in parameters.h (RF=64 prevents scheduler crash) ──────
_params_h = f"{HLS_DIR}/firmware/parameters.h"
with open(_params_h) as _f:
    _ph = _f.read()
_ph = _ph.replace('static const unsigned reuse_factor = 1;',
                  'static const unsigned reuse_factor = 64;')
with open(_params_h, 'w') as _f:
    _f.write(_ph)
print(f"Patched {_params_h}: reuse_factor=64")

# ── Patch array partition threshold in both TCL files ────────────────────────
import re as _re
for _tcl_path in [f"{HLS_DIR}/project.tcl", f"{HLS_DIR}/build_prj.tcl"]:
    with open(_tcl_path) as _f:
        _tcl = _f.read()
    _tcl = _re.sub(r'set maximum_size \d+', 'set maximum_size 16384', _tcl)
    _tcl = _re.sub(r'catch \{config_array_partition -maximum_size \$maximum_size\}',
                   'config_array_partition -maximum_size 16384', _tcl)
    with open(_tcl_path, 'w') as _f:
        _f.write(_tcl)
    print(f"Patched {_tcl_path}")

# ── Synthesis ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running Vivado HLS synthesis (~30-60 min)...")
print("="*60)
report = hls_model.build(csim=False, synth=True, cosim=False, export=False)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SYNTHESIS RESULTS")
print("="*60)
if report and "CSynthesisReport" in report:
    r = report["CSynthesisReport"]
    print(f"Target clock:  {CLOCK_NS} ns  ({1000/CLOCK_NS:.0f} MHz)")
    print(f"Estimated:     {r.get('EstimatedClockPeriod','N/A')} ns")
    print(f"Latency:       {r.get('LatencyMin','N/A')}–{r.get('LatencyMax','N/A')} cycles")
    print(f"Interval:      {r.get('IntervalMin','N/A')}–{r.get('IntervalMax','N/A')} cycles")
    print()
    print("Resource utilization:")
    print(f"  {'Resource':<12} {'Used':>8}  {'Available':>10}  {'%':>6}")
    print(f"  {'-'*40}")
    for res in ["BRAM_18K", "DSP48E", "FF", "LUT"]:
        used  = r.get(res, "N/A")
        avail = r.get(f"{res}_AVAILABLE", "N/A")
        try:
            pct = f"{100*int(used)/int(avail):.1f}%"
        except Exception:
            pct = "N/A"
        print(f"  {res:<12} {str(used):>8}  {str(avail):>10}  {pct:>6}")
else:
    print("Report:", report)
    print(f"Check {HLS_DIR}/myproject_prj/solution1/syn/report/ for raw reports")

print(f"\nFull project: {HLS_DIR}/")
