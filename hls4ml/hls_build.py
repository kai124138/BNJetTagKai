"""Vivado HLS synthesis for the DeepSets jet tagger (io_parallel).

What it does
    Re-runs the io_parallel conversion (fast, ~1 min) with the SAME config the
    C-sim was verified against, patches reuse_factor/array-partition in the
    generated project, then calls hls_model.build(synth=True) and prints the
    estimated clock, latency, interval, and LUT/FF/DSP/BRAM utilization.

Run (from repo root, on a machine with Vivado 2020.1)
    python hls4ml/hls_build.py        # ~30-60 min

Needs
    Vivado 2020.1 (vivado_hls; path hardcoded at VIVADO_BIN — edit for your
    machine), patched hls4ml, TensorFlow, the model h5. NOTE 2023.2 uses
    vitis_hls which this script does not target.

Outputs
    Synthesized project in models/hls4ml_deepsets_v2/ and a printed resource/
    latency summary. As of 2026-05-31 this has NOT yet been run to completion —
    it is the current top item in docs/NEXT_STEPS.md. Target: II=1, latency
    < 100 cycles (see docs/paper_notes_2510.24784.md).

Precision
    All precision lives in bnjettag/hls_precision.py (io_parallel profile). This
    script only adds ReuseFactor=64 on top. Do NOT re-inline configs here.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# vivado_hls is in Vivado 2020.1 (2023.2 uses vitis_hls which is not available here)
VIVADO_BIN = "/data/software/xilinx/Vivado/2020.1/bin"
os.environ["PATH"] = VIVADO_BIN + ":" + os.environ.get("PATH", "")

import numpy as np
import tensorflow as tf
import hls4ml

from bnjettag.hls_precision import build_hls_config  # single source of precision truth

MODEL_PATH = "models/deepsets_d64_l3_ffn128/deepsets_clean.h5"
HLS_DIR    = "models/hls4ml_deepsets_v2"
PART       = "xcvu9p-flgb2104-2L-e"
CLOCK_NS   = 5

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ── Build config (single source of truth: bnjettag/hls_precision.py) ──────────
# Identical io_parallel precision to hls_convert_v2.py / hls_trace.py — the same
# config the C-sim was verified against, so synthesis can't drift from it.
cfg = build_hls_config(model, io_type="io_parallel")
cfg["Model"]["ReuseFactor"] = 64  # RF=64 prevents the HLS scheduler crash

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
