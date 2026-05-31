"""Single source of truth for the DeepSets hls4ml precision configuration.

Why this file exists
--------------------
The per-layer precision config (LayerNorm LUT ranges/table types + Dense
result/accum types) used to be copy-pasted inline into four scripts:
``hls4ml/hls_convert_v2.py``, ``hls4ml/hls_trace.py``, ``hls4ml/hls_build.py``
(all io_parallel) and ``hls4ml/hls_convert_iostream.py`` (io_stream). The copies
drifted (e.g. ``hls_build.py`` stopped setting the LN ``scale``/``bias``/``table``
keys, and risked baking a different config into firmware than the one the C-sim
verified). Keeping precision in ONE place removes that whole class of bug.

There are two genuinely different profiles — this is NOT pure duplication:

* ``io_parallel`` — tight types (``ap_fixed<16,x>``), used by the verified C-sim
  path (``hls_convert_v2`` / ``hls_trace``) and synthesis (``hls_build``).
* ``io_stream``   — wider types (``ap_fixed<32,16>`` etc.) with more integer
  headroom that the streaming dataflow path needs.

Both are defined below. ``build_hls_config(model, io_type=...)`` returns a fully
populated hls4ml config dict; the scripts just call it.

If you change a precision value, change it HERE — all four scripts pick it up.

See ``docs/hls4ml_precision_bugs.md`` and ``docs/GLOSSARY.md`` for the reasoning
behind each value (LUT range mismatch, accum sizing, the head_fc2 artifact).
"""

import hls4ml

TABLE_SIZE = 4096

# ── LayerNorm configs ─────────────────────────────────────────────────────────
# table_range_power2: LUT covers [0, 2^-p) for the 1/sqrt(var) inverse-sqrt table.
#   p > 0  → small variances (input_norm, ~0.009-0.046): tighten to keep resolution.
#   p < 0  → large variances (blocks 1/2, final, var up to ~3800): widen to 2^12.
# accum: must hold sum_cache2 (= dim * max_diff^2, integer range) AND k_inv=1/64
#   (>=6 frac bits) AND var=sum_cache2*k_inv without truncating small values to 0.
# table: the inverse-sqrt LUT entry type (needs enough frac bits for small results).
#
# input_norm FIX (2026-05-31): range tightened from 2^0 to 2^-4 (=0.0625). The tiny
# observed variances (~0.009-0.046) had occupied only the bottom ~4.6% of a [0,1)
# table, so the steep low-index region was undersampled and 1/sqrt(var) read ~2x
# high (output doubled). Tightening puts the full variance band across the table.

_LN_IO_PARALLEL = {
    "input_norm":       {"table_range_power2":  4, "accum": "ap_fixed<32,10>", "table": "ap_fixed<18,6>"},
    "ds_block_0_norm1": {"table_range_power2":  0, "accum": "ap_fixed<32,15>", "table": "ap_fixed<16,6>"},
    "ds_block_1_norm1": {"table_range_power2": -12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
    "ds_block_2_norm1": {"table_range_power2": -12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
    "final_norm":       {"table_range_power2": -12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
}

# io_stream needs extra integer headroom on input_norm's accum.
_LN_IO_STREAM = {
    "input_norm":       {"table_range_power2":  4, "accum": "ap_fixed<32,16>", "table": "ap_fixed<18,6>"},
    "ds_block_0_norm1": {"table_range_power2":  0, "accum": "ap_fixed<32,15>", "table": "ap_fixed<16,6>"},
    "ds_block_1_norm1": {"table_range_power2": -12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
    "ds_block_2_norm1": {"table_range_power2": -12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
    "final_norm":       {"table_range_power2": -12, "accum": "ap_fixed<32,23>", "table": "ap_fixed<24,8>"},
}

# ── Dense / pointwise result precision ──────────────────────────────────────────
# Lessons: (1) accum_t must hold N_inputs * max_input, not just the result range;
# (2) *_fc2_linear and head_fc2_linear are separate activation layers with their
#     own result type that defaults to ap_fixed<16,6> (+/-32) and must be set;
# (3) head_fc2 bias ~= -64 overflows ap_fixed<16,6> -> needs I>=7.
_DENSE_RESULT_IO_PARALLEL = {
    "input_proj":               "ap_fixed<16,6>",
    "ds_block_0_fc1":           "ap_fixed<16,11>",
    "ds_block_0_fc2":           "ap_fixed<16,9>",   # Keras [-159,121] -> I=8 (256>159)
    "ds_block_0_fc2_linear":    "ap_fixed<16,9>",
    "ds_block_0_add":           "ap_fixed<16,9>",
    "ds_block_1_fc1":           "ap_fixed<16,8>",
    "ds_block_1_fc2":           "ap_fixed<16,7>",   # Keras [-37,23] -> I=6 (64>37)
    "ds_block_1_fc2_linear":    "ap_fixed<16,7>",
    "ds_block_1_add":           "ap_fixed<16,9>",
    "ds_block_2_fc1":           "ap_fixed<16,8>",
    "ds_block_2_fc2":           "ap_fixed<16,8>",   # Keras [-99,114] -> I=7 (128>114)
    "ds_block_2_fc2_linear":    "ap_fixed<16,8>",
    "ds_block_2_add":           "ap_fixed<16,9>",
    "head_fc1":                 "ap_fixed<16,9>",
    "head_fc2":                 "ap_fixed<16,8>",   # FP32 weights; bias ~=-64 needs I=7+
    "head_fc2_linear":          "ap_fixed<16,8>",   # model output; must hold [-40,-38]
    "global_average_pooling1d": "ap_fixed<16,9>",
}

_DENSE_RESULT_IO_STREAM = {k: "ap_fixed<32,16>" for k in _DENSE_RESULT_IO_PARALLEL}
# io_stream uses a slightly narrower integer width on the float input projection's
# result than the rest (matches the original inline tune).
_DENSE_RESULT_IO_STREAM["input_proj"] = "ap_fixed<32,12>"

_DENSE_ACCUM_IO_PARALLEL = {
    "ds_block_0_fc1": "ap_fixed<24,10>",  # 64 inputs * max~6 = 384; max=512
    "ds_block_0_fc2": "ap_fixed<24,12>",  # 128 inputs * max~15 = 1920; max=2048
    "ds_block_1_fc1": "ap_fixed<24,10>",
    "ds_block_1_fc2": "ap_fixed<24,10>",
    "ds_block_2_fc1": "ap_fixed<24,10>",
    "ds_block_2_fc2": "ap_fixed<24,12>",  # 128 inputs * max~11 = 1408; max=2048
    "head_fc1":       "ap_fixed<24,10>",
    "head_fc2":       "ap_fixed<24,12>",  # 64 inputs, FP32 weights
}

_DENSE_ACCUM_IO_STREAM = {k: "ap_fixed<48,24>" for k in _DENSE_ACCUM_IO_PARALLEL}

# Layers that must NOT get ternary (ap_int<2>) weights: the float first projection,
# the float head, linear activations, GAP, and the residual adds.
NO_TERNARY = {
    "input_proj", "head_fc2", "head_fc2_linear", "global_average_pooling1d",
    "ds_block_0_fc2_linear", "ds_block_1_fc2_linear", "ds_block_2_fc2_linear",
    "ds_block_0_add", "ds_block_1_add", "ds_block_2_add",
}

_PROFILES = {
    "io_parallel": {
        "ln": _LN_IO_PARALLEL,
        "dense_result": _DENSE_RESULT_IO_PARALLEL,
        "dense_accum": _DENSE_ACCUM_IO_PARALLEL,
        "ln_io_prec": "ap_fixed<16,6>",      # result/scale/bias for LN layers
        "head_fc2_wb": "ap_fixed<16,8>",     # head_fc2 weight/bias
        "input_proj_wb": None,               # io_parallel does not widen input_proj
        "input_proj_accum": None,
    },
    "io_stream": {
        "ln": _LN_IO_STREAM,
        "dense_result": _DENSE_RESULT_IO_STREAM,
        "dense_accum": _DENSE_ACCUM_IO_STREAM,
        "ln_io_prec": "ap_fixed<16,6>",
        "head_fc2_wb": "ap_fixed<32,16>",
        "input_proj_wb": "ap_fixed<32,16>",  # io_stream keeps the float proj wide
        "input_proj_accum": "ap_fixed<48,24>",
    },
}


def build_hls_config(model, io_type="io_parallel", default_precision="ap_fixed<16,6>"):
    """Return a fully populated hls4ml ``hls_config`` dict for the DeepSets model.

    Parameters
    ----------
    model : keras.Model
        The loaded DeepSets model (used for name-granularity config extraction).
    io_type : {"io_parallel", "io_stream"}
        Which precision profile to apply. See module docstring.
    default_precision : str
        Model-level default precision.

    The two profiles are intentionally different (io_stream needs wider types);
    do not collapse them. Both are byte-for-byte equivalent to the configs the
    scripts used inline before this module existed.
    """
    if io_type not in _PROFILES:
        raise ValueError(f"io_type must be one of {sorted(_PROFILES)}, got {io_type!r}")
    p = _PROFILES[io_type]

    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    cfg["Model"]["Precision"]["default"] = default_precision

    # ── LayerNorm layers ──
    for ln, lncfg in p["ln"].items():
        cfg["LayerName"][ln]["table_range_power2"] = lncfg["table_range_power2"]
        cfg["LayerName"][ln]["table_size"] = TABLE_SIZE
        cfg["LayerName"][ln]["Precision"] = {
            "result":  p["ln_io_prec"],
            "scale":   p["ln_io_prec"],
            "bias":    p["ln_io_prec"],
            "table":   lncfg["table"],
            # table_t mirrors 'table'; required so the patched _set_type_t('table')
            # in layers.py honors our table precision instead of the backend
            # default ap_ufixed<8,5> (see docs/hls4ml_layernorm_patches.md).
            "table_t": lncfg["table"],
            "accum":   lncfg["accum"],
        }

    # ── Dense / pointwise result precision (+ ternary weights) ──
    for layer_name, prec in p["dense_result"].items():
        cfg["LayerName"].setdefault(layer_name, {})
        cfg["LayerName"][layer_name].setdefault("Precision", {})
        cfg["LayerName"][layer_name]["Precision"]["result"] = prec
        if layer_name not in NO_TERNARY:
            cfg["LayerName"][layer_name]["Precision"]["weight"] = "ap_int<2>"

    # ── input_proj: io_stream keeps the float projection wide ──
    if p["input_proj_wb"] is not None:
        for key in ("weight", "bias"):
            cfg["LayerName"]["input_proj"]["Precision"][key] = p["input_proj_wb"]
    if p["input_proj_accum"] is not None:
        cfg["LayerName"]["input_proj"]["Precision"]["accum"] = p["input_proj_accum"]

    # ── head_fc2: FP32 weight/bias (bias ~= -64 overflows ap_fixed<16,6>) ──
    for key in ("weight", "bias"):
        cfg["LayerName"]["head_fc2"]["Precision"][key] = p["head_fc2_wb"]

    # ── Dense accumulator precision ──
    for layer_name, prec in p["dense_accum"].items():
        cfg["LayerName"].setdefault(layer_name, {})
        cfg["LayerName"][layer_name].setdefault("Precision", {})
        cfg["LayerName"][layer_name]["Precision"]["accum"] = prec

    return cfg


def enable_trace(cfg):
    """Set Trace=True on every named layer (used by hls_trace.py)."""
    for ln in cfg["LayerName"]:
        cfg["LayerName"][ln]["Trace"] = True
    return cfg
