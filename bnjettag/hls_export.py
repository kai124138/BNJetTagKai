"""hls4ml config export + rough FPGA resource estimate for a trained model."""

import os

import numpy as np

from .layers import BitLinear


def write_hls4ml_config(model, args, tag, act_bits=8, fp_edges=True):
    """Write an hls4ml-compatible YAML config and an FPGA resource estimate.

    hls4ml (Fastml, CMS L1T group) consumes this YAML to emit Vivado HLS
    firmware.  We cannot guarantee bit-perfect synthesis without running
    hls4ml ourselves, but the config captures every precision decision so
    a hardware engineer can proceed without re-reading the Python source.

    Reference: Duarte et al. 2018 (JINST 13 P07027); Fahim et al. 2021
    (arXiv:2101.05108); hls4ml docs at https://fastmachinelearning.org/hls4ml
    """
    import yaml

    cfg_path = f"{tag}_hls4ml_config.yaml"
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

    # Per-layer precision map
    # ternary W: ap_int<2> (values -1,0,+1 fit in 2 signed bits)
    # FP32   W: ap_fixed<16,6> (typical for HLS4ML Dense in L1T context)
    # int8   A: ap_int<8>
    # fp32   A: ap_fixed<16,6>
    w_tern  = "ap_int<2>"
    w_fp    = "ap_fixed<16,6>"
    a_str   = f"ap_int<{act_bits}>" if act_bits == 8 else "ap_fixed<16,6>"

    layer_prec = {}
    fp_names = {"input_proj", "head_fc2"} if fp_edges else set()
    for layer in model.layers:
        if not hasattr(layer, "kernel"):
            continue
        is_fp = layer.name in fp_names
        wt    = w_fp if is_fp else w_tern
        bt    = w_fp  # biases always FP
        layer_prec[layer.name] = {
            "Precision": {"weight": wt, "bias": bt, "result": a_str}
        }

    # Count ternary vs FP params for resource estimate
    n_ternary_params = sum(
        int(np.prod(layer.kernel.shape))
        for layer in model.layers
        if hasattr(layer, "kernel") and layer.name not in fp_names
        and isinstance(layer, BitLinear)
    )
    n_fp_params = sum(
        int(np.prod(layer.kernel.shape))
        for layer in model.layers
        if hasattr(layer, "kernel") and layer.name in fp_names
    )

    # Rough FPGA LUT estimate:
    # ternary multiply = 2 LUTs (compare to 0, negate if -1)
    # FP16 multiply    ~ 3 DSPs or ~30 LUTs (using LUT mult)
    # We report LUTs only (no DSPs for ternary path).
    lut_ternary = n_ternary_params * 2
    lut_fp      = n_fp_params * 30
    lut_total   = lut_ternary + lut_fp
    # VU9P has 1,182,240 LUTs — give usage fraction
    vu9p_luts   = 1_182_240
    lut_pct     = 100.0 * lut_total / vu9p_luts

    resource_est = {
        "device":           "xcvu9p-flgb2104-2L-e",
        "ternary_params":   int(n_ternary_params),
        "fp_params":        int(n_fp_params),
        "lut_estimate":     int(lut_total),
        "lut_pct_vu9p":     round(lut_pct, 3),
        "note": ("Ternary weights cost ~2 LUTs each (no DSP). "
                 "FP layers estimated at 30 LUTs/param. "
                 "Actual usage depends on pipeline depth and reuse factor."),
    }

    config = {
        "backend":      "Vivado",
        "project_name": "BNJetTag",
        "output_dir":   "hls4ml_prj",
        "part":         "xcvu9p-flgb2104-2L-e",
        "clock_period": 5,
        "io_type":      "io_stream",
        "hls_config": {
            "Model": {
                "Precision":      a_str,
                "ReuseFactor":    1,
                "Strategy":       "Latency",
            },
            "LayerName": layer_prec,
        },
        "model_info": {
            "n_params":        model.count_params(),
            "input_shape":     list(model.input_shape[1:]),
            "output_shape":    list(model.output_shape[1:]),
            "weight_bits":     1,
            "activation_bits": act_bits,
            "fp_edge_layers":  list(fp_names),
            "v_eps":           float(getattr(args, "qv_eps", 2e-6)),
        },
        "resource_estimate": resource_est,
    }

    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nHLS4ML config written to {cfg_path}")
    print(f"  Estimated LUTs on VU9P: {lut_total:,}  ({lut_pct:.2f}% of {vu9p_luts:,})")
    print(f"  Weight precision  : ternary={w_tern}  FP-edge={w_fp}")
    print(f"  Activation precision: {a_str}")
    return cfg_path
