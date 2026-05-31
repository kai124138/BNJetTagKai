"""Quick shape/weight sanity check for the jet tagger (no data needed)."""

import numpy as np

from .config import (
    N_PART_PER_JET, N_FEAT,
    FP_EDGES, QAT_ENABLED, STOCH_ROUND, ACT_QAT_ENABLED,
)
from .layers import AbsMeanQuantizer, BitLinear
from .models import (
    build_bitnet_jet_tagger, build_deepsets_jet_tagger,
    build_particle_bitnet_tagger,
)


def sanity_check(fp_edges=True, arch="bitnet"):
    """
    Verify input/output shapes, per-layer ternary/FP status, and weight values.
    Run with:  python train.py --sanity [--arch {bitnet,deepsets,particle}]
    """
    print("=" * 70)
    print(f"Jet Tagger — sanity check  (arch={arch})")
    print("=" * 70)

    FP_EDGES.assign(fp_edges)
    if arch == "deepsets":
        model = build_deepsets_jet_tagger(fp_edges=fp_edges)
    elif arch == "particle":
        model = build_particle_bitnet_tagger(fp_edges=fp_edges)
    else:
        model = build_bitnet_jet_tagger(fp_edges=fp_edges)
    model.summary()

    # Shape check
    dummy = np.random.randn(8, N_PART_PER_JET, N_FEAT).astype(np.float32)
    out   = model(dummy, training=False)
    assert out.shape == (8, 1), f"Wrong output shape: {out.shape}"
    print(f"\n✓  Input  shape : {dummy.shape}")
    print(f"✓  Output shape : {out.shape}  (raw logit, no sigmoid)")

    # Manually apply the ternary constraint using model.submodules so nested
    # BitLinear layers inside BitMHSA/BitFFN are reached (model.layers is
    # shallow — it only sees top-level layers in the functional graph).
    QAT_ENABLED.assign(True)
    STOCH_ROUND.assign(False)    # deterministic for the check
    q = AbsMeanQuantizer()
    for sub in model.submodules:
        if isinstance(sub, BitLinear):
            sub.kernel.assign(q(sub.kernel))

    # Build weight-name → submodule map to retrieve eps for nested BitLinear
    wname_to_sub = {}
    for sub in model.submodules:
        for w in sub.weights:
            if "kernel" in w.name:
                wname_to_sub[w.name] = sub

    # Per-layer table: name, ternary?, eps, params
    fp_layer_names  = {"input_proj", "head_fc2"} if fp_edges else set()
    print(f"\n{'Kernel':<40} {'Ternary?':<12} {'eps':>8} {'Params':>8}")
    print("-" * 72)
    n_ternary_layers, n_fp_layers = 0, 0
    ternary_ok = True
    seen = set()
    for layer in model.layers:
        for w in layer.weights:
            if "kernel" not in w.name or w.name in seen:
                continue
            seen.add(w.name)
            vals       = w.numpy()
            n_params_w = int(np.prod(vals.shape))
            unique_v   = np.unique(np.round(vals, 4))
            is_ternary = set(unique_v).issubset({-1.0, 0.0, 1.0})
            is_edge    = any(fp_name in w.name for fp_name in fp_layer_names)
            tern_str   = "yes" if is_ternary else "no (FP32)"
            # Retrieve eps from the actual BitLinear submodule (handles nesting)
            sub_layer = wname_to_sub.get(w.name)
            eps_str   = f"{sub_layer.eps:.0e}" if isinstance(sub_layer, BitLinear) else "—"
            print(f"  {w.name:<38} {tern_str:<12} {eps_str:>8} {n_params_w:>8,}")
            if is_edge:
                n_fp_layers += 1
                if is_ternary:
                    print(f"  ✗ {w.name} should be FP32 but is ternary!")
                    ternary_ok = False
            else:
                n_ternary_layers += 1
                if not is_ternary:
                    print(f"  ✗ {w.name}: expected ternary, got {unique_v[:5]}")
                    ternary_ok = False

    if ternary_ok:
        print("✓  Ternary/FP edge assignment is correct")

    # int8 activation quantization sanity: output must be finite and within ±10× of FP32
    ACT_QAT_ENABLED.assign(False)
    out_fp32 = model(dummy, training=False).numpy()
    ACT_QAT_ENABLED.assign(True)
    out_int8 = model(dummy, training=False).numpy()
    ACT_QAT_ENABLED.assign(False)
    assert np.all(np.isfinite(out_int8)), "int8 path produced non-finite output!"
    ratio = np.abs(out_int8) / (np.abs(out_fp32) + 1e-8)
    assert np.all(ratio < 10.0), f"int8/FP32 ratio out of bounds: max={ratio.max():.2f}"
    print("✓  int8 activation path: finite, within 10× of FP32")

    # FPGA resource estimate (rough); use submodules to capture nested BitLinear
    n_ternary_params = sum(
        int(np.prod(sub.kernel.shape))
        for sub in model.submodules
        if isinstance(sub, BitLinear) and sub.name not in fp_layer_names
    )
    n_fp_params = sum(
        int(np.prod(layer.kernel.shape))
        for layer in model.layers
        if hasattr(layer, "kernel") and layer.name in fp_layer_names
    )
    lut_est = n_ternary_params * 2 + n_fp_params * 30
    print(f"\n  FPGA resource estimate (Xilinx VU9P):")
    print(f"    ternary params : {n_ternary_params:>7,}  × 2 LUT  = {n_ternary_params*2:>7,} LUTs")
    print(f"    FP-edge params : {n_fp_params:>7,}  × 30 LUT = {n_fp_params*30:>7,} LUTs")
    print(f"    total estimate : {lut_est:>7,} LUTs  "
          f"({100.*lut_est/1_182_240:.2f}% of VU9P)")

    n_params_total = model.count_params()
    act_str = "8"
    print(f"\nBitNet jet tagger ready: {n_params_total:,} params, "
          f"{n_ternary_layers} ternary layers, {n_fp_layers} FP layers, "
          f"W1A{act_str}")
    print("=" * 70)
