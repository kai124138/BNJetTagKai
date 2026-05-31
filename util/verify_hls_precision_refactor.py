"""Verify bnjettag.hls_precision reproduces the OLD inline configs byte-for-byte.

Strategy: stub hls4ml with a fake config_from_keras_model that returns a skeleton
config containing all the real layer names. Run (a) the OLD inline logic copied
verbatim from each script and (b) the new build_hls_config(), then deep-diff.

No TensorFlow / real hls4ml needed: the model arg is unused by the stub.
"""
import copy
import sys
import types

# All named layers the real model exposes (from the trace output).
LAYER_NAMES = [
    "input_proj", "input_norm",
    "ds_block_0_norm1", "ds_block_0_fc1", "ds_block_0_fc2",
    "ds_block_0_fc2_linear", "ds_block_0_add",
    "ds_block_1_norm1", "ds_block_1_fc1", "ds_block_1_fc2",
    "ds_block_1_fc2_linear", "ds_block_1_add",
    "ds_block_2_norm1", "ds_block_2_fc1", "ds_block_2_fc2",
    "ds_block_2_fc2_linear", "ds_block_2_add",
    "global_average_pooling1d", "final_norm",
    "head_fc1", "head_fc2", "head_fc2_linear",
]


def fresh_skeleton():
    return {
        "Model": {"Precision": {"default": None}, "Strategy": "Latency"},
        "LayerName": {name: {} for name in LAYER_NAMES},
    }


# Install a fake hls4ml module so both the module and the inline code can call it.
def install_fake_hls4ml():
    fake = types.ModuleType("hls4ml")
    fake.utils = types.ModuleType("hls4ml.utils")
    fake.converters = types.ModuleType("hls4ml.converters")
    fake.utils.config_from_keras_model = lambda model, granularity=None: fresh_skeleton()
    sys.modules["hls4ml"] = fake
    sys.modules["hls4ml.utils"] = fake.utils
    sys.modules["hls4ml.converters"] = fake.converters
    return fake


# ── OLD inline logic, io_parallel (verbatim from hls_convert_v2.py) ──
def old_io_parallel():
    import hls4ml
    LN_CONFIGS = {
        'input_norm':      {'table_range_power2':  4, 'accum': 'ap_fixed<32,10>', 'table': 'ap_fixed<18,6>'},
        'ds_block_0_norm1':{'table_range_power2':  0, 'accum': 'ap_fixed<32,15>', 'table': 'ap_fixed<16,6>'},
        'ds_block_1_norm1':{'table_range_power2': -12,'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
        'ds_block_2_norm1':{'table_range_power2': -12,'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
        'final_norm':      {'table_range_power2': -12,'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
    }
    cfg = hls4ml.utils.config_from_keras_model(None, granularity="name")
    cfg["Model"]["Precision"]["default"] = "ap_fixed<16,6>"
    for ln, lncfg in LN_CONFIGS.items():
        cfg["LayerName"][ln].update({
            "table_range_power2": lncfg["table_range_power2"],
            "table_size": 4096,
            "Precision": {
                "result":  "ap_fixed<16,6>",
                "scale":   "ap_fixed<16,6>",
                "bias":    "ap_fixed<16,6>",
                "table":   lncfg["table"],
                "table_t": lncfg["table"],
                "accum":   lncfg["accum"],
            },
        })
    dense_result_prec = {
        "input_proj":          "ap_fixed<16,6>",
        "ds_block_0_fc1":      "ap_fixed<16,11>",
        "ds_block_0_fc2":      "ap_fixed<16,9>",
        "ds_block_0_fc2_linear": "ap_fixed<16,9>",
        "ds_block_0_add":      "ap_fixed<16,9>",
        "ds_block_1_fc1":      "ap_fixed<16,8>",
        "ds_block_1_fc2":      "ap_fixed<16,7>",
        "ds_block_1_fc2_linear": "ap_fixed<16,7>",
        "ds_block_1_add":      "ap_fixed<16,9>",
        "ds_block_2_fc1":      "ap_fixed<16,8>",
        "ds_block_2_fc2":      "ap_fixed<16,8>",
        "ds_block_2_fc2_linear": "ap_fixed<16,8>",
        "ds_block_2_add":      "ap_fixed<16,9>",
        "head_fc1":            "ap_fixed<16,9>",
        "head_fc2":            "ap_fixed<16,8>",
        "head_fc2_linear":     "ap_fixed<16,8>",
        "global_average_pooling1d": "ap_fixed<16,9>",
    }
    for layer_name, prec in dense_result_prec.items():
        if layer_name not in cfg["LayerName"]:
            cfg["LayerName"][layer_name] = {}
        if "Precision" not in cfg["LayerName"][layer_name]:
            cfg["LayerName"][layer_name]["Precision"] = {}
        cfg["LayerName"][layer_name]["Precision"]["result"] = prec
        if layer_name not in ("input_proj", "head_fc2", "head_fc2_linear", "global_average_pooling1d",
                              "ds_block_0_fc2_linear", "ds_block_1_fc2_linear", "ds_block_2_fc2_linear",
                              "ds_block_0_add", "ds_block_1_add", "ds_block_2_add"):
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
    return cfg


# ── OLD inline logic, io_stream (verbatim from hls_convert_iostream.py) ──
def old_io_stream():
    import hls4ml
    LN_CONFIGS = {
        'input_norm':      {'table_range_power2':  4,  'accum': 'ap_fixed<32,16>', 'table': 'ap_fixed<18,6>'},
        'ds_block_0_norm1':{'table_range_power2':  0,  'accum': 'ap_fixed<32,15>', 'table': 'ap_fixed<16,6>'},
        'ds_block_1_norm1':{'table_range_power2': -12, 'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
        'ds_block_2_norm1':{'table_range_power2': -12, 'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
        'final_norm':      {'table_range_power2': -12, 'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
    }
    cfg = hls4ml.utils.config_from_keras_model(None, granularity="name")
    cfg["Model"]["Precision"]["default"] = "ap_fixed<16,6>"
    for ln, lncfg in LN_CONFIGS.items():
        cfg["LayerName"][ln].update({
            "table_range_power2": lncfg["table_range_power2"],
            "table_size": 4096,
            "Precision": {
                "result":  "ap_fixed<16,6>",
                "scale":   "ap_fixed<16,6>",
                "bias":    "ap_fixed<16,6>",
                "table":   lncfg["table"],
                "table_t": lncfg["table"],
                "accum":   lncfg["accum"],
            },
        })
    dense_result_prec = {k: "ap_fixed<32,16>" for k in [
        "input_proj", "ds_block_0_fc1", "ds_block_0_fc2", "ds_block_0_fc2_linear",
        "ds_block_0_add", "ds_block_1_fc1", "ds_block_1_fc2", "ds_block_1_fc2_linear",
        "ds_block_1_add", "ds_block_2_fc1", "ds_block_2_fc2", "ds_block_2_fc2_linear",
        "ds_block_2_add", "head_fc1", "head_fc2", "head_fc2_linear",
        "global_average_pooling1d"]}
    dense_result_prec["input_proj"] = "ap_fixed<32,12>"  # matches the real inline file
    for layer_name, prec in dense_result_prec.items():
        if layer_name not in cfg["LayerName"]:
            cfg["LayerName"][layer_name] = {}
        if "Precision" not in cfg["LayerName"][layer_name]:
            cfg["LayerName"][layer_name]["Precision"] = {}
        cfg["LayerName"][layer_name]["Precision"]["result"] = prec
        if layer_name not in ("input_proj", "head_fc2", "head_fc2_linear", "global_average_pooling1d",
                              "ds_block_0_fc2_linear", "ds_block_1_fc2_linear", "ds_block_2_fc2_linear",
                              "ds_block_0_add", "ds_block_1_add", "ds_block_2_add"):
            cfg["LayerName"][layer_name]["Precision"]["weight"] = "ap_int<2>"
    for key in ("weight", "bias"):
        cfg["LayerName"]["input_proj"]["Precision"][key] = "ap_fixed<32,16>"
    cfg["LayerName"]["input_proj"]["Precision"]["accum"] = "ap_fixed<48,24>"
    for key in ("weight", "bias"):
        cfg["LayerName"]["head_fc2"]["Precision"][key] = "ap_fixed<32,16>"
    dense_accum_prec = {k: "ap_fixed<48,24>" for k in [
        "ds_block_0_fc1", "ds_block_0_fc2", "ds_block_1_fc1", "ds_block_1_fc2",
        "ds_block_2_fc1", "ds_block_2_fc2", "head_fc1", "head_fc2"]}
    for layer_name, prec in dense_accum_prec.items():
        if layer_name not in cfg["LayerName"]:
            cfg["LayerName"][layer_name] = {}
        if "Precision" not in cfg["LayerName"][layer_name]:
            cfg["LayerName"][layer_name]["Precision"] = {}
        cfg["LayerName"][layer_name]["Precision"]["accum"] = prec
    return cfg


def deep_diff(a, b, path=""):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                diffs.append(f"{path}.{k}: MISSING in new")
            elif k not in b:
                diffs.append(f"{path}.{k}: EXTRA in new")
            else:
                diffs += deep_diff(a[k], b[k], f"{path}.{k}")
    elif a != b:
        diffs.append(f"{path}: old={a!r} new={b!r}")
    return diffs


def main():
    install_fake_hls4ml()
    # Import the module directly by path to avoid bnjettag/__init__.py, which
    # imports layers.py -> tensorflow (not needed and not installed here).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hls_precision", "bnjettag/hls_precision.py")
    hls_precision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hls_precision)

    failures = 0
    for name, old_fn in [("io_parallel", old_io_parallel), ("io_stream", old_io_stream)]:
        old_cfg = old_fn()
        new_cfg = hls_precision.build_hls_config(None, io_type=name)
        diffs = deep_diff(old_cfg, new_cfg, name)
        if diffs:
            failures += 1
            print(f"[FAIL] {name}: {len(diffs)} difference(s):")
            for d in diffs:
                print("   ", d)
        else:
            print(f"[OK]   {name}: new module config is byte-for-byte identical to old inline")

    # Also confirm the io_parallel inline copy in hls_build.py would now MATCH
    # (it previously diverged: it omitted scale/bias/table in the LN Precision).
    # The new module sets those, so build.py adopting the module is an upgrade.
    print("\nNote: hls_build.py previously omitted LN scale/bias/table keys; the")
    print("module sets them, so wiring build.py to the module fixes that drift.")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
