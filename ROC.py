import argparse
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import h5py
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from tensorflow.keras.models import load_model

from bnjettag.layers import (
    AbsMeanQuantizer,
    BitFFN,
    BitLinear,
    BitMHSA,
    BitTransformerBlock,
    RMSNorm,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# New May-2026 dataset (key "jet_constituents", shape N×141; label in column 140).
# Paths mirror bnjettag/config.py; override with --sig-test / --qcd-test if data moved.
_DATA_ROOT = (
    "/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM"
    "/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data"
)
DEFAULT_SIG_TEST = f"{_DATA_ROOT}/test_merged/phi60_bbbb_merged_testPart.h5"
DEFAULT_QCD_TEST = f"{_DATA_ROOT}/QCD_Pt15To3000_Flat_PU200/Bkg_test.h5"
DEFAULT_BITNET_MODEL = (
    "models/transformer_d64_l3_ffn128_kd/"
    "noNorm_train_d64_l3_ffn128_bitnetJetTagModel.h5"
)
DEFAULT_FP32_MODEL = (
    "models/transformer_fp32_d64_l3_ffn128/"
    "transformer_fp32_d64_l3_ffn128.keras"
)

N_PART_PER_JET = 10
N_FEAT = 14
BKG_EFF_TARGETS = (0.01, 0.001)


def tpr_at_fpr(fpr, tpr, target):
    """Interpolate signal efficiency at a fixed background efficiency."""
    return float(np.interp(target, fpr, tpr))


parser = argparse.ArgumentParser(description="Plot BitNet jet-tagger ROC curve.")
parser.add_argument("--sig-test", default=DEFAULT_SIG_TEST,
                    help="HDF5 file with signal test data.")
parser.add_argument("--qcd-test", default=DEFAULT_QCD_TEST,
                    help="HDF5 file with QCD background test data.")
parser.add_argument("--bitnet-model", default=DEFAULT_BITNET_MODEL,
                    help="Path to the trained BitNet model.")
parser.add_argument("--fp32-model", default=DEFAULT_FP32_MODEL,
                    help="Path to the FP32 baseline model (optional).")
args = parser.parse_args()

with h5py.File(args.sig_test, "r") as hf:
    dataset = hf["jet_constituents"][:]
with h5py.File(args.qcd_test, "r") as hf:
    dataset_qcd = hf["jet_constituents"][:]

dataset = np.concatenate((dataset, dataset_qcd))
np.random.default_rng(42).shuffle(dataset)

# Label is at column 140; particle features are columns 0-139.
x_test = dataset[:, 0:140].reshape(-1, N_PART_PER_JET, N_FEAT)
y_test = dataset[:, 140]

custom_objects = {
    "AbsMeanQuantizer": AbsMeanQuantizer,
    "BitLinear": BitLinear,
    "RMSNorm": RMSNorm,
    "BitMHSA": BitMHSA,
    "BitFFN": BitFFN,
    "BitTransformerBlock": BitTransformerBlock,
}
plt.figure(figsize=(8, 6))

models = [
    ("BitNet d64 l3 KD", args.bitnet_model, custom_objects),
]
if os.path.exists(args.fp32_model):
    models.append(("FP32 transformer d64 l3", args.fp32_model, None))
else:
    print(f"FP32 baseline not found, skipping: {args.fp32_model}")

for model_label, model_path, model_custom_objects in models:
    model = load_model(model_path, custom_objects=model_custom_objects, compile=False)
    scores = tf.sigmoid(model.predict(x_test, verbose=0)).numpy().ravel()
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_value = auc(fpr, tpr)
    target_tprs = {
        target: tpr_at_fpr(fpr, tpr, target) for target in BKG_EFF_TARGETS
    }

    print(f"{model_label} AUC: {auc_value:.6f}")
    for target, eff_sig in target_tprs.items():
        print(f"{model_label} signal efficiency at bkg_eff={target:g}: {eff_sig:.6f}")

    label = (
        f"{model_label}, AUC={auc_value:.3f}, "
        f"sig eff @1% bkg={target_tprs[0.01]:.3f}, "
        f"@0.1% bkg={target_tprs[0.001]:.3f}"
    )
    plt.plot(fpr, tpr, label=label)

    for target, eff_sig in target_tprs.items():
        plt.scatter([target], [eff_sig], s=35)
        plt.annotate(
            f"{eff_sig:.3f}",
            xy=(target, eff_sig),
            xytext=(8, -18 if target == 0.01 else 10),
            textcoords="offset points",
            fontsize=9,
        )

plt.xlabel("Background Efficiency", fontsize=16)
plt.ylabel("Signal Efficiency", fontsize=16)
plt.title("BitNet Jet Tagger ROC Curve", fontsize=16, weight="bold")
plt.legend(loc="best")
plt.xscale("log")
plt.xlim(1e-4, 1.0)
plt.ylim(0.0, 1.02)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
plt.savefig("ROCCurve.png", dpi=150)
