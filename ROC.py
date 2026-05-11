import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import h5py
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from tensorflow.keras.models import load_model

from qkerasModel import (
    AbsMeanQuantizer,
    BitFFN,
    BitLinear,
    BitMHSA,
    BitTransformerBlock,
    RMSNorm,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIG_TEST = (
    "/home/users/russelld/L1JetTagDaniel/hls4mlModifications"
    "/10-08-23/02-02_datasets/ReversedPhi_Eta/4c_4b_testData.h5"
)
QCD_TEST = (
    "/home/users/russelld/L1JetTagDaniel/hls4mlModifications"
    "/10-08-23/02-02_datasets/ReversedPhi_Eta/QCD"
    "/testingDatapt20_vDter_wEdits4ff.h5"
)
BITNET_MODEL = (
    "models/transformer_d64_l3_ffn128_kd/"
    "noNorm_train_d64_l3_ffn128_bitnetJetTagModel.h5"
)

N_PART_PER_JET = 10
N_FEAT = 14
BKG_EFF_TARGETS = (0.01, 0.001)


def tpr_at_fpr(fpr, tpr, target):
    """Interpolate signal efficiency at a fixed background efficiency."""
    return float(np.interp(target, fpr, tpr))


with h5py.File(SIG_TEST, "r") as hf:
    dataset = hf["Testing Data"][:]
with h5py.File(QCD_TEST, "r") as hf:
    dataset_qcd = hf["Testing Data"][:]

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
model = load_model(BITNET_MODEL, custom_objects=custom_objects, compile=False)

scores = tf.sigmoid(model.predict(x_test)).numpy().ravel()
fpr, tpr, _ = roc_curve(y_test, scores)
auc_value = auc(fpr, tpr)
target_tprs = {target: tpr_at_fpr(fpr, tpr, target) for target in BKG_EFF_TARGETS}

print(f"BitNet d64 l3 KD AUC: {auc_value:.6f}")
for target, eff_sig in target_tprs.items():
    print(f"Signal efficiency at bkg_eff={target:g}: {eff_sig:.6f}")

plt.figure(figsize=(8, 6))
label = (
    f"BitNet d64 l3 KD, AUC={auc_value:.3f}, "
    f"sig eff @1% bkg={target_tprs[0.01]:.3f}, "
    f"@0.1% bkg={target_tprs[0.001]:.3f}"
)
plt.plot(fpr, tpr, label=label)

for target, eff_sig in target_tprs.items():
    plt.scatter([target], [eff_sig], s=35)
    plt.annotate(
        f"{eff_sig:.3f} @ {100 * target:g}% bkg",
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
