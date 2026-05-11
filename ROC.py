import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib
import h5py
from numpy import expand_dims
import numpy as np
matplotlib.use("Agg")
import sys, os, numpy
import tensorflow
from qkerasModel import (AbsMeanQuantizer, BitLinear, RMSNorm,
                         BitMHSA, BitFFN, BitTransformerBlock)

SIG_TEST = ("/home/users/russelld/L1JetTagDaniel/hls4mlModifications"
            "/10-08-23/02-02_datasets/ReversedPhi_Eta/4c_4b_testData.h5")
QCD_TEST = ("/home/users/russelld/L1JetTagDaniel/hls4mlModifications"
            "/10-08-23/02-02_datasets/ReversedPhi_Eta/QCD"
            "/testingDatapt20_vDter_wEdits4ff.h5")
BITNET_MODEL = (
    "models/transformer_d64_l3_ffn128_kd/"
    "noNorm_train_d64_l3_ffn128_bitnetJetTagModel.h5"
)

with h5py.File(SIG_TEST, "r") as hf:
    dataset = hf["Testing Data"][:]
with h5py.File(QCD_TEST, "r") as hf:
    datasetQCD = hf["Testing Data"][:]

dataset = np.concatenate((dataset, datasetQCD))
np.random.shuffle(dataset)

N_PART_PER_JET = 10
N_FEAT = 14
# label is at column 140; particle features are columns 0-139
A = dataset[:, 0:140].reshape(-1, N_PART_PER_JET, N_FEAT)
b = dataset[:, 140]

# Second Dataset
#with h5py.File("/data/t3home000/aidandc/testingDataHHThreeV.h5", "r") as hf:
 #   dataset1 = hf["Testing Data"][:]

#A1 = dataset1[:, 0 : len(dataset1[0]) - 1]
#b1 = dataset1[:, len(dataset1[0]) - 1]
#A1 = expand_dims(A1, axis=3)

# Third Dataset
#with h5py.File("/data/t3home000/aidandc/testingDataHHFiveV.h5", "r") as hf:
 #   dataset2 = hf["Testing Data"][:]

#A2 = dataset2[:, 0 : len(dataset2[0]) - 1]
#b2 = dataset2[:, len(dataset2[0]) - 1]
#A2 = expand_dims(A2, axis=3)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

# Create plot for ROC
plt.figure(1)
#plt.plot([0, 1], [0, 1], "k--")

# Load in respective model for the datasets
#model1 = load_model("modelOne.h5")
#model2 = load_model("modelTwo.h5")
#model3 = load_model("modelThree.h5")
custom_objects = {
    "AbsMeanQuantizer": AbsMeanQuantizer,
    "BitLinear": BitLinear,
    "RMSNorm": RMSNorm,
    "BitMHSA": BitMHSA,
    "BitFFN": BitFFN,
    "BitTransformerBlock": BitTransformerBlock,
}
model1 = load_model(BITNET_MODEL, custom_objects=custom_objects, compile=False)

# Creating ROC curves based on model predictions for each dataset
import tensorflow as tf
Ab_pred_keras = tf.sigmoid(model1.predict(A)).numpy().ravel()
fpr_Ab, tpr_Ab, thresholds_Ab = roc_curve(b, Ab_pred_keras)
auc_Ab = auc(fpr_Ab, tpr_Ab)
plt.plot(fpr_Ab, tpr_Ab, label="BitNet d64 l3, AUC={:.3f}".format(auc_Ab))

#fpr_Bc, tpr_Bc, thresholds_Bc = roc_curve(b1, Bc_pred_keras)
#auc_Bc = auc(fpr_Bc, tpr_Bc)
#plt.plot(fpr_Bc, tpr_Bc, label="dZ+dXY 3 Vertex (area={:.3f})".format(auc_Bc))

#Cd_pred_keras = model3.predict(A2).ravel()
#fpr_Cd, tpr_Cd, thresholds_Cd = roc_curve(b2, Cd_pred_keras)
#auc_Cd = auc(fpr_Cd, tpr_Cd)
#plt.plot(fpr_Cd, tpr_Cd, label="dZ+dXY 5 Vertex (area={:.3f})".format(auc_Cd))

# Establish labels and save image
plt.xlabel("Background Efficiency", fontsize=16)
plt.ylabel("Signal Efficiency", fontsize=16)
#plt.axvline(x=0.01, ymin=0, ymax=0.59, color="red")
#plt.axhline(y=0.6, xmin=0, xmax=0.573, color="red")
plt.title("BitNet Jet Tagger ROC Curve", fontsize=16, weight="bold")
plt.legend(loc="best")
plt.xscale("log")
plt.grid(True)
plt.savefig("ROCCurve.png")
