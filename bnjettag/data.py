"""
Dataset loading and evaluation helpers (new May-2026 merged format).

HDF5 key contract:
  * particle files (trainPart / testPart) → key "jet_constituents"  (N, 141)
  * jet-level files (trainJet)            → key "train_jet_data"     (N, 4)
  * jet-level test files                  → key "test_jet_data"      (N, 4)

The flat (N, 141) particle array packs 10 particles × 14 features (140) plus a
trailing label column. `load_train_split` reproduces the signal+background
concatenate → shuffle → reshape pipeline used by the sweep modes.
"""

import os

import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .config import (
    N_PART_PER_JET, N_FEAT, TEST_CATEGORIES, DEFAULT_BKG_TEST_PART,
)
from .losses import _tpr_at_fpr


def load_train_split(args, val_frac=0.20):
    """Load + merge signal/background, shuffle, and return a train/val split.

    Mirrors the pipeline used in the LR/WD and W&B sweeps (no pT-reweighting
    or kinematics plotting — those live in the main training loop).

    Returns (X_tr, y_tr, X_vl, y_vl) as float32 arrays.
    """
    with h5py.File(args.SignalTrainFile,       "r") as hf:
        dataset    = hf["jet_constituents"][:]
    with h5py.File(args.BkgTrainFile,          "r") as hf:
        datasetQCD = hf["jet_constituents"][:]
    with h5py.File(args.sig_jetData_TrainFile, "r") as hf:
        sampleData = hf["train_jet_data"][:]
    with h5py.File(args.bkg_jetData_TrainFile, "r") as hf:
        sampleDataQCD = hf["train_jet_data"][:]

    dataset    = np.concatenate((dataset, datasetQCD))
    sampleData = np.concatenate((sampleData, sampleDataQCD))
    fullData   = np.concatenate((dataset, sampleData), axis=1)
    np.random.shuffle(fullData)
    dataset    = fullData[:, 0:141]
    fullData_X = dataset[:, :-1].reshape(-1, N_PART_PER_JET, N_FEAT).astype(np.float32)
    fullData_y = dataset[:, -1].astype(np.float32)

    n_val = int(val_frac * len(fullData_X))
    X_tr  = fullData_X[:-n_val]
    y_tr  = fullData_y[:-n_val]
    X_vl  = fullData_X[-n_val:]
    y_vl  = fullData_y[-n_val:]
    return X_tr, y_tr, X_vl, y_vl


def evaluate_test_categories(model, test_dir, bkg_test_part=DEFAULT_BKG_TEST_PART):
    """Evaluate model on every signal test category under test_dir.

    Each category is evaluated against the shared background test file so
    that y_true always contains both classes (required for roc_auc_score).

    Returns a dict keyed by "{mass}_{decay}" with sub-keys
    "auc", "tpr_1e2", "tpr_1e3" for each category found on disk.
    Missing files are silently skipped so a partial test dir still works.

    File naming convention:
        {test_dir}/{mass}_{decay}_merged_testPart.h5  →  jet_constituents (N, 141)
        bkg_test_part                                 →  jet_constituents (M, 141)
    """
    # Load background test once and reuse across all categories
    if not os.path.exists(bkg_test_part):
        print(f"  [test eval] WARNING: background test file not found: {bkg_test_part}")
        bkg_X, bkg_y = None, None
    else:
        with h5py.File(bkg_test_part, "r") as hf:
            bkg_raw = hf["jet_constituents"][:]
        bkg_X = bkg_raw[:, :-1].reshape(-1, N_PART_PER_JET, N_FEAT).astype(np.float32)
        bkg_y = bkg_raw[:, -1].astype(np.float32)

    results = {}
    for mass, decay in TEST_CATEGORIES:
        part_path = os.path.join(test_dir, f"{mass}_{decay}_merged_testPart.h5")
        if not os.path.exists(part_path):
            print(f"  [test eval] skipping {mass}_{decay} — file not found")
            continue
        with h5py.File(part_path, "r") as hf:
            sig_raw = hf["jet_constituents"][:]
        sig_X = sig_raw[:, :-1].reshape(-1, N_PART_PER_JET, N_FEAT).astype(np.float32)
        sig_y = sig_raw[:, -1].astype(np.float32)

        # Combine signal + background so both classes are present
        if bkg_X is not None:
            X_te = np.concatenate([sig_X, bkg_X], axis=0)
            y_te = np.concatenate([sig_y, bkg_y], axis=0)
        else:
            X_te, y_te = sig_X, sig_y

        prob    = tf.sigmoid(model(X_te, training=False)).numpy().ravel()
        auc     = float(roc_auc_score(y_te, prob))
        tpr_1e2 = _tpr_at_fpr(y_te, prob, 1e-2)
        tpr_1e3 = _tpr_at_fpr(y_te, prob, 1e-3)
        key = f"{mass}_{decay}"
        results[key] = {"auc": auc, "tpr_1e2": tpr_1e2, "tpr_1e3": tpr_1e3}
        print(f"  [test] {key:<16}  AUC={auc:.4f}  "
              f"TPR@1e-2={tpr_1e2:.4f}  TPR@1e-3={tpr_1e3:.4f}")
    return results
