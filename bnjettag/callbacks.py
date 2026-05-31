"""Training callbacks for the BitNet jet tagger."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve


class AUCReshapingCallback(tf.keras.callbacks.Callback):
    """Per-epoch positive reweighting at the operating FPR point.

    After each Stage-3 epoch:
      1. Compute score threshold τ that achieves FPR = fpr_thresh on validation.
      2. Identify positive training samples with score < τ (false negatives).
      3. Multiply their sample_weight by `boost`, clamped at `cap`.

    Panambur et al. (2023), DOI 10.1038/s41598-023-48482-x.
    """

    def __init__(self, model, X_tr, y_tr, X_vl, y_vl,
                 fpr_thresh=0.01, boost=2.0, cap=8.0):
        super().__init__()
        self._model     = model
        self.X_tr       = X_tr
        self.y_tr       = y_tr
        self.X_vl       = X_vl
        self.y_vl       = y_vl
        self.fpr_thresh = fpr_thresh
        self.boost      = boost
        self.cap        = cap
        self.sample_weights = np.ones(len(y_tr), dtype=np.float32)
        self.tau        = None

    def on_epoch_end(self, epoch, logs=None):
        # Find τ: smallest threshold at which val FPR ≤ fpr_thresh
        vl_score = tf.sigmoid(
            self._model(self.X_vl, training=False)).numpy().ravel()
        fpr_arr, _, thresh_arr = roc_curve(
            self.y_vl, vl_score, drop_intermediate=False)
        idx = int(np.searchsorted(fpr_arr, self.fpr_thresh, side="right")) - 1
        idx = max(0, min(idx, len(thresh_arr) - 1))
        self.tau = float(thresh_arr[idx])

        # Score positive training samples
        pos_mask = self.y_tr == 1.0
        tr_score = tf.sigmoid(
            self._model(self.X_tr, training=False)).numpy().ravel()
        fn_mask = pos_mask & (tr_score < self.tau)

        # Boost false-negative positives; clamp cumulative boost at cap
        self.sample_weights[fn_mask] = np.minimum(
            self.sample_weights[fn_mask] * self.boost, self.cap
        )
