"""
Loss functions and ROC helpers for the BitNet jet tagger.

  * focal_loss         — focal binary cross-entropy from logits
  * pauc_loss_fn       — one-way partial-AUC surrogate (top-K hard negatives)
  * pauc2way_loss_fn   — two-way partial-AUC surrogate (hard neg + hard pos)
  * _tpr_at_fpr        — interpolate TPR at a target FPR on the empirical ROC
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve


def focal_loss(gamma=1.0, alpha=0.5):
    """
    Focal loss for binary classification.
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=1 down-weights easy examples moderately.
    alpha=0.5 gives equal class weighting.
    """
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        # Sigmoid probability from raw logit
        p       = tf.sigmoid(y_pred)
        p_t     = tf.where(tf.equal(y_true, 1.0), p, 1.0 - p)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        # Binary cross-entropy from logits for numerical stability
        bce     = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        focal   = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss_fn


# ══════════════════════════════════════════════════════════════════════════════
# PARTIAL-AUC LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def pauc_loss_fn(y_true, y_logit, fpr_thresh=0.01):
    """One-way pAUC surrogate via top-K hard negatives.
    Yao, Lin, Yang (2022), arXiv:2203.01505. Equivalent to LibAUC pAUCLoss 1-way."""
    pos      = tf.boolean_mask(y_logit, tf.equal(y_true, 1.0))
    neg      = tf.boolean_mask(y_logit, tf.equal(y_true, 0.0))
    def _compute():
        K        = tf.maximum(1, tf.cast(
                       tf.cast(tf.size(neg), tf.float32) * fpr_thresh, tf.int32))
        hard_neg, _ = tf.math.top_k(neg, k=K)
        diff     = tf.expand_dims(hard_neg, 0) - tf.expand_dims(pos, 1) + 1.0
        return tf.reduce_mean(tf.square(tf.nn.relu(diff)))
    return tf.cond(
        tf.logical_or(tf.equal(tf.size(neg), 0), tf.equal(tf.size(pos), 0)),
        lambda: tf.constant(0.0),
        _compute,
    )


def pauc2way_loss_fn(y_true, y_logit, fpr_thresh=0.01, tpr_floor=0.80):
    """Two-way pAUC surrogate: hard negatives (top-K FPR) + hard positives (bottom-K TPR).
    Yang et al. TPAMI 2022, arXiv:2206.11655."""
    pos      = tf.boolean_mask(y_logit, tf.equal(y_true, 1.0))
    neg      = tf.boolean_mask(y_logit, tf.equal(y_true, 0.0))
    def _compute():
        K_neg    = tf.maximum(1, tf.cast(
                       tf.cast(tf.size(neg), tf.float32) * fpr_thresh, tf.int32))
        K_pos    = tf.maximum(1, tf.cast(
                       tf.cast(tf.size(pos), tf.float32) * (1.0 - tpr_floor), tf.int32))
        hard_neg, _ = tf.math.top_k(neg, k=K_neg)
        # Bottom-K positives: negate, top-K, negate back
        hard_pos, _ = tf.math.top_k(-pos, k=K_pos)
        hard_pos    = -hard_pos
        diff     = tf.expand_dims(hard_neg, 0) - tf.expand_dims(hard_pos, 1) + 1.0
        return tf.reduce_mean(tf.square(tf.nn.relu(diff)))
    return tf.cond(
        tf.logical_or(tf.equal(tf.size(neg), 0), tf.equal(tf.size(pos), 0)),
        lambda: tf.constant(0.0),
        _compute,
    )


def _tpr_at_fpr(y_true, y_score, fpr_target):
    """Return TPR interpolated at fpr_target using the empirical ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_target, fpr, tpr))
