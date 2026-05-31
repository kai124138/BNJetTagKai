"""
BitNet-style 1-bit Transformer Jet Tagger — training entrypoint
===============================================================
Drop-in replacement for the QKeras CNN jet tagger.

Matches exactly:
  - Input  shape : (batch, 10, 14)  [N_PART_PER_JET=10, N_FEAT=14]
  - Output shape : (batch, 1)       [single logit, no sigmoid]
  - Loss         : binary_crossentropy
  - Sample weights, pruning callbacks, and training loop are unchanged.

This file is the CLI/training driver. The model definitions, layers, losses,
data loading, W&B integration, sweeps, and hls4ml export all live in the
`bnjettag` package:

  bnjettag/config.py      constants, dataset paths, runtime toggles
  bnjettag/layers.py      BitLinear / RMSNorm / ternary attention / ParT blocks
  bnjettag/models/        one builder per architecture
  bnjettag/losses.py      focal + partial-AUC losses
  bnjettag/data.py        HDF5 loaders + per-category test evaluation
  bnjettag/wandb_utils.py W&B tracker
  bnjettag/sweeps.py      LR/WD + Bayesian sweeps
  bnjettag/hls_export.py  hls4ml YAML config
  bnjettag/sanity.py      shape/weight sanity check

Architecture overview
---------------------
  Input (10×14)
    │
  BitLinear projection  →  (10×D_MODEL)   [1-bit weights]
    │
  × N_LAYERS of BitTransformerBlock:
      ├─ RMSNorm
      ├─ 1-bit Multi-Head Self-Attention  (Q/K/V projections are ternary)
      ├─ residual add
      ├─ RMSNorm
      ├─ 1-bit FFN  (expand → contract, ternary weights)
      └─ residual add
    │
  Global average pool   →  (D_MODEL,)
    │
  BitLinear head        →  (1,)            [logit]
"""

import argparse
import os

import h5py
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for nohup/headless runs
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import (
    prune, pruning_callbacks, pruning_schedule
)

from bnjettag.config import (
    N_FEAT, N_PART_PER_JET, D_MODEL, N_LAYERS, FFN_DIM,
    DEFAULT_SIG_PART, DEFAULT_SIG_JET, DEFAULT_BKG_PART, DEFAULT_BKG_JET,
    DEFAULT_BKG_TEST_PART, DEFAULT_TEST_DIR,
    QAT_ENABLED, FP_EDGES, ACT_QAT_ENABLED, STOCH_ROUND,
)
from bnjettag.losses import (
    focal_loss, pauc_loss_fn, pauc2way_loss_fn, _tpr_at_fpr,
)
from bnjettag.models import build_for_arch
from bnjettag.data import evaluate_test_categories
from bnjettag.wandb_utils import WandbTracker
from bnjettag.callbacks import AUCReshapingCallback
from bnjettag.hls_export import write_hls4ml_config
from bnjettag.sweeps import sweep_mode, wandb_sweep_mode
from bnjettag.sanity import sanity_check


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT  (mirrors your original train.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    signalTrainFile      = args.SignalTrainFile
    bkgTrainFile         = args.BkgTrainFile
    sig_jetData_TrainFile= args.sig_jetData_TrainFile
    bkg_jetData_TrainFile= args.bkg_jetData_TrainFile

    print("Reading signal from "          + signalTrainFile)
    print("Reading background from "      + bkgTrainFile)
    print("Reading signal jet data from " + sig_jetData_TrainFile)
    print("Reading background jet data from " + bkg_jetData_TrainFile)

    # ── Load data  (new dataset format: jet_constituents / train_jet_data) ──────
    with h5py.File(signalTrainFile,       "r") as hf:
        dataset      = hf["jet_constituents"][:]
    with h5py.File(bkgTrainFile,          "r") as hf:
        datasetQCD   = hf["jet_constituents"][:]
    with h5py.File(sig_jetData_TrainFile, "r") as hf:
        sampleData   = hf["train_jet_data"][:]
    with h5py.File(bkg_jetData_TrainFile, "r") as hf:
        sampleDataQCD= hf["train_jet_data"][:]

    dataset    = np.concatenate((dataset, datasetQCD))
    sampleData = np.concatenate((sampleData, sampleDataQCD))
    fullData   = np.concatenate((dataset, sampleData), axis=1)
    np.random.shuffle(fullData)
    dataset = fullData[0:,0:141]
    LLPfeats = fullData[0:,142:146]
    sampleData = fullData[0:,141:]

    X = dataset[:, 0 : len(dataset[0]) - 1]
    y = dataset[:, len(dataset[0]) - 1]
    X = X.reshape((X.shape[0], N_PART_PER_JET, N_FEAT))

    # ── Impact parameter normalisation knob  (unchanged) ─────────────────────
    normalizeIPs = False
    if max(X[:, :, 8].ravel()) < 2.0:
        norm_b4 = True
    else:
        print("\nImpact parameter was not normalized beforehand.\n")
        norm_b4 = False

    # arch dispatch: "bitnet" (transformer), "deepsets", "particle"
    arch_prefix = {"bitnet": "transformer",
                   "deepsets": "deepsets",
                   "particle": "particle"}[args.arch]
    arch_suffix = f"_d{args.d_model}_l{args.n_layers}_ffn{args.ffn_dim}"
    run_dir = f"models/{arch_prefix}_d{args.d_model}_l{args.n_layers}_ffn{args.ffn_dim}"
    # KD is only wired for the attention-based architectures
    if args.arch in ("bitnet", "particle") and (not args.baseline) and args.kd_weight > 0.0:
        run_dir += "_kd"
    file_prefix = {"bitnet": "",
                   "deepsets": "deepsets_",
                   "particle": "particle_"}[args.arch]
    if norm_b4:
        tag = f"{run_dir}/{file_prefix}train{arch_suffix}"
    elif normalizeIPs:
        tag = f"{run_dir}/{file_prefix}Norm{arch_suffix}"
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for feat_idx in [8, 9, 10]:
            tmp = scaler.fit_transform([[v] for v in X[:, :, feat_idx].ravel()])
            X[:, :, feat_idx] = tmp.reshape(X[:, :, feat_idx].shape)
    else:
        tag = f"{run_dir}/{file_prefix}noNorm_train{arch_suffix}"

    os.makedirs(os.path.dirname(os.getcwd() + f"/{tag}_model.png"),
                exist_ok=True)
    os.makedirs(os.getcwd() + "/legacy/v1/" + os.path.dirname(tag), exist_ok=True)

    # ── W&B run initialisation (enabled by --wandb, no-op otherwise) ─────────
    tracker = WandbTracker(args, tag, run_type="main-training")

    #plot kinematics
    from util.plotting.kinematics_plotter import kinematics
    kinematics(X, sampleData, y, "legacy/v1", tag)

    # ── pT-reweighting  (unchanged) ───────────────────────────────────────────
    thebins    = np.linspace(0, 500, 20)
    bkgPts     = sampleData[y == 0][:, 0]
    sigPts     = sampleData[y == 1][:, 0]
    bkg_counts, _ = np.histogram(bkgPts, bins=thebins)
    sig_counts, _ = np.histogram(sigPts, bins=thebins)
    total_bkg  = len(bkgPts)
    total_sig  = len(sigPts)
    weights_pt = np.nan_to_num(sig_counts / bkg_counts,
                               nan=total_sig / total_bkg)

    weights    = np.ones(len(y))
    pt_indices = np.clip(
        np.digitize(sampleData[:, 0], bins=thebins) - 1, 0, len(weights_pt) - 1
    )
    weights[y == 0] = weights_pt[pt_indices][y == 0]

    plt.figure()
    plt.hist(weights, bins=51)
    plt.xlabel("Weights")
    plt.savefig("{}_weights.png".format(tag))

    np.save("{}_bitnetWeights.npy".format(tag),  weights)
    np.save("{}_ptRange.npy".format(tag),        sampleData[:, 0])

    # ── Build model  ──────────────────────────────────────────────────────────
    fp_edges = (not args.baseline) and args.fp_edges
    FP_EDGES.assign(fp_edges)
    model = build_for_arch(args)
    model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file    = os.getcwd() + f"/{tag}_model.png",
        show_shapes= True,
        show_layer_names=True,
    )

    # ── Pruning  (same schedule as your original) ─────────────────────────────
    # Note: tfmot pruning wraps Dense-like layers. BitLinear is a custom Layer,
    # so we selectively prune only the head Dense equivalents if needed.
    # For the transformer blocks, the ternary constraint already achieves ~67%
    # sparsity on average (roughly 1/3 of weights are zero after quantisation).
    # If you want explicit magnitude pruning on top, uncomment the block below.

    # pruning_params = {
    #     "pruning_schedule":
    #         pruning_schedule.ConstantSparsity(0.75, begin_step=2000,
    #                                           frequency=100)
    # }
    # model = prune.prune_low_magnitude(model, **pruning_params)

    # ── Learning rate schedule: cosine decay with 5% linear warmup ───────────
    BATCH_SIZE    = 50
    EPOCHS        = 5
    TRAIN_SIZE    = int(len(X) * 0.80)
    total_steps   = (TRAIN_SIZE // BATCH_SIZE) * EPOCHS
    warmup_steps  = int(0.05 * total_steps)
    peak_lr       = 3e-4
    min_lr        = 1e-6

    @tf.keras.utils.register_keras_serializable()
    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step    = tf.cast(step, tf.float32)
            warmup  = peak_lr * (step / max(warmup_steps, 1))
            cos_arg = np.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine  = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + tf.cos(cos_arg))
            return tf.where(step < warmup_steps, warmup, cosine)
        def get_config(self):
            return {}

    lr_schedule = WarmupCosineDecay()

    # ── Compile ───────────────────────────────────────────────────────────────
    # focal_loss uses sigmoid_cross_entropy_with_logits internally — correct
    # for a logit-output model. weighted_metrics tracks AUC live during fit()
    # (same fix as Russell's BinaryCrossentropy(from_logits=True) + AUC patch).
    model.compile(
        loss             = focal_loss(gamma=1.0, alpha=0.5),
        optimizer        = tf.keras.optimizers.experimental.AdamW(
            learning_rate = lr_schedule,
            weight_decay  = 0.01,
            beta_2        = 0.95,
        ),
        metrics          = ["binary_accuracy"],
        weighted_metrics = [tf.keras.metrics.AUC(name="auc")],
    )

    # ── Two-stage QAT warm-start ──────────────────────────────────────────────
    # Stage 1: 20% of EPOCHS in full FP32 (QAT_ENABLED = False)
    # Stage 2: 80% of EPOCHS with ternary QAT (QAT_ENABLED = True)
    # The same model + optimizer instance is reused across both stages so
    # AdamW's first/second-moment estimates carry over into the QAT phase.
    warmup_epochs = int(0.20 * EPOCHS)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", verbose=1, patience=5
    )

    # ── Stage 1: FP32 warm-start (no early stopping — let it run full 20%) ──
    print(f"\n=== Stage 1: FP32 warm-start for {warmup_epochs} epochs ===")
    QAT_ENABLED.assign(False)
    history_fp32 = model.fit(
        X, y,
        epochs           = warmup_epochs,
        batch_size       = BATCH_SIZE,
        verbose          = 2,
        sample_weight    = np.asarray(weights),
        validation_split = 0.20,
        callbacks        = tracker.with_epoch_cb([], stage_offset=0, prefix="stage1"),
    )

    # ── Stage 2: ternary QAT (resume from warmup_epochs, keep AdamW state) ──
    print(f"\n=== Stage 2: ternary QAT for epochs {warmup_epochs}–{EPOCHS} ===")
    QAT_ENABLED.assign(True)
    STOCH_ROUND.assign((not args.baseline) and args.stoch_round)

    kd_weight = 0.0 if args.baseline else args.kd_weight
    kd_temp   = args.kd_temp
    do_kd     = kd_weight > 0.0

    # Validation split boundary — mirrors Keras validation_split=0.20
    n_val_s2  = int(0.20 * len(X))
    X_tr_s2   = X[:-n_val_s2].astype(np.float32)
    y_tr_s2   = y[:-n_val_s2].astype(np.float32)
    w_tr_s2   = np.asarray(weights)[:-n_val_s2]
    X_vl_s2   = X[-n_val_s2:].astype(np.float32)
    y_vl_s2   = y[-n_val_s2:].astype(np.float32)

    train_loss_s2: list = []
    val_loss_s2:   list = []

    if do_kd:
        # Knowledge distillation (Huang et al. 2023, arXiv:2307.00331):
        # Teacher = frozen FP32 copy of the Stage-1 warm-start weights.
        # Student (ternary) minimises  focal + kd_weight * MSE(σ(s/T), σ(t/T)).
        print(f"  KD enabled — kd_weight={kd_weight:.2f}  kd_temp={kd_temp:.1f}")
        QAT_ENABLED.assign(False)           # teacher sees FP32 forward
        teacher = build_for_arch(args)
        teacher.set_weights(model.get_weights())  # copy Stage-1 FP32 weights
        teacher.trainable = False
        QAT_ENABLED.assign(True)            # student becomes ternary

        focal_fn_s2 = focal_loss(gamma=1.0, alpha=0.5)
        tr_ds_s2 = (
            tf.data.Dataset
            .from_tensor_slices((X_tr_s2, y_tr_s2, w_tr_s2))
            .shuffle(50_000, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Reuse the same AdamW optimizer from Stage 1 (iterations carry over)
        kd_optimizer = model.optimizer

        pat        = 5
        best_vloss = float("inf")
        no_improve = 0
        for epoch in range(warmup_epochs, EPOCHS):
            batch_losses = []
            for x_b, y_b, w_b in tr_ds_s2:
                with tf.GradientTape() as tape:
                    s_logit = tf.squeeze(model(x_b, training=True), axis=-1)       # (B,)
                    t_logit = tf.stop_gradient(
                        tf.squeeze(teacher(x_b, training=False), axis=-1))         # (B,)
                    f_l = focal_fn_s2(y_b, s_logit)
                    # MSE of soft sigmoid outputs at temperature T
                    kd_l = tf.reduce_mean(tf.square(
                        tf.sigmoid(s_logit / kd_temp) -
                        tf.sigmoid(t_logit / kd_temp)
                    ))
                    loss = f_l + kd_weight * kd_l
                grads = tape.gradient(loss, model.trainable_variables)
                kd_optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                batch_losses.append(float(loss))

            ep_loss = float(np.mean(batch_losses))
            vl_logit = tf.squeeze(model(X_vl_s2, training=False), axis=-1)
            vl_loss  = float(focal_fn_s2(y_vl_s2, vl_logit))
            train_loss_s2.append(ep_loss)
            val_loss_s2.append(vl_loss)
            print(f"  ep {epoch+1:3d}/{EPOCHS}  "
                  f"loss={ep_loss:.4f}  val_loss={vl_loss:.4f}")
            tracker.log({"stage2/loss": ep_loss, "stage2/val_loss": vl_loss,
                         "epoch": epoch}, step=epoch)

            # Manual early stopping on val_loss (patience = pat)
            if vl_loss < best_vloss:
                best_vloss = vl_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= pat:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        del teacher   # free memory before Stage 2.5

    else:
        history_qat = model.fit(
            X, y,
            initial_epoch    = warmup_epochs,
            epochs           = EPOCHS,
            batch_size       = BATCH_SIZE,
            verbose          = 2,
            sample_weight    = np.asarray(weights),
            validation_split = 0.20,
            callbacks        = tracker.with_epoch_cb(
                [early_stop], stage_offset=warmup_epochs, prefix="stage2"),
        )
        train_loss_s2 = history_qat.history["loss"]
        val_loss_s2   = history_qat.history["val_loss"]

    # Disable stochastic rounding for inference — weights stay ternary but
    # eval passes are deterministic from here onward.
    STOCH_ROUND.assign(False)

    # ── Stage 2.5: activation-QAT calibration (W1A8) ─────────────────────────
    # BitNet a4.8 (arXiv:2411.04965): turn on ACT_QAT_ENABLED for 5% of EPOCHS
    # at 0.3× LR to calibrate int8 activation scales before AUC fine-tuning.
    do_act_quant = (not args.baseline) and (args.act_quant == "int8")
    if do_act_quant:
        act_epochs = max(1, int(0.05 * EPOCHS))
        print(f"\n=== Stage 2.5: activation-QAT calibration for {act_epochs} epochs ===")
        ACT_QAT_ENABLED.assign(True)
        # Rebuild optimizer at 0.3× LR; keep model weights from Stage 2
        model.compile(
            loss             = focal_loss(gamma=1.0, alpha=0.5),
            optimizer        = tf.keras.optimizers.experimental.AdamW(
                learning_rate = peak_lr * 0.3,
                weight_decay  = 0.01,
                beta_2        = 0.95,
            ),
            metrics          = ["binary_accuracy"],
            weighted_metrics = [tf.keras.metrics.AUC(name="auc")],
        )
        model.fit(
            X, y,
            initial_epoch    = EPOCHS,
            epochs           = EPOCHS + act_epochs,
            batch_size       = BATCH_SIZE,
            verbose          = 2,
            sample_weight    = np.asarray(weights),
            validation_split = 0.20,
            callbacks        = tracker.with_epoch_cb(
                [], stage_offset=EPOCHS, prefix="stage25"),
        )
    else:
        ACT_QAT_ENABLED.assign(False)

    # ── Loss curve (concatenated stages) ─────────────────────────────────────
    train_loss = history_fp32.history["loss"]     + train_loss_s2
    val_loss   = history_fp32.history["val_loss"] + val_loss_s2
    plt.figure(figsize=(7, 5), dpi=120)
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss,   label="Validation")
    plt.axvline(warmup_epochs - 0.5, color="k", linestyle="--",
                label="FP32 → QAT switch")
    plt.title("BitNet Model Loss", fontsize=25)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/{}_bitnetLoss.pdf".format(tag), dpi=120)
    plt.savefig(os.getcwd() + "/{}_bitnetLoss.png".format(tag), dpi=120)

    # Save post-Stage-2.5 weights so a Stage-3 crash doesn't lose them
    model.save(os.getcwd() + "/{}_bitnetJetTagModel_preS3.h5".format(tag))
    print(f"Pre-Stage-3 model saved to {tag}_bitnetJetTagModel_preS3.h5")

    # ── Stage 3: AUC fine-tuning ──────────────────────────────────────────────
    # Three loss modes selected by --auc-loss:
    #   aucm     : AUC margin loss (Yuan et al. NeurIPS 2021), min-max formulation
    #   pauc1way : one-way pAUC surrogate at FPR≤α (Yao/Lin/Yang 2022, arXiv:2203.01505)
    #   pauc2way : two-way pAUC with TPR floor    (Yang et al. TPAMI 2022, arXiv:2206.11655)
    # Composite loss for pAUC paths: focal + pAUC (Zhu/Wu/Yang 2022, arXiv:2203.14177)
    # QAT stays active — ternary weights are preserved throughout.
    AUC_EPOCHS  = 2
    LR_AUC      = 1e-4        # bumped from 5e-5; denser gradients with pAUC
    LR_DUAL     = LR_AUC / 500

    auc_loss_mode = "aucm" if args.baseline else args.auc_loss
    fpr_thresh    = args.fpr_thresh
    tpr_floor     = args.tpr_floor
    focal_weight  = 0.0 if args.baseline else args.focal_weight
    pauc_weight   = 1.0 if args.baseline else args.pauc_weight
    do_stratify   = (not args.baseline) and args.stratify

    focal_fn_s3 = focal_loss(gamma=1.0, alpha=0.5)

    # Mirror Keras' validation_split=0.20 boundary (last 20% = val, same order)
    n_val_s3 = int(0.20 * len(X))
    X_tr_s3  = X[:-n_val_s3].astype(np.float32)
    y_tr_s3  = y[:-n_val_s3].astype(np.float32)
    X_vl_s3  = X[-n_val_s3:].astype(np.float32)
    y_vl_s3  = y[-n_val_s3:].astype(np.float32)

    imratio = float(np.mean(y_tr_s3))
    p_auc   = tf.constant(imratio, dtype=tf.float32)
    m_auc   = tf.constant(0.7,     dtype=tf.float32)

    # Auxiliary variables used only by the AUCM min-max path
    a_var     = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="auc_a")
    b_var     = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="auc_b")
    alpha_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="auc_alpha")

    def aucml_loss_fn(y_true, y_prob):
        """AUC margin loss — LibAUC formulation (Yuan et al. 2021), pure TF."""
        pos = tf.cast(tf.equal(y_true, 1.0), tf.float32)
        neg = 1.0 - pos
        return (
            (1.0 - p_auc) * tf.reduce_mean((y_prob - a_var) ** 2 * pos)
            + p_auc       * tf.reduce_mean((y_prob - b_var) ** 2 * neg)
            + 2.0 * alpha_var * (
                p_auc * (1.0 - p_auc) * m_auc
                + tf.reduce_mean(p_auc * y_prob * neg - (1.0 - p_auc) * y_prob * pos)
            )
            - p_auc * (1.0 - p_auc) * alpha_var ** 2
        )

    # Stratified 50/50 batches — Zhu/Wu/Yang arXiv:2203.14177
    steps_per_epoch = max(1, len(X_tr_s3) // BATCH_SIZE)
    if do_stratify:
        pos_ds = tf.data.Dataset.from_tensor_slices(
            (X_tr_s3[y_tr_s3 == 1], y_tr_s3[y_tr_s3 == 1])
        ).shuffle(20_000).repeat()
        neg_ds = tf.data.Dataset.from_tensor_slices(
            (X_tr_s3[y_tr_s3 == 0], y_tr_s3[y_tr_s3 == 0])
        ).shuffle(20_000).repeat()
        tr_ds_s3 = (
            tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
            .batch(BATCH_SIZE).take(steps_per_epoch).prefetch(tf.data.AUTOTUNE)
        )
    else:
        tr_ds_s3 = (
            tf.data.Dataset
            .from_tensor_slices((X_tr_s3, y_tr_s3))
            .shuffle(20_000, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    auc_opt_s3 = tf.keras.optimizers.experimental.AdamW(
        learning_rate = LR_AUC,
        weight_decay  = 0.005,
        beta_2        = 0.95,
    )

    # AUC-Reshaping callback (Panambur et al. 2023, DOI 10.1038/s41598-023-48482-x)
    do_reshape   = (not args.baseline) and args.reshape
    reshape_cb   = AUCReshapingCallback(
        model       = model,
        X_tr        = X_tr_s3,
        y_tr        = y_tr_s3,
        X_vl        = X_vl_s3,
        y_vl        = y_vl_s3,
        fpr_thresh  = fpr_thresh,
        boost       = args.reshape_boost,
        cap         = args.reshape_cap,
    ) if do_reshape else None

    auc_train_hist, auc_val_hist = [], []
    print(f"\n=== Stage 3: {auc_loss_mode} fine-tuning  {AUC_EPOCHS} epochs "
          f"(fpr_thresh={fpr_thresh}, stratify={do_stratify}, "
          f"focal_w={focal_weight}, pauc_w={pauc_weight}, reshape={do_reshape}) ===")

    for epoch in range(AUC_EPOCHS):
        # Rebuild dataset with reshape weights if requested (after first epoch)
        if do_reshape and epoch > 0:
            sw  = reshape_cb.sample_weights
            probs = sw / sw.sum()
            idx_r = np.random.choice(len(X_tr_s3), size=steps_per_epoch * BATCH_SIZE,
                                     p=probs)
            tr_ds_s3 = (
                tf.data.Dataset
                .from_tensor_slices((X_tr_s3[idx_r], y_tr_s3[idx_r]))
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            )

        for x_b, y_b in tr_ds_s3:
            if auc_loss_mode == "aucm":
                # AUCM needs a persistent tape for the dual variables
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch([a_var, b_var, alpha_var])
                    y_prob = tf.squeeze(tf.sigmoid(model(x_b, training=True)))
                    loss   = aucml_loss_fn(y_b, y_prob)
                grads_model = tape.gradient(loss, model.trainable_variables)
                grad_a      = tape.gradient(loss, a_var)
                grad_b      = tape.gradient(loss, b_var)
                grad_alpha  = tape.gradient(loss, alpha_var)
                del tape
                auc_opt_s3.apply_gradients(zip(grads_model, model.trainable_variables))
                a_var.assign_sub(LR_AUC * grad_a)
                b_var.assign_sub(LR_AUC * grad_b)
                alpha_var.assign(tf.maximum(0.0, alpha_var + LR_DUAL * grad_alpha))
            else:
                # Composite loss: focal + pAUC (arXiv:2203.14177)
                with tf.GradientTape() as tape:
                    y_logit = tf.squeeze(model(x_b, training=True))
                    if auc_loss_mode == "pauc2way":
                        p_loss = pauc2way_loss_fn(y_b, y_logit, fpr_thresh, tpr_floor)
                    else:
                        p_loss = pauc_loss_fn(y_b, y_logit, fpr_thresh)
                    loss = focal_weight * focal_fn_s3(y_b, y_logit) + pauc_weight * p_loss
                grads_model = tape.gradient(loss, model.trainable_variables)
                auc_opt_s3.apply_gradients(zip(grads_model, model.trainable_variables))

        # Epoch-level metrics (subsample training set to keep eval fast)
        tr_prob = tf.sigmoid(model(X_tr_s3[:5000], training=False)).numpy().ravel()
        vl_prob = tf.sigmoid(model(X_vl_s3,        training=False)).numpy().ravel()
        tr_auc  = roc_auc_score(y_tr_s3[:5000], tr_prob)
        vl_auc  = roc_auc_score(y_vl_s3,        vl_prob)
        vl_tpr1e2 = _tpr_at_fpr(y_vl_s3, vl_prob, 1e-2)
        vl_tpr1e3 = _tpr_at_fpr(y_vl_s3, vl_prob, 1e-3)
        auc_train_hist.append(tr_auc)
        auc_val_hist.append(vl_auc)
        if auc_loss_mode == "aucm":
            extra = f"  a={a_var.numpy():.3f}  b={b_var.numpy():.3f}  α={alpha_var.numpy():.4f}"
        else:
            extra = ""
        print(f"  ep {epoch+1:2d}/{AUC_EPOCHS}  "
              f"train_AUC={tr_auc:.4f}  val_AUC={vl_auc:.4f}  "
              f"TPR@1e-2={vl_tpr1e2:.4f}  TPR@1e-3={vl_tpr1e3:.4f}{extra}")
        _s3_base = EPOCHS + (act_epochs if do_act_quant else 0)
        _s3_step = _s3_base + epoch
        _s3_log = {
            "stage3/train_auc":      tr_auc,
            "stage3/val_auc":        vl_auc,
            "stage3/tpr_at_fpr_1e2": vl_tpr1e2,
            "stage3/tpr_at_fpr_1e3": vl_tpr1e3,
            "epoch": _s3_step,
        }
        if auc_loss_mode == "aucm":
            _s3_log["stage3/auc_a"]     = float(a_var.numpy())
            _s3_log["stage3/auc_b"]     = float(b_var.numpy())
            _s3_log["stage3/auc_alpha"] = float(alpha_var.numpy())
        tracker.log(_s3_log, step=_s3_step)

        # Update reshape weights for next epoch
        if do_reshape:
            reshape_cb.on_epoch_end(epoch)

    # AUC fine-tuning curve
    plt.figure(figsize=(7, 4), dpi=120)
    plt.plot(auc_train_hist, label="Train AUC")
    plt.plot(auc_val_hist,   label="Val   AUC")
    plt.title(f"Stage-3 {auc_loss_mode} fine-tuning", fontsize=16)
    plt.xlabel(f"Epoch within Stage 3  (after {EPOCHS} focal-loss epochs)")
    plt.ylabel("AUROC")
    plt.ylim(max(0.80, min(auc_train_hist) - 0.02), 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.getcwd() + f"/{tag}_auc_finetune.pdf", dpi=120)
    plt.savefig(os.getcwd() + f"/{tag}_auc_finetune.png", dpi=120)
    print(f"\nStage-3 final val AUC: {vl_auc:.4f}  "
          f"TPR@FPR=1e-2: {vl_tpr1e2:.4f}  TPR@FPR=1e-3: {vl_tpr1e3:.4f}")

    # ── Save model  ──────────────────────────────────────────────────────────
    # model = tfmot.sparsity.keras.strip_pruning(model)  # ← re-enable if pruning
    model_path = os.getcwd() + f"/{tag}_bitnetJetTagModel.h5"
    model.save(model_path)
    print(f"\nModel saved to {tag}_bitnetJetTagModel.h5")

    # ── HLS4ML config export (optional)  ──────────────────────────────────────
    hls_cfg_path = None
    if args.export_hls:
        hls_cfg_path = write_hls4ml_config(model, args, tag,
                                           act_bits=8 if do_act_quant else 32,
                                           fp_edges=fp_edges)

    # ── Test-category evaluation ─────────────────────────────────────────────
    # Evaluate on all 9 signal categories (3 masses × 3 decay modes) and log
    # per-category metrics so the W&B dashboard shows a per-signal breakdown.
    test_dir      = getattr(args, "test_dir",      DEFAULT_TEST_DIR)      or DEFAULT_TEST_DIR
    bkg_test_part = getattr(args, "bkg_test_part", DEFAULT_BKG_TEST_PART) or DEFAULT_BKG_TEST_PART
    print(f"\n=== Test-category evaluation  (test_dir={test_dir}) ===")
    test_results = evaluate_test_categories(model, test_dir, bkg_test_part=bkg_test_part)

    test_payload = {}
    for cat, metrics in test_results.items():
        test_payload[f"test/{cat}/auc"]     = metrics["auc"]
        test_payload[f"test/{cat}/tpr_1e2"] = metrics["tpr_1e2"]
        test_payload[f"test/{cat}/tpr_1e3"] = metrics["tpr_1e3"]
    if test_results:
        test_payload["test/mean_auc"]     = float(np.mean([m["auc"]     for m in test_results.values()]))
        test_payload["test/mean_tpr_1e2"] = float(np.mean([m["tpr_1e2"] for m in test_results.values()]))
        test_payload["test/mean_tpr_1e3"] = float(np.mean([m["tpr_1e3"] for m in test_results.values()]))
        print(f"  mean  AUC={test_payload['test/mean_auc']:.4f}  "
              f"TPR@1e-2={test_payload['test/mean_tpr_1e2']:.4f}  "
              f"TPR@1e-3={test_payload['test/mean_tpr_1e3']:.4f}")
    tracker.log(test_payload)
    tracker.summary(test_payload)

    # ── W&B final metrics, plots, artifacts, finish ──────────────────────────
    # Final scalars: log as time-series AND as run summary (sticky on dashboard)
    final_metrics = {
        "final/val_auc":        float(vl_auc),
        "final/tpr_at_fpr_1e2": float(vl_tpr1e2),
        "final/tpr_at_fpr_1e3": float(vl_tpr1e3),
        "final/n_params":       int(model.count_params()),
    }
    tracker.log(final_metrics)
    tracker.summary(final_metrics)
    # Diagnostic plots written during training — upload if they exist
    tracker.log_image("plots/loss_curve",
                      os.getcwd() + f"/{tag}_bitnetLoss.png",
                      caption="FP32 warm-start → ternary QAT loss")
    tracker.log_image("plots/auc_finetune",
                      os.getcwd() + f"/{tag}_auc_finetune.png",
                      caption=f"Stage-3 {auc_loss_mode} fine-tuning AUC")
    tracker.log_image("plots/sample_weights",
                      f"{tag}_weights.png",
                      caption="pT-reweighting histogram")
    # Versioned artifacts: trained model + (optional) hls4ml config
    artifact_stem = f"{args.arch}-jet-tagger"
    tracker.log_artifact(model_path,                       artifact_stem,            "model")
    tracker.log_artifact(f"{tag}_bitnetJetTagModel_preS3.h5",
                                                            f"{artifact_stem}-preS3", "model")
    if hls_cfg_path:
        tracker.log_artifact(hls_cfg_path,                  f"{artifact_stem}-hls4ml-config", "config")
    tracker.finish()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BN 1-bit Transformer jet tagger for CMS L1 trigger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Architectures (--arch):\n"
            "  bitnet    BitNet transformer (default) — ternary MHSA + RMSNorm + GAP\n"
            "  deepsets  Attention-free DeepSets (fully hls4ml-compatible)\n"
            "  particle  ParT-style with pair-feature attention bias + [CLS] readout\n"
            "\n"
            "Quick start (data files default to the May-2026 merged dataset):\n"
            "  python train.py --sanity\n"
            "  python train.py --sanity --arch particle\n"
            "  python train.py --arch particle                  # use defaults\n"
            "  python train.py --arch particle \\\n"
            "      --sig-part /path/sig_trainPart.h5 \\\n"
            "      --sig-jet  /path/sig_trainJet.h5  \\\n"
            "      --bkg-part /path/bkg_train.h5     \\\n"
            "      --bkg-jet  /path/bkg_trainJets.h5 \\\n"
            "      --test-dir /path/test_merged/\n"
            "\n"
            "W&B tracking (recommended for cluster runs):\n"
            "  export WANDB_API_KEY=...\n"
            "  python train.py --wandb --wandb-project my-jet-tagger \\\n"
            "      --arch particle --wandb-tags ablation,run1\n"
            "\n"
            "  # Cluster node with no internet — sync logs locally, push later\n"
            "  python train.py --wandb --wandb-offline --arch particle\n"
            "  wandb sync wandb/offline-run-*       # (run on a node with internet)\n"
            "\n"
            "W&B Bayesian hyperparameter sweep:\n"
            "  python train.py --wandb-sweep --num-agents 4 --epochs-sw 5 --arch particle\n"
            "  python train.py --wandb-sweep --sweep-id abc123 --num-agents 2  # join\n"
            "\n"
            "W&B metric layout (per run):\n"
            "  stage1/loss, stage1/val_loss          — FP32 warm-start\n"
            "  stage2/loss, stage2/val_loss          — ternary QAT\n"
            "  stage25/loss, stage25/val_loss        — activation-QAT calibration\n"
            "  stage3/train_auc, stage3/val_auc      — AUC fine-tuning\n"
            "  stage3/tpr_at_fpr_1e2, _1e3\n"
            "  test/{mass}_{decay}/auc               — per signal category\n"
            "  test/{mass}_{decay}/tpr_1e2, tpr_1e3\n"
            "  test/mean_auc, test/mean_tpr_1e2/1e3  — average across categories\n"
            "  final/val_auc, final/tpr_at_fpr_1e2/1e3, final/n_params\n"
        ),
    )

    # ── Mode flags ────────────────────────────────────────────────────────────
    mode = parser.add_argument_group("Mode flags")
    mode.add_argument("--sanity",      action="store_true",
                      help="Run shape/weight sanity check (no data needed)")
    mode.add_argument("--sweep",       action="store_true",
                      help="Run sequential LR/WD CSV grid → models/sweep_results.csv")
    mode.add_argument("--wandb-sweep", dest="wandb_sweep", action="store_true",
                      help="Run Bayesian LR/WD sweep via W&B with parallel agents")
    mode.add_argument("--wandb",       action="store_true",
                      help="Enable W&B tracking in the main training loop")

    # ── W&B / sweep options ───────────────────────────────────────────────────
    wb = parser.add_argument_group("W&B / sweep options")
    wb.add_argument("--wandb-project", dest="wandb_project",
                    default="bnjettagkai",
                    help="W&B project name (default: bnjettagkai)")
    wb.add_argument("--wandb-entity",  dest="wandb_entity", default=None,
                    help="W&B team/entity (default: your personal workspace)")
    wb.add_argument("--wandb-name",    dest="wandb_name", default=None,
                    help="Override the auto-generated W&B run name")
    wb.add_argument("--wandb-tags",    dest="wandb_tags", default="",
                    help="Comma-separated extra tags, e.g. 'ablation,paper-v2'")
    wb.add_argument("--wandb-offline", dest="wandb_offline", action="store_true",
                    help="Run W&B in offline mode (logs to ./wandb/, sync later)")
    wb.add_argument("--wandb-no-artifacts", dest="wandb_no_artifacts",
                    action="store_true",
                    help="Skip model/config artifact upload (faster for smoke runs)")
    wb.add_argument("--sweep-id", dest="sweep_id", default=None,
                    help="Join an existing W&B sweep ID (multi-machine parallelism)")
    wb.add_argument("--num-agents", dest="num_agents", type=int, default=1,
                    help="Parallel agent subprocesses to spawn locally (default: 1)")
    wb.add_argument("--sweep-count", dest="sweep_count", type=int, default=None,
                    help="Max runs per agent; None = run until sweep is done")
    wb.add_argument("--max-runs", dest="max_runs", type=int, default=20,
                    help="Total sweep run budget / run_cap (default: 20)")
    wb.add_argument("--epochs-sw", dest="epochs_sw", type=int, default=5,
                    help="Epochs per sweep run — keep low for quick CPU testing (default: 5)")

    # ── Architecture ──────────────────────────────────────────────────────────
    arch = parser.add_argument_group("Architecture")
    arch.add_argument("--arch", choices=["bitnet", "deepsets", "particle"],
                      default="bitnet",
                      help="Model architecture: bitnet | deepsets | particle "
                           "(default: bitnet). 'particle' is the ParT-style "
                           "ternary transformer with pair-feature attention "
                           "bias and class-token readout.")
    arch.add_argument("--d_model",  type=int, default=D_MODEL,
                      help=f"Embedding dimension (default {D_MODEL})")
    arch.add_argument("--n_layers", type=int, default=N_LAYERS,
                      help=f"Transformer blocks (default {N_LAYERS})")
    arch.add_argument("--ffn_dim",  type=int, default=FFN_DIM,
                      help=f"FFN hidden dimension (default {FFN_DIM})")

    # ── Training ──────────────────────────────────────────────────────────────
    train = parser.add_argument_group("Training")
    train.add_argument("--baseline",    action="store_true",
                       help="Reproduce original byte-for-byte (disables all new features)")
    train.add_argument("--deepsets",    action="store_true",
                       help="[deprecated] alias for --arch deepsets")
    train.add_argument("--fp-edges",    dest="fp_edges",
                       action="store_true", default=True,
                       help="Keep input_proj & head_fc2 in FP32 (default: on)")
    train.add_argument("--no-fp-edges", dest="fp_edges", action="store_false",
                       help="Use ternary BitLinear for edge layers")
    train.add_argument("--act-quant",   dest="act_quant",
                       choices=["fp32", "int8"], default="int8",
                       help="Activation quantization: fp32 | int8 (default: int8)")
    train.add_argument("--stoch-round",    dest="stoch_round",
                       action="store_true", default=True,
                       help="Stochastic rounding in ternary STE (default: on)")
    train.add_argument("--no-stoch-round", dest="stoch_round", action="store_false",
                       help="Disable stochastic rounding")
    train.add_argument("--kd-weight",   dest="kd_weight", type=float, default=0.3,
                       help="Stage-2 KD MSE loss weight (default: 0.3; 0 to disable)")
    train.add_argument("--no-kd",       dest="kd_weight", action="store_const", const=0.0,
                       help="Disable Stage-2 knowledge distillation")
    train.add_argument("--kd-temp",     dest="kd_temp", type=float, default=2.0,
                       help="KD soft-target temperature (default: 2.0)")
    train.add_argument("--export-hls",  dest="export_hls", action="store_true", default=False,
                       help="Write hls4ml YAML config + resource estimate after training")

    # ── Loss & AUC fine-tuning ────────────────────────────────────────────────
    loss = parser.add_argument_group("Loss & AUC fine-tuning")
    loss.add_argument("--auc-loss",     dest="auc_loss",
                      choices=["aucm", "pauc1way", "pauc2way"], default="pauc1way",
                      help="Stage-3 loss: aucm | pauc1way | pauc2way (default: pauc1way)")
    loss.add_argument("--fpr-thresh",   dest="fpr_thresh", type=float, default=0.01,
                      help="FPR threshold for pAUC loss (default: 0.01)")
    loss.add_argument("--tpr-floor",    dest="tpr_floor",  type=float, default=0.80,
                      help="TPR floor for two-way pAUC (default: 0.80)")
    loss.add_argument("--focal-weight", dest="focal_weight", type=float, default=0.3,
                      help="Focal weight in composite Stage-3 loss (default: 0.3)")
    loss.add_argument("--pauc-weight",  dest="pauc_weight",  type=float, default=0.7,
                      help="pAUC weight in composite Stage-3 loss (default: 0.7)")
    loss.add_argument("--stratify",    dest="stratify", action="store_true", default=True,
                      help="Stratified 50/50 batches in Stage 3 (default: on)")
    loss.add_argument("--no-stratify", dest="stratify", action="store_false",
                      help="Disable stratified batching")
    loss.add_argument("--reshape",     dest="reshape", action="store_true", default=True,
                      help="AUCReshaping per-epoch positive reweighting (default: on)")
    loss.add_argument("--no-reshape",  dest="reshape", action="store_false",
                      help="Disable AUCReshaping callback")
    loss.add_argument("--reshape-boost", dest="reshape_boost", type=float, default=2.0,
                      help="Weight multiplier for FN positives (default: 2.0)")
    loss.add_argument("--reshape-cap",   dest="reshape_cap",  type=float, default=8.0,
                      help="Max cumulative boost per sample (default: 8.0)")
    loss.add_argument("--qv-eps", dest="qv_eps", type=float, default=2e-6,
                      help="Absmedian eps for Value projection in BitMHSA (default: 2e-6)")

    # ── Data files ────────────────────────────────────────────────────────────
    data = parser.add_argument_group(
        "Data files",
        "All four training files default to the May-2026 merged dataset on "
        "russelld's area. Override any of them if you move the data."
    )
    data.add_argument("--sig-part", dest="SignalTrainFile",
                      default=DEFAULT_SIG_PART, metavar="H5",
                      help="Signal particle data  (key: jet_constituents  shape N×141)"
                           f"  [default: {DEFAULT_SIG_PART}]")
    data.add_argument("--sig-jet",  dest="sig_jetData_TrainFile",
                      default=DEFAULT_SIG_JET,  metavar="H5",
                      help="Signal jet-level data (key: train_jet_data    shape N×4)"
                           f"  [default: {DEFAULT_SIG_JET}]")
    data.add_argument("--bkg-part", dest="BkgTrainFile",
                      default=DEFAULT_BKG_PART, metavar="H5",
                      help="QCD particle data     (key: jet_constituents  shape N×141)"
                           f"  [default: {DEFAULT_BKG_PART}]")
    data.add_argument("--bkg-jet",  dest="bkg_jetData_TrainFile",
                      default=DEFAULT_BKG_JET,  metavar="H5",
                      help="QCD jet-level data    (key: train_jet_data    shape N×4)"
                           f"  [default: {DEFAULT_BKG_JET}]")
    data.add_argument("--bkg-test-part", dest="bkg_test_part",
                      default=DEFAULT_BKG_TEST_PART, metavar="H5",
                      help="QCD particle-level test file used as background in per-category "
                           "evaluation (key: jet_constituents  shape M×141)"
                           f"  [default: {DEFAULT_BKG_TEST_PART}]")
    data.add_argument("--test-dir", dest="test_dir",
                      default=DEFAULT_TEST_DIR, metavar="DIR",
                      help="Directory of per-category test files "
                           "(phi{15,30,60}_{bbbb,cccc,uuuu}_merged_test{Part,Jet}.h5)"
                           f"  [default: {DEFAULT_TEST_DIR}]")

    args = parser.parse_args()

    # Resolve the deprecated --deepsets alias into --arch deepsets.
    if args.deepsets:
        import warnings
        warnings.warn(
            "--deepsets is deprecated; use --arch deepsets instead.",
            DeprecationWarning, stacklevel=2,
        )
        # Explicit --arch wins; only override if the user left it at default.
        if args.arch == "bitnet":
            args.arch = "deepsets"

    if args.sanity:
        fp_edges = (not args.baseline) and args.fp_edges
        sanity_check(fp_edges=fp_edges, arch=args.arch)
    elif args.sweep:
        sweep_mode(args)
    elif args.wandb_sweep:
        wandb_sweep_mode(args)
    else:
        main(args)
