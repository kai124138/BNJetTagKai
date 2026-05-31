"""
Hyperparameter sweeps for the BitNet jet tagger.

  * sweep_mode        — sequential 3×3 LR/WD grid → models/sweep_results.csv
  * _run_single_agent — one wandb.agent worker (runs inside a child subprocess)
  * wandb_sweep_mode  — orchestrate a W&B Bayesian sweep with parallel agents
"""

import os
import sys

import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .config import QAT_ENABLED
from .losses import focal_loss, _tpr_at_fpr
from .models import build_for_arch
from .data import load_train_split


# ══════════════════════════════════════════════════════════════════════════════
# LR/WD SWEEP  (BitNet b1.58 Reloaded, arXiv:2407.09527)
# ══════════════════════════════════════════════════════════════════════════════

def sweep_mode(args):
    """Train a 3×3 LR/WD grid for 10% of EPOCHS; rank by TPR@FPR=1e-2.
    BitNet b1.58 Reloaded (arXiv:2407.09527): small-net QAT is sensitive to LR/WD."""
    import csv

    # Load data (same pipeline as main, without kinematics/pT-reweighting)
    X_tr, y_tr, X_vl, y_vl = load_train_split(args, val_frac=0.20)

    BATCH_SIZE_SW = 50
    EPOCHS_SW     = args.epochs_sw          # default 5 for quick CPU testing
    lr_grid       = [5e-5, 1e-4, 3e-4]
    wd_grid       = [1e-3, 1e-2, 5e-2]

    os.makedirs("models", exist_ok=True)
    csv_path = "models/sweep_results.csv"
    rows = []

    for lr in lr_grid:
        for wd in wd_grid:
            print(f"\n── sweep lr={lr:.0e}  wd={wd:.0e} ──")
            # Fresh model + optimizer for each config
            QAT_ENABLED.assign(False)
            m = build_for_arch(args)
            m.compile(
                loss             = focal_loss(gamma=1.0, alpha=0.5),
                optimizer        = tf.keras.optimizers.experimental.AdamW(
                    learning_rate = lr, weight_decay = wd, beta_2 = 0.95),
                metrics          = ["binary_accuracy"],
                weighted_metrics = [tf.keras.metrics.AUC(name="auc")],
            )
            # Stage 1 (20% of sweep epochs): FP32 warm-start
            s1_ep = max(1, EPOCHS_SW // 5)
            m.fit(X_tr, y_tr, epochs=s1_ep, batch_size=BATCH_SIZE_SW,
                  verbose=0, validation_split=0.10)
            # Stage 2: ternary QAT
            QAT_ENABLED.assign(True)
            hist = m.fit(X_tr, y_tr, initial_epoch=s1_ep, epochs=EPOCHS_SW,
                         batch_size=BATCH_SIZE_SW, verbose=0, validation_split=0.10)

            vl_prob = tf.sigmoid(m(X_vl, training=False)).numpy().ravel()
            auroc       = roc_auc_score(y_vl, vl_prob)
            tpr_1e2     = _tpr_at_fpr(y_vl, vl_prob, 1e-2)
            tpr_1e3     = _tpr_at_fpr(y_vl, vl_prob, 1e-3)
            final_vloss = hist.history["val_loss"][-1]
            rows.append(dict(lr=lr, wd=wd, final_val_loss=final_vloss,
                             auroc=auroc, tpr_at_fpr_1e2=tpr_1e2,
                             tpr_at_fpr_1e3=tpr_1e3))
            print(f"  → AUROC={auroc:.4f}  TPR@1e-2={tpr_1e2:.4f}  TPR@1e-3={tpr_1e3:.4f}")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lr","wd","final_val_loss","auroc",
                                           "tpr_at_fpr_1e2","tpr_at_fpr_1e3"])
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["tpr_at_fpr_1e2"])
    print(f"\nSweep results saved to {csv_path}")
    print(f"Best config: lr={best['lr']:.0e}  wd={best['wd']:.0e}"
          f"  TPR@1e-2={best['tpr_at_fpr_1e2']:.4f}  AUROC={best['auroc']:.4f}")
    QAT_ENABLED.assign(True)   # restore for any subsequent run


# ══════════════════════════════════════════════════════════════════════════════
# WANDB SWEEP  (parallel agents via subprocess.Popen — TF-safe)
# ══════════════════════════════════════════════════════════════════════════════

def _kd_finetune(model, X_tr, y_tr, args, start_ep, end_ep,
                 batch_size, kd_weight, kd_temp):
    """Ternary Stage-2 with knowledge distillation from a frozen FP32 teacher.

    Mirrors the KD recipe in train.main(): teacher = FP32 copy of the Stage-1
    warm-start weights; student (ternary) minimises
    focal + kd_weight * MSE of temperature-softened sigmoid outputs.
    """
    QAT_ENABLED.assign(False)                     # teacher forward stays FP32
    teacher = build_for_arch(args)
    teacher.set_weights(model.get_weights())      # copy Stage-1 FP32 weights
    teacher.trainable = False
    QAT_ENABLED.assign(True)                      # student becomes ternary

    focal_fn = focal_loss(gamma=1.0, alpha=0.5)
    ds = (tf.data.Dataset
          .from_tensor_slices((X_tr.astype("float32"), y_tr.astype("float32")))
          .shuffle(50_000, reshuffle_each_iteration=True)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))
    opt = model.optimizer                         # reuse Stage-1 AdamW

    for _ in range(start_ep, end_ep):
        for x_b, y_b in ds:
            with tf.GradientTape() as tape:
                s_logit = tf.squeeze(model(x_b, training=True), axis=-1)
                t_logit = tf.stop_gradient(
                    tf.squeeze(teacher(x_b, training=False), axis=-1))
                f_l  = focal_fn(y_b, s_logit)
                kd_l = tf.reduce_mean(tf.square(
                    tf.sigmoid(s_logit / kd_temp) -
                    tf.sigmoid(t_logit / kd_temp)))
                loss = f_l + kd_weight * kd_l
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
    del teacher


def _run_single_agent(args):
    """Single wandb.agent worker — runs inside each spawned subprocess.

    Each subprocess loads data independently so TF graphs are never forked.
    Reads lr/wd from wandb.config on each run; logs 4 metrics back to W&B.
    """
    import wandb

    # ── Load data ─────────────────────────────────────────────────────────────
    X_tr, y_tr, X_vl, y_vl = load_train_split(args, val_frac=0.20)

    BATCH_SIZE_SW = 50
    EPOCHS_SW     = args.epochs_sw

    # Honour --wandb-offline inside child agents (must be set before init)
    if getattr(args, "wandb_offline", False):
        os.environ.setdefault("WANDB_MODE", "offline")

    sweep_tags = [f"arch:{args.arch}", f"d{args.d_model}",
                  f"l{args.n_layers}", f"ffn{args.ffn_dim}", "sweep"]
    if (args.wandb_tags or "").strip():
        sweep_tags += [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    def train_run():
        # Only non-swept context goes in the init config. lr / wd / d_model /
        # n_layers / batch_size / kd_weight are chosen by the sweep controller
        # and read from run.config below.
        with wandb.init(tags=sweep_tags,
                        config={"arch": args.arch,
                                "ffn_dim": args.ffn_dim,
                                "baseline": args.baseline}) as run:
            cfg = run.config
            lr         = cfg.lr
            wd         = cfg.wd
            batch_size = int(cfg.get("batch_size", BATCH_SIZE_SW))
            kd_weight  = float(cfg.get("kd_weight", 0.0))
            kd_temp    = float(getattr(args, "kd_temp", 2.0))

            # Apply swept architecture knobs before building the model.
            args.d_model  = int(cfg.get("d_model",  args.d_model))
            args.n_layers = int(cfg.get("n_layers", args.n_layers))

            QAT_ENABLED.assign(False)
            m = build_for_arch(args)
            m.compile(
                loss             = focal_loss(gamma=1.0, alpha=0.5),
                optimizer        = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=lr, weight_decay=wd, beta_2=0.95),
                metrics          = ["binary_accuracy"],
                weighted_metrics = [tf.keras.metrics.AUC(name="auc")],
            )

            # Stage 1: FP32 warm-start (~20% of sweep epochs)
            s1_ep = max(1, EPOCHS_SW // 5)
            m.fit(X_tr, y_tr, epochs=s1_ep, batch_size=batch_size,
                  verbose=0, validation_split=0.10)

            # Stage 2: ternary QAT. With kd_weight > 0, distil from a frozen
            # FP32 copy of the Stage-1 weights (same recipe as train.main()).
            if kd_weight > 0.0:
                _kd_finetune(m, X_tr, y_tr, args, s1_ep, EPOCHS_SW,
                             batch_size, kd_weight, kd_temp)
            else:
                QAT_ENABLED.assign(True)
                m.fit(X_tr, y_tr, initial_epoch=s1_ep, epochs=EPOCHS_SW,
                      batch_size=batch_size, verbose=0, validation_split=0.10)

            focal_eval  = focal_loss(gamma=1.0, alpha=0.5)
            vl_logit    = tf.squeeze(m(X_vl, training=False), axis=-1)
            vl_prob     = tf.sigmoid(vl_logit).numpy().ravel()
            auroc       = float(roc_auc_score(y_vl, vl_prob))
            tpr_1e2     = _tpr_at_fpr(y_vl, vl_prob, 1e-2)
            tpr_1e3     = _tpr_at_fpr(y_vl, vl_prob, 1e-3)
            final_vloss = float(focal_eval(y_vl.astype("float32"), vl_logit))

            wandb.log({
                "auroc":          auroc,
                "tpr_at_fpr_1e2": tpr_1e2,
                "tpr_at_fpr_1e3": tpr_1e3,
                "final_val_loss": final_vloss,
            })
            print(f"  run done  d{args.d_model} l{args.n_layers} "
                  f"bs{batch_size} kd{kd_weight:.2f}  lr={lr:.0e} wd={wd:.0e}"
                  f"  AUROC={auroc:.4f}  TPR@1e-2={tpr_1e2:.4f}")

    wandb.agent(args.sweep_id, function=train_run,
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                count=args.sweep_count)


def wandb_sweep_mode(args):
    """Orchestrate a W&B Bayesian sweep with parallel subprocess agents.

    Parent path  (num_agents > 1, or no sweep_id yet):
      1. Validate data paths.
      2. Create sweep (or join existing via --sweep-id).
      3. Spawn N child subprocesses each targeting that sweep.
      4. Wait for all children.

    Child path  (sweep_id set AND num_agents == 1):
      Calls _run_single_agent() directly — no further forking.

    This split avoids forking TensorFlow/CUDA contexts, which corrupts GPU
    memory and causes silent graph errors with multiprocessing.Pool.

    Multi-machine parallelism:
      Run once without --sweep-id → note the printed ID.
      Re-run on each machine with --sweep-id <id> --num-agents <N>.

    Ref: https://docs.wandb.ai/models/sweeps/parallelize-agents
    """
    import subprocess
    import wandb

    # ── Child process path: run a single agent directly ───────────────────────
    if args.sweep_id and args.num_agents == 1:
        _run_single_agent(args)
        return

    # ── Parent path: validate data, create sweep, spawn children ─────────────
    for attr in ("SignalTrainFile", "BkgTrainFile",
                 "sig_jetData_TrainFile", "bkg_jetData_TrainFile"):
        path = getattr(args, attr)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    sweep_config = {
        "method":  "bayes",
        "metric":  {"name": "tpr_at_fpr_1e2", "goal": "maximize"},
        "run_cap": args.max_runs,
        "parameters": {
            "lr":         {"values": [5e-5, 1e-4, 3e-4]},
            "wd":         {"values": [1e-3, 1e-2, 5e-2]},
            "d_model":    {"values": [32, 64, 128]},
            "n_layers":   {"values": [2, 4, 6]},
            "batch_size": {"values": [256, 512, 1024]},
            "kd_weight":  {"distribution": "uniform", "min": 0.0, "max": 0.5},
        },
    }

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Joining existing W&B sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config,
                               project=args.wandb_project,
                               entity=args.wandb_entity or None)
        print(f"Initialised W&B sweep: {sweep_id}")
        print(f"  Re-run on another machine with: --wandb-sweep --sweep-id {sweep_id}")

    # ── Build base command forwarding all relevant flags to children ──────────
    # sys.argv[0] is the entry script (train.py); each child re-enters it as a
    # single agent. Using __file__ here would point at this module, which has
    # no __main__ guard.
    entry = os.path.abspath(sys.argv[0])
    base_cmd = [
        sys.executable, entry,
        "--wandb-sweep",
        "--sweep-id",      sweep_id,
        "--wandb-project", args.wandb_project,
        "--num-agents",    "1",
        "--sweep-count",   str(args.sweep_count) if args.sweep_count else "1",
        "--epochs-sw",     str(args.epochs_sw),
        "--d_model",       str(args.d_model),
        "--n_layers",      str(args.n_layers),
        "--ffn_dim",       str(args.ffn_dim),
        "--kd-weight",     str(args.kd_weight),
        "--qv-eps",        str(args.qv_eps),
        "--act-quant",     args.act_quant,
        "--auc-loss",      args.auc_loss,
        "--arch",          args.arch,
        # Named data file args (match new CLI)
        "--sig-part",  args.SignalTrainFile,
        "--sig-jet",   args.sig_jetData_TrainFile,
        "--bkg-part",  args.BkgTrainFile,
        "--bkg-jet",   args.bkg_jetData_TrainFile,
        "--test-dir",  args.test_dir,
    ]
    # Boolean flags with explicit on/off forms
    base_cmd += ["--fp-edges"    if args.fp_edges    else "--no-fp-edges"]
    base_cmd += ["--stoch-round" if args.stoch_round else "--no-stoch-round"]
    base_cmd += ["--baseline"] if args.baseline else []
    # Forward W&B options so child runs land in the right project/entity
    if args.wandb_entity:
        base_cmd += ["--wandb-entity",  args.wandb_entity]
    if args.wandb_tags:
        base_cmd += ["--wandb-tags",    args.wandb_tags]
    if args.wandb_offline:
        base_cmd += ["--wandb-offline"]
    if args.wandb_no_artifacts:
        base_cmd += ["--wandb-no-artifacts"]

    # ── Spawn N subprocesses ──────────────────────────────────────────────────
    n_agents = args.num_agents
    procs = []
    for i in range(n_agents):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i % max(1, n_agents))
        print(f"Spawning agent {i}  (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
        procs.append(subprocess.Popen(base_cmd, env=env))

    # ── Wait for all agents ───────────────────────────────────────────────────
    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            print(f"[WARNING] Agent {i} exited with code {rc}")

    print(f"\nAll {n_agents} agent(s) finished for sweep {sweep_id}.")
