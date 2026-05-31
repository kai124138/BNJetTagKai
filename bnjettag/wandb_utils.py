"""
Weights & Biases integration for the BitNet jet tagger.

  * WandbEpochLogger — Keras callback logging per-epoch metrics with a
    continuous global step across training stages.
  * WandbTracker     — single source of truth for run init, live logging,
    image/artifact upload, and the epoch-callback helpers. A no-op when
    `args.wandb` is False, so call sites need no guards.
"""

import os

import tensorflow as tf

from .config import DEFAULT_TEST_DIR


# ══════════════════════════════════════════════════════════════════════════════
# W&B EPOCH LOGGER  (used by Stage 1 / 2.5 model.fit callbacks)
# ══════════════════════════════════════════════════════════════════════════════

class WandbEpochLogger(tf.keras.callbacks.Callback):
    """Log per-epoch Keras metrics to W&B with a continuous global step.

    Args:
        stage_offset: Added to the Keras epoch counter so the W&B x-axis is
                      continuous across stages (pass warmup_epochs for Stage 2,
                      EPOCHS for Stage 2.5, etc.).
        prefix:       Metric key prefix, e.g. "stage1" → "stage1/loss".
    """

    def __init__(self, stage_offset: int = 0, prefix: str = "stage"):
        super().__init__()
        self.stage_offset = stage_offset
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        step = self.stage_offset + epoch
        payload = {f"{self.prefix}/{k}": v for k, v in (logs or {}).items()}
        payload["epoch"] = step
        wandb.log(payload, step=step)


class WandbTracker:
    """Centralised W&B integration — single source of truth for all logging.

    All training code goes through this class so enabling/disabling W&B is a
    single `--wandb` flag and the W&B contract (run name, tags, config,
    artifact policy) lives in one place. When `args.wandb` is False every
    method is a no-op, so call sites need no guards.

    Required `args` attributes (CLI):
        wandb, wandb_project, wandb_entity, wandb_name, wandb_tags,
        wandb_offline, wandb_no_artifacts, plus all standard training knobs.

    Typical usage:
        tracker = WandbTracker(args, tag, run_type="train")
        ...
        history = model.fit(..., callbacks=tracker.with_epoch_cb(cbs, 0, "stage1"))
        tracker.log({"stage2/loss": v}, step=epoch)
        tracker.log_image("plots/loss", "loss.png")
        tracker.log_artifact("model.h5", "bitnet-jet-tagger", type_="model")
        tracker.finish()
    """

    def __init__(self, args, tag: str, run_type: str = "train"):
        self.enabled = bool(getattr(args, "wandb", False))
        self.args    = args
        self.tag     = tag
        self._wandb  = None
        if not self.enabled:
            return
        # Offline mode for clusters with no internet egress — must be set
        # before `wandb.init` is called.
        if getattr(args, "wandb_offline", False):
            os.environ.setdefault("WANDB_MODE", "offline")
        import wandb as _wb
        self._wandb = _wb

        # ── Run name + tags ──────────────────────────────────────────────
        run_name = getattr(args, "wandb_name", None) or tag.replace("/", "_")
        tags = [run_type, f"arch:{args.arch}",
                f"d{args.d_model}", f"l{args.n_layers}", f"ffn{args.ffn_dim}"]
        if getattr(args, "baseline", False):
            tags.append("baseline")
        if getattr(args, "kd_weight", 0.0) > 0.0 and args.arch != "deepsets":
            tags.append("kd")
        extra_tags = (getattr(args, "wandb_tags", "") or "").strip()
        if extra_tags:
            tags.extend(t.strip() for t in extra_tags.split(",") if t.strip())

        # ── Hyperparameter config snapshot ───────────────────────────────
        cfg = {
            # Architecture
            "arch":          args.arch,
            "d_model":       args.d_model,
            "n_layers":      args.n_layers,
            "ffn_dim":       args.ffn_dim,
            "fp_edges":      args.fp_edges,
            "act_quant":     args.act_quant,
            "qv_eps":        args.qv_eps,
            "stoch_round":   args.stoch_round,
            # Training schedule
            "epochs":        200,
            "batch_size":    50,
            "peak_lr":       3e-4,
            "weight_decay":  0.01,
            # Loss / AUC fine-tuning
            "kd_weight":     args.kd_weight,
            "kd_temp":       args.kd_temp,
            "auc_loss":      args.auc_loss,
            "fpr_thresh":    args.fpr_thresh,
            "tpr_floor":     args.tpr_floor,
            "focal_weight":  args.focal_weight,
            "pauc_weight":   args.pauc_weight,
            "stratify":      args.stratify,
            "reshape":       args.reshape,
            "baseline":      args.baseline,
            "run_tag":       tag,
            # Dataset provenance
            "sig_part_file": os.path.basename(getattr(args, "SignalTrainFile", "")),
            "sig_jet_file":  os.path.basename(getattr(args, "sig_jetData_TrainFile", "")),
            "bkg_part_file": os.path.basename(getattr(args, "BkgTrainFile", "")),
            "bkg_jet_file":  os.path.basename(getattr(args, "bkg_jetData_TrainFile", "")),
            "test_dir":      os.path.basename(getattr(args, "test_dir", DEFAULT_TEST_DIR) or ""),
        }

        _wb.init(
            project = args.wandb_project,
            entity  = getattr(args, "wandb_entity", None) or None,
            name    = run_name,
            tags    = tags,
            config  = cfg,
        )

    # ── Live logging ─────────────────────────────────────────────────────
    def log(self, payload: dict, step: int | None = None) -> None:
        if not self._active():
            return
        if step is not None:
            self._wandb.log(payload, step=step)
        else:
            self._wandb.log(payload)

    def summary(self, payload: dict) -> None:
        """Write to `wandb.run.summary` — final scalars that should not be
        time-series (e.g. best val AUC). Idempotent across re-logs."""
        if not self._active():
            return
        for k, v in payload.items():
            self._wandb.run.summary[k] = v

    # ── Files / artifacts ────────────────────────────────────────────────
    def log_image(self, key: str, path: str, caption: str | None = None) -> None:
        if not self._active() or not os.path.exists(path):
            return
        self._wandb.log({key: self._wandb.Image(path, caption=caption)})

    def log_artifact(self, path: str, name: str, type_: str = "model") -> None:
        """Upload a file as a versioned W&B Artifact (skipped if
        --wandb-no-artifacts is set, e.g. for fast smoke runs)."""
        if not self._active() or not os.path.exists(path):
            return
        if getattr(self.args, "wandb_no_artifacts", False):
            return
        art = self._wandb.Artifact(name=name, type=type_)
        art.add_file(path)
        self._wandb.log_artifact(art)

    # ── Keras callback helpers ───────────────────────────────────────────
    def epoch_callback(self, stage_offset: int = 0,
                       prefix: str = "stage") -> "tf.keras.callbacks.Callback | None":
        """Return a WandbEpochLogger, or None when W&B is disabled."""
        return WandbEpochLogger(stage_offset=stage_offset, prefix=prefix) \
            if self.enabled else None

    def with_epoch_cb(self, base_cbs: list, stage_offset: int = 0,
                      prefix: str = "stage") -> list:
        """Append a W&B epoch logger to `base_cbs` if enabled, else return
        `base_cbs` unchanged. Convenience for `model.fit(callbacks=...)`."""
        cb = self.epoch_callback(stage_offset, prefix)
        return base_cbs + [cb] if cb is not None else list(base_cbs)

    def finish(self) -> None:
        if self._active():
            self._wandb.finish()

    # ── Internal ─────────────────────────────────────────────────────────
    def _active(self) -> bool:
        return self.enabled and self._wandb is not None and self._wandb.run is not None
