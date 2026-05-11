import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import h5py
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from tensorflow.keras.layers import (
    Add,
    Dense,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_PART_PER_JET = 10
N_FEAT = 14

DATA_DIR = (
    "/home/users/russelld/L1JetTagDaniel/hls4mlModifications"
    "/10-08-23/02-02_datasets/ReversedPhi_Eta"
)
DEFAULT_SIGNAL_TRAIN = f"{DATA_DIR}/4c_4b_trainData.h5"
DEFAULT_BKG_TRAIN = f"{DATA_DIR}/QCD/trainingDatapt20_vDter_wEdits4ff.h5"
DEFAULT_SIGNAL_JETDATA = f"{DATA_DIR}/4c_4b_sampleData.h5"
DEFAULT_BKG_JETDATA = f"{DATA_DIR}/QCD/sampleDatapt20_vDter_wEdits4ff.h5"
DEFAULT_SIGNAL_TEST = f"{DATA_DIR}/4c_4b_testData.h5"
DEFAULT_BKG_TEST = f"{DATA_DIR}/QCD/testingDatapt20_vDter_wEdits4ff.h5"
DEFAULT_OUTPUT_DIR = "models/transformer_fp32_d64_l3_ffn128"
DEFAULT_MODEL_NAME = "transformer_fp32_d64_l3_ffn128.keras"


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, min_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = max(int(warmup_steps), 1)
        self.total_steps = max(int(total_steps), self.warmup_steps + 1)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.peak_lr * (step / float(self.warmup_steps))
        cos_arg = np.pi * (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        cosine = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(cos_arg)
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


def focal_loss(gamma=1.0, alpha=0.5):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")

    def loss(y_true, y_logit):
        y_true = tf.cast(y_true, tf.float32)
        y_logit = tf.squeeze(y_logit, axis=-1)
        ce = bce(y_true, y_logit)
        prob = tf.sigmoid(y_logit)
        p_t = y_true * prob + (1.0 - y_true) * (1.0 - prob)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * ce)

    return loss


def load_training_data(signal_train, bkg_train, signal_jetdata, bkg_jetdata, seed):
    """Mirror qkerasModel.py data concatenation, shuffle, reshape, and pT weights."""
    print("Reading signal from " + signal_train)
    print("Reading background from " + bkg_train)
    print("Reading signal jet data from " + signal_jetdata)
    print("Reading background jet data from " + bkg_jetdata)

    with h5py.File(signal_train, "r") as hf:
        dataset = hf["Training Data"][:]
    with h5py.File(bkg_train, "r") as hf:
        dataset_qcd = hf["Training Data"][:]
    with h5py.File(signal_jetdata, "r") as hf:
        sample_data = hf["Sample Data"][:]
    with h5py.File(bkg_jetdata, "r") as hf:
        sample_data_qcd = hf["Sample Data"][:]

    dataset = np.concatenate((dataset, dataset_qcd))
    sample_data = np.concatenate((sample_data, sample_data_qcd))
    full_data = np.concatenate((dataset, sample_data), axis=1)
    rng = np.random.default_rng(seed)
    rng.shuffle(full_data)

    dataset = full_data[:, 0:141]
    sample_data = full_data[:, 141:]

    x = dataset[:, :-1].reshape(-1, N_PART_PER_JET, N_FEAT).astype(np.float32)
    y = dataset[:, -1].astype(np.float32)

    if max(x[:, :, 8].ravel()) >= 2.0:
        print("\nImpact parameter was not normalized beforehand.\n")

    bins = np.linspace(0, 500, 20)
    bkg_pts = sample_data[y == 0][:, 0]
    sig_pts = sample_data[y == 1][:, 0]
    bkg_counts, _ = np.histogram(bkg_pts, bins=bins)
    sig_counts, _ = np.histogram(sig_pts, bins=bins)
    total_bkg = len(bkg_pts)
    total_sig = len(sig_pts)
    weights_pt = np.nan_to_num(
        sig_counts / bkg_counts,
        nan=total_sig / total_bkg,
        posinf=total_sig / total_bkg,
        neginf=total_sig / total_bkg,
    )

    weights = np.ones(len(y), dtype=np.float32)
    pt_indices = np.clip(
        np.digitize(sample_data[:, 0], bins=bins) - 1, 0, len(weights_pt) - 1
    )
    weights[y == 0] = weights_pt[pt_indices][y == 0]
    return x, y, weights, sample_data


def load_test_data(signal_test, bkg_test):
    with h5py.File(signal_test, "r") as hf:
        dataset = hf["Testing Data"][:]
    with h5py.File(bkg_test, "r") as hf:
        dataset_qcd = hf["Testing Data"][:]
    dataset = np.concatenate((dataset, dataset_qcd))
    x = dataset[:, 0:140].reshape(-1, N_PART_PER_JET, N_FEAT).astype(np.float32)
    y = dataset[:, 140].astype(np.float32)
    return x, y


def transformer_block(x, d_model, n_heads, ffn_dim, dropout, name):
    norm1 = LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(x)
    attn = MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
        dropout=dropout,
        name=f"{name}_mha",
    )(norm1, norm1)
    x = Add(name=f"{name}_attn_add")([x, attn])

    norm2 = LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(x)
    ffn = Dense(ffn_dim, activation="relu", name=f"{name}_ffn1")(norm2)
    ffn = Dense(d_model, name=f"{name}_ffn2")(ffn)
    return Add(name=f"{name}_ffn_add")([x, ffn])


def build_transformer_fp32(d_model=64, n_layers=3, ffn_dim=128, n_heads=4, dropout=0.0):
    inputs = Input(shape=(N_PART_PER_JET, N_FEAT), name="particles")
    x = Dense(d_model, name="input_proj")(inputs)
    x = LayerNormalization(epsilon=1e-6, name="input_norm")(x)

    for i in range(n_layers):
        x = transformer_block(x, d_model, n_heads, ffn_dim, dropout, f"fp32_block_{i}")

    x = LayerNormalization(epsilon=1e-6, name="final_norm")(x)
    x = GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    x = Dense(d_model, activation="relu", name="head_fc1")(x)
    outputs = Dense(1, name="head_fc2")(x)
    return Model(inputs=inputs, outputs=outputs, name="transformer_fp32_jet_tagger")


def tpr_at_fpr(fpr, tpr, target):
    return float(np.interp(target, fpr, tpr))


def evaluate(model, signal_test, bkg_test):
    x_test, y_test = load_test_data(signal_test, bkg_test)
    scores = tf.sigmoid(model.predict(x_test, verbose=0)).numpy().ravel()
    fpr, tpr, _ = roc_curve(y_test, scores)
    metrics = {
        "auc": float(auc(fpr, tpr)),
        "sig_eff_at_bkg_eff_0p01": tpr_at_fpr(fpr, tpr, 0.01),
        "sig_eff_at_bkg_eff_0p001": tpr_at_fpr(fpr, tpr, 0.001),
    }
    return metrics


def save_history(history, output_dir):
    hist_path = output_dir / "history.csv"
    keys = sorted(history.history.keys())
    with hist_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + keys)
        writer.writeheader()
        for idx in range(len(history.history[keys[0]])):
            row = {"epoch": idx + 1}
            row.update({key: history.history[key][idx] for key in keys})
            writer.writerow(row)


def plot_weights(weights, output_dir):
    plt.figure()
    plt.hist(weights, bins=51)
    plt.xlabel("Weights")
    plt.tight_layout()
    plt.savefig(output_dir / "pt_weights.png")
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / args.model_name

    x, y, weights, sample_data = load_training_data(
        args.signal_train,
        args.bkg_train,
        args.signal_jetdata,
        args.bkg_jetdata,
        args.seed,
    )
    plot_weights(weights, output_dir)
    np.save(output_dir / "pt_weights.npy", weights)
    np.save(output_dir / "pt_range.npy", sample_data[:, 0])

    model = build_transformer_fp32(
        d_model=args.d_model,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    model.summary()

    train_size = int(len(x) * (1.0 - args.validation_split))
    total_steps = max(1, (train_size // args.batch_size) * args.epochs)
    lr_schedule = WarmupCosineDecay(
        peak_lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=int(args.warmup_fraction * total_steps),
        total_steps=total_steps,
    )
    model.compile(
        loss=focal_loss(gamma=1.0, alpha=0.5),
        optimizer=tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            beta_2=0.95,
        ),
        metrics=["binary_accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]
    history = model.fit(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_weight=weights,
        validation_split=args.validation_split,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(model_path)
    save_history(history, output_dir)

    metrics = evaluate(model, args.signal_test, args.bkg_test)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Saved model: {model_path}")
    print(f"AUC: {metrics['auc']:.6f}")
    print(f"Signal efficiency at bkg_eff=0.01: {metrics['sig_eff_at_bkg_eff_0p01']:.6f}")
    print(f"Signal efficiency at bkg_eff=0.001: {metrics['sig_eff_at_bkg_eff_0p001']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FP32 transformer jet tagger")
    parser.add_argument("--signal-train", default=DEFAULT_SIGNAL_TRAIN)
    parser.add_argument("--bkg-train", default=DEFAULT_BKG_TRAIN)
    parser.add_argument("--signal-jetdata", default=DEFAULT_SIGNAL_JETDATA)
    parser.add_argument("--bkg-jetdata", default=DEFAULT_BKG_JETDATA)
    parser.add_argument("--signal-test", default=DEFAULT_SIGNAL_TEST)
    parser.add_argument("--bkg-test", default=DEFAULT_BKG_TEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
