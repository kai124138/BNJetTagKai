# Knowledge Distillation Pipeline

This documents the Stage-2 KD path used for the current transformer run in
`models/transformer_d64_l3_ffn128_kd/`.

## Teacher and Student

The teacher is not a separate `.h5` file loaded from disk. During training,
`qkerasModel.py` first runs a Stage-1 FP32 warm start with quantization disabled.
When Stage 2 begins, the script builds a fresh model with the same architecture,
copies the Stage-1 weights into it, marks it non-trainable, and evaluates it
with `QAT_ENABLED=False`.

Relevant code:

```python
QAT_ENABLED.assign(False)           # teacher sees FP32 forward
teacher = build_bitnet_jet_tagger(
    d_model  = args.d_model,
    n_layers = args.n_layers,
    ffn_dim  = args.ffn_dim,
    fp_edges = fp_edges,
    v_eps    = 0.0 if args.baseline else args.qv_eps,
)
teacher.set_weights(model.get_weights())  # copy Stage-1 FP32 weights
teacher.trainable = False
QAT_ENABLED.assign(True)            # student becomes ternary
```

The student is the original model object that continues training after Stage 1.
Once `QAT_ENABLED` is set to `True`, the `BitLinear` layers use ternary
weights in the forward pass while preserving straight-through gradients.

Stage-2 setup:

```python
print(f"\n=== Stage 2: ternary QAT for epochs {warmup_epochs}–{EPOCHS} ===")
QAT_ENABLED.assign(True)
STOCH_ROUND.assign((not args.baseline) and args.stoch_round)
```

## Loss

The implementation uses focal loss plus an MSE knowledge-distillation term on
temperature-scaled sigmoid probabilities:

```text
loss = focal_loss + kd_weight * mean((sigmoid(student_logit / T)
                                      - sigmoid(teacher_logit / T))^2)
```

The exact training-loop code is:

```python
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
```

This differs from the generic weighted form
`alpha * KD_loss + (1 - alpha) * student_BCE`; in this code the supervised term
is the full focal loss and the KD term is added with coefficient `kd_weight`.

## Defaults and Run Values

The KD command-line defaults are the values used for the current KD run:

```python
parser.add_argument("--kd-weight", dest="kd_weight", type=float, default=0.3,
                    help="Weight for KD MSE loss in Stage 2 (default: 0.3; 0 to disable)")
parser.add_argument("--no-kd", dest="kd_weight", action="store_const", const=0.0,
                    help="Disable Stage-2 knowledge distillation")
parser.add_argument("--kd-temp", dest="kd_temp", type=float, default=2.0,
                    help="Temperature for KD soft targets (default: 2.0)")
```

Inside `main(args)`:

```python
kd_weight = 0.0 if args.baseline else args.kd_weight
kd_temp   = args.kd_temp
do_kd     = kd_weight > 0.0
```

For the current run:

- `--kd-weight 0.3`
- `--kd-temp 2.0`
- `--qv-eps 2e-6`
- `--d_model 64 --n_layers 3 --ffn_dim 128`

## Training Command

The README records the historical command as:

```bash
python qkerasModel.py \
  --d_model 64 --n_layers 3 --ffn_dim 128 \
  --qv-eps 2e-6 \
  --kd-weight 0.3 --kd-temp 2.0 \
  /home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/ReversedPhi_Eta/4c_4b_trainData.h5 \
  /home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/ReversedPhi_Eta/4c_4b_testData.h5
```

Current `qkerasModel.py` expects four positional data files:

```text
SignalTrainFile BkgTrainFile sig_jetData_TrainFile bkg_jetData_TrainFile
```

The matching explicit command for the current 4c/4b + QCD files is:

```bash
python qkerasModel.py \
  --d_model 64 --n_layers 3 --ffn_dim 128 \
  --qv-eps 2e-6 \
  --kd-weight 0.3 --kd-temp 2.0 \
  /home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/ReversedPhi_Eta/4c_4b_trainData.h5 \
  /home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/ReversedPhi_Eta/QCD/trainingDatapt20_vDter_wEdits4ff.h5 \
  /home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/ReversedPhi_Eta/4c_4b_sampleData.h5 \
  /home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/ReversedPhi_Eta/QCD/sampleDatapt20_vDter_wEdits4ff.h5
```

## Artifacts

Current KD transformer artifacts live in:

```text
models/transformer_d64_l3_ffn128_kd/
```

Important files:

- `noNorm_train_d64_l3_ffn128_bitnetJetTagModel.h5`: final KD-trained model
  after Stage 3 AUC fine-tuning.
- `noNorm_train_d64_l3_ffn128_bitnetJetTagModel_preS3.h5`: checkpoint after
  Stage 2 / KD, before Stage 3.
- `noNorm_train_d64_l3_ffn128_bitnetLoss.pdf`: Stage-1 + Stage-2 loss curve,
  with the FP32-to-QAT switch marked.
- `noNorm_train_d64_l3_ffn128_auc_finetune.pdf`: Stage-3 AUC fine-tuning curve.
- `noNorm_train_d64_l3_ffn128_bitnetWeights.npy` and
  `noNorm_train_d64_l3_ffn128_ptRange.npy`: pT reweighting artifacts.
- `bitnet_d64_l3.onnx`: ONNX export of the transformer run.
