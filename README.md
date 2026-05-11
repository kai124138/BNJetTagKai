# BNJetTagKai

A 1-bit (BitNet-style) jet tagger for the CMS L1 trigger, trained on Russell's
4c/4b dataset and being prepared for FPGA deployment via hls4ml.

This is a fork of the original BNJetTag work, with two main additions:

1. **A BitNet transformer jet tagger** (`qkerasModel.py`) — ternary `{-1, 0, +1}`
   weights with absmean quantization, RMSNorm, multi-head self-attention,
   trained end-to-end with knowledge distillation. Achieves **AUC ≈ 0.989** on
   the 4c/4b validation split.

2. **An attention-free "Deep Sets" variant** of the same architecture, built
   specifically because hls4ml does not support transformer self-attention.
   This is the model being converted to firmware. The Deep Sets variant uses
   the same BitFFN + LayerNorm blocks plus `GlobalAveragePooling1D` over the
   particle dimension.

The active work right now is in `hls4ml/` — converting `deepsets_clean.h5` to
HLS C++ and getting bit-accurate C-simulation against Keras before running
Vivado synthesis.

---

## Repository layout

```
BNJetTagKai/
├── qkerasModel.py          BitNet transformer + DeepSets variant training script
├── ROC.py                  ROC-curve evaluation for trained models
├── pfTuple*.root           Small input data (full datasets are external)
├── ROCCurve.png            Current best ROC plot
├── environment.yml         conda/micromamba environment
│
├── bitnet/                 Trained models and training artifacts
│   ├── deepsets_clean.h5                        ← model going into hls4ml
│   ├── bitnet_d64_l3.onnx                       ← ONNX export of transformer
│   ├── deepsets_noNorm_train_d64_l3_ffn128_*    ← DeepSets training run outputs
│   └── noNorm_train_d64_l3_ffn128_*             ← Transformer training run outputs
│
├── dataForgeScripts/       Jet reconstruction from CMS Ntuples
│   ├── dataForge.py
│   └── removeBackground.py
│
├── util/plotting/          Plot helpers (kinematics, etc.)
│
├── hls4ml/                 hls4ml conversion + synthesis scripts (see README inside)
│   ├── hls_convert_v2.py
│   ├── hls_trace.py
│   ├── hls_debug.py
│   └── hls_build.py
│
├── patches/hls4ml/         Source-level patches to hls4ml that MUST be applied
│   ├── nnet_layernorm.h.md
│   ├── core_templates.py.md
│   └── layers.py.md
│
├── docs/                   Technical write-ups
│   ├── hls4ml_layernorm_patches.md
│   └── hls4ml_precision_bugs.md
│
└── legacy/                 Older / superseded code kept for reference
    ├── HLS_qk_Roc_Tracing.py
    └── v1/
```

---

## What's been done

### 1. BitNet transformer training (complete — AUC 0.989)

`qkerasModel.py` implements a drop-in BitNet replacement for the original
QKeras CNN tagger. It matches the same I/O shape `(batch, 10, 14) → (batch, 1)`
and is trained with:

- Ternary `{-1, 0, +1}` weights via absmean quantization (straight-through
  estimator for gradients).
- Multi-head self-attention with ternary Q/K/V projections.
- RMSNorm and residual connections.
- Optional knowledge distillation from a float teacher (`--kd-weight`,
  `--kd-temp`).
- Pruning callbacks from `tensorflow_model_optimization`.
- Focal loss option.

Training command used for the current best run:

```bash
nohup python qkerasModel.py \
  --d_model 64 --n_layers 3 --ffn_dim 128 \
  --qv-eps 2e-6 \
  --kd-weight 0.3 --kd-temp 2.0 \
  /home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/ReversedPhi_Eta/4c_4b_trainData.h5 \
  /home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/ReversedPhi_Eta/4c_4b_testData.h5 \
  > training_kd_run2.log 2>&1 &
```

Result: **AUC ≈ 0.989** (see `ROCCurve.png`).

### 2. Attention-free DeepSets variant (complete)

After confirming that hls4ml has no support for `MultiHeadAttention`, a
parallel architecture was added: the same BitFFN blocks (BitLinear →
activation → BitLinear) wrapped around LayerNorm with residual connections,
but with `GlobalAveragePooling1D` over the particle dimension instead of
attention. This model is saved as `bitnet/deepsets_clean.h5` and is the one
targeting FPGA synthesis.

### 3. hls4ml conversion (in progress — accuracy debugging)

Stock hls4ml has three independent bugs in its `LayerNormalization` support
that prevent correct C-simulation output for any model whose post-residual
variance significantly exceeds 1:

| Issue                                              | Fix                                |
| -------------------------------------------------- | ---------------------------------- |
| `table_range_power2` typed `unsigned`              | change to `int`                    |
| UB integer bit-shift for negative powers           | replace with `pow(2.0f, …)` float  |
| `table_t` precision hardcoded, ignores config dict | add `TypeAttribute('table_t')` + `_set_type_t('table')` |

The full diagnostic write-up is in `docs/hls4ml_precision_bugs.md`. The exact
patches are in `patches/hls4ml/` and must be applied to your local hls4ml
clone before `hls_convert_v2.py` will produce correct output.

Current state (`hls_convert_v2.py` with all patches applied):
- `input_proj`: correlation 1.000 ✓
- `input_norm`: correlation 0.955, with a remaining ~2× amplification under
  investigation
- Downstream layers: still need work — `input_norm` amplification cascades

The remaining hypothesis (from `docs/hls4ml_precision_bugs.md`) is that
`accum_t` resolution is still insufficient for the smallest variances
(~0.009), causing the variance to round to roughly ¼ of its true value and
giving 2× amplification in `1/sqrt(var)`.

### 4. Synthesis (queued behind accuracy)

`hls4ml/hls_build.py` is ready to run Vivado HLS (`xcvu9p-flgb2104-2L-e`,
5 ns clock) once C-simulation matches Keras. Two synthesis attempts have been
made so far; both got past the schedulability check and into loop unrolling
before being stopped to retune precision.

---

## Setup

### Environment

Install `micromamba`:

```bash
# Linux x86_64
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba
source ~/.bashrc
```

Create and activate the environment:

```bash
micromamba env create -f environment.yml
micromamba activate <env_name>
```

### Install hls4ml (with patches)

```bash
mkdir software && cd software
git clone https://github.com/fastmachinelearning/hls4ml.git
cd hls4ml
pip install -e .
```

Then apply the three patches in `patches/hls4ml/`. They are documented as
before/after snippets — see `patches/hls4ml/README.md` for details. Without
these patches the LayerNorm output will be wrong.

### Vivado HLS (for synthesis only)

`hls_build.py` expects Vivado 2020.1 at `/data/software/xilinx/Vivado/2020.1/bin`.
The Vitis HLS in 2023.2 is *not* compatible with the hls4ml writer used here.

---

## Usage

### Reconstruct jets from Ntuples

```bash
cd dataForgeScripts
python3 dataForge.py <path/to/ntuple.root> QCDpt30 30 50 0
#                    file              tag      pTcut trainPct usePuppi
```

### Remove background-matched signal jets

```bash
python3 removeBackground.py <signal_train.h5> <signal_test.h5>
```

### Train the model

```bash
python3 qkerasModel.py \
  --d_model 64 --n_layers 3 --ffn_dim 128 \
  <SignalTrain.h5> <BackgroundTrain.h5> <SignalJetData.h5> <BackgroundJetData.h5>
```

### Evaluate ROC

```bash
python3 ROC.py
# Edit the file's hardcoded test-data paths first.
```

### Convert to HLS and synthesize

```bash
# 1. Convert + C-simulate (fast)
python hls4ml/hls_convert_v2.py

# 2. If accuracy is off, trace per-layer
python hls4ml/hls_trace.py

# 3. Synthesize (slow, ~30–60 min)
python hls4ml/hls_build.py
```

See `hls4ml/README.md` for more.

---

## References

- Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*
  (2024) — for the BitNet ternary quantization scheme.
- [fastmachinelearning/hls4ml](https://github.com/fastmachinelearning/hls4ml)
- Russell's L1JetTag dataset and the upstream BNJetTag project at
  `Brainz22/BNJetTag`.
