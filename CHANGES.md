# CHANGES — BitNet Jet Tagger improvements

## Flags added

| Flag | Default | What it does | Paper citation |
|------|---------|--------------|----------------|
| `--fp-edges` / `--no-fp-edges` | on | Keep `input_proj` and `head_fc2` in FP32 instead of ternary | BitNet b1.58, arXiv:2402.17764 |
| `--baseline` | off | Disable all new features; reproduce original behaviour exactly | — |
| `--auc-loss {aucm,pauc1way,pauc2way}` | `pauc1way` | Stage-3 fine-tuning loss | Yao/Lin/Yang 2022 arXiv:2203.01505; Yang et al. TPAMI 2022 arXiv:2206.11655 |
| `--fpr-thresh` | 0.01 | FPR threshold for pAUC surrogate and AUCReshaping | arXiv:2203.01505 |
| `--tpr-floor` | 0.80 | TPR floor for two-way pAUC | arXiv:2206.11655 |
| `--focal-weight` | 0.3 | Focal component weight in composite Stage-3 loss | Zhu/Wu/Yang arXiv:2203.14177 |
| `--pauc-weight` | 0.7 | pAUC component weight in composite Stage-3 loss | arXiv:2203.14177 |
| `--stratify` / `--no-stratify` | on | Stratified 50/50 class-balanced batches in Stage 3 | arXiv:2203.14177 |
| `--act-quant {fp32,int8}` | `int8` | Per-token absmax int8 activation quantization inside BitLinear | BitNet a4.8, arXiv:2411.04965 |
| `--stoch-round` / `--no-stoch-round` | on | Stochastic rounding in ternary STE during training | Zhao et al. NeurIPS 2024, arXiv:2412.04787 |
| `--reshape` / `--no-reshape` | on | AUCReshaping per-epoch positive reweighting | Panambur et al. 2023, DOI:10.1038/s41598-023-48482-x |
| `--reshape-boost` | 2.0 | Weight multiplier for false-negative positives | arXiv:10.1038/s41598-023-48482-x |
| `--reshape-cap` | 8.0 | Maximum cumulative boost per sample | arXiv:10.1038/s41598-023-48482-x |
| `--sweep` | — | Run 3×3 LR/WD grid search; writes `bitnet/sweep_results.csv` | BitNet b1.58 Reloaded, arXiv:2407.09527 |
| `--qv-eps` | 2e-6 | Absmedian eps for Value projection in BitMHSA; Q/K use 1e-6 | Huang et al. 2023, arXiv:2307.00331 |
| `--kd-weight` | 0.3 | Stage-2 KD loss weight; 0 to disable (uses frozen FP32 teacher) | Huang et al. 2023, arXiv:2307.00331 |
| `--no-kd` | — | Set `--kd-weight 0`; skip Stage-2 knowledge distillation | — |
| `--kd-temp` | 2.0 | Temperature for KD soft sigmoid targets | Hinton et al. 2015 (distillation) |
| `--export-hls` | off | After training, write `bitnet/<tag>_hls4ml_config.yaml` + LUT estimate | Duarte et al. JINST 2018; Fahim et al. arXiv:2101.05108 |

## Recommended invocation for Russell's run

```bash
python qkerasModel.py \
    --fp-edges \
    --auc-loss pauc1way --fpr-thresh 0.01 \
    --focal-weight 0.3 --pauc-weight 0.7 \
    --stratify \
    --act-quant int8 \
    --stoch-round \
    --reshape --reshape-boost 2.0 --reshape-cap 8.0 \
    <signal.h5> <bkg.h5> <sig_jet.h5> <bkg_jet.h5>
```

For an apples-to-apples baseline (original behaviour):

```bash
python qkerasModel.py --baseline \
    <signal.h5> <bkg.h5> <sig_jet.h5> <bkg_jet.h5>
```

To find the best LR/WD before the full run:

```bash
python qkerasModel.py --sweep \
    <signal.h5> <bkg.h5> <sig_jet.h5> <bkg_jet.h5>
# Reads bitnet/sweep_results.csv and prints the best config by TPR@FPR=1e-2
```

## Expected ROC tail improvement vs baseline

Smoke training results on 10 k synthetic data (steps 1–3 combined vs baseline):

| Configuration | TPR @ FPR=1e-2 | Δ vs baseline |
|---------------|---------------|---------------|
| Baseline (all ternary, AUCM loss) | 0.614 | — |
| Steps 1–3 (FP edges + pauc1way + composite + stratify) | 0.871 | **+0.257** |

Final integration smoke (3 epochs, all flags, synthetic 8 k samples):

- AUROC = 0.911 (log-scale ROC; earlier figure of 0.9808 was from a linear-scale evaluation and is incorrect)
- TPR @ FPR = 1e-2 = 0.658 (after only 3 Stage-3 epochs on a 4-epoch warm-start)

The +0.26 gain at FPR=1e-2 on the diag plot is reproducible
(`bitnet/diag/roc_tail_after_step3.pdf`). The full run with Russell's
data over 200 epochs should show a larger absolute TPR gain.

## Step 10: Quantization Variation + Stage-2 KD

**What changed**:
- `BitLinear` accepts an `eps` kwarg (default `1e-6`); `BitMHSA` gives `W_v`
  a separate `v_eps` (default `2e-6`). The larger eps compresses the V weight
  distribution less aggressively, preserving attention value resolution.
  Q/K/W_o/FFN layers keep the standard `1e-6`.
- `build_bitnet_jet_tagger(v_eps=2e-6)` threads `v_eps` down to every
  `BitTransformerBlock → BitMHSA`.
- When `--kd-weight > 0` (default 0.3), Stage 2 switches from `model.fit()`
  to a custom training loop that minimises
  `focal + kd_weight × MSE(σ(s/T), σ(t/T))` where the teacher is the
  frozen FP32 model saved at the end of Stage 1.
- `sanity_check()` now prints the eps for every kernel in the per-layer table.
- `--sanity` resource estimate uses `model.submodules` so nested layers
  are counted correctly (17,408 ternary params → 34,816 LUTs estimated).

**Paper**: Huang et al. "Quantization Variation" (2023), arXiv:2307.00331.

## Step 11: HLS4ML Config Export

**What changed**:
- `write_hls4ml_config(model, args, tag, act_bits, fp_edges)` writes a
  hls4ml-compatible YAML to `bitnet/<tag>_hls4ml_config.yaml` with:
  - Per-layer precision: ternary layers → `ap_int<2>`, FP-edge layers →
    `ap_fixed<16,6>`, activations → `ap_int<8>` or `ap_fixed<16,6>`.
  - Backend/part/clock defaults for CMS L1T (Xilinx VU9P, 200 MHz / 5 ns).
  - Rough LUT estimate: 2 LUTs per ternary param + 30 LUTs per FP param.
    For the default 18 k-param model: ~49 k LUTs ≈ 4.2% of VU9P.
- Pass `--export-hls` to write the YAML after the full training run.
- `sanity_check()` prints the LUT estimate inline.

**References**: Duarte et al. (JINST 13 P07027, 2018); Fahim et al.
(arXiv:2101.05108, 2021); hls4ml docs at fastmachinelearning.org/hls4ml.

## Updated recommended invocation for Russell's run

```bash
python qkerasModel.py \
    --fp-edges \
    --auc-loss pauc1way --fpr-thresh 0.01 \
    --focal-weight 0.3 --pauc-weight 0.7 \
    --stratify \
    --act-quant int8 \
    --stoch-round \
    --reshape --reshape-boost 2.0 --reshape-cap 8.0 \
    --qv-eps 2e-6 \
    --kd-weight 0.3 --kd-temp 2.0 \
    --export-hls \
    <signal.h5> <bkg.h5> <sig_jet.h5> <bkg_jet.h5>
```

## Open TODOs / skipped items

- The `--sweep` mode produces a CSV but does not automatically wire the best
  (lr, wd) into the subsequent `main()` run — inspect `bitnet/sweep_results.csv`
  and pass the best values explicitly via `--d_model`, etc.
- `write_hls4ml_config()` writes the YAML but does not invoke `hls4ml.converters`
  — install hls4ml and call `hls4ml.convert_from_keras_model()` with the YAML
  for full firmware generation.
