# Notes on arXiv:2510.24784

Verified from page 1 with:

```bash
wget https://arxiv.org/pdf/2510.24784 -O /tmp/paper.pdf
pdftotext -layout -l 1 /tmp/paper.pdf - | head -30
```

Title: **Sub-microsecond Transformers for Jet Tagging on FPGAs**

Authors: Lauri Laatu, Chang Sun, Arianna Cox, Abhijith Gandrakota,
Benedikt Maier, Jennifer Ngadiuba, Zhiqiang Que, Wayne Luk,
Maria Spiropulu, and Alexander Tapper.

## Architecture

- The paper studies small encoder-only transformer models for FPGA jet tagging.
- The main attention baseline uses vanilla multi-head attention with a single
  attention head and no positional encoding, so the model behaves like a Set
  Transformer over particles.
- Inputs are particle sequences with maximum lengths of 8, 16, 32, or 64,
  sorted by transverse momentum.
- Each particle has three features: `pT`, `eta`, and `phi`.
- They also implement Linformer-style linear attention, projecting keys and
  values to a lower dimension to reduce the `O(n^2)` attention cost to
  approximately `O(k*n)`.
- Compression uses High Granularity Quantization (HGQ), including per-parameter
  bitwidth optimization and pruning through an EBOPs target. Constant matrix
  vector multiplies are optimized with `da4ml`.

## Dataset

- The benchmark is the common hls4ml LHC jet tagging dataset.
- It has five balanced jet classes: gluon, light quark, W, Z, and top.
- The paper reports 620,000 training jets and 260,000 test jets.
- This differs from BNJetTagKai's current 4c/4b LLP-vs-QCD setup, so the
  metrics are useful for presentation style and FPGA targets, not direct
  numerical comparison.

## Metrics and ROC Presentation

- Figure 1 reports classification accuracy as a function of maximum input
  particles.
- Figure 2 shows ROC curves for 8, 16, 32, and 64 particles.
- The ROC plots use true positive rate on the x-axis and false positive rate
  on a logarithmic y-axis, with one-vs-rest AUC labels for each jet class and
  for both MHA and Linformer.
- The paper emphasizes AUC in the ROC legend, but the low false-positive-rate
  region is visually important because the y-axis spans roughly `1` down to
  `1e-4`.
- For BNJetTagKai, `ROC.py` now keeps AUC as a secondary metric and explicitly
  reports signal efficiency at background efficiencies of `0.01` and `0.001`.

## FPGA Targets and Reported Numbers

- Hardware target: Xilinx XCU250.
- Toolchain: Vitis HLS and Vivado, through hls4ml.
- All reported transformer and HGQ Deep Sets rows have initiation interval
  `II = 1` and `DSP = 0`.
- The synthesis table reports latency in nanoseconds, LUT usage in thousands,
  II, and DSP. It does not report FF or BRAM.

Selected Table 1 values:

| Model | Particles | Accuracy (%) | Latency (ns) | LUT (k) | II | DSP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-Head Attention | 8 | 66.3 | 104 | 246 | 1 | 0 |
| Multi-Head Attention | 16 | 72.3 | 98 | 279 | 1 | 0 |
| Multi-Head Attention | 32 | 77.0 | 83 | 180 | 1 | 0 |
| Multi-Head Attention | 64 | 77.9 | 44 | 47 | 1 | 0 |
| Linformer | 8 | 66.3 | 110 | 230 | 1 | 0 |
| Linformer | 16 | 72.8 | 103 | 246 | 1 | 0 |
| Linformer | 32 | 78.4 | 140 | 267 | 1 | 0 |
| Linformer | 64 | 79.8 | 78 | 202 | 1 | 0 |
| Deep Sets (HGQ) | 8 | 64.7 | 49 | 177 | 1 | 0 |
| Deep Sets (HGQ) | 16 | 70.1 | 53 | 205 | 1 | 0 |
| Deep Sets (HGQ) | 32 | 77.4 | 53 | 256 | 1 | 0 |
| Deep Sets (HGQ) | 64 | 79.4 | 44 | 191 | 1 | 0 |
| MLP Mixer | 16 | 71.7 | 68 | 75 | 1 | 0 |
| MLP Mixer | 32 | 78.0 | 62 | 63 | 1 | 0 |
| MLP Mixer | 64 | 79.7 | 72 | 159 | 1 | 0 |
| Deep Sets (QKeras) | 8 | 64.0 | 95 | 386 | 3 | 626 |
| Deep Sets (QKeras) | 16 | 69.4 | 115 | 747 | 3 | 555 |
| Deep Sets (QKeras) | 32 | 75.9 | 130 | 903 | 2 | 434 |

Relevant target for Russell's BNJetTagKai review: aim for `II = 1`, latency
below 100 cycles if possible, and resource usage low enough to be plausible for
the intended L1 trigger FPGA.
