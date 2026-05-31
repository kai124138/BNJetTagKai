# Codex Session Summary - 2026-05-11

This session worked through Russell's feedback items 1-6 for BNJetTagKai.
No pushes were run from Codex; commits were made locally after explicit user
approval.

## Commit Stack

```text
4c02c5f Add io_stream hls4ml conversion
a1ded6a Document knowledge distillation pipeline
f3625b8 Add FP32 transformer baseline scaffold
df367ec Align ROC metrics with benchmark paper
5d2257e Document hls4ml attention support
aa65681 Organize model artifacts by run
1dc59bf Update README.md
```

## Item 1 - Organize Model Artifacts By Run

Commit: `aa65681 Organize model artifacts by run`

- Renamed the flat `bitnet/` artifact directory to `models/`.
- Grouped existing artifacts by run:
  - `models/transformer_d64_l3_ffn128_kd/`
  - `models/deepsets_d64_l3_ffn128/`
- Updated path references across code and documentation.
- Updated `.gitignore` so generated HLS projects under `models/hls4ml_*` stay
  untracked.
- Added `models/README.md` describing the run-folder naming convention.
- Verified moved Keras model paths load correctly.
- `python ROC.py` no longer had path errors; the only issue observed at that
  point was the known TensorFlow/protobuf environment issue.

## Item 2 - Check hls4ml Attention Support

Commit: `5d2257e Document hls4ml attention support`

- Probed hls4ml `main` and actual post-1.0 tags in isolated scratch checkouts.
- Kept the patched `software/hls4ml/` checkout untouched.
- Recorded exact tags, SHAs, and registry findings in
  `docs/hls4ml_attention_support.md`.
- Finding: standard Keras `Attention` and `MultiHeadAttention` remain
  unsupported in the checked hls4ml versions. Newer registries include HGQ
  attention-related handlers, but not standard Keras attention handlers.

## Item 3 - Align ROC Metrics With Benchmark Paper

Commit: `df367ec Align ROC metrics with benchmark paper`

- Downloaded arXiv `2510.24784` and verified the page-1 title/authors before
  writing notes.
- Added `docs/paper_notes_2510.24784.md` covering:
  - architecture,
  - dataset,
  - ROC/metric presentation,
  - FPGA targets and reported numbers.
- Updated `ROC.py` to report:
  - AUC,
  - signal efficiency at `bkg_eff=0.01`,
  - signal efficiency at `bkg_eff=0.001`.
- Regenerated `ROCCurve.png`.
- `python ROC.py` ran successfully and reported for the BitNet KD model:
  - AUC: `0.989171`
  - signal efficiency at `bkg_eff=0.01`: `0.727899`
  - signal efficiency at `bkg_eff=0.001`: `0.376544`

## Item 4 - FP32 Transformer Baseline Scaffold

Commit: `f3625b8 Add FP32 transformer baseline scaffold`

- Added `training/transformer_fp32.py`.
- The script uses the same model I/O shape as `qkerasModel.py`: `(10, 14)` to
  one logit.
- Reuses the existing 4c/4b data flow and pT reweighting approach from
  `qkerasModel.py`.
- Saves the future trained model as:
  `models/transformer_fp32_d64_l3_ffn128/transformer_fp32_d64_l3_ffn128.keras`
- Updated `ROC.py` to include the FP32 model in the plot when that file exists.
- Did not run training because no GPU was visible from the shell:
  TensorFlow saw `0` physical GPUs.
- Recorded the deferred training blocker in `docs/codex_questions.md`.

## Item 5 - Document Knowledge Distillation

Commit: `a1ded6a Document knowledge distillation pipeline`

- Added `docs/knowledge_distillation.md`.
- Documented:
  - frozen FP32 Stage-1 teacher,
  - ternary BitNet/QAT student,
  - implemented focal-plus-KD loss formula,
  - default KD parameters `--kd-weight 0.3` and `--kd-temp 2.0`,
  - training command,
  - KD artifact locations.
- Included short quoted code blocks from `qkerasModel.py`.

## Item 6 - io_stream hls4ml Conversion

Commit: `4c02c5f Add io_stream hls4ml conversion`

- Added `hls4ml/hls_convert_iostream.py`.
- Deterministic paths:
  - input Keras model:
    `models/deepsets_d64_l3_ffn128/deepsets_clean.h5`
  - existing io_parallel output:
    `models/hls4ml_deepsets_v2/`
  - new io_stream output:
    `models/hls4ml_deepsets_iostream/`
- Kept generated HLS output uncommitted through `.gitignore`.
- The first io_stream conversion failed to compile because generated C++ called
  LayerNorm with `hls::stream` objects, while the local LayerNorm patch only
  provided an array overload.
- Added a script-side generated-header patch that injects the missing LayerNorm
  stream overload before C-sim. This leaves `software/hls4ml/` untouched.
- Started from the v2 precision config, then retuned dense and input-projection
  precision.
- C-sim comparison:

| Variant | Output directory | Noise correlation | Noise MAE | Physics correlation | Physics MAE |
| ------- | ---------------- | ----------------- | --------- | ------------------- | ----------- |
| `io_parallel` | `models/hls4ml_deepsets_v2/` | `0.969850` | `0.383298` | `0.997195` | `0.544165` |
| `io_stream` | `models/hls4ml_deepsets_iostream/` | `0.972328` | `0.302683` | `0.999144` | `0.322316` |

The io_stream noise correlation remains below `0.99`, but it is consistent
with the known io_parallel v2 baseline limitation. The physics correlation is
above `0.99`, and MAE improved relative to v2.

## Deferred Work

### Item 4 Training

FP32 training is deferred until GPU access is available. The training script is
ready, but the model artifact has not been produced.

### Item 7 Synthesis

Synthesis for both io_parallel and io_stream has not been run yet. The current
state is ready for synthesis with v2 and io_stream C-sim correlations reported.
The pending note is also recorded in `docs/codex_questions.md`.

## Current Uncommitted Changes

At the end of the session, `docs/codex_questions.md` had an uncommitted Item 7
deferred-synthesis note. This summary file is also uncommitted until explicitly
staged and committed.
