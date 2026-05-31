# Repo Map — BNJetTagKai

Last updated: 2026-05-31

One-line index of every file so you (human or LLM) can orient WITHOUT reading
the code. Read `RULES.md` first, this second, then `docs/NEXT_STEPS.md`. For the
"why" behind non-obvious decisions, see `docs/GLOSSARY.md`.

The project: BitNet-quantized (ternary {-1,0,+1}) jet taggers for the CMS
Level-1 trigger, deployed to FPGA via hls4ml. The transformer is the accuracy
model (AUC ~= 0.989); the attention-free DeepSets variant is the FPGA target
(hls4ml has no MultiHeadAttention support).

---

## Read-first / governance

| File | What it is |
|------|------------|
| `RULES.md` | The contract every contributor/LLM must follow first: per-session workflow, mandatory CHANGELOG + NEXT_STEPS updates, commit conventions, honesty rules. |
| `AGENTS.md` | Pointer that sends auto-discovering agent tools to `RULES.md`. |
| `README.md` | Project overview + setup/run instructions. |
| `docs/REPO_MAP.md` | This file. |
| `docs/NEXT_STEPS.md` | Current status + prioritized roadmap. The "what do I do now" answer. |
| `docs/CHANGELOG.md` | Dated, newest-first session patch notes. |
| `docs/GLOSSARY.md` | Decided questions / non-obvious facts (so they aren't re-litigated). |

## Training (the Keras side)

| File | What it is |
|------|------------|
| `train.py` | Training entrypoint. Builds + trains a tagger; CLI `--arch {bitnet,deepsets,particle}`. ~900 lines. |
| `ROC.py` | Evaluate a trained model and plot/AUC the ROC on the validation set. Source of the real AUC ~= 0.989 number. |
| `bnjettag/config.py` | Shared constants, dataset paths, and the module-level `tf.Variable` toggles (ternary/FP32/act-quant). Single source of truth for hyperparams. |
| `bnjettag/data.py` | HDF5 dataset loading + evaluation helpers (May-2026 merged format). |
| `bnjettag/layers.py` | Custom Keras layers: `AbsMeanQuantizer`, `BitLinear`, `RMSNorm`, `BitMHSA`, `BitFFN`, `BitTransformerBlock`. ~620 lines. Imports TensorFlow. |
| `bnjettag/losses.py` | Focal loss + partial-AUC surrogate losses + ROC helpers. |
| `bnjettag/callbacks.py` | Training callbacks (e.g. `AUCReshapingCallback` — per-epoch hard-negative reweighting). |
| `bnjettag/sanity.py` | Quick shape/weight sanity check (no data needed). |
| `bnjettag/sweeps.py` | Hyperparameter sweeps (sequential grid + W&B Bayesian). |
| `bnjettag/wandb_utils.py` | Weights & Biases integration (no-op when `--wandb` off). |
| `bnjettag/hls_export.py` | Writes an hls4ml YAML config + rough FPGA resource estimate from a trained model (training-side helper; distinct from the hls4ml/ scripts). |

## Model architectures

| File | What it is |
|------|------------|
| `bnjettag/models/__init__.py` | `build_for_arch(args)` dispatcher. |
| `bnjettag/models/transformer.py` | `--arch bitnet` (default): ternary MHSA + BitFFN transformer. The accuracy model. |
| `bnjettag/models/deepsets.py` | `--arch deepsets`: attention-free, fully hls4ml-compatible. **The FPGA target.** |
| `bnjettag/models/particle.py` | `--arch particle`: ParT-style ternary transformer with pair-feature attention bias + [CLS] readout. |

## hls4ml (the FPGA side)

| File | What it is |
|------|------------|
| `bnjettag/hls_precision.py` | **Single source of truth for the DeepSets hls4ml precision config.** `build_hls_config(model, io_type=...)` returns the full config; two profiles (`io_parallel`, `io_stream`). The four scripts below import from here — change precision HERE, nowhere else. |
| `hls4ml/hls_convert_v2.py` | Primary io_parallel C-sim: convert + compare HLS vs Keras (noise + physics). The "does HLS still match Keras?" check. |
| `hls4ml/hls_trace.py` | Per-layer trace table to find the first diverging layer. The precision-debugging tool. |
| `hls4ml/hls_build.py` | Vivado 2020.1 synthesis (~30-60 min). Same config as v2 + ReuseFactor=64. NOT yet run to completion (top NEXT_STEPS item). |
| `hls4ml/hls_convert_iostream.py` | io_stream (streaming dataflow) C-sim. Wider precision profile; patches in a missing LayerNorm stream overload. |
| `hls4ml/hls_debug.py` | **[LEGACY]** early wide-precision debug scratchpad. Superseded by hls_trace.py / hls_convert_v2.py. |
| `hls4ml/setup_hls4ml.sh` | One-shot bootstrap: clone hls4ml @ pinned v1.3.0 SHA, apply patches, editable install. Run this on a fresh machine first. |
| `hls4ml/README.md` | hls4ml-specific usage notes. |

## hls4ml source patches

| File | What it is |
|------|------------|
| `patches/hls4ml/apply_patches.py` | Idempotent, anchored applier for the three LayerNorm source patches. Tested against a mock tree. |
| `patches/hls4ml/README.md` | What the patches are and why. |
| `docs/hls4ml_layernorm_patches.md` | Patch rationale (the three stock-hls4ml LayerNorm bugs). |
| `docs/hls4ml_precision_bugs.md` | The full bug-diagnosis story; Step 7 = the input_norm 2x fix. |
| `docs/hls4ml_attention_support.md` | Why hls4ml can't do the transformer's attention (→ DeepSets). |

## Data pipeline

| File | What it is |
|------|------------|
| `dataForgeScripts/dataForge.py` | Build the HDF5 datasets from CMS ROOT files (signal/background jet constituents). |
| `dataForgeScripts/removeBackground.py` | Background-removal / split helper for the dataset. |
| `docs/DATASETS.md` | Dataset format + paths reference. |
| `util/plotting/kinematics_plotter.py` | Plot jet/particle kinematics for a dataset. |

## Deferred / scaffold

| File | What it is |
|------|------------|
| `training/transformer_fp32.py` | FP32 baseline scaffold — never trained (no GPU available). Gives the float reference ROC curve when run. |

## Other docs

| File | What it is |
|------|------------|
| `docs/knowledge_distillation.md` | The KD-from-float-teacher pipeline. |
| `docs/paper_notes_2510.24784.md` | Notes on arXiv:2510.24784 ("Sub-microsecond Transformers for Jet Tagging on FPGAs"); benchmark targets. |
| `docs/codex_questions.md` | Open questions left for/by Codex. |
| `docs/session_summary_2026-05-11.md` | Codex session summary (Russell feedback items 1-6). |

## Infra / config

| File | What it is |
|------|------------|
| `environment.yml` | Conda environment spec. |
| `.gitignore` | Ignores `models/*` except `models/MODEL.md`; ignores generated hls4ml project dirs. |
| `models/MODEL.md` | Documents the gitignored trained model a fresh clone needs (`models/deepsets_d64_l3_ffn128/deepsets_clean.h5`). |
| `util/verify_hls_precision_refactor.py` | Test harness proving `hls_precision.py` reproduces the old inline configs byte-for-byte (stubs hls4ml; no TF/hls4ml needed). |
