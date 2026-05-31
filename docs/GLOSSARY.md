# Glossary & Decided Questions

Last updated: 2026-05-31

Non-obvious facts and already-settled questions about this project. Read this
before "fixing" something that looks wrong — several things that look like bugs
are intentional, and several numbers that look bad are artifacts. The point of
this file is so the next person/LLM doesn't re-derive or re-litigate them.

---

## Models & terminology

- **Two models, different jobs.** The **BitNet transformer** (`--arch bitnet`)
  is the accuracy model: ternary MHSA + BitFFN, trained with knowledge
  distillation, **AUC ~= 0.989** (from `ROC.py` on the full validation set).
  The **DeepSets** variant (`--arch deepsets`) is attention-free and is the
  **FPGA deployment target**. They share the BitFFN/LayerNorm blocks.
- **Why DeepSets for FPGA, not the transformer?** Current hls4ml releases do
  **not support standard Keras MultiHeadAttention**. DeepSets replaces attention
  with global average pooling over particles, so it converts cleanly. This is a
  hard constraint, not a preference — see `docs/hls4ml_attention_support.md`.
- **Ternary = {-1, 0, +1}.** "1-bit"/BitNet here means absmean-quantized ternary
  weights (`ap_int<2>` in HLS), not literal single-bit.
- **Input shape is (batch, 10, 14):** 10 particles per jet, 14 features each.

## hls4ml conversion — the big settled facts

- **The input_norm 2x amplification is FIXED and VERIFIED (2026-05-31).** This
  was the milestone the project was stuck on. Root cause was a LUT **range**
  mismatch (NOT accum_t resolution, an earlier wrong hypothesis): the tiny
  per-sample variances (~0.009-0.046) occupied only the bottom ~4.6% of a [0,1)
  inverse-sqrt table, so the steep low-index region was undersampled and
  1/sqrt(var) read ~2x high, doubling the output. Fix: `table_range_power2 = 4`
  (range [0, 2^-4)) + `table_t = ap_fixed<18,6>`. Full story in
  `docs/hls4ml_precision_bugs.md` (Step 7).
- **`head_fc2` correlation of 0.979 is NOT an accuracy problem.** It is a
  tight-output-cluster artifact: the output band is only ~2.5 wide and there's a
  ~0.3 systematic bias from FP32 weights/bias landing in `ap_fixed<16,8>`. With
  such a narrow spread, Pearson correlation looks low even though absolute error
  is tiny. Confirmed harmless because the **AUC matches Keras within 0.008**.
  Don't "chase" this number. Optional cosmetic fix: widen head_fc2
  result/weight/bias to `ap_fixed<18,8>` (in `bnjettag/hls_precision.py`).
- **The physics-data AUC of ~0.44 is MEANINGLESS as a quality metric.** It comes
  from a 46-jet smoke test in the convert scripts. It exists ONLY to prove the
  HLS model tracks the Keras model (it does: corr 0.995, AUC matches within
  0.008). The real model AUC (~0.989) comes from `ROC.py` on the full validation
  set. Never quote 0.44 as the model's quality.
- **Two precision profiles exist on purpose.** `io_parallel` uses tight types
  (`ap_fixed<16,x>`); `io_stream` uses wider types (`ap_fixed<32,16>` etc.) for
  streaming-dataflow headroom. This is intentional, not drift. Both live in
  `bnjettag/hls_precision.py`; the verified C-sim path is io_parallel.
- **Per-layer LUT ranges are mandatory in BOTH directions.** Small variances
  (input_norm) need `table_range_power2 > 0`; large ones (blocks 1/2, final,
  var up to ~3800) need `< 0`. A single global range breaks one end or the other.

## Precision config — where it lives now

- **All hls4ml precision is in `bnjettag/hls_precision.py` (since 2026-05-31).**
  It used to be copy-pasted inline into four scripts and the copies drifted
  (e.g. `hls_build.py` had stopped setting LN scale/bias/table keys — it would
  have baked a different config into firmware than the C-sim verified). If you
  change any precision value, change it THERE. The four scripts just call
  `build_hls_config(model, io_type=...)`.
- **`hls_model.compile()` regenerates firmware.** Never hand-edit
  `defines.h` / `parameters.h`; changes go in the Python config or the hls4ml
  source patches (`patches/hls4ml/`).

## Three patched hls4ml bugs (LayerNorm inverse-sqrt LUT)

In `patches/hls4ml/` (applied by `setup_hls4ml.sh`). See
`docs/hls4ml_layernorm_patches.md`:
1. `table_range_power2` was typed unsigned but needs to be a signed int (to
   allow negative powers for large-variance layers).
2. An undefined-behavior bit-shift in the table-range computation — replaced
   with a float `pow`.
3. The `table_t` precision configuration path was unreachable — fixed with a
   `TypeAttribute` + `_set_type_t('table')` in `layers.py`. This is why the
   configs must set an explicit `table_t` key (the module does this).

## Environment realities (so you don't waste time)

- **The agent sandbox has neither TensorFlow nor hls4ml installed.** Code
  changes are syntax-checked and config-equivalence-tested (see
  `util/verify_hls_precision_refactor.py`, which stubs hls4ml). Actual C-sim /
  trace / synthesis runs happen on the owner's machine (patched hls4ml + model
  + Vivado). When you change precision code, you can prove config equivalence
  but you CANNOT run the C-sim here — hand it to the owner to run.
- **No GPU is visible from the shell** (TF saw 0 GPUs), which is why the FP32
  baseline (`training/transformer_fp32.py`) is still untrained.
- **Synthesis needs Vivado 2020.1** (`vivado_hls`), not 2023.2 (`vitis_hls`).
  The path is hardcoded in `hls_build.py` (`VIVADO_BIN`) — edit for your machine.
- **`DATA_DIR` is hardcoded** in the convert scripts for the physics comparison.
  If absent, the scripts still run the noise-only C-sim.

## A fresh clone can't run anything until...

1. `bash hls4ml/setup_hls4ml.sh` installs the patched hls4ml.
2. The trained model is placed at
   `models/deepsets_d64_l3_ffn128/deepsets_clean.h5` (it's gitignored — see
   `models/MODEL.md`).

## Benchmark target (for the writeup)

From arXiv:2510.24784 "Sub-microsecond Transformers for Jet Tagging on FPGAs"
(`docs/paper_notes_2510.24784.md`): aim for **II = 1**, latency **< 100 cycles**.
Their HGQ Deep Sets baseline hits ~44-53 ns latency, ~177-256k LUT, DSP = 0 on
an XCU250 — a yardstick (different dataset, so not a direct accuracy comparison).
