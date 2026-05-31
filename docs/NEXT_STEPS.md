# BNJetTagKai — Status & Next Steps

Last updated: 2026-05-31

This is the working roadmap: an honest answer to "does it work yet, and what's
next." Read this first when you come back to the project.

---

## TL;DR — does it work?

**Training works. hls4ml conversion is now VERIFIED end-to-end on C-sim.
Synthesis is the only remaining step.**

- The BitNet transformer trains to **AUC ≈ 0.989** — done.
- The attention-free DeepSets variant (the one going to FPGA) trains — done.
- The hls4ml C-simulation runs, the three stock-hls4ml LayerNorm bugs are
  patched, and the `input_norm` 2× amplification is **fixed and confirmed**.
  The 2026-05-31 trace run came back clean (see below) — this is the milestone
  the project was stuck on.
- FPGA synthesis (`hls_build.py`) is wired up but has **never been run to
  completion**. We don't have latency/LUT numbers yet. **This is now the top
  item.**

So: the hard part is done. What remains is running synthesis and recording the
resource/latency numbers.

### Verified C-sim results (2026-05-31)
- `hls_trace.py`: `input_norm` corr **0.955 → 1.000**;
  `ds_block_0_norm1` corr **−0.06 → 1.000**; all layers **corr 1.000** through
  `head_fc1`; `head_fc2` corr **0.979** (a tight-output-cluster artifact, not a
  real accuracy problem — see gotchas); Final Corr **0.9788**.
- `hls_convert_v2.py` (io_parallel): physics correlation **0.995**, MAE 0.578;
  HLS AUC **0.4505** vs Keras **0.4429** → "✓ matches within tolerance."
  (The 0.44 AUC is meaningless as a quality number — only 46 jets, a smoke
  test. It only proves HLS tracks Keras. Real model AUC ≈ 0.989 comes from
  `ROC.py` on the full validation set.)

---

## What's up next (in priority order)

### 1. Run synthesis and capture resource/latency numbers  ← do this first
C-sim is clean (verified 2026-05-31), so synthesis is the next real step.

```bash
python hls4ml/hls_build.py     # needs Vivado 2020.1; ~30–60 min
```

**Capture:** estimated clock, latency (cycles + ns), II, and LUT/FF/DSP/BRAM.
Save the report and add a synthesis-results section to `hls4ml/README.md`.
Benchmark target (from `docs/paper_notes_2510.24784.md`, arXiv:2510.24784,
"Sub-microsecond Transformers for Jet Tagging on FPGAs"): aim for **II = 1**,
latency **< 100 cycles** if achievable, and resource usage plausible for an L1
trigger FPGA. Their HGQ Deep Sets baseline hits ~44–53 ns latency, ~177–256k
LUT, II = 1, DSP = 0 on an XCU250 — a useful yardstick (different dataset, so
not a direct accuracy comparison).

### 2. (Optional) Tighten `head_fc2` and run io_stream
Neither blocks synthesis; both are polish.

- **`head_fc2` corr 0.979** is a tight-output-cluster artifact (output band only
  ~2.5 wide, ~0.3 systematic bias from FP32 weights/bias landing in
  `ap_fixed<16,8>`), confirmed harmless by the AUC matching within 0.008. If you
  want it cosmetically clean, widen `head_fc2`'s result/weight/bias to
  `ap_fixed<18,8>` in all four config copies and re-trace.
- **io_stream path:** `python hls4ml/hls_convert_iostream.py` for the streaming
  dataflow variant (different latency/resource trade-off than io_parallel).

### 3. (Deferred) Train the FP32 baseline
`training/transformer_fp32.py` is ready but was never trained — no GPU was
visible from the shell (TF saw 0 GPUs). Run it when you have GPU access so the
ROC plot has the float reference curve.

```bash
python training/transformer_fp32.py <train.h5> <test.h5> ...
```

---

## Things to watch out for (gotchas already hit)

- **Three copies of the precision config exist** — `hls_convert_v2.py`,
  `hls_convert_iostream.py`, `hls_trace.py`, and `hls_build.py` each inline
  their own `LN_CONFIGS`. They're now consistent for `input_norm`, but any
  future precision change has to be made in **all** of them. (Worth refactoring
  into a single shared `bnjettag/hls_precision.py` someday — see below.)
- **`hls_model.compile()` regenerates firmware** — never hand-edit
  `defines.h` / `parameters.h`; fixes go in the Python config or the hls4ml
  source patches.
- **Per-layer ranges are mandatory in both directions** — small variances need
  `table_range_power2 > 0`, large ones need `< 0`. A single global range breaks
  one end or the other.
- **A fresh clone can't run anything until** (a) `bash hls4ml/setup_hls4ml.sh`
  installs the patched hls4ml and (b) the model is placed at
  `models/deepsets_d64_l3_ffn128/deepsets_clean.h5` (see `models/MODEL.md`).
- **Physics-data path is hardcoded** in the convert scripts (`DATA_DIR`).
  Edit it for your machine; if absent, the scripts still do the noise C-sim.

---

## Nice-to-haves (not blocking)

- Refactor the four duplicated `LN_CONFIGS` / `dense_*_prec` blocks into one
  shared module so precision lives in a single place.
- Make `DATA_DIR` an env var / CLI arg instead of a hardcoded path.
- Add a tiny smoke test that asserts the generated `defines.h` contains the
  expected `*_table_t` typedefs (catches "config silently ignored" regressions).

---

## Draft abstract (for a writeup / poster)

> **BitNet-quantized jet taggers for the CMS Level-1 trigger via hls4ml.**
> We present a 1-bit (ternary {−1, 0, +1}) jet tagger targeting the sub-
> microsecond latency budget of the CMS Level-1 trigger. A BitNet-style
> transformer with absmean-quantized Q/K/V projections, RMSNorm, and multi-head
> self-attention, trained end-to-end with knowledge distillation from a float
> teacher, reaches AUC ≈ 0.989 on a 4c/4b long-lived-particle-vs-QCD benchmark.
> Because current hls4ml releases do not support standard Keras attention, we
> derive an attention-free "Deep Sets" variant that reuses the same ternary
> BitFFN and LayerNorm blocks with global average pooling over particles, and
> target it for FPGA deployment. We document and fix three numerical bugs in
> hls4ml's LayerNormalization inverse-square-root lookup — a sign-typing error
> and an undefined-behavior bit-shift in the table-range computation, and an
> unreachable table-precision configuration path — and introduce per-layer LUT
> range and precision tuning that handles per-sample variances spanning four
> orders of magnitude (≈0.009 to ≈3800). [TODO once measured: report final
> C-simulation fidelity to the Keras model and the synthesized latency,
> initiation interval, and resource utilization on the target Xilinx FPGA.]

The bracketed sentence is intentionally a placeholder — fill it in after steps
2 and 3 produce real numbers. Don't claim synthesis results before running
`hls_build.py`.

---

## Pointers

- Bug diagnosis story: `docs/hls4ml_precision_bugs.md` (Step 7 = the
  `input_norm` fix)
- Patch rationale: `docs/hls4ml_layernorm_patches.md`
- Apply patches: `patches/hls4ml/apply_patches.py` / `hls4ml/setup_hls4ml.sh`
- Benchmark paper notes: `docs/paper_notes_2510.24784.md`
- Attention-support findings: `docs/hls4ml_attention_support.md`
