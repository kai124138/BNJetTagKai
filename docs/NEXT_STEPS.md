# BNJetTagKai — Status & Next Steps

Last updated: 2026-05-31

This is the working roadmap: an honest answer to "does it work yet, and what's
next." Read this first when you come back to the project.

---

## TL;DR — does it work?

**Training works. hls4ml conversion is close but not yet verified end-to-end.
Synthesis has not been run.**

- The BitNet transformer trains to **AUC ≈ 0.989** — done.
- The attention-free DeepSets variant (the one going to FPGA) trains — done.
- The hls4ml C-simulation runs, the three stock-hls4ml LayerNorm bugs are
  patched, and we just fixed the `input_norm` 2× amplification (the last known
  per-layer divergence). **This still needs to be confirmed by a trace run** —
  see step 1 below. Nothing about the project is "automatically" working until
  that trace comes back clean; the code change is sound on the math but
  unverified on hardware-accurate C-sim.
- FPGA synthesis (`hls_build.py`) is wired up but has **never been run to
  completion**. We don't have latency/LUT numbers yet.

So: not done, but the remaining path is short and well-understood.

---

## What's up next (in priority order)

### 1. Verify the `input_norm` fix  ← do this first
Run the per-layer trace on a machine with the patched hls4ml + the model file:

```bash
bash hls4ml/setup_hls4ml.sh      # if hls4ml isn't installed/patched yet
python hls4ml/hls_trace.py
```

**Success looks like:** `input_norm` correlation jumps from 0.955 toward ~1.0,
and the layers after it (`ds_block_0_norm1` onward, which were −0.06 / 0.009)
recover now that the upstream 2× is gone.

- If `input_norm` is clean but a *downstream* LayerNorm still diverges, that's
  the next layer to tune (same playbook: profile its variance, set its
  `table_range_power2` / `table_t`). Capture the new trace table.
- If `input_norm` is still off, the range pick needs adjusting — re-profile its
  actual per-sample variance and confirm `max var < 2^-4 = 0.0625`.

### 2. Confirm full-model C-sim correlation
Once the trace is clean, run the end-to-end conversion + comparison:

```bash
python hls4ml/hls_convert_v2.py            # io_parallel
python hls4ml/hls_convert_iostream.py      # io_stream (optional)
```

**Target:** noise-input correlation > 0.99 (was stuck at ~0.97 because of the
`input_norm` bug) and physics correlation > 0.99 (already there at 0.997).
Record the new numbers in `hls4ml/README.md` and `docs/hls4ml_precision_bugs.md`.

### 3. Run synthesis and capture resource/latency numbers
Only after C-sim is clean — synthesizing the buggy config wastes ~30–60 min.

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

### 4. (Deferred) Train the FP32 baseline
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
