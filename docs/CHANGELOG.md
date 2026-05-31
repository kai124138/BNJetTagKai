# Changelog

Session-level patch notes. Newest first. See `RULES.md` for the required
format — every session that changes the repo appends an entry here.

---

## 2026-05-31 — Record verified hls4ml C-sim results (input_norm fix confirmed)

**Who:** Claude (LLM)
**Commits:** (this commit)

### What changed
- Recorded the verified C-sim/trace results in `docs/NEXT_STEPS.md`: the
  `input_norm` fix is confirmed working on hardware-accurate C-sim.
- Marked the "verify input_norm fix" and "confirm full-model C-sim" items DONE
  and promoted "run synthesis" to the top item.
- Added an optional `head_fc2`-tightening + io_stream item.
- Markdown only; no code changed.

### Why
- The owner ran `hls_trace.py` and `hls_convert_v2.py` on a patched machine and
  pasted the output. Per `RULES.md`, verified results must be logged so the next
  session starts from the real state, not the unverified one.

### Verified (measured, by the owner on a patched machine)
- `hls_trace.py`: `input_norm` corr **0.955 → 1.000**;
  `ds_block_0_norm1` **−0.06 → 1.000**; all layers **1.000** through `head_fc1`;
  `head_fc2` **0.979** (tight-cluster artifact, not an accuracy problem);
  Final Corr **0.9788**.
- `hls_convert_v2.py` (io_parallel): physics corr **0.995**, MAE 0.578;
  HLS AUC **0.4505** vs Keras **0.4429** → matches within tolerance (0.008).
  Note: the 0.44 AUC is a 46-jet smoke test, not a quality metric — it only
  proves HLS tracks Keras. Real AUC ≈ 0.989 is from `ROC.py` on full validation.

### Still broken / unfinished
- Synthesis (`hls_build.py`) still never run — no latency/resource numbers. Now
  the top item in `NEXT_STEPS.md`.
- FP32 baseline (`training/transformer_fp32.py`) still untrained (needs GPU).

---

## 2026-05-31 — Add contributor rules + changelog

**Who:** Claude (LLM)
**Commits:** (this commit)

### What changed
- Added `RULES.md`: the canonical "read this first" contract for humans and
  LLMs — required per-session workflow, mandatory patch-notes + next-steps
  updates, and version-control conventions.
- Added this `docs/CHANGELOG.md` and seeded it with the recent sessions.
- Added `AGENTS.md` pointing auto-discovering agent tools to `RULES.md`.
- Added a "read first" pointer near the top of `README.md`.

### Why
- The repo is worked on in short bursts by different people/LLMs. Without an
  enforced habit of leaving patch notes and an updated "what's next," each new
  session starts blind. The owner asked for a rules file that the next LLM (or
  person) must follow so context carries forward.

### Verified
- Markdown only; no code changed. Nothing to run.

### Still broken / unfinished
- Nothing from this change. Project-level open items are unchanged — see
  `docs/NEXT_STEPS.md` (verify the `input_norm` trace is the current top item).

---

## 2026-05-31 — Fix hls4ml input_norm 2× amplification + reproducibility

**Who:** Claude (LLM)
**Commits:** `31418be`, `4d03f71`

### What changed
- Fixed the `input_norm` ~2× output amplification: set its LayerNorm
  `table_range_power2` from `0` to `4` (LUT range `[0, 2^-4)`) and widened
  `table_t` to `ap_fixed<18,6>`. Applied consistently across
  `hls_convert_v2.py`, `hls_convert_iostream.py`, `hls_trace.py`, and
  `hls_build.py`.
- Added an explicit `table_t` key to the LayerNorm precision dicts so the
  patched `_set_type_t('table')` path is guaranteed to fire.
- Added `patches/hls4ml/apply_patches.py`: an idempotent, anchored applier for
  the three LayerNorm source patches (previously only prose `.md` snippets).
- Added `hls4ml/setup_hls4ml.sh`: one-shot bootstrap (clone hls4ml @ pinned
  v1.3.0, apply patches, editable install).
- Added `models/MODEL.md` + a `.gitignore` negation documenting the gitignored
  model artifact a fresh clone needs.
- Added `docs/NEXT_STEPS.md` roadmap and a draft abstract.

### Why
- `input_norm` was the first diverging layer (corr 0.955, clean 2× over-
  amplification). Root cause was a LUT *range* mismatch (not `accum_t`
  resolution as previously hypothesized): tiny variances (~0.009–0.046) occupied
  only the bottom ~4.6% of a `[0,1)` table, so the steep low-index region of
  `1/sqrt(var)` was undersampled and read ~2× high.
- The prose-only patches and gitignored model made a fresh clone unrunnable.

### Verified
- All edited Python syntax-checked; `setup_hls4ml.sh` `bash -n` clean.
- `apply_patches.py` tested against a mock hls4ml tree: applies all three
  patches correctly and is idempotent on a second run.
- `.gitignore` negation confirmed: only `models/MODEL.md` is tracked; `*.h5`
  and generated project dirs stay ignored.
- **NOT verified:** the actual C-sim/trace improvement — requires patched
  hls4ml + the model file + TensorFlow, none available in this environment. The
  fix is sound on the LUT-indexing math but needs a real `hls_trace.py` run to
  confirm on hardware-accurate C-sim.

### Still broken / unfinished
- Run `python hls4ml/hls_trace.py` to confirm `input_norm` corr → ~1.0 and the
  downstream cascade clears. This is the current top item in `NEXT_STEPS.md`.
- Synthesis (`hls_build.py`) still never run — no latency/resource numbers.
- FP32 baseline (`training/transformer_fp32.py`) still untrained (needs GPU).

---

## 2026-05-11 — Russell feedback items 1–6 (Codex)

**Who:** Codex (LLM)
**Commits:** `aa65681`, `1dc59bf`, `5d2257e`, `df367ec`, `f3625b8`, `a1ded6a`,
`4c02c5f`

### What changed
- Reorganized model artifacts by run under `models/`.
- Documented hls4ml attention support findings (`docs/hls4ml_attention_support.md`).
- Aligned ROC metrics with the benchmark paper; added `docs/paper_notes_2510.24784.md`.
- Added FP32 transformer baseline scaffold (`training/transformer_fp32.py`).
- Documented the knowledge-distillation pipeline (`docs/knowledge_distillation.md`).
- Added io_stream hls4ml conversion (`hls4ml/hls_convert_iostream.py`).

### Why
- Working through Russell's review items 1–6 for BNJetTagKai.

### Verified
- `python ROC.py` ran and reported AUC 0.989171 for the BitNet KD model.
- io_parallel and io_stream C-sim correlations recorded in the session summary
  (physics corr 0.997 / 0.999).

### Still broken / unfinished
- Item 4 FP32 training deferred (no GPU visible).
- Item 7 synthesis deferred (not run).
- Full details in `docs/session_summary_2026-05-11.md`.
