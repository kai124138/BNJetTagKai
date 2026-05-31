# hls4ml Source Patches

The DeepSets jet tagger uses `LayerNormalization`, which has several issues in
stock hls4ml that prevent it from producing numerically correct C-simulation
output for our input distributions. These patches must be applied to your local
hls4ml clone before running `hls4ml/hls_convert_v2.py`.

## Why these patches are needed

1. **`table_range_power2` was `unsigned`** — we need negative values to extend
   the inverse-sqrt LUT range beyond `[0, 1]`. With our model, residual blocks
   produce per-sample variances up to ~3800 at later LayerNorms, so we need the
   table to cover `[0, 4096]` (i.e. `table_range_power2 = -12`).

2. **Index computation used UB bit-shift** — replaced with a `pow(2.0f, …)`
   float multiply so negative powers actually scale down correctly.

3. **`table_t` precision was hardcoded and ignored by config** — stock hls4ml
   uses `ap_ufixed<8,5>` regardless of what you put in the `Precision` dict.
   We need `ap_fixed<24,8>` (or similar) for layers with wide variance range so
   the `1/sqrt(var)` values don't get quantized to garbage. The patch adds
   `TypeAttribute('table_t')` to `LayerNormalization._expected_attributes` and
   calls `_set_type_t('table')` in `initialize()`, mirroring how `accum_t` is
   already plumbed.

## Files patched

- `hls4ml/templates/vivado/nnet_utils/nnet_layernorm.h`
- `hls4ml/backends/vivado/passes/core_templates.py`
- `hls4ml/model/layers.py`

See the individual `.md` files in this directory for the exact diffs.

## Apply

### Recommended: automated (from repo root)

The one-shot bootstrap clones hls4ml at a pinned commit, applies all three
patches, and does the editable install:

```bash
bash hls4ml/setup_hls4ml.sh
```

If you already have a clone, apply just the patches with the idempotent applier:

```bash
python patches/hls4ml/apply_patches.py --hls4ml-root software/hls4ml
cd software/hls4ml && pip install -e .
```

The applier uses anchored string replacement (not line numbers), so it is
resilient to upstream drift and safe to run twice — already-patched files are
skipped. If an anchor can't be found it fails loudly and points you back to the
`.md` diffs below for the manual edit.

### Manual fallback

After cloning hls4ml under `software/hls4ml/`, manually edit the three files
listed above using the diffs in this directory. Then re-install:

```bash
cd software/hls4ml
pip install -e .
```

## Pinned hls4ml commit

These patches were developed against the `master` branch of
`fastmachinelearning/hls4ml` as of late 2025. If the upstream code has moved
significantly, line numbers may differ — the change descriptions explain
*what* to change so you can find the right lines.
