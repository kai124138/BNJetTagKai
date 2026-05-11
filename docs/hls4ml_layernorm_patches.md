# hls4ml LayerNormalization Patches — Technical Notes

This document records the three source-level patches to `hls4ml` that were
needed to get the DeepSets jet tagger's LayerNormalization layers to produce
correct output in HLS C-simulation. The patches themselves live in
`patches/hls4ml/`; this file explains *why* each one is needed.

## Background — how hls4ml implements LayerNorm

For each sample, LayerNormalization computes:

```
mean = sum(x) / n
var  = sum((x - mean)^2) / n
y    = (x - mean) * (1 / sqrt(var + eps)) * scale + bias
```

`1 / sqrt(var + eps)` is implemented as a **lookup table** indexed by the
computed `var`. The LUT covers the range `[0, max_val)` where:

```
max_val = 2 ^ (-table_range_power2)
```

So `table_range_power2 = 0` covers `[0, 1)` (the default), `-12` covers
`[0, 4096)`, etc. The index into the table is:

```
index = var * table_size / max_val
      = var * table_size * 2^(table_range_power2)
```

There are three things hls4ml gets wrong for our use case.

## Issue 1 — `table_range_power2` typed as `unsigned`

In `hls4ml/templates/vivado/nnet_utils/nnet_layernorm.h` the value is declared
`static const unsigned table_range_power2`. The Python template in
`backends/vivado/passes/core_templates.py` similarly emits `unsigned`. This
means negative values silently wrap to huge positive ones, producing nonsense
addresses into the LUT.

**Why we need negative values:** the residual blocks in DeepSets produce
activations with values up to ±200 by the time we reach `final_norm`, giving
per-sample variances up to ~3800. The default range of `[0, 1)` covers none of
that — every variance saturates at the highest table index, yielding a single
`1/sqrt(~1)` ≈ 1 instead of the correct `1/sqrt(3800)` ≈ 0.016. The fix is to
allow `table_range_power2 = -12` (range `[0, 4096)`) for those layers.

**Patch:** change `unsigned` to `int` in both files.

## Issue 2 — UB integer bit-shift for negative powers

`nnet_layernorm.h` computed the inverse range factor as
`1 << (-table_range_power2)` and indexed the LUT with
`(var * table_size) >> (-table_range_power2)`. These shifts have undefined
behavior for negative shift amounts, and even for positive ones they can't
represent the multiplication needed when `table_range_power2 < 0`.

**Patch:** replace the bit-shifts with a float `pow`:

```cpp
float inv_range_inv = pow(2.0f, (float)(int)CONFIG_T::table_range_power2);
int   index         = (float)(var) * (float)(CONFIG_T::table_size) * inv_range_inv;
```

This correctly handles both positive and negative `table_range_power2`.

## Issue 3 — `table_t` precision is unreachable from config

Per-layer precision configuration in hls4ml goes through a `_set_type_t(name)`
method that reads from the `LayerName[...]['Precision']` dict and creates a
`TypeAttribute(name + '_t')` on the layer. The mechanism is already wired up
for `accum_t` on LayerNormalization, but **not for `table_t`** — that one is
silently hardcoded to the backend default `ap_ufixed<8,5>`.

For our model, `ap_ufixed<8,5>` is woefully inadequate for the large-variance
layers: `1/sqrt(3800) ≈ 0.0162` rounds to ~0.0156 (next representable value),
introducing a ~3.4% error per layer that compounds across the network.

**Patch:** in `hls4ml/model/layers.py`, add `TypeAttribute('table_t')` to
`LayerNormalization._expected_attributes` and call `self._set_type_t('table')`
in `initialize()`. After this, you can set `'table': 'ap_fixed<24,8>'` in the
Precision dict and have it actually be respected. Verify with:

```bash
grep "_table_t" bitnet/hls4ml_deepsets_v2/firmware/defines.h
# Expected:
#   typedef ap_fixed<16,6> input_norm_table_t;
#   typedef ap_fixed<24,8> ds_block_1_norm1_table_t;
#   ...
```

## Why three patches and not one

These are three independent issues that all happen to manifest at the same
layer:

- (1) and (2) are about the *range* of variances the LUT can represent.
- (3) is about the *resolution* of the LUT output values.

You need all three to get correct output for any model whose post-residual
variance significantly exceeds 1.

## Failed approaches (recorded so we don't repeat them)

- **Post-conversion monkey-patching of `node.types['table_t'].precision`:**
  `hls_model.compile()` calls `write()` which regenerates `defines.h` from
  scratch, blowing away any manual edits.

- **Using `backend.convert_precision_string()`:** returns a plain
  `FixedPrecisionType` lacking the HLS-wrapped `definition_cpp()` method,
  causing `AttributeError: 'FixedPrecisionType' object has no attribute
  'definition_cpp'` at write time.

- **Editing `defines.h` directly after conversion:** same problem —
  regenerated on every `compile()`.

- **Using `float` for `accum_t`:** causes a compile error because the table
  multiplication becomes ambiguous between `float * ap_ufixed` and the HLS
  fixed-point operator overloads.

The only approach that works is patching `layers.py` at the source so the
correct types are baked in from `convert_from_keras_model()` onward.
