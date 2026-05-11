# hls4ml Precision Bugs — Diagnosis Log

This is the chronological story of debugging numerical accuracy in the hls4ml
C-simulation of the DeepSets jet tagger. It complements
`hls4ml_layernorm_patches.md` (which is the *what*) by recording the *how* —
the symptoms we saw and how we localized each issue.

## Starting symptom

After the initial conversion, Keras and HLS predictions were not just slightly
different — they were uncorrelated:

```
Keras logits: [-40.3, -34.6]      (tight cluster)
HLS   logits: [-4.9,  13.3]       (totally different range)
Correlation:  -0.02
MAE:          37
```

A correlation near zero means the HLS output is unrelated to the input, not
just shifted. Something fundamental was broken.

## Step 1 — Eliminate the obvious

Tried `ap_fixed<32,16>` as the global default precision to rule out generic
overflow. Predictions improved only slightly. So the issue wasn't simple
quantization at the dense layers.

## Step 2 — Layer-by-layer trace

Wrote `hls4ml/hls_trace.py` to compare each Keras intermediate output against
the corresponding HLS C-sim trace. This was the key diagnostic.

```
Layer                K_range            H_range            Corr
input_proj           [-1.014,  1.019]   [-1.026,  1.011]   1.000  ✓
input_norm           [-6.216,  5.938]   [-13.247, 11.559]  0.955  ~2× amp
ds_block_0_norm1     [-8.549,  9.068]   [-31.809, 31.166]  -0.060 ✗
ds_block_0_fc1       [-15.659, 14.847]  [-32.000, 31.969]  0.009  ✗
```

The first divergence is at `input_norm`. Everything downstream is corrupted by
its 2× amplification.

## Step 3 — Inspect generated firmware

`bitnet/hls4ml_deepsets_v2/firmware/defines.h` showed:

```cpp
typedef ap_ufixed<8,5> input_norm_table_t;
```

Even though `hls_convert_v2.py` set `'table': 'ap_fixed<16,6>'` in the
Precision config dict, the generated firmware still had the default
`ap_ufixed<8,5>`. The config was being silently ignored.

## Step 4 — Find where the precision is set

In `hls4ml/model/layers.py`, `_set_type_t(name)` reads precision from the
config and creates a `TypeAttribute`. There were calls for `accum_t` (working)
but **none for `table_t`** on LayerNormalization. That's why the config was
ignored — the code path that would have honored it didn't exist.

→ Applied **Patch 3** (add `TypeAttribute('table_t')` and
`self._set_type_t('table')` to `LayerNormalization`). Verified `defines.h`
now shows `typedef ap_fixed<16,6> input_norm_table_t;`.

## Step 5 — Discover the range bug

After Patch 3, `input_norm` correlation improved but later layers were still
broken. The post-residual LayerNorms have per-sample variance up to ~3800,
which is way outside the default LUT range `[0, 1)`. Setting
`table_range_power2 = -12` (range `[0, 4096)`) should fix this — but the field
was declared `unsigned`, so `-12` wrapped to a huge positive value.

→ Applied **Patches 1 + 2** (change `unsigned` to `int` in both the template
header and the Python template emitter, and replace the UB bit-shift index
computation with a float `pow`).

## Step 6 — Per-layer tuning

With all three patches in place, the remaining work was choosing the right
`table_range_power2`, `accum_t`, and `table_t` per LayerNorm based on the
empirical variance range. The values in `hls4ml/hls_convert_v2.py` are the
result of profiling each layer's variance distribution on a noise input:

| Layer              | Observed var | `table_range_power2` | Reasoning                                 |
| ------------------ | ------------ | -------------------- | ----------------------------------------- |
| `input_norm`       | 0.009–0.046  | 0                    | range `[0,1)` already covers it           |
| `ds_block_0_norm1` | ~0.83        | 0                    | still under 1                              |
| `ds_block_1_norm1` | ~2642        | -12                  | needs `[0, 4096)`                         |
| `ds_block_2_norm1` | ~2575        | -12                  | same                                       |
| `final_norm`       | ~3680        | -12                  | same                                       |

A single global `table_range_power2 = -16` (range `[0, 65536)`) was tested
early on and it broke the small-variance layers: with `input_norm` variance
0.04, the index becomes `0.04 * 4096 / 65536 ≈ 0` and the table lookup
returns `1/sqrt(0.001) ≈ 31.6` — orders of magnitude wrong. Per-layer
configuration is mandatory.

## Lessons learned

- **A correlation of -0.02 is a "totally broken" signal, not a "needs more
  bits" signal.** Don't reach for wider precision first; trace layer-by-layer
  to find the first divergence.

- **Trust but verify the generated firmware.** Reading `defines.h` after
  conversion is fast and catches "config was silently ignored" bugs that no
  amount of Python-side debugging would catch.

- **`hls_model.compile()` regenerates everything.** Any post-hoc patching of
  generated files will be overwritten — fixes must go into the source.

- **A 2× amplification at one layer cascades.** Don't try to fix downstream
  layers until the first diverging layer is correct.
