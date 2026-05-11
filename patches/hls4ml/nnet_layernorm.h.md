# Patch: `hls4ml/templates/vivado/nnet_utils/nnet_layernorm.h`

Fixes table-range type and lookup-index computation for LayerNormalization.

## Change 1 — `table_range_power2` type (around line 25)

Change the type from `unsigned` to `int` so it can hold negative values, which
are needed to extend the `1/sqrt(var)` LUT range beyond `[0, 1]`.

```cpp
// BEFORE
static const unsigned table_range_power2 = 0;

// AFTER
static const int table_range_power2 = 0;  // negative = larger variance range
```

## Change 2 — Inverse-range computation (around line 54)

The original used a left-shift on `table_range_power2` to compute the scaling
factor for table indexing. With negative values this is undefined behavior.
Replace with a float `pow`.

```cpp
// BEFORE
unsigned inv_range_inv = 1 << (-CONFIG_T::table_range_power2);

// AFTER
float inv_range_inv = pow(2.0f, (float)(int)CONFIG_T::table_range_power2);
```

## Change 3 — Index computation (around line 92)

Replace the integer-shift index calculation with a float multiplication using
the value from Change 2.

```cpp
// BEFORE
int index = (var * CONFIG_T::table_size) >> (-CONFIG_T::table_range_power2);

// AFTER
int index = (float)(var) * (float)(CONFIG_T::table_size) * inv_range_inv;
```

After these changes, negative `table_range_power2` values (e.g. `-12` for a
range of `[0, 4096]`) work correctly during C-simulation.
