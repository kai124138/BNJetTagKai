# Patch: `hls4ml/model/layers.py`

Without this patch, the `table` precision for LayerNormalization cannot be set
through the standard `Precision` config dict — it stays at the hardcoded
backend default (`ap_ufixed<8,5>`), which is far too narrow for our 1/sqrt
values. The patch wires `table_t` through the same config mechanism that
already exists for `accum_t`.

## Change — `LayerNormalization` class

Add `TypeAttribute('table_t')` to `_expected_attributes` and call
`self._set_type_t('table')` from `initialize()`.

```python
class LayerNormalization(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('seq_len'),
        Attribute('axis', value_type=int, default=2),
        Attribute('epsilon_power_of_10', value_type=int, default=3),
        WeightAttribute('scale'),
        WeightAttribute('bias'),
        TypeAttribute('scale'),
        TypeAttribute('bias'),
        TypeAttribute('table_t'),  # <-- ADD THIS LINE
    ]

    def initialize(self):
        # ... existing initialize body ...
        self.add_weights_variable(name='scale', var_name='s{index}', data=scale)
        self.add_weights_variable(name='bias',  var_name='b{index}', data=bias)
        # ADD THIS LINE: allow 'table' precision to be configured via LayerName config
        self._set_type_t('table')
```

After applying this patch, you can specify `'table': 'ap_fixed<24,8>'` inside
the `LayerName[...]['Precision']` config dict in `hls4ml/hls_convert_v2.py`
and it will actually be honored in the generated `defines.h`.

You can verify the patch worked by checking the generated firmware:

```bash
grep "_table_t" models/hls4ml_deepsets_v2/firmware/defines.h
# Expected: typedef ap_fixed<24,8> ds_block_1_norm1_table_t;
#           (and similar for other LNs, matching what you configured)
```
