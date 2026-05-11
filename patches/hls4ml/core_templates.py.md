# Patch: `hls4ml/backends/vivado/passes/core_templates.py`

The C++ template string that generates the LayerNorm config struct must emit
`int` instead of `unsigned` for `table_range_power2`, matching the header
patch.

## Change — LayerNorm template (around line 155)

```python
# BEFORE
static const unsigned table_range_power2 = {table_range_power2};

# AFTER
static const int table_range_power2 = {table_range_power2};  // negative = larger variance range
```

Find the multi-line string that defines the LayerNormalization config_struct
template and update this single line.
