# Model Artifacts

Model artifacts are grouped by training run rather than by file type.

Directory names use:

```text
<architecture>_d<d_model>_l<n_layers>_ffn<ffn_dim>[_kd|_fp32]
```

- `transformer_d64_l3_ffn128_kd/` contains the current BitNet transformer run.
  The flat transformer files that previously lived in `bitnet/` were produced
  by the KD workflow and are treated as the KD run.
- `deepsets_d64_l3_ffn128/` contains the attention-free DeepSets run used for
  hls4ml conversion, including `deepsets_clean.h5`.

Generated hls4ml project directories live directly under `models/` and are
ignored by git because they can be regenerated from the conversion scripts.
