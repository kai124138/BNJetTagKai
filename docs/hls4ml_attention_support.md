# hls4ml Attention Support Check

Date: 2026-05-11

This records a fresh check of upstream hls4ml attention-layer support using
scratch checkouts and venvs under `software/hls4ml_attention_probe/`. The
patched local clone at `software/hls4ml/` was not modified.

## Versions Checked

Tag discovery command:

```bash
git ls-remote --tags https://github.com/fastmachinelearning/hls4ml.git \
  | grep -v '\^{}' | awk -F/ '{print $NF}' | sort -V | tail -10
```

Output:

```text
v0.7.0rc1
v0.7.1
v0.8.0
v0.8.0rc1
v0.8.1
v1.0.0
v1.1.0
v1.2.0
v1.3.0
v.0.1.0
```

The repository no longer has a `master` branch; `git ls-remote --symref ... HEAD`
reports `refs/heads/main`, so `main` was checked as the current development
branch.

| Requested ref | Checked out ref | SHA | hls4ml version | Keras v2 registry | Keras v3 registry |
| --- | --- | --- | --- | ---: | ---: |
| master/current | `main` | `a3064d50d9edff806964520015bbb96085ba10d4` | `0.1.0.dev1+ga3064d50d` | 70 | 102 |
| `v1.3.0` | `v1.3.0` | `fb91b2eb92b69d4c56e37b4867655844debee52b` | `1.3.0` | 70 | 102 |
| `v1.2.0` | `v1.2.0` | `b493ea7fd282ed969e250a94a0738f29f144ac7e` | `1.2.0` | 70 | 102 |
| `v1.1.0` | `v1.1.0` | `ad86cb68374e28b68bbecc02aecb99b0bc426516` | `1.1.0` | 65 | not present |

## Attention Registry Entries

No checked version registers standard Keras `Attention` or
`MultiHeadAttention` in the Keras v2 converter registry.

| Ref | Keras v2 attention-like entries | Keras v3 attention-like entries | Standard Keras attention support |
| --- | --- | --- | --- |
| `main` | none | `hgq.layers.attn.linformer.QLinformerAttention`, `hgq.layers.attn.mha.QMultiHeadAttention` | no |
| `v1.3.0` | none | `hgq.layers.linformer_attention.QLinformerAttention`, `hgq.layers.multi_head_attention.QMultiHeadAttention` | no |
| `v1.2.0` | none | `hgq.layers.linformer_attention.QLinformerAttention`, `hgq.layers.multi_head_attention.QMultiHeadAttention` | no |
| `v1.1.0` | none | Keras v3 registry not present | no |

The HGQ attention handlers are for HGQ/HGQ2 quantized attention classes, not
the standard TensorFlow/Keras `MultiHeadAttention` used by the transformer
model in `qkerasModel.py`.

Because no standard Keras `Attention` or `MultiHeadAttention` handler was
registered in any checked version, no two-block transformer conversion was
attempted. A toy conversion would fail at unsupported-layer dispatch rather
than testing an implemented attention path.

## Keras v2 Supported Layer Lists

### `main` / current development branch

```text
Activation, Add, Average, AveragePooling1D, AveragePooling2D, BatchNormalization, Bidirectional, BinaryDense, Concatenate, Conv1D, Conv2D, Cropping1D, Cropping2D, Dense, DepthwiseConv1D, DepthwiseConv2D, Dot, ELU, Embedding, FixedPointQuantizer, Flatten, Functional, GRU, GarNet, GarNetStack, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, HGQ>FixedPointQuantizer, HGQ>UnaryLUT, InputLayer, LSTM, LayerNormalization, LeakyReLU, MaxPooling1D, MaxPooling2D, Maximum, Minimum, Multiply, PReLU, Permute, QActivation, QBatchNormalization, QConv1D, QConv2D, QConv2DBatchnorm, QDense, QDepthwiseConv2D, QGRU, QLSTM, QSeparableConv1D, QSeparableConv2D, QSimpleRNN, ReLU, Reshape, SeparableConv1D, SeparableConv2D, Sequential, SimpleRNN, Softmax, Subtract, TernaryDense, ThresholdedReLU, TimeDistributed, UnaryLUT, UpSampling1D, UpSampling2D, ZeroPadding1D, ZeroPadding2D
```

### `v1.3.0`

```text
Activation, Add, Average, AveragePooling1D, AveragePooling2D, BatchNormalization, Bidirectional, BinaryDense, Concatenate, Conv1D, Conv2D, Cropping1D, Cropping2D, Dense, DepthwiseConv1D, DepthwiseConv2D, Dot, ELU, Embedding, FixedPointQuantizer, Flatten, Functional, GRU, GarNet, GarNetStack, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, HGQ>FixedPointQuantizer, HGQ>UnaryLUT, InputLayer, LSTM, LayerNormalization, LeakyReLU, MaxPooling1D, MaxPooling2D, Maximum, Minimum, Multiply, PReLU, Permute, QActivation, QBatchNormalization, QConv1D, QConv2D, QConv2DBatchnorm, QDense, QDepthwiseConv2D, QGRU, QLSTM, QSeparableConv1D, QSeparableConv2D, QSimpleRNN, ReLU, Reshape, SeparableConv1D, SeparableConv2D, Sequential, SimpleRNN, Softmax, Subtract, TernaryDense, ThresholdedReLU, TimeDistributed, UnaryLUT, UpSampling1D, UpSampling2D, ZeroPadding1D, ZeroPadding2D
```

### `v1.2.0`

```text
Activation, Add, Average, AveragePooling1D, AveragePooling2D, BatchNormalization, Bidirectional, BinaryDense, Concatenate, Conv1D, Conv2D, Cropping1D, Cropping2D, Dense, DepthwiseConv1D, DepthwiseConv2D, Dot, ELU, Embedding, FixedPointQuantizer, Flatten, Functional, GRU, GarNet, GarNetStack, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, HGQ>FixedPointQuantizer, HGQ>UnaryLUT, InputLayer, LSTM, LayerNormalization, LeakyReLU, MaxPooling1D, MaxPooling2D, Maximum, Minimum, Multiply, PReLU, Permute, QActivation, QBatchNormalization, QConv1D, QConv2D, QConv2DBatchnorm, QDense, QDepthwiseConv2D, QGRU, QLSTM, QSeparableConv1D, QSeparableConv2D, QSimpleRNN, ReLU, Reshape, SeparableConv1D, SeparableConv2D, Sequential, SimpleRNN, Softmax, Subtract, TernaryDense, ThresholdedReLU, TimeDistributed, UnaryLUT, UpSampling1D, UpSampling2D, ZeroPadding1D, ZeroPadding2D
```

### `v1.1.0`

```text
Activation, Add, Average, AveragePooling1D, AveragePooling2D, BatchNormalization, BinaryDense, Concatenate, Conv1D, Conv2D, Dense, DepthwiseConv1D, DepthwiseConv2D, Dot, ELU, Embedding, FixedPointQuantizer, Flatten, Functional, GRU, GarNet, GarNetStack, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, HGQ>FixedPointQuantizer, HGQ>UnaryLUT, InputLayer, LSTM, LeakyReLU, MaxPooling1D, MaxPooling2D, Maximum, Minimum, Multiply, PReLU, Permute, QActivation, QBatchNormalization, QConv1D, QConv2D, QConv2DBatchnorm, QDense, QDepthwiseConv2D, QGRU, QLSTM, QSeparableConv1D, QSeparableConv2D, QSimpleRNN, ReLU, Reshape, SeparableConv1D, SeparableConv2D, Sequential, SimpleRNN, Softmax, Subtract, TernaryDense, ThresholdedReLU, UnaryLUT, UpSampling1D, UpSampling2D, ZeroPadding1D, ZeroPadding2D
```

## Conclusion

Current upstream hls4ml releases after `v1.0.0`, plus the current development
branch, still do not support the standard Keras attention layers needed for
the BitNet transformer. The DeepSets HLS path remains necessary unless the
model is rewritten to use HGQ/HGQ2 attention layers and that path is validated
separately.
