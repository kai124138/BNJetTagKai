"""Trace intermediate HLS layer outputs vs Keras to find first diverging layer."""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf, numpy as np, hls4ml

model = tf.keras.models.load_model('bitnet/deepsets_clean.h5', compile=False)
rng = np.random.default_rng(42)
X = rng.normal(0, 0.1, size=(8, 10, 14)).astype(np.float32)

cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')

cfg['Model']['Precision']['default'] = 'ap_fixed<16,6>'

LN_CONFIGS = {
    # accum must handle:
    #   sum_cache2 = dim * max_diff^2  (integer range)
    #   k_inv = 1/dim = 1/64 = 0.015625 = 2^-6  (needs >=6 fractional bits)
    #   var = sum_cache2 * k_inv  (needs enough frac bits so small variances don't truncate to 0)
    # ap_fixed<32,10>: max=511 (>256=64*4), 22 frac bits (k_inv=4194304/2^22 exact, var=0.001 -> 4194 ok)
    'input_norm':      {'table_range_power2':  0, 'accum': 'ap_fixed<32,10>', 'table': 'ap_fixed<16,6>'},
    # ap_fixed<32,15>: max=16383 (>12544=64*196), 17 frac bits (k_inv=2048/2^17 exact)
    'ds_block_0_norm1':{'table_range_power2':  0, 'accum': 'ap_fixed<32,15>', 'table': 'ap_fixed<16,6>'},
    'ds_block_1_norm1':{'table_range_power2': -12,'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
    'ds_block_2_norm1':{'table_range_power2': -12,'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
    'final_norm':      {'table_range_power2': -12,'accum': 'ap_fixed<32,23>', 'table': 'ap_fixed<24,8>'},
}
for ln, lncfg in LN_CONFIGS.items():
    cfg['LayerName'][ln]['table_range_power2'] = lncfg['table_range_power2']
    cfg['LayerName'][ln]['table_size'] = 4096
    cfg['LayerName'][ln]['Precision'] = {
        'result': 'ap_fixed<16,6>',
        'scale':  'ap_fixed<16,6>',
        'bias':   'ap_fixed<16,6>',
        'table':  lncfg['table'],
        'accum':  lncfg['accum'],
    }
    # Also set table_t key (used by _set_type_t in patched layers.py)
    if 'table_t' not in cfg['LayerName'][ln].get('Precision', {}):
        cfg['LayerName'][ln]['Precision']['table_t'] = lncfg['table']

dense_result_prec = {
    'input_proj':      'ap_fixed<16,6>',
    # fc1: 64 inputs * max~6 = 384 theoretical; accum set separately
    'ds_block_0_fc1':  'ap_fixed<16,11>',
    # fc2_linear: linear activation output after fc2 Dense computation
    # must hold the same range as fc2 result (the Dense saves its output before linear)
    # ds_block_0 fc2: Keras [-159,121] -> need I>=8 (2^8=256>159); use ap_fixed<16,9>
    'ds_block_0_fc2':  'ap_fixed<16,9>',
    'ds_block_0_fc2_linear': 'ap_fixed<16,9>',
    'ds_block_0_add':  'ap_fixed<16,9>',
    'ds_block_1_fc1':  'ap_fixed<16,8>',
    # ds_block_1 fc2: Keras [-37,23] -> I=6 (max 64>37) -> ap_fixed<16,7>
    'ds_block_1_fc2':  'ap_fixed<16,7>',
    'ds_block_1_fc2_linear': 'ap_fixed<16,7>',
    'ds_block_1_add':  'ap_fixed<16,9>',
    'ds_block_2_fc1':  'ap_fixed<16,8>',
    # ds_block_2 fc2: Keras [-99,114] -> I=7 (max 128>114) -> ap_fixed<16,8>
    'ds_block_2_fc2':  'ap_fixed<16,8>',
    'ds_block_2_fc2_linear': 'ap_fixed<16,8>',
    'ds_block_2_add':  'ap_fixed<16,9>',
    'head_fc1':        'ap_fixed<16,9>',
    # head_fc2: FP32 weights, bias can be large (≈-64 observed); use ap_fixed<16,8> for weight/bias
    'head_fc2':        'ap_fixed<16,8>',
    # head_fc2_linear: linear activation = model output; result_t must hold [-40,-38] range
    # default ap_fixed<16,6> (min=-32) wraps -40 → +24; use ap_fixed<16,8> (min=-128)
    'head_fc2_linear': 'ap_fixed<16,8>',
    'global_average_pooling1d': 'ap_fixed<16,9>',
}
for layer_name, prec in dense_result_prec.items():
    if layer_name not in cfg['LayerName']:
        cfg['LayerName'][layer_name] = {}
    if 'Precision' not in cfg['LayerName'][layer_name]:
        cfg['LayerName'][layer_name]['Precision'] = {}
    cfg['LayerName'][layer_name]['Precision']['result'] = prec
    # ternary weights only for the actual Dense compute layers (not linear activations / add)
    if layer_name not in ('input_proj', 'head_fc2', 'head_fc2_linear', 'global_average_pooling1d',
                          'ds_block_0_fc2_linear', 'ds_block_1_fc2_linear', 'ds_block_2_fc2_linear',
                          'ds_block_0_add', 'ds_block_1_add', 'ds_block_2_add'):
        cfg['LayerName'][layer_name]['Precision']['weight'] = 'ap_int<2>'

# head_fc2: FP32 weights/bias must be stored with enough precision
# bias ≈ -64 overflows default ap_fixed<16,6> (max ±32) → must use ap_fixed<16,8> (max ±128)
for key in ('weight', 'bias'):
    cfg['LayerName']['head_fc2']['Precision'][key] = 'ap_fixed<16,8>'

# Dense accumulator precision: must hold N_inputs * max_input without overflow
# fc2 layers: 128 inputs; fc1 layers: 64 inputs
# accum is separate from result — overflow in accum corrupts result regardless of result_t width
dense_accum_prec = {
    # fc1: 64 inputs * max~6 = 384; ap_fixed<24,10> has max=512 > 384
    'ds_block_0_fc1': 'ap_fixed<24,10>',
    # fc2: 128 inputs * max~15 = 1920; ap_fixed<24,12> has max=2048 > 1920
    'ds_block_0_fc2': 'ap_fixed<24,12>',
    # fc1/fc2 for blocks 1 and 2 (inputs after LN ~[-2,2] / ReLU output)
    'ds_block_1_fc1': 'ap_fixed<24,10>',
    'ds_block_1_fc2': 'ap_fixed<24,10>',
    'ds_block_2_fc1': 'ap_fixed<24,10>',
    'ds_block_2_fc2': 'ap_fixed<24,12>',
    # head_fc1: 64 inputs * max~3 (after GAP+LN) = 192; ap_fixed<24,9> max=256 > 192
    'head_fc1':       'ap_fixed<24,10>',
    # head_fc2: 64 inputs * max~15 * max_weight; ap_fixed<24,12> safe
    'head_fc2':       'ap_fixed<24,12>',
}
for layer_name, prec in dense_accum_prec.items():
    if layer_name not in cfg['LayerName']:
        cfg['LayerName'][layer_name] = {}
    if 'Precision' not in cfg['LayerName'][layer_name]:
        cfg['LayerName'][layer_name]['Precision'] = {}
    cfg['LayerName'][layer_name]['Precision']['accum'] = prec

for ln in cfg['LayerName']:
    cfg['LayerName'][ln]['Trace'] = True

print('Converting and compiling...')
hls_m = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir='/tmp/hls_trace_v2',
    backend='Vivado', io_type='io_parallel',
    part='xcvu9p-flgb2104-2L-e', clock_period=5)
hls_m.compile()

hls_pred, hls_trace = hls_m.trace(X)

print('%-35s %10s %10s | %10s %10s | %6s' % ('Layer', 'K_min', 'K_max', 'H_min', 'H_max', 'Corr'))
print('-' * 90)
for layer in model.layers[1:]:
    try:
        sub = tf.keras.Model(inputs=model.input, outputs=layer.output)
        k = sub.predict(X, verbose=0).ravel()
        h = hls_trace.get(layer.name)
        if h is None:
            print('%-35s  (no HLS trace)' % layer.name)
            continue
        h = np.array(h).ravel()
        c = np.corrcoef(k, h)[0, 1] if len(k) > 1 else float('nan')
        flag = ' <- DIVERGE' if abs(c) < 0.9 else ''
        print('%-35s %10.3f %10.3f | %10.3f %10.3f | %6.3f%s' % (
            layer.name, k.min(), k.max(), h.min(), h.max(), c, flag))
    except Exception as e:
        print('%-35s  error: %s' % (layer.name, e))

keras_final = model.predict(X, verbose=0).ravel()
hls_final   = np.array(hls_pred).ravel()
print('\nFinal: Keras=[%.3f, %.3f]  HLS=[%.3f, %.3f]  Corr=%.4f' % (
    keras_final.min(), keras_final.max(), hls_final.min(), hls_final.max(),
    np.corrcoef(keras_final, hls_final)[0,1]))
