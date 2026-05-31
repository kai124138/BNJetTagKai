"""
BitNet-style 1-bit Transformer jet tagger for the CMS L1 trigger.

Package layout
--------------
  config.py      constants, dataset paths, runtime toggles (QAT/FP/act/stoch)
  layers.py      BitLinear, RMSNorm, ternary MHSA/FFN, ParT pair-bias + [CLS]
  losses.py      focal loss, partial-AUC surrogates, ROC helpers
  callbacks.py   AUCReshapingCallback
  data.py        HDF5 loading + per-category test evaluation
  wandb_utils.py WandbTracker / WandbEpochLogger
  sweeps.py      LR/WD grid + W&B Bayesian sweep
  hls_export.py  hls4ml YAML config + FPGA resource estimate
  sanity.py      shape/weight sanity check
  models/        one builder per architecture (transformer/deepsets/particle)

The training entrypoint is the top-level `train.py`, not this package.

`CUSTOM_OBJECTS` collects every custom layer/constraint so saved `.h5` models
can be reloaded with `load_model(path, custom_objects=CUSTOM_OBJECTS)`.
"""

from .layers import (
    AbsMeanQuantizer, BitLinear, RMSNorm,
    BitMHSA, BitFFN, BitTransformerBlock,
    PairFeatures, PairBiasMHSA, PairBiasTransformerBlock,
    ClassToken, ClassAttentionBlock,
)
from .models import (
    build_bitnet_jet_tagger, build_deepsets_jet_tagger,
    build_particle_bitnet_tagger, build_for_arch,
)

CUSTOM_OBJECTS = {
    "AbsMeanQuantizer":          AbsMeanQuantizer,
    "BitLinear":                 BitLinear,
    "RMSNorm":                   RMSNorm,
    "BitMHSA":                   BitMHSA,
    "BitFFN":                    BitFFN,
    "BitTransformerBlock":       BitTransformerBlock,
    "PairFeatures":              PairFeatures,
    "PairBiasMHSA":              PairBiasMHSA,
    "PairBiasTransformerBlock":  PairBiasTransformerBlock,
    "ClassToken":                ClassToken,
    "ClassAttentionBlock":       ClassAttentionBlock,
}

__all__ = list(CUSTOM_OBJECTS) + [
    "CUSTOM_OBJECTS",
    "build_bitnet_jet_tagger",
    "build_deepsets_jet_tagger",
    "build_particle_bitnet_tagger",
    "build_for_arch",
]
