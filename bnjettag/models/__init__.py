"""
Model builders for the BitNet jet tagger, one architecture per module:

  * transformer.build_bitnet_jet_tagger    — --arch bitnet   (default)
  * deepsets.build_deepsets_jet_tagger      — --arch deepsets
  * particle.build_particle_bitnet_tagger   — --arch particle

`build_for_arch(args)` dispatches on `args.arch` using the standard CLI knobs.
"""

from .transformer import build_bitnet_jet_tagger
from .deepsets import build_deepsets_jet_tagger
from .particle import build_particle_bitnet_tagger

__all__ = [
    "build_bitnet_jet_tagger",
    "build_deepsets_jet_tagger",
    "build_particle_bitnet_tagger",
    "build_for_arch",
]


def build_for_arch(args):
    """Build a model matching `args.arch` using the standard CLI knobs.

    Centralises the dispatch so sweep_mode, wandb_sweep, and main() all
    instantiate the architecture consistently.
    """
    fp_edges = (not args.baseline) and args.fp_edges
    v_eps    = 1e-6 if args.baseline else args.qv_eps
    arch     = getattr(args, "arch", "bitnet")
    if arch == "deepsets":
        return build_deepsets_jet_tagger(
            d_model  = args.d_model,
            n_layers = args.n_layers,
            ffn_dim  = args.ffn_dim,
            fp_edges = fp_edges,
        )
    if arch == "particle":
        return build_particle_bitnet_tagger(
            d_model  = args.d_model,
            n_layers = args.n_layers,
            ffn_dim  = args.ffn_dim,
            fp_edges = fp_edges,
            v_eps    = v_eps,
        )
    return build_bitnet_jet_tagger(
        d_model  = args.d_model,
        n_layers = args.n_layers,
        ffn_dim  = args.ffn_dim,
        fp_edges = fp_edges,
        v_eps    = v_eps,
    )
