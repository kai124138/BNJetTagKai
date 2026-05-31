#!/usr/bin/env bash
# Bootstrap a patched hls4ml clone so a fresh checkout can run the conversion.
#
# This clones hls4ml at a pinned commit known to have LayerNormalization support
# (v1.3.0, verified in docs/hls4ml_attention_support.md), then applies the three
# LayerNormalization source patches via patches/hls4ml/apply_patches.py and does
# an editable install.
#
# Run from the repository root:
#     bash hls4ml/setup_hls4ml.sh
#
# Override the pin or location with env vars:
#     HLS4ML_REF=main HLS4ML_DIR=software/hls4ml bash hls4ml/setup_hls4ml.sh
set -euo pipefail

# Resolve repo root as the parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HLS4ML_DIR="${HLS4ML_DIR:-software/hls4ml}"
# v1.3.0 — has LayerNormalization; verified in docs/hls4ml_attention_support.md.
HLS4ML_REF="${HLS4ML_REF:-fb91b2eb92b69d4c56e37b4867655844debee52b}"
HLS4ML_URL="https://github.com/fastmachinelearning/hls4ml.git"

echo "==> Target hls4ml clone: ${HLS4ML_DIR}"
echo "==> Pinned ref:          ${HLS4ML_REF}"

if [ -d "${HLS4ML_DIR}/.git" ]; then
    echo "==> Clone already exists; fetching pinned ref"
    git -C "${HLS4ML_DIR}" fetch --depth 1 origin "${HLS4ML_REF}" 2>/dev/null || \
        git -C "${HLS4ML_DIR}" fetch origin
    git -C "${HLS4ML_DIR}" checkout "${HLS4ML_REF}"
else
    echo "==> Cloning hls4ml"
    mkdir -p "$(dirname "${HLS4ML_DIR}")"
    git clone "${HLS4ML_URL}" "${HLS4ML_DIR}"
    git -C "${HLS4ML_DIR}" checkout "${HLS4ML_REF}"
fi

echo "==> Applying LayerNormalization patches"
python3 patches/hls4ml/apply_patches.py --hls4ml-root "${HLS4ML_DIR}"

echo "==> Installing hls4ml (editable)"
pip install -e "${HLS4ML_DIR}"

cat <<EOF

Done. Patched hls4ml installed from ${HLS4ML_DIR}.

Next:
  1. Place the trained model at models/deepsets_d64_l3_ffn128/deepsets_clean.h5
     (see models/MODEL.md — it is gitignored as a large binary).
  2. Run the conversion + C-sim:
       python hls4ml/hls_convert_v2.py
  3. Sanity-check the patch landed:
       grep "_table_t" models/hls4ml_deepsets_v2/firmware/defines.h
EOF
