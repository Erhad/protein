#!/bin/bash
# Launch ESM2-15B mean-pool embedding on 4 GPUs, merge, and send.
# Run this on a RunPod instance with 4x A100-80GB (or similar).
#
# Usage:
#   bash precompute/launch_esm2_15b.sh

set -e
cd /workspace

# Store HF model cache on /workspace (large network volume) not container disk
export HF_HOME=/workspace/hf_cache

# ── 1. Clone / update repo ────────────────────────────────────────────────────
if [ -d /workspace/protein ]; then
    echo "=== Pulling latest repo ==="
    cd /workspace/protein && git pull && cd /workspace
else
    echo "=== Cloning repo ==="
    git clone https://github.com/Erhad/protein.git /workspace/protein
fi

# ── 2. Install deps ───────────────────────────────────────────────────────────
echo "=== Installing deps ==="
pip install transformers accelerate pandas numpy -q

# ── 3. Pre-download model once before workers start (avoids 4×30GB simultaneous downloads) ───
echo "=== Pre-downloading ESM2-15B (~30GB, once) ==="
python - <<'EOF'
from transformers import AutoTokenizer, EsmModel
import torch
print("Downloading tokenizer...")
AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
print("Downloading model weights (this takes a while)...")
EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D", torch_dtype=torch.float16, use_safetensors=True)
print("Download complete.")
EOF

# ── 4. Launch 4 workers ───────────────────────────────────────────────────────
cd /workspace/protein/v1
echo "=== Launching 4 workers ==="
python precompute/runpod_esm2_15b.py --rank 0 --world 4 > /tmp/esm2_15b_rank0.log 2>&1 &
python precompute/runpod_esm2_15b.py --rank 1 --world 4 > /tmp/esm2_15b_rank1.log 2>&1 &
python precompute/runpod_esm2_15b.py --rank 2 --world 4 > /tmp/esm2_15b_rank2.log 2>&1 &
python precompute/runpod_esm2_15b.py --rank 3 --world 4 > /tmp/esm2_15b_rank3.log 2>&1 &

echo "Workers running. Logs: /tmp/esm2_15b_rank{0..3}.log"
echo "Monitor with: tail -f /tmp/esm2_15b_rank0.log"
wait

echo "=== All workers done. Merging... ==="
python precompute/merge_esm2_15b.py

echo "=== Sending to Mac (paste each code shown below) ==="
runpodctl send /workspace/protein/v1/data/gb1/embeddings_esm2_15b_meanpool.npy
runpodctl send /workspace/protein/v1/data/trpb/embeddings_esm2_15b_meanpool.npy

echo "=== Done! ==="
