#!/bin/bash
# Launch ESM2-15B embedding workers on 4 GPUs, merge, and send.
# Run on a RunPod instance with 4x A100-40GB or A100-80GB.
#
# Usage:
#   bash precompute/launch_esm2_15b.sh

set -e
cd /workspace

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
pip install transformers accelerate pandas numpy datasets -q

# ── 3. Pre-download model once (avoids 4x simultaneous 30GB downloads) ────────
echo "=== Pre-downloading ESM2-15B (~30GB) ==="
python - <<'EOF'
from huggingface_hub import snapshot_download
print("Downloading ESM2-15B safetensors only (~30GB)...")
snapshot_download(
    "facebook/esm2_t48_15B_UR50D",
    cache_dir="/workspace/hf_cache",
    ignore_patterns=["*.bin", "*.bin.index.json"],
)
print("Download complete.")
EOF

# ── 4. Launch 4 workers — all missing jobs ────────────────────────────────────
cd /workspace/protein/v1
JOBS="gb1 trpb tev t7"

echo "=== Launching 4 workers for: $JOBS ==="
for rank in 0 1 2 3; do
    python precompute/runpod_esm2_15b.py \
        --rank $rank --world 4 \
        --only $JOBS \
        > /tmp/esm2_15b_rank${rank}.log 2>&1 &
done

echo "Workers running. Monitor: tail -f /tmp/esm2_15b_rank0.log"
wait

# ── 5. Merge ──────────────────────────────────────────────────────────────────
echo "=== Merging ==="
python precompute/merge_esm2_15b.py

# ── 6. Send to Mac ────────────────────────────────────────────────────────────
echo "=== Sending files (paste each code on your Mac) ==="
runpodctl send /workspace/protein/v1/data/gb1/embeddings_esm2_15b_4site.npy
runpodctl send /workspace/protein/v1/data/trpb/embeddings_esm2_15b_4site.npy
runpodctl send /workspace/protein/v1/data/tev/embeddings_esm2_15b_meanpool.npy
runpodctl send /workspace/protein/v1/data/tev/embeddings_esm2_15b_4site.npy
runpodctl send /workspace/protein/v1/data/t7/embeddings_esm2_15b_meanpool.npy
runpodctl send /workspace/protein/v1/data/t7/embeddings_esm2_15b_3site.npy

echo "=== Done! ==="
