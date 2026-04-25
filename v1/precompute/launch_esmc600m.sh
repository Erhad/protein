#!/bin/bash
# Launch ESMc-600M embedding workers on 4 GPUs, merge, and send.
# Run on a RunPod instance (even 4x T4 works — model is only 1.2GB).
#
# Usage:
#   bash precompute/launch_esmc600m.sh

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
pip install esm datasets pandas numpy -q

# ── 3. Launch 4 workers — all missing jobs ────────────────────────────────────
cd /workspace/protein/v1
JOBS="gb1 trpb tev t7"

echo "=== Launching 4 workers for: $JOBS ==="
for rank in 0 1 2 3; do
    python precompute/runpod_esmc600m.py \
        --rank $rank --world 4 \
        --only $JOBS \
        > /tmp/esmc_rank${rank}.log 2>&1 &
done

echo "Workers running. Monitor: tail -f /tmp/esmc_rank0.log"
wait

# ── 4. Merge ──────────────────────────────────────────────────────────────────
echo "=== Merging ==="
python precompute/merge_esmc600m.py

# ── 5. Send to Mac ────────────────────────────────────────────────────────────
echo "=== Sending files (paste each code on your Mac) ==="
runpodctl send /workspace/protein/v1/data/gb1/embeddings_esmc600m_meanpool.npy
runpodctl send /workspace/protein/v1/data/trpb/embeddings_esmc600m_meanpool.npy
runpodctl send /workspace/protein/v1/data/tev/embeddings_esmc600m_meanpool.npy
runpodctl send /workspace/protein/v1/data/tev/embeddings_esmc600m_4site.npy
runpodctl send /workspace/protein/v1/data/t7/embeddings_esmc600m_meanpool.npy
runpodctl send /workspace/protein/v1/data/t7/embeddings_esmc600m_3site.npy

echo "=== Done! ==="
