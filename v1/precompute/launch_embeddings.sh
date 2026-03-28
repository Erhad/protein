#!/bin/bash
# Launch 4 parallel embedding processes, one per GPU.
# Run this from /workspace after cloning the repo.
#
# Usage: bash precompute/launch_embeddings.sh

WORLD=4
SCRIPT="precompute/runpod_embeddings_15B.py"

pip install -q "transformers<4.48.2" "huggingface_hub<1.0" accelerate datasets

pids=()
for rank in 0 1 2 3; do
    echo "Starting rank $rank..."
    python $SCRIPT --rank $rank --world $WORLD --batch_size 32 \
        > /workspace/embeddings/log_rank${rank}.txt 2>&1 &
    pids+=($!)
done

echo "All 4 processes launched: ${pids[@]}"
echo "Monitor with: tail -f /workspace/embeddings/log_rank*.txt"

# Wait for all
for pid in "${pids[@]}"; do
    wait $pid && echo "PID $pid done" || echo "PID $pid FAILED"
done

echo "All done. Run: python precompute/merge_embeddings.py"
