#!/bin/bash
# Launch 4 parallel ESMc-600M embedding workers, then merge and send.
cd /workspace/protein/v1
pip install esm datasets -q

echo "=== Launching 4 workers ==="
python precompute/runpod_esmc600m.py --rank 0 --world 4 > /tmp/esmc_rank0.log 2>&1 &
python precompute/runpod_esmc600m.py --rank 1 --world 4 > /tmp/esmc_rank1.log 2>&1 &
python precompute/runpod_esmc600m.py --rank 2 --world 4 > /tmp/esmc_rank2.log 2>&1 &
python precompute/runpod_esmc600m.py --rank 3 --world 4 > /tmp/esmc_rank3.log 2>&1 &

echo "Workers launched. Waiting..."
wait
echo "=== All workers done. Merging... ==="

python precompute/merge_embeddings.py

echo "=== Sending files (paste each code to your Mac) ==="
runpodctl send /workspace/data/gb1/embeddings_esmc600m_4site.npy
runpodctl send /workspace/data/trpb/embeddings_esmc600m_4site.npy

echo "=== All done! ==="
