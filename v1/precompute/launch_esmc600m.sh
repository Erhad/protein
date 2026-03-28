#!/bin/bash
# Launch 4 parallel ESMc-600M embedding workers (one per GPU)
cd /workspace/protein/v1
pip install esm datasets -q

python precompute/runpod_esmc600m.py --rank 0 --world 4 > /tmp/esmc_rank0.log 2>&1 &
python precompute/runpod_esmc600m.py --rank 1 --world 4 > /tmp/esmc_rank1.log 2>&1 &
python precompute/runpod_esmc600m.py --rank 2 --world 4 > /tmp/esmc_rank2.log 2>&1 &
python precompute/runpod_esmc600m.py --rank 3 --world 4 > /tmp/esmc_rank3.log 2>&1 &

echo "All 4 workers launched. Monitor with:"
echo "  tail -f /tmp/esmc_rank*.log"
wait
echo "All done!"
