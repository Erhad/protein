"""
Run many experiments in parallel using multiprocessing.

Usage:
    # 1000 random runs across all landscapes + batch sizes
    python experiments/run_batch.py --method random --seeds 1000

    # Specific conditions
    python experiments/run_batch.py --method evolvepro --landscapes gfp --batch_sizes 96 --seeds 10

    # All conditions for a method
    python experiments/run_batch.py --method boes_ts --seeds 10
"""

import argparse
import os
import sys
from itertools import product
from multiprocessing import Pool, cpu_count

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.run_single import run, load_landscape

ALL_LANDSCAPES  = ["gfp", "gb1", "trpb", "gb1_esmc", "t7_onehot", "tev_onehot"]
ALL_BATCH_SIZES = [1, 16, 96]

_worker_preloaded = None

def _init_worker(preloaded, limit_threads=True):
    global _worker_preloaded
    _worker_preloaded = preloaded
    if limit_threads:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass


def _run_job(args):
    landscape, method, batch_size, seed, zs_predictor, cluster_init, double_mut_init, track_calibration = args
    try:
        return run(landscape, method, batch_size, seed, zs_predictor, cluster_init, double_mut_init,
                   track_calibration=track_calibration, _preloaded=_worker_preloaded)
    except Exception as e:
        print(f"  FAILED {landscape} {method} {batch_size} seed={seed}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",        required=True)
    parser.add_argument("--landscapes",    nargs="+", default=ALL_LANDSCAPES)
    parser.add_argument("--batch_sizes",   nargs="+", type=int, default=ALL_BATCH_SIZES)
    parser.add_argument("--seeds",         type=int, default=10)
    parser.add_argument("--workers",       type=int, default=1)
    parser.add_argument("--zs_predictor",  default=None,
                        help="ZS predictor for guided init (e.g. ev-esm-esmif_score)")
    parser.add_argument("--cluster_init",   action="store_true",
                        help="Use cluster-stratified initialization")
    parser.add_argument("--double_mut_init", action="store_true",
                        help="Li et al ds-ev: sample from <=2-mut variants proportional to ev ZS score")
    parser.add_argument("--track_calibration", action="store_true",
                        help="Save RF/DNN calibration stats per round to results/calibration/")
    args = parser.parse_args()

    jobs = [
        (landscape, args.method, batch_size, seed,
         args.zs_predictor, args.cluster_init, args.double_mut_init, args.track_calibration)
        for landscape, batch_size, seed
        in product(args.landscapes, args.batch_sizes, range(args.seeds))
    ]

    print(f"Jobs: {len(jobs)}  Workers: {args.workers}")
    print(f"Landscapes: {args.landscapes}  Batch sizes: {args.batch_sizes}  "
          f"Seeds: 0–{args.seeds-1}")

    # Load landscape data once in the main process; workers inherit via initializer.
    # For pods we always have a single landscape, so this is one NFS read total.
    if len(args.landscapes) == 1:
        print(f"Pre-loading landscape '{args.landscapes[0]}' into RAM...", flush=True)
        preloaded = load_landscape(args.landscapes[0])
        print(f"  Loaded: emb={preloaded[0].shape} fitness={preloaded[1].shape}", flush=True)
    else:
        preloaded = None  # multi-landscape: fall back to per-call loading

    if args.workers == 1:
        _init_worker(preloaded, limit_threads=False)  # no nested parallelism; let torch use all cores
        results = [_run_job(job) for job in jobs]
    else:
        with Pool(args.workers, initializer=_init_worker, initargs=(preloaded,)) as pool:
            results = pool.map(_run_job, jobs, chunksize=1)

    done = sum(1 for r in results if r)
    print(f"\nCompleted {done}/{len(jobs)} runs.")


if __name__ == "__main__":
    main()
