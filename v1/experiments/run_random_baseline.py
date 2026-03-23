"""
Fast vectorized random baseline.

For each seed: shuffle the full landscape index array once.
The first TOTAL_BUDGET indices of that shuffle IS the complete run.

Output: results/raw/{landscape}_random.jsonl
  One JSON record per line, one per seed:
  {
    "landscape": "gfp",
    "method": "random",
    "seed": 0,
    "selection_order": [4821, 302, 19044, ...]   # length = TOTAL_BUDGET
  }

From selection_order you can compute any metric at any n_evaluated:
    labeled = selection_order[:n]
    max_fitness = fitness[labeled].max()

Usage:
    python experiments/run_random_baseline.py --landscape gfp --seeds 1000
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANDSCAPE_CFG = {
    "gfp":  {"fitness_csv": "data/gfp/sequences_scores.csv",  "fitness_col": "score"},
    "gb1":  {"fitness_csv": "data/gb1/gb1_fitness.csv",        "fitness_col": "label"},
    "trpb": {"fitness_csv": "data/trpb/trpb_fitness.csv",      "fitness_col": "label"},
}

TOTAL_BUDGET = 480


def run(landscape: str, n_seeds: int):
    cfg = LANDSCAPE_CFG[landscape]
    fitness = pd.read_csv(
        os.path.join(ROOT, cfg["fitness_csv"])
    )[cfg["fitness_col"]].values.astype(np.float32)
    n_total = len(fitness)
    print(f"Landscape: {landscape}  N={n_total:,}  global_max={fitness.max():.4f}")

    out_path = os.path.join(ROOT, "results", "raw", f"{landscape}_random.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            order = rng.permutation(n_total)[:TOTAL_BUDGET]
            record = {
                "landscape":       landscape,
                "method":          "random",
                "seed":            seed,
                "selection_order": order.tolist(),
            }
            f.write(json.dumps(record) + "\n")

            if seed % 100 == 0:
                print(f"  seed {seed}/{n_seeds}")

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Done. {out_path}  ({size_mb:.1f} MB,  {n_seeds} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--landscape", required=True, choices=list(LANDSCAPE_CFG))
    parser.add_argument("--seeds",     type=int, default=1000)
    args = parser.parse_args()
    run(args.landscape, args.seeds)
