"""
Per-round metric computation.

Output schema (one row per round, per run):
  landscape, method, batch_size, seed, round, n_evaluated,
  max_fitness_raw, norm_max_fitness, mean_top10_fitness,
  frac_top1pct, frac_top0p1pct, wall_seconds_round

Path: results/raw/{landscape}_{method}_{batch_size}_seed{seed:04d}.csv
"""

import math
import numpy as np


def precompute_landscape_stats(fitness: np.ndarray) -> dict:
    """
    Call once per landscape before any runs.
    Returns a dict of precomputed values needed every round.
    """
    n = len(fitness)
    sorted_idx = np.argsort(fitness)[::-1]
    return {
        "global_max":      float(fitness.max()),
        "top1pct_set":     set(sorted_idx[: math.ceil(0.01  * n)].tolist()),
        "top0p1pct_set":   set(sorted_idx[: math.ceil(0.001 * n)].tolist()),
    }


def compute_round_metrics(
    landscape:       str,
    method:          str,
    batch_size:      int,
    seed:            int,
    round_idx:       int,
    labeled_indices: list,        # all labeled indices so far (into full landscape array)
    selected_indices: list,       # indices selected THIS round (subset of labeled)
    full_fitness:    np.ndarray,
    landscape_stats: dict,        # from precompute_landscape_stats()
    wall_seconds:    float,
) -> dict:
    """Return one row dict matching the output schema."""
    labeled_fitness = full_fitness[labeled_indices]
    labeled_set = set(labeled_indices)
    n_eval = len(labeled_indices)

    max_raw = float(labeled_fitness.max())
    norm_max = max_raw / landscape_stats["global_max"]

    top10 = float(np.sort(labeled_fitness)[-10:].mean()) if n_eval >= 10 else max_raw

    frac_top1 = (
        len(labeled_set & landscape_stats["top1pct_set"])
        / len(landscape_stats["top1pct_set"])
    )
    frac_top0p1 = (
        len(labeled_set & landscape_stats["top0p1pct_set"])
        / len(landscape_stats["top0p1pct_set"])
    )

    return {
        "landscape":          landscape,
        "method":             method,
        "batch_size":         batch_size,
        "seed":               seed,
        "round":              round_idx,
        "n_evaluated":        n_eval,
        "max_fitness_raw":    max_raw,
        "norm_max_fitness":   norm_max,
        "mean_top10_fitness": top10,
        "frac_top1pct":       frac_top1,
        "frac_top0p1pct":     frac_top0p1,
        "wall_seconds_round": wall_seconds,
        "selected_indices":   "|".join(map(str, selected_indices)),
    }
