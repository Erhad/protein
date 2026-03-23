# Thesis Experiment Plan — V1
**Last updated:** 2026-03-22 (revised after pilot results)
**Experiment:** Acquisition function × batch size interaction across surrogate architectures and protein landscapes

---

## 0. Pilot Results — Key Findings (GB1, EVOLVEpro, seeds 0–4)

Before the main experiment, 8 pilot runs were completed on GB1 (5 × batch=96, 3 × batch=16).

**Finding 1 — batch=96:** EVOLVEpro achieves mean normalised fitness 0.868 ± 0.066 after 480 labels. This is competitive with ALDE's middle-tier configurations (~0.80–0.88) and clearly above the DE greedy-walk baseline (~0.67). However, 4/5 seeds converge to rank #5 (norm 0.835); only 1/5 finds the global max (norm 1.000). The distribution is bimodal, not smooth.

**Finding 2 — batch=16 (critical):** All 3 seeds converge to rank #5 with zero variance from round 8 (~224 labels) onward. The method finds the identical variant across all seeds and then wastes the remaining 256 label budget confirming the same local basin. Std drops to exactly 0.000.

**Interpretation:** Pure greedy exploitation creates a confirmation loop — the RF becomes confident about the rank #5 region early and exclusively exploits it. Small batches accelerate this pathology (more frequent model updates → faster commitment). This is the local optima problem EVOLVEpro's abstract claims to solve but provides no algorithmic mechanism for.

**Implication for experimental design:** The acquisition function is the critical variable, not (only) the surrogate. UCB and Thompson sampling are one-line changes on top of the existing RF and directly test whether exploration resolves the convergence failure. This is the cleanest ablation the thesis can make.

---

## 1. Research Question

> Does acquisition function (greedy vs UCB vs Thompson sampling) interact with batch size to determine MLDE success — and does surrogate architecture (RF vs GP) modulate this interaction across landscape types?

The primary ablation is now **acquisition function × batch size** (the pilot results motivate this directly). The secondary comparison is **surrogate type** (RF vs GP) at fixed acquisition. EVOLVEpro-greedy is expected to degrade severely at small batches due to early commitment to local optima. RF+TS and GP+TS are expected to escape this through exploration. The key question is whether exploration matters more at small or large batch sizes and whether landscape ruggedness amplifies the effect.

**Reference baselines (not reimplemented):**
- ALDE (Yang et al. 2024) — published GB1 and TrpB results used directly if conditions are matched (see Section 6.3)
- Random — always included as lower bound

---

## 2. Methods to Implement

The methods are now organised as two surrogate families, each with an acquisition function ablation. This gives a clean 2×3 design (surrogate × acquisition) that directly addresses the pilot finding.

### 2.1 RF Family (EVOLVEpro surrogate)

All three variants share identical surrogate, encoding, and hyperparameters. Only the acquisition function differs.

**Shared:**
- **Surrogate:** Random Forest (`RandomForestRegressor`, n_estimators=100, friedman_mse criterion)
- **Encoding:** ESM2-650M mean-pooled full-sequence embeddings (1280-dim) — for ALL landscapes
- **Preprocessing:** StandardScaler fit on training set only
- **Note on published model:** EVOLVEpro paper used ESM-2 15B (optimal). 650M is ~10pp worse on their DMS benchmarks. Standardised to 650M for hardware feasibility and fair comparison. Acknowledged in thesis.

**2.1a RF-Greedy (EVOLVEpro as published)**
- Acquisition: `argsort(mean_prediction)[-batch_size:]`
- UQ used: None
- Status: **implemented, pilot runs complete**

**2.1b RF-UCB**
- Acquisition: `argsort(mean + beta * std)[-batch_size:]`
- Std from: variance across tree predictions = `np.std([t.predict(X) for t in rf.estimators_], axis=0)`
- Beta: 2.0 (default; can sweep 1.0/2.0/3.0 as sensitivity check on GB1 only)
- UQ used: Frequentist tree variance
- Status: **one-line change from RF-Greedy, implement next**

**2.1c RF-TS (Thompson Sampling)**
- Acquisition: randomly sample one tree from the forest, use its predictions as acquisition score
- Implementation: `tree = rng.choice(rf.estimators_); scores = tree.predict(X_pool)`
- UQ used: Frequentist — stochastic tree selection approximates posterior sampling
- Parameter-free (no beta)
- Status: **one-line change, implement alongside RF-UCB**

**Novel contribution:** All three RF variants have never been tested on GB1, TrpB, or avGFP with this exact setup. The RF-UCB and RF-TS variants are the thesis's methodological contribution on top of the EVOLVEpro baseline.

---

### 2.2 BOES (Shen et al., BIBM 2024)

- **Surrogate:** Gaussian Process (GPyTorch, exact GP)
- **Kernel:** Matérn 3/2. Initial implementation: single isotropic length scale (one global ℓ, as in paper). Ablation (GB1 only): ARD Matérn (one ℓ per dimension).
- **Encoding:**
  - GB1 / TrpB: ESM2-650M **per-residue at the 4 targeted sites** → 4 × 1280 = **5120-dim** input. Apply StandardScaler before GP fitting (critical — GP kernels are not scale-invariant).
  - GFP: ESM2-650M **mean-pooled** full-sequence (1280-dim). Mean-pool is appropriate here; per-site encoding is undefined for a full-protein landscape with variable mutation positions.
- **Acquisition function (two variants):**
  - `boes_ei`: Expected Improvement — faithful to published BOES. EI incumbent = max fitness in labeled set.
  - `boes_ts`: Thompson Sampling — the thesis addition. Requires GP posterior sampling (see note below).
- **UQ:** Bayesian posterior variance
- **TS implementation for GP:** Use Matheron's rule (exact, tractable up to ~10k candidates). For GB1's 149k candidate pool at batch=96: use Random Fourier Feature (RFF) approximation or subsample the candidate pool to 20k without replacement each round.
- **Note on BIBM 2024:** The paper used ESM-1b (not ESM2) and only tested EI (not TS). The thesis extends this by (1) using ESM2-650M for consistency across methods and (2) adding TS as a second acquisition function.
- **Scaling warning:** Exact GP is O(n³) in training set size. Cap kernel hyperparameter restarts to 5. Profile at batch=1 before full run — this is the computational bottleneck.

---

### 2.3 DNN-ESM2 (Deprioritised — Future Work)

Originally planned as an optional third arm. **Deprioritised** in favour of the RF acquisition function ablation (Section 2.1b/c), which is more directly motivated by the pilot results and cheaper to implement.

Rationale for deferral: the RF-UCB vs RF-TS ablation cleanly isolates the acquisition function effect within a fixed surrogate. Adding a DNN surrogate at the same time confounds surrogate quality with acquisition quality. The thesis is stronger with a clean 2-surrogate × 3-acquisition design than a 3-surrogate × mixed-acquisition design.

If time permits after the core experiment: implement DNN-TS on GB1 only, at batch=96, as a single data point comparing against ALDE's DNN-TS published number. This would be one additional method × one landscape × one batch size = 30 runs, low marginal cost.

---

### 2.4 ALDE — Reference Baseline Only (Yang et al., bioRxiv July 2024)

**Not reimplemented.** Published results from ALDE's Fig 4D (GB1 and TrpB) are used as a comparison point, subject to the conditions in Section 6.3.

Key facts for interpretation:
- Encoding: one-hot at the 4 targeted sites (4 × 20 = 80-dim). **Not applicable to GFP.**
- Surrogate: DNN ensemble of 5 MLPs (256→128→64→1)
- Acquisition: Thompson Sampling
- Landscapes: GB1 (149k variants), TrpB (160k variants). GFP not tested.
- Simulation setup: **96 initial samples + 4 rounds × 96 = 480 total evaluations**, 70 random seeds
- Metric: **normalized max fitness** = max fitness found / global landscape maximum
- Key finding: Frequentist DNN ensemble outperforms Bayesian GP; ESM2 embeddings could not be used with GP (too high-dimensional)

---

## 3. ESM2 Model Choice

**Standardise to ESM2-650M (`esm2_t33_650M_UR50D`) throughout.** Output: 1280-dim per residue.

Why 650M not larger:
- 3B and 15B require multi-GPU; impractical for precomputing 149k GB1 embeddings
- ESM2-650M is the standard in ALDE (Yang 2024), Shen 2024, and other recent work
- EVOLVEpro optimal was 15B, but 650M is ~10pp worse — noted in thesis, acceptable tradeoff

**⚠ Open question to resolve before experiments:** The existing GFP embeddings in `archive/data/esm_embeddings_mean_pooled.npz` are suspected to be 320-dim (from the 8M ESM2 model, not 650M) based on the discrepancy flagged in TABLE_papers_and_datasets.md. Load the file and check `arr.shape[1]`. If 320, recompute with `esm2_t33_650M_UR50D` before running anything.

**Embedding modes by landscape:**

| Landscape | Method | Mode | Dims |
|---|---|---|---|
| GB1 | BOES | Per-residue at 4 targeted sites | 4 × 1280 = **5120** |
| GB1 | EVOLVEpro / DNN-ESM2 | Mean-pool full sequence | **1280** |
| TrpB | BOES | Per-residue at 4 targeted sites | 4 × 1280 = **5120** |
| TrpB | EVOLVEpro / DNN-ESM2 | Mean-pool full sequence | **1280** |
| GFP | All methods | Mean-pool full sequence | **1280** |

Precompute and cache all embeddings before any experiments. Store as `.npz`. Never recompute mid-experiment.

For GB1/TrpB 4-site embeddings: run the full 56-aa (GB1) or full-length (TrpB) sequence through ESM2, extract the embedding vectors at the four targeted residue positions, concatenate → 5120-dim vector per variant.

---

## 4. Protein Landscapes

### 4.1 GFP (avGFP) — already have

- **Source:** Sarkisyan et al. 2016
- **Files in archive:** `archive/data/sequence_and_score.csv` (58,417 variants), `archive/data/esm_embeddings_mean_pooled.npz`
- **Action:** Copy to `v1/data/gfp/`. **Verify embedding dimensions first** (see Section 3 note).
- **Pool:** Observed variants only — cannot propose unseen sequences. Pool-based optimization.
- **Fitness normalisation for comparison with ALDE:** ALDE did not run on GFP, so no direct paper comparison here. Use raw log-fluorescence scores from the CSV.

### 4.2 GB1 — need to download

- **Source:** Wu et al. 2016
- **Download:** FLIP benchmark: https://github.com/J-SNACKKB/FLIP/tree/main/splits/gb1
  - Use the full dataset CSV (not a specific split). Filter: keep rows where all 4 sites are standard AAs and fitness is not NaN. Target: 149,361 variants.
  - Alternatively: HuggingFace `SaProtHub/Dataset-GB1-fitness`
- **Fitness column:** enrichment score (log scale); the global maximum is the variant V39E/D40G/G41A/V54L
- **Targeted sites:** V39, D40, G41, V54 (0-indexed positions 38, 39, 40, 53 in the 56-aa sequence)
- **4-site embedding for BOES:** extract ESM2 per-residue vectors at these positions, concatenate → 5120-dim
- **Action:** Download → `v1/data/gb1/gb1_fitness.csv`. Compute embeddings → `v1/data/gb1/embeddings_esm2_650m_4site.npz` and `v1/data/gb1/embeddings_esm2_650m_meanpool.npz`

### 4.3 TrpB — need to download

- **Source:** Johnston et al. 2024; sub-variant navigability from Li et al. 2024 (ftALDE paper)
- **Download:** https://github.com/fhalab/TrpB (or paper supplementary)
- **Sub-variant selection:** Choose one sub-variant with fraction active ~10–15% (intermediate navigability — harder than trivial, easier than near-impossible). This contrasts with GB1 which has ~8% variants above 1% of maximum fitness. Lock this in before running — changing it invalidates all TrpB results.
- **Targeted sites:** 4 positions (confirm from Johnston et al. 2024 paper)
- **Action:** Download → `v1/data/trpb/trpb_fitness.csv`. Compute both embedding types (4-site and mean-pool) as for GB1.

---

## 5. Libraries

```bash
# Core ML
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gpytorch botorch          # GP / BOES
pip install scikit-learn              # RF (EVOLVEpro), metrics
pip install numpy pandas scipy

# ESM2 (choose one)
pip install fair-esm                  # Meta's original package
# OR: pip install esm                 # newer API

# Experiment management
pip install pyyaml                    # config files
pip install wandb                     # optional run tracking

# Analysis / plotting
pip install matplotlib seaborn
pip install statsmodels               # confidence intervals
pip install datasets                  # HuggingFace (for GB1 download)
```

Hardware: GPU required for ESM2 embedding precomputation (~2–4 hours per landscape at 150k sequences for ESM2-650M). After that, all surrogate methods fit comfortably on CPU at the training set sizes used here.

---

## 6. Experimental Design

### 6.1 Budget Alignment with ALDE

**This is critical for the ALDE comparison.** ALDE ran: `96 initial + 4 rounds × 96 = 480 total evaluations, 70 seeds`.

To enable direct comparison of our GB1 and TrpB results against ALDE's published Fig. 4D numbers:
- Set **initial training size = 96** (matching ALDE)
- Set **total evaluations = 480** (matching ALDE)
- At **batch=96**: this is exactly 4 rounds of 96 (ALDE-compatible)
- At **batch=16**: this is 24 rounds of 16 (same total budget)
- At **batch=1**: this is 384 rounds of 1 (same total budget — computationally expensive, profile first)
- Use **normalized max fitness** as primary metric: `norm_fitness = max_fitness_found / global_landscape_maximum`
- Run **≥ 30 seeds** (ideally 70 to exactly match ALDE, but 30 is sufficient for CIs)

**Note:** The batch=1 condition at 384 rounds is the computational bottleneck for GP (BOES). Profile one full run before committing all seeds. If wall-clock is prohibitive, reduce the batch=1 experiment to GB1 only with a clear note.

### 6.2 Parameters

| Parameter | Value | Notes |
|---|---|---|
| Initial training set size | **96** | Matches ALDE; changed from prior plan of 16 |
| Total evaluations | **480** | Matches ALDE (96 + 4×96); changed from prior plan of 208 |
| Batch sizes | **1, 16, 96** | Gives 384, 24, 4 rounds respectively |
| Seeds | **30–70** | 30 minimum for CIs; 70 matches ALDE exactly |
| Landscapes | GB1, TrpB, GFP | |
| Methods | RF-Greedy, RF-UCB, RF-TS, BOES-EI, BOES-TS | |
| Random baseline | Always included | Single run per landscape; greedy random sampling |
| UCB beta | 2.0 | Sensitivity sweep (1.0/2.0/3.0) on GB1 only if time permits |

**Total runs (5 methods × 3 landscapes × 3 batch sizes × 30 seeds):** 1,350 runs minimum. RF methods are fast (seconds/round); GP methods are the bottleneck. Profile BOES at batch=1 before committing.

**Priority order if time is constrained:**
1. GB1, all 5 methods, batch=96 (directly comparable to ALDE — most important)
2. GB1, all 5 methods, batch=16 (most interesting given pilot finding)
3. TrpB, all 5 methods, batch=96
4. GFP, all 5 methods, batch=96
5. All remaining batch=1 and batch=16 conditions

### 6.3 ALDE Comparability Conditions

To cite ALDE's published GB1 and TrpB numbers directly (from their Fig. 4D or Table S3), the following conditions must hold:

| Condition | ALDE paper | Our setup | Compatible? |
|---|---|---|---|
| Initial training size | 96 | 96 | ✓ |
| Total evaluations | 480 | 480 | ✓ |
| Batch size | 96 | 96 (one of three conditions) | ✓ for batch=96 only |
| Seeds | 70 | 30–70 | ✓ (note if fewer) |
| Metric | Normalized max fitness | Normalized max fitness | ✓ |
| Landscape | GB1, TrpB | GB1, TrpB | ✓ |
| Landscape version | Full 149,361 variants | Full 149,361 variants | ✓ (verify same source) |
| Oracle | True fitness from dataset | True fitness from dataset | ✓ |

**Result:** At batch=96 on GB1 and TrpB, our EVOLVEpro, BOES-EI, and BOES-TS results are directly comparable to ALDE's published DNN-ensemble+TS numbers. ALDE's performance number at round 4 (n=480) can be plotted as a reference horizontal line on the learning curve figure.

**Caveat to state in thesis:** "ALDE numbers are reproduced from Yang et al. (2024) Table/Fig X and were run at identical conditions (96-sample batches, 70 seeds). Our methods are reimplemented from scratch; ALDE's number is cited directly to avoid re-implementing a method that uses one-hot encoding, which is not applicable to the GFP landscape."

### 6.4 Per-Run Loop

```
INPUT: landscape (pool of variants + oracle fitness), method, batch_size, seed
FIXED: initial_n = 96, total_budget = 480

1. INIT
   - Set random seed globally (torch, numpy, random)
   - Sample initial_n = 96 variants uniformly at random
   - labeled_set = {(index, fitness)} for initial 96 variants
   - remaining_pool = all variants not in labeled_set

2. LOOP until len(labeled_set) >= total_budget:
   a. Train surrogate on labeled_set
      - EVOLVEpro: fit RF (n_estimators=100, StandardScaler on inputs)
      - BOES: fit GP (Matérn 3/2, optimize kernel hyperparams via marginal likelihood, max 5 restarts)
      - DNN-ESM2: train 5 MLPs from scratch (independent seeds per member)

   b. Predict on remaining_pool
      - EVOLVEpro: RF mean prediction
      - BOES: GP posterior mean + variance
      - DNN-ESM2: mean + variance across 5 MLPs

   c. Acquire next batch of size batch_size
      - EVOLVEpro: top-k by predicted mean (greedy)
      - BOES-EI: Expected Improvement (incumbent = max fitness in labeled_set)
      - BOES-TS: Thompson Sampling from GP posterior (Matheron's rule or RFF)
      - DNN-ESM2: TS (randomly selected ensemble member per candidate)

   d. Oracle lookup: get true fitness for selected indices from dataset

   e. Add selected variants to labeled_set, remove from remaining_pool

   f. RECORD round metrics (see Section 7)

3. OUTPUT: per-round CSV
```

### 6.5 Controlled Variables (Do Not Vary)

- ESM2 model: always `esm2_t33_650M_UR50D`
- Input standardisation: always StandardScaler fit on training set only, applied to val+test/pool
- No warm-starting of any surrogate between rounds
- Initial set always random (no zero-shot enrichment in core experiments)
- Landscape dataset always full filtered version (no sub-sampling)

---

## 7. Output Data Schema

### 7.1 Per-round record — save as CSV

Path: `results/raw/{landscape}_{method}_{batch_size}_seed{seed:04d}.csv`

| Column | Type | Description |
|---|---|---|
| `landscape` | str | `gb1`, `trpb`, `gfp` |
| `method` | str | `evolvepro`, `boes_ei`, `boes_ts`, `dnn_esm2`, `random` |
| `batch_size` | int | 1, 16, or 96 |
| `seed` | int | 0–69 |
| `round` | int | Round index (0 = after initial 96 random; 1 = after first batch...) |
| `n_evaluated` | int | Total variants labeled including initial set |
| `max_fitness_raw` | float | Maximum raw fitness in labeled set so far |
| `norm_max_fitness` | float | `max_fitness_raw / global_landscape_maximum` — primary comparison metric |
| `mean_top10_fitness` | float | Mean fitness of top-10 variants in labeled set |
| `frac_top1pct` | float | Fraction of true top-1% of landscape found so far |
| `frac_top0p1pct` | float | Fraction of true top-0.1% of landscape found |
| `wall_seconds_round` | float | Training + acquisition time for this round (not cumulative) |

### 7.2 How to compute each metric

- `global_landscape_maximum`: precomputed once per landscape. For GB1: `max(full_fitness_array)`. Store in the landscape config YAML.
- `norm_max_fitness`: `max(labeled_fitness) / global_landscape_maximum`
- `frac_top1pct`: precompute `top1pct_set = set(argsort(full_fitness)[-ceil(0.01*N):])`. Each round: `len(labeled_indices ∩ top1pct_set) / len(top1pct_set)`
- `frac_top0p1pct`: same with `ceil(0.001*N)`

### 7.3 Derived analysis outputs (post-processing)

1. **Learning curves** — `norm_max_fitness` vs `n_evaluated`, mean ± 95% CI across seeds. One figure per landscape. All methods on same axes. Horizontal reference line for ALDE (GB1 and TrpB only, batch=96 condition).

2. **Batch size comparison table** — For each (method, landscape): `norm_max_fitness` at n_evaluated=480 for each batch size. Shows the cost of sequential vs batch. Key finding to reproduce/extend: RF-Greedy at batch=16 collapses (pilot: norm 0.835, zero variance); RF-TS should recover.

3. **Acquisition × batch interaction** — Primary thesis figure. Does the gap between greedy and TS/UCB grow as batch size shrinks? Bar chart: acquisition functions on x-axis, bars per batch size, per landscape. This is the core contribution.

4. **Rank distribution histograms** — For RF-Greedy vs RF-TS at batch=16: histogram of final rank achieved across 30 seeds. Expected result: RF-Greedy bimodal (all at rank #5); RF-TS spread across top-10. This directly visualises the local optima escape.

5. **Compute cost table** — `mean wall_seconds_per_round` × n_rounds at batch=1 per method. Shows practical tradeoff between exploration quality and computational cost.

6. **Landscape difficulty vs method** — Does greedy degrade relative to TS more on GFP (rugged, sparse) than on GB1 (one dominant peak)?

---

## 8. Folder Structure

```
/protein/
├── archive/                        ← rename from aws_lambda_bo/ (keep intact)
│
├── thesis/                         ← existing lit review — unchanged
│   ├── TABLE_papers_and_datasets.md
│   ├── lit_review_*.md
│   └── aalto/
│
└── v1/                             ← all thesis experiment code
    ├── THESIS_PLAN.md              ← this file
    │
    ├── data/
    │   ├── gfp/
    │   │   ├── sequences_scores.csv            ← copy from archive/data/
    │   │   └── embeddings_esm2_650m_meanpool.npz  ← copy + verify dims (must be 1280)
    │   ├── gb1/
    │   │   ├── gb1_fitness.csv                 ← download (149,361 rows)
    │   │   ├── embeddings_esm2_650m_4site.npz  ← compute (149k × 5120)
    │   │   └── embeddings_esm2_650m_meanpool.npz ← compute (149k × 1280)
    │   └── trpb/
    │       ├── trpb_fitness.csv                ← download
    │       ├── embeddings_esm2_650m_4site.npz  ← compute
    │       └── embeddings_esm2_650m_meanpool.npz ← compute
    │
    ├── precompute/
    │   ├── compute_esm2_embeddings.py  ← generic: takes landscape name, computes both modes
    │   └── check_gfp_embeddings.py     ← quick sanity check: load archive .npz, print shape
    │
    ├── methods/
    │   ├── base.py                 ← abstract Optimizer: fit(), predict(), select_batch()
    │   ├── evolvepro.py            ← RF + ESM2 mean-pool + greedy
    │   ├── boes.py                 ← GP + Matérn + EI or TS (acquisition_fn arg)
    │   └── dnn_ensemble.py         ← 5-MLP + ESM2 mean-pool + TS (optional)
    │
    ├── data_utils/
    │   ├── loaders.py              ← load CSV + embeddings, return structured LandscapeData
    │   └── metrics.py              ← norm_max_fitness, frac_topk, top10 mean
    │
    ├── experiments/
    │   ├── configs/
    │   │   ├── gb1.yaml            ← n_variants, targeted_sites, global_max, embedding_paths
    │   │   ├── trpb.yaml
    │   │   └── gfp.yaml
    │   ├── run_single.py           ← run one (landscape, method, batch_size, seed) → CSV
    │   └── run_all.py              ← sweep all conditions, parallelise across seeds
    │
    ├── analysis/
    │   ├── aggregate.py            ← concatenate raw CSVs → summary parquet
    │   ├── plot_learning_curves.py
    │   ├── plot_batch_comparison.py
    │   └── generate_tables.py      ← LaTeX table output for thesis
    │
    └── results/
        ├── raw/                    ← one CSV per run
        └── figures/
```

### What to copy from archive

| From `archive/` | To `v1/` | Action |
|---|---|---|
| `data/sequence_and_score.csv` | `data/gfp/sequences_scores.csv` | Direct copy |
| `data/esm_embeddings_mean_pooled.npz` | `data/gfp/embeddings_esm2_650m_meanpool.npz` | Copy + verify shape[1] == 1280 |
| `preprocessing/2-generate_esm_embeddings.py` | `precompute/` reference | Rewrite cleanly for all three landscapes |
| `src/core/models/random_forest.py` | reference only | UCB logic → strip to pure RF+StandardScaler |

**Do not copy:** AWS infrastructure (`lambda_function.py`, Dockerfile, aws configs), `feature_stats_optimizer.py`, old result dumps.

---

## 9. Pre-Setup Checklist

- [ ] Rename `aws_lambda_bo/` → `archive/`
- [ ] Run `check_gfp_embeddings.py`: load archive `.npz`, print `arr.shape`. If `(58417, 320)` not `(58417, 1280)`, flag for recompute.
- [ ] Download GB1 dataset from FLIP. Verify 149,361 rows with fitness column.
- [ ] Identify TrpB sub-variant with ~10–15% active fraction. Lock in choice.
- [ ] Download TrpB dataset for chosen sub-variant.
- [ ] Compute GB1 ESM2 embeddings (both 4-site and mean-pool). Cache as `.npz`. One-time GPU job.
- [ ] Compute TrpB ESM2 embeddings. Same.
- [ ] Recompute GFP embeddings if 320-dim bug confirmed.
- [ ] Install GPyTorch and BoTorch. Test GP fit on 100 random points from GB1.
- [ ] Profile one full run: BOES-EI on GB1, batch=1, seed=0. Record wall-clock per round. Extrapolate to 384 rounds. If > 8 hours total, decide whether to include batch=1 for BOES.
- [ ] Profile EVOLVEpro on GB1, batch=1, seed=0. RF at batch=1 with 384 rounds should be fast.
- [ ] Confirm global_landscape_maximum for each landscape and write to config YAMLs.

---

## 10. Open Questions to Resolve Before Running

1. **GFP embedding dimensions** — 320-dim or 1280-dim? If 320, from which model? Check archive preprocessing script for the ESM model string used. Recompute from scratch with `esm2_t33_650M_UR50D` if unclear.

2. **TrpB sub-variant** — Which of the Johnston et al. 2024 sub-variants to use? Check Li et al. 2024 (ftALDE) SI table for navigability scores per sub-variant. Pick one now.

3. **BOES TS at batch=96 on GB1 (149k candidates)** — Matheron's rule requires O(n²) memory for the posterior covariance over all candidates. At n=149k, this is infeasible. Options: (a) RFF approximation, (b) subsample candidates to 20k per round, (c) use inducing point GP (SVGP via GPyTorch). Decide before implementing. Likely: subsample 20k (simplest, defensible, similar to what ALDE does implicitly by not evaluating the full pool per round).

4. **Embedding for EVOLVEpro on GB1/TrpB** — EVOLVEpro uses mean-pool (1280-dim), not 4-site embeddings. This is a deliberate choice to test whether mean-pool generalises. But the mean-pool smears out all positional information from the 4 sites that matter. Acknowledge in thesis that this may disadvantage EVOLVEpro on combinatorial landscapes. Alternative (sensitivity check only): also run EVOLVEpro with 4-site embeddings on GB1 and compare.

5. **ALDE as reference baseline — exact Fig. 4D values** — Extract the numerical values from ALDE paper Fig. 4D for (GB1, batch=96, DNN-ens+TS, round 4). These are the numbers to plot as the reference horizontal line. If the figure is not readable at sufficient precision, use the supplementary table if available, or note "approximately X ± Y from visual inspection of Fig. 4D."
