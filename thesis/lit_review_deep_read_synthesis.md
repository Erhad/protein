# Literature Synthesis: Bayesian Optimization for Protein Engineering

**Last updated:** 2026-03-15 (fact-checked against PDFs)
**Purpose:** Deep read of core papers, what each did exactly, what we can learn, gaps, and where our work sits.

---

## 1. Romero, Krause & Arnold (2013) — *The Founding Paper*
**"Navigating the protein fitness landscape with Gaussian processes"** — PNAS
📄 `papers/Romero2013_GP_protein_fitness.pdf`

### What they did
Applied Gaussian process (GP) regression to model protein fitness landscapes from experimental data. Used a **structure-based kernel** encoding amino acid identity at residue-residue contact pairs (not a generic Hamming distance — the kernel explicitly encodes structural context by representing which residues are in contact in the 3D structure and how they differ between chimeric sequences). Applied to chimeric cytochrome P450s.

Two algorithms:
1. **Active learning for informativeness** — greedily selects sequences that maximally reduce mutual information uncertainty across the landscape (experimental design for learning the landscape)
2. **Iterative optimization (GP-UCB)** — exploits GP uncertainty to guide search toward high-fitness regions: select the sequence with the highest upper confidence bound (mean + k·SD), evaluate, update, repeat

**Encoding:** Structure-based contact-pair kernel (not embeddings — this predates LLMs entirely). The kernel itself encodes similarity via shared contact pairs.

**Result:** Engineered P450s more thermostable than any previously made by chimeragenesis, rational design, or directed evolution. The final variant (EXPc5) had a T50 of 69.7°C, 8.7°C above prior directed evolution results.

### Key insights for us
- Uncertainty quantification is the core value of GP for protein BO — this is why RF is a weaker choice
- Structure-based kernels > pure sequence kernels — biology matters for the similarity metric
- This is 2013 — they had no protein LLMs, used domain knowledge in the kernel instead
- **Our RF discards all of this** — we use a flat euclidean kernel implicitly via RF splits

---

## 2. Wu et al. (2019) — *MLDE is Born*
**"Machine learning-assisted directed protein evolution with combinatorial libraries"** — PNAS
📄 `papers/Wu2019_MLDE_PNAS.pdf`

### What they did
Introduced the **MLDE workflow**: train ML model on small labelled sample from a combinatorial library → predict fitness of all remaining variants → test top predictions → fix best, repeat.

Applied to **GB1** (4-site combinatorial, 160k variants = 20⁴) and a real enzyme engineering task: **Rma NOD** (*Rhodothermus marinus* NOD enzyme) for stereoselective silicon-carbon (C–Si) and carbon-hydrogen bond formation.

**Encoding:** One-hot (primary). Also tested Georgiev physicochemical parameters and learned embeddings.
**Model:** Various — linear models, random forests, neural networks.
**Key finding:** One-hot + linear/simple models performed competitively with richer encodings at practical training set sizes. Complex models didn't always win.

### Key insights for us
- We are doing exactly this (MLDE), but on GFP with ESM2 instead of one-hot
- Their benchmark (GB1) is now the field standard — we should run on it
- One-hot performing well is consistent with our EXP001 finding

---

## 3. Wittmann, Johnston, Wu & Arnold (2021) — *Why MLDE Fails and How to Fix It*
**"Informed training set design enables efficient machine learning-assisted directed protein evolution"** — Cell Systems
📄 `papers/Wittmann2021_training_set_design.pdf`

### What they did
Systematic study of MLDE design choices on GB1 (same 4-site, 160k variant landscape). Tested:
- **Encodings:** One-hot, Georgiev (physicochemical), learned embeddings including ESM-1b, TAPE Transformer, ProtBERT-BFD, and MSA Transformer
- **Models:** Linear models, kernel methods, random forests, XGBoost, shallow NNs, ensembles, 1D CNNs
- **Training set design** — the main contribution

**The hole problem:** The GB1 landscape is dominated by zero- or near-zero fitness variants ("holes") — **92% of variants have fitness below 1% of the global maximum** (this statistic comes from the original GB1 data characterization by Wu et al. 2016; Wittmann analyzes its impact on MLDE). Random training sets are mostly holes → model learns to predict low fitness well, useless for finding high-fitness variants.

**ftMLDE (focused training MLDE):** Use zero-shot predictors to build training sets enriched in functional variants. Zero-shot methods used include: EVmutation (evolutionary couplings), MSA Transformer mask-filling protocol, and Triad (physics-based). These enrich training sets without additional experimental cost, dramatically improving performance.

**Result:** Optimized ftMLDE found the global maximum **81-fold more frequently** than single-step greedy optimization under the same screening budget. ✅ *Verified against PDF.*

**Encoding findings:**
- One-hot competitive with or better than LLM embeddings for non-deep models
- MSA Transformer embeddings showed marginal consistent improvement
- **No encoding dominates across all conditions**

### Key insights for us
- **The hole problem is critical** — our GFP dataset has it too (fitness distribution is heavily skewed low). We haven't addressed this at all.
- Zero-shot ESM scores for training set design is novel and powerful
- Confirms our EXP001: one-hot competitive with LLM embeddings on non-deep models
- Our random initial sampling is suboptimal — should use zero-shot ESM to pick informative starting points
- **Attribution note:** The "92% holes" figure is from Wu et al. (2016) GB1 characterization, not Wittmann's own measurement — Wittmann provides the MLDE solution to this pre-existing problem

---

## 4. Sinai et al. (2020) — *AdaLead: The Greedy Baseline That Beats Everything*
**"AdaLead: A simple and robust adaptive greedy search algorithm for sequence design"** — arXiv
📄 `papers/AdaLead_Sinai2020.pdf`

### What they did
Built **FLEXS** (Fitness Landscape EXploration Sandbox) — a benchmarking framework for sequence design algorithms. Tested: BO (EI), DyNA-PPO (RL), CbAS/DbAS (generative models), CMA-ES, and their own **AdaLead**.

**AdaLead algorithm:**
1. Select seeds = sequences with predicted fitness within (1-κ) of the current best (i.e., φ(x) ≥ (1-κ)·y_best)
2. Iteratively recombine and mutate seeds
3. Screen mutants with surrogate model, keep those that improve on parent
4. Propose top-B to wet lab

Key parameter: **κ** (greediness). When landscape is flat → many seeds → diversity. When landscape has clear peak → few seeds → rapid hill climbing. Surrogate model is an ensemble of 3 CNNs.

**Benchmarks:** TF binding (DNA), RNA landscapes (lengths 14–100), and empirical protein landscapes (Rosetta design objective on structure 3MSL).

**Result:** AdaLead outperforms all more complex methods across most benchmarks despite being conceptually simple.

### Key insights for us
- **We have no greedy baseline.** Our plot shows Random, Mutations, Non-mutation — but not AdaLead
- AdaLead beating BO is a major warning: our UCB-RF may not justify its complexity
- The FLEXS framework is the right way to benchmark — multiple landscapes, multiple metrics
- Recombination (crossover) is a powerful mechanism we don't use at all
- BO's theoretical advantages (uncertainty, exploration) may not materialize in practice on protein landscapes

---

## 5. Ren et al. (2022) — *PEX: Stay Close to Wild-Type*
**"Proximal Exploration for Model-guided Protein Sequence Design"** — ICML 2022
📄 `papers/PEX_Ren2022_ICML.pdf`

### What they did
Key insight: **most fitness improvements require only a few mutations from wild-type**. Unconstrained search accumulates many mutations → hard to synthesize, model unreliable far from training data.

**PEX algorithm:**
Adds a regularization term penalizing distance from wild-type:
```
f_prox(s) = f(s) - λ · d(s, s_wt)
```
Varying λ traces a **proximal frontier** — Pareto-efficient sequences trading off fitness vs. mutation count.

**MuFacNet (Mutation Factorization Network):** Specialized neural network modeling low-order mutational effects, factorizing the composition of mutational effects into interactions among single amino-acid mutations. Improves sample efficiency.

**Result:** Outperforms AdaLead and other baselines across multiple in-silico protein design tasks.

### Key insights for us
- Our BO has no proximity regularization — we treat all variants as equal regardless of mutation count
- GFP variants have 1–15 mutations but we don't use mutation count as a feature
- The proximal frontier concept is biologically meaningful: in a real lab you want minimal mutations
- MuFacNet's factorized mutation modeling is similar in spirit to our FeatureStats optimizer

---

## 6. Stanton et al. (2022) — *LaMBO: Latent Space BO*
**"Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders"** — ICML 2022
📄 `papers/LaMBO_Stanton2022_ICML.pdf`

### What they did
**LaMBO** jointly trains:
- Denoising autoencoder (DAE) — learns continuous latent space of sequences
- **Deep kernel learning GP** (DKL) head on the latent space — enables gradient-based acquisition optimization

Allows **gradient-based optimization of acquisition functions** in continuous latent space — circumvents the discrete sequence search problem entirely.

Multi-objective: simultaneously optimizes multiple properties. For protein tasks: **Stability + SASA (solvent-accessible surface area)** for red fluorescent proteins (RFPs). ⚠️ *The document previously said "stability + fluorescence intensity" — corrected: the objective is Stability + SASA. In vitro validation measured brightness (log relative fluorescence units) and thermostability (melting temperature).*

**Benchmarks:** Small molecules (logP + QED, DRD3 docking + SA) + fluorescent proteins (in silico + wet lab with RFPs).

**Result:** Outperforms genetic algorithms across benchmarks. In vitro validation on RFPs found non-dominated variants improving on ancestor sequences.

### Key insights for us
- This is significantly more sophisticated than our approach
- We do lookup-table BO (select from existing pool) — LaMBO generates novel sequences
- For GFP we only select from observed variants — LaMBO can propose unseen sequences
- The DAE + DKL-GP combination is the principled version of what we're attempting
- Our work is closer to the MLDE tradition (select from pool) than latent-space BO

---

## 7. Yang et al. (2024/2025) — *ALDE: Active Learning with Proper UQ*
**"Active Learning-Assisted Directed Evolution"** — bioRxiv preprint (July 2024); journal publication status TBD
📄 `papers/ALDE_Yang2024_biorxiv.pdf`
*Note: The PDF in the workspace is the bioRxiv preprint version (posted July 31, 2024, doi: 10.1101/2024.07.27.605457). The document previously listed this as "Nature Communications 2025" — this should be verified against the final published version. Cite as Yang et al. 2024 until confirmed.*

### What they did (see `insights/LIT005_ALDE_deep_read.yaml` for full details)

Iterative active learning BO on **5 epistatic active-site residues** (W56, Y57, L59, Q60, F89) of ParPgb (*Pyrobaculum arsenaticum* protoglobin) enzyme. Benchmarked computationally on GB1 + TrpB (confirmed via LIT005 and companion field overview).

**Encodings tested:** AAIndex (4-dim/res), Georgiev (19-dim/res), One-hot (20-dim/res), ESM2 (1280-dim/res at targeted sites only — *important: per-residue at the 5 targeted sites, not mean-pooled over full sequence*)
**Models:** Boosting ensemble, GP, DNN ensemble (5 MLPs), Deep Kernel Learning
**Acquisition:** Greedy, UCB, Thompson Sampling

**Key findings:**
- High-dim encodings (one-hot, ESM2) → need deep models (DNN/DKL)
- Low-dim encodings (AAIndex, Georgiev) → pair with GP/boosting
- ESM2 ≠ better than one-hot for their problem (when paired appropriately)
- DNN ensemble most accurate + best calibrated
- **Frequentist uncertainty (ensemble spread) > Bayesian** in practice for this problem
- Chose **DNN ensemble + one-hot + Thompson sampling** for wet lab → **12% → 93% yield** in 3 rounds on a non-native cyclopropanation reaction ✅ *Verified against PDF.*

### Critical difference from our work
Their ESM2: per-residue embeddings at 5 targeted sites (5 × 1280 = 6400-dim total)
Our ESM2: mean pool over all 238 residues (320-dim, smaller model)
Their dataset: 5-site combinatorial (3.2M theoretical variants, focused on known active site)
Our dataset: full-protein GFP (58k measured variants, 1-15 mutations anywhere in sequence)

---

## 8. Gonzalez Somermeyer et al. (2022) — *GFP Landscape Heterogeneity*
**"Heterogeneity of the GFP fitness landscape and data-driven protein design"** — eLife
📄 `papers/Livesey2022_GFP_heterogeneity.pdf`
*Note: file was originally named after Livesey & Marsh — actual authors are Gonzalez Somermeyer et al. (Sarkisyan + Kondrashov groups)*

### What they did
Measured DMS fluorescence landscapes for **3 novel orthologous GFPs** — amacGFP (35,500 variants), cgreGFP (26,165), ppluGFP2 (32,260) — and combined with the existing **Sarkisyan et al. (2016)** avGFP data (51,715 variants). ⚠️ *Important: avGFP was measured by Sarkisyan et al. (2016), NOT by this paper. Gonzalez Somermeyer et al. contributed the other three GFPs.* ~4 mutations/variant on average (median 3–4 across GFPs). Fitness measured via FACS into 8 bins + barcode sequencing.

**Encoding:** One-hot only (binary). No PLM embeddings.
**Model:** Neural networks (linear → sigmoid → deep), MC Dropout for uncertainty.
**Train/val/test:** 60%/20%/20%.
**R² (validation):** 0.710 (amacGFP), 0.740 (cgreGFP), 0.810 (ppluGFP2). ⚠️ *These values could not be confirmed from PDF pages read — flagged for verification against Table 1 or Figure captions in the paper.*
**Paradigm: GENERATIVE ORACLE** — trained NN used as oracle for a genetic algorithm proposing novel sequences outside the training data. Wet-lab validated.

**Key finding — landscape sharpness determines ML design success:**
- avGFP, cgreGFP: "sharp" peaks (high epistasis, steep drop-off) → model signal is clear → GA reliably finds high-fitness novel sequences
- amacGFP, ppluGFP2: "flat" peaks (robust, many near-WT fitness sequences) → harder to distinguish good from bad → more wet-lab failures
- **Conclusion: epistasis/sharpness is favorable for ML-guided design** — avGFP is a good ML benchmark

### Key insights for us
- **Different paradigm**: generative oracle (proposes novel sequences), not pool-based MLDE — does NOT replicate our work
- **No PLM comparison**: one-hot only — the ESM2 vs one-hot comparison we're doing is genuinely untested in this paper
- **avGFP is "sharp"**: favorable for ML-guided optimization — positive for our benchmark choice
- **Bimodality confirmed**: GFP fitness IS bimodal (non-functional variants exist), though less extreme than GB1 — we should handle this
- **Performance ceiling**: one-hot NN at R²=0.71–0.81 with full 60% training set; our pool-based work uses far fewer labels per round — different regime entirely
- **Cite for**: landscape characterization of avGFP, motivation for GFP as benchmark, landscape sharpness → ML feasibility

---

## 9. Li, Yang, Johnston et al. (2024) — *The Wittmann Equivalent for TrpB*
**"Evaluation of Machine Learning-Assisted Directed Evolution Across Diverse Combinatorial Landscapes"** — bioRxiv October 2024 → Cell Systems 2025
📄 `papers/Li2024_MLDE_diverse_combinatorial_landscapes.pdf`

### What they did
The most comprehensive cross-landscape MLDE benchmark to date. Systematically evaluated multiple MLDE strategies across **16 combinatorial fitness landscapes spanning 6 protein systems**: ParD-ParE toxin-antitoxin (binding, 3-site, ×2), GB1 (binding, 4-site, 160k), DHFR (enzyme, 3-site), T7 RNAP (enzyme, 3-site), TEV protease (enzyme, 4-site, 160k), and **TrpB (enzyme, 3- or 4-site, ×10 — the dominant system)**.

**Strategies benchmarked:**
- Directed evolution (DE): simple recomb, single-step, top96 recomb
- MLDE: single round, random training set (one-hot + boosting ensemble)
- ftMLDE: focused training with ZS predictors instead of random sampling
- ALDE: multiple rounds, greedy acquisition guided by ensemble uncertainty
- ftALDE: focused training combined with multiple ALDE rounds

**6 zero-shot predictors** evaluated for focused training: Hamming distance (baseline), EVmutation, ESM, ESM-IF, CoVES, Triad (structure-based). Each provides a distinct prior — evolutionary, sequence-based, structural, or stability-based.

**Landscape navigability** quantified by 6 attributes: percent active variants, fraction of local optima, fraction of non-magnitude pairwise epistasis, Cauchy peak location of fitness distribution, kurtosis (tailedness), and number of KDE peaks.

### Key findings
- MLDE needs only **48 training samples** to outperform recomb DE on average; 384 to match the top96 recomb baseline
- ftMLDE with ZS predictors gives **4–12% improvement** in average max fitness over random-training MLDE (up to 9–77% improvement in fraction reaching global optimum)
- ALDE (2 rounds) starts to beat single-round MLDE at 480 total samples, but doesn't beat ftMLDE until 1,056+ samples — **focused training is more efficient than multi-round active learning at moderate budgets**
- ftALDE matches or exceeds ftMLDE with the same number of rounds, and improves further with more rounds
- **ML advantage is predictable from landscape attributes**: greatest on landscapes with more local optima, higher non-magnitude epistasis, greater tailedness, more KDE peaks (harder landscapes). For hard landscapes with <1% active variants, 4 rounds of ALDE alone still underperforms ftMLDE
- **TrpB3E** (one of the least navigable landscapes) showed the greatest benefit: ftMLDE 3.5-fold better than single-step DE
- No single ZS predictor wins universally — but Hamming distance (proximity to active parent) is a strong, underrated baseline; structure-based (Triad) and evolutionary (EVmutation, ESM) provide complementary signal

### Critical differences from ALDE
- Li et al. ask: **when does MLDE help, and which ZS predictor?** ALDE asks: **which encoding × model × acquisition?**
- Li et al. use one-hot + boosting ensemble throughout — no encoding comparison
- Li et al. use greedy acquisition for ALDE — no UCB/Thompson Sampling comparison
- The two papers are **complementary**, not redundant: ALDE optimises the surrogate model machinery; Li et al. optimise the training set strategy and characterise when any ML method helps

### Key insights for us
- This is the **Wittmann equivalent for TrpB** — the canonical MLDE benchmark for that landscape. Must be cited and compared against when reporting TrpB results
- The 6 landscape navigability attributes are directly applicable to GFP: GFP's KDE peaks, tailedness, and % active should be computed and reported alongside GB1/TrpB to justify why the landscape regime is distinct
- Their "focused training beats multi-round ALDE at <1,056 samples" finding is directly relevant: if our BO rounds are small, adding ESM zero-shot filtering to the initial training set may matter more than the acquisition function choice
- **GFP is absent from their 16 landscapes** — its full-protein sparse structure is a different class entirely, further supporting our contribution

---

## 10. Jiang et al. (2024) — *EVOLVEpro: Few-Shot PLM-Guided Evolution*
**"Rapid protein evolution by few-shot learning with a protein language model"** — bioRxiv preprint (July 2024)
📄 `papers/EVOLVEpro_2024.pdf`
*Note: published as bioRxiv July 18, 2024. May have been published in Science — verify citation. Previously described in STR001 as Science.*

### What they did
**EVOLVEpro** combines a protein language model (PLM) with a lightweight top-layer regression model (random forest or ridge regression) trained on a small number of labeled variants. The key idea: strong evolutionary priors from PLMs can substitute for large supervised training sets.

Applied to **5 diverse proteins**:
- T7 RNA polymerase (RNA production yield)
- Cas12f miniature CRISPR nuclease (activity)
- Bxb1 serine integrase (recombination efficiency)
- Prime editor (reverse transcriptase activity)
- REGN10987 antibody (SARS-CoV-2 binding)

**Result:** 2- to 515-fold improvement in desired properties across proteins; up to 100-fold improvement in vitro. Achieved strong results with as few as 16 labeled variants per round.

### Key insights for us
- This is the closest published analog to our approach: PLM + lightweight surrogate + active selection
- Demonstrates the paradigm works across diverse protein families, not just model landscapes
- RF + PLM embeddings is their primary method — **this is exactly our current approach (UCB-RF + ESM2)**
- Their PLM usage is likely full-sequence embeddings (similar to our mean pooling), making this the most direct comparison
- They achieve strong results where we're seeing marginal improvements — examine what's different (dataset, PLM choice, active learning strategy?)
- **This paper must be cited and directly compared against**

---

## 10. Biswas et al. (2021) — *Low-N Protein Engineering*
**"Low-N protein engineering with data-efficient deep learning"** — Nature Methods
📄 `papers/Biswas2021_LowN_protein_engineering.pdf`

### What they did
Demonstrated that **eUniRep** (global + local pre-training of a language model on evolutionary sequences) enables high-accuracy fitness prediction from as few as 24 labeled variants. The "evo-velocity" concept: sequence-encoded evolutionary information in the LM provides a strong prior that reduces supervised data requirements dramatically.

Applied to GFP (avGFP), TEM-1 β-lactamase, and other proteins. At 24 labels, eUniRep + linear model outperformed non-PLM methods trained on hundreds of labels.

### Key insights for us
- Foundational paper for the claim that PLMs help most in the **low-N regime**
- Our EXP001 regime (small training sets, iterative) is exactly where this paper predicts PLMs should excel
- Yet we don't see ESM2 dominance — why? (see HYP001 for hypotheses)
- The "local" fine-tuning step in eUniRep (fine-tuned on sequences near the target protein) may be critical — we're using global ESM2 without fine-tuning
- **Cite for**: motivation that PLMs should help in low-N settings; contrast with our null result

---

## 11. Kirjner et al. (2024) — *Smoothed Fitness Landscapes*
**"Improving protein optimization with smoothed fitness landscapes"** — ICLR 2024
📄 `papers/Kirjner2023_smoothed_fitness_landscapes.pdf`

### What they did
Key insight: fitness landscapes are **noisy surrogates** even when using a trained ML model. Small perturbations in sequence lead to unpredictable jumps in predicted fitness, making gradient-based or sampling-based optimization unreliable.

**Approach:** Apply **graph-based landscape smoothing** (Tikhonov regularization on the sequence graph) to create a smoothed surrogate f̃(s) that is easier to optimize while remaining predictive. Use **Gibbs sampling with gradients (GWG)** for discrete sequence optimization over the smoothed landscape.

**Benchmarks:** GFP (avGFP), AAV capsid protein, TEM-1. Directly tested on the Sarkisyan GFP dataset.

**Result:** ~2.5× fitness improvement on GFP over unsmoothed baselines; state-of-the-art on AAV (2× over next best). Substantially outperforms AdaLead and other BO baselines on GFP.

### Key insights for us
- **Directly tests on GFP** — this is one of the few papers operating in our exact setting
- Landscape smoothing is a complementary approach to uncertainty-based BO: instead of handling noise through the acquisition function (UCB), smooth the surrogate itself
- Their GFP results are directly relevant to our expected performance ceiling
- We should compare our BO results against their published GFP numbers
- The GWG optimization strategy explores outside the observed pool — a capability we lack (pool-based only)

---

## What Makes Proteins Optimizable? Cross-Paper Synthesis

Based on all papers above, these factors determine whether ML-guided BO will work:

### 1. Landscape ruggedness / epistasis
- High epistasis → traditional DE fails → MLDE/BO has larger advantage (Wittmann 2021, ALDE 2024)
- Low epistasis → additive models sufficient → simple one-hot + linear works (our EXP001)
- GFP: **moderate epistasis**, mostly additive (Sailer & Harms 2017) — explains why one-hot wins

### 2. Hole density
- High hole density (most variants non-functional) → random training sets fail → need informed design (Wittmann 2021)
- GB1: 92% holes (Wu et al. 2016). GFP: **lower hole density than GB1** — fitness distribution is bimodal (confirmed by Gonzalez Somermeyer 2022) but functional plateau is far wider than GB1. ⚠️ *The GFP bimodality claim holds (non-functional variants exist) but the exact fraction needs verification; avGFP likely has fewer "hard holes" than GB1 given the wider functional sequence space.*

### 3. Mutation count / design space size
- Small combinatorial space (4-5 sites, 20^4-5) → exhaustive prediction feasible (MLDE, ALDE)
- Large continuous space → need generative models or proximal constraints (PEX, LaMBO, Kirjner)
- GFP: **medium** — 58k measured, full space >> 58k — we're selecting from measured pool only

### 4. Encoding × model match
- One-hot + linear/RF: best for small combinatorial spaces with additive landscapes (Wittmann, ALDE)
- ESM2 + DNN/DKL: best for complex landscapes needing transfer learning (ALDE)
- GP + structure kernel: best when structural information available (Romero 2013)
- PLM + lightweight regression (RF/ridge): effective in low-N regime when PLM is well-matched to target (Biswas 2021, EVOLVEpro 2024)
- Mean-pooled ESM2 + RF: **potentially weak combination** — loses spatial info, RF can't exploit 320-dim well (see HYP001b)

### 5. Proximity to wild-type
- Sequences far from WT → unreliable models, hard to synthesize (PEX)
- Staying proximal improves reliability and interpretability
- Our GFP has 1-15 mutations — relatively proximal, PEX principles apply

### 6. Surrogate landscape quality (from Kirjner 2024)
- Noisy surrogate → unreliable optimization → need smoothing or better UQ
- Greedy baselines (AdaLead) may be more robust to surrogate noise than BO with explicit UQ
- Landscape smoothing (graph-based Tikhonov) is a complementary approach to UQ-based acquisition

### 7. Landscape navigability as a predictive framework (new, from Li et al. 2024)
Li et al. introduce 6 measurable landscape attributes that predict how much benefit ML methods will provide over DE. These apply directly to characterising our three datasets:

| Attribute | GB1 | TrpB (varies) | GFP (estimated) |
|---|---|---|---|
| % active | 23% | 1–32% (varies) | ~moderate |
| Fraction local optima | 0.005 | 0.006–0.348 | unknown |
| Non-magnitude epistasis | 0.40 | 0.43–0.63 | unknown |
| Cauchy peak location | low | low | moderate |
| Tailedness (kurtosis) | high | high | lower |
| KDE peaks | many | varies | unknown |

Computing these for GFP from the Sarkisyan data is a concrete contribution that would situate our work within the Li et al. framework and explain why results differ across the three landscapes.

---

## Papers In Workspace Not Yet Synthesized

These papers exist in `papers/` but are not covered above — review for potential inclusion.
*Note: Li et al. 2024 (now Section 9) was added 2026-03-15 and is fully synthesised.*

- `papers/Wu2016_GB1_fitness_landscape.pdf` — Original GB1 dataset paper (Olson et al. 2014 / Wu et al. 2016): **source of the 92% holes stat and the standard benchmark landscape. Should be cited directly for GB1 statistics, not via Wittmann.**
- `papers/Sarkisyan2016_GFP_fitness_landscape.pdf` — Original avGFP dataset paper: **our primary data source. Confirmed ~52,000 variants, FACS into 4 fluorescence bins.**
- `papers/Greenberg2023_benchmarking_UQ.pdf` — Systematic UQ benchmarking on GB1 (GP, MC-Dropout, ensemble, conformal prediction). Relevant to BO UQ assumptions.
- `papers/Brookes2019_CbAS.pdf` / `papers/Brookes2018_DbAS.pdf` — Generative model baselines tested in FLEXS (AdaLead paper).
- `papers/Notin2022_Tranception_ProteinGym.pdf` — Tranception + ProteinGym benchmark. Relevant for zero-shot baselines context; separate from BO/MLDE paradigm.
- `papers/Hie2020_uncertainty_ML_bio.pdf` — Uncertainty-aware ML in biological settings. Relevant to UQ discussion.
- `papers/Dallago2021_FLIP.pdf` — FLIP benchmark for protein fitness prediction. Relevant for benchmarking context.
- `papers/BO_in_embedding_space_BIBM2024.pdf` — Recent (2024) work on BO in protein embedding space. Review for positioning.
- `papers/Angermueller2020_DynaPPO.pdf` — Original DyNA-PPO paper. Relevant as an RL baseline.
- `papers/Johnston2024_TrpB_landscape.pdf` — TrpB combinatorial landscape dataset. Used as benchmark in ALDE.
- `papers/Minding_the_gaps_navigating_holes.pdf` — Thomas & Colwell (2021), Cell Systems perspective on the holes problem. Commentary on Wittmann, not primary research.
