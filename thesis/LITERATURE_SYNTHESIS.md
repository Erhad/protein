# Literature Synthesis: Bayesian Optimization for Protein Engineering

**Last updated:** 2026-03-14
**Purpose:** Deep read of core papers, what each did exactly, what we can learn, gaps, and where our work sits.

---

## 1. Romero, Krause & Arnold (2013) — *The Founding Paper*
**"Navigating the protein fitness landscape with Gaussian processes"** — PNAS
📄 `papers/Romero2013_GP_protein_fitness.pdf`

### What they did
Applied Gaussian process (GP) regression to model protein fitness landscapes from experimental data. Used a **structure-based kernel** (not sequence Hamming distance — residue-residue contact map distance) to capture how mutations affect function through structural context. Applied to chimeric cytochrome P450s.

Two algorithms:
1. **Active learning for informativeness** — select sequences that maximally reduce model uncertainty
2. **Iterative optimization** — exploit GP uncertainty to guide search toward high-fitness regions (UCB-style)

**Encoding:** Structure-based contact map kernel (not embeddings — this predates LLMs entirely). The kernel itself encodes the similarity.

**Result:** Engineered P450s more thermostable than anything previously made by directed evolution, rational design, or chimeragenesis.

### Key insights for us
- Uncertainty quantification is the core value of GP for protein BO — this is why RF is a weaker choice
- Structure-based kernels > pure sequence kernels — biology matters for the similarity metric
- This is 2013 — they had no protein LLMs, used domain knowledge in the kernel instead
- **Our RF discards all of this** — we use a flat euclidean kernel implicitly via RF splits

---

## 2. Wu et al. (2019) — *MLDE is Born*
**"Machine learning-assisted directed protein evolution with combinatorial libraries"** — PNAS
📄 `papers/Wu2019_MLDE_PNAS.pdf` *(note: download was incomplete — 7KB)*

### What they did
Introduced the **MLDE workflow**: train ML model on small labelled sample from a combinatorial library → predict fitness of all remaining variants → test top predictions → fix best, repeat.

Applied to **GB1** (4-site combinatorial, 160k variants) and a real enzyme engineering task (silicon-carbon bond formation).

**Encoding:** One-hot (primary). Also tested physicochemical encodings.
**Model:** Various — including linear models, random forests, neural networks.
**Key finding:** One-hot + linear/simple models worked well. Complex models didn't always win.

### Key insights for us
- We are doing exactly this (MLDE), but on GFP with ESM2 instead of one-hot
- Their benchmark (GB1) is now the field standard — we should run on it
- One-hot performing well is consistent with our EXP001 finding

---

## 3. Wittmann, Johnston, Wu & Arnold (2021) — *Why MLDE Fails and How to Fix It*
**"Informed training set design enables efficient MLDE"** — Cell Systems
📄 `papers/Wittmann2021_training_set_design.pdf`

### What they did
Systematic study of MLDE design choices on GB1 (same 4-site, 160k variant landscape). Tested:
- **Encodings:** One-hot, physicochemical (AAIndex, Georgiev), learned embeddings (including ESM-1b among others) %% CHECK: verify exact number/list of embeddings tested
- **Models:** Ensembles, XGBoost, neural networks
- **Training set design** — the main contribution

**The hole problem:** Combinatorial libraries are dominated by zero- or near-zero fitness variants ("holes" — 92% of GB1 is below 1% of max fitness). Random training sets are mostly holes → model learns to predict low fitness well, useless for finding high-fitness variants.

**ftMLDE (focused training MLDE):** Use zero-shot predictors (ESM log-likelihoods, evolutionary scores) to build training sets enriched in functional variants. Dramatically improves performance.

**Encoding findings:**
- One-hot competitive with or better than LLM embeddings for non-deep models
- LLM embeddings help when paired with appropriate models
- **No encoding dominates across all conditions**

**Result:** Optimized MLDE found global maximum far more frequently than traditional DE. %% CHECK: "81×" figure needs verification against the actual paper

### Key insights for us
- **The hole problem is critical** — our GFP dataset has it too (fitness distribution is heavily skewed low). We haven't addressed this at all.
- Zero-shot ESM scores for training set design is novel and powerful
- Confirms our EXP001: one-hot competitive with LLM embeddings on non-deep models
- Our random initial sampling is suboptimal — should use zero-shot ESM to pick informative starting points

---

## 4. Sinai et al. (2020) — *AdaLead: The Greedy Baseline That Beats Everything*
**"AdaLead: A simple and robust adaptive greedy search algorithm for sequence design"** — arXiv
📄 `papers/AdaLead_Sinai2020.pdf`

### What they did
Built **FLEXS** (Fitness Landscape EXploration Sandbox) — a benchmarking framework for sequence design algorithms. Tested: BO (EI), DyNA-PPO (RL), CbAS/DbAS (generative models), CMA-ES, and their own **AdaLead**.

**AdaLead algorithm:**
1. Select seeds = sequences within (1-κ) of current best fitness
2. Iteratively recombine and mutate seeds
3. Screen mutants with surrogate model, keep those that improve on parent
4. Propose top-B to wet lab

Key parameter: **κ** (greediness). When landscape is flat → many seeds → diversity. When landscape has clear peak → few seeds → rapid hill climbing.

**Benchmarks:** TF binding (DNA), RNA landscapes (size 14–100), and empirical protein landscapes.

**Result:** AdaLead outperforms all more complex methods across most benchmarks despite being conceptually simple.

### Key insights for us
- **We have no greedy baseline.** Our plot shows Random, Mutations, Non-mutation — but not AdaLead
- AdaLead beating BO is a major warning: our UCB-RF may not justify its complexity
- The FLEXS framework is the right way to benchmark — multiple landscapes, multiple metrics
- Recombination (crossover) is a powerful mechanism we don't use at all
- BO's theoretical advantages (uncertainty, exploration) may not materialize in practice on protein landscapes

---

## 5. Ren et al. (2022) — *PEX: Stay Close to Wild-Type*
**"Proximal Exploration for Model-guided Protein Sequence Design"** — ICML
📄 `papers/PEX_Ren2022_ICML.pdf`

### What they did
Key insight: **most fitness improvements require only a few mutations from wild-type**. Unconstrained search accumulates many mutations → hard to synthesize, model unreliable far from training data.

**PEX algorithm:**
Adds a regularization term penalizing distance from wild-type:
```
f_prox(s) = f(s) - λ · d(s, s_wt)
```
Varying λ traces a **proximal frontier** — Pareto-efficient sequences trading off fitness vs. mutation count.

**MuFacNet:** Specialized neural network modeling low-order mutational effects (factorized by mutation identity), improving sample efficiency.

**Result:** Outperforms AdaLead and other baselines across multiple in-silico protein design tasks.

### Key insights for us
- Our BO has no proximity regularization — we treat all variants as equal regardless of mutation count
- GFP variants have 1–15 mutations but we don't use mutation count as a feature
- The proximal frontier concept is biologically meaningful: in a real lab you want minimal mutations
- MuFacNet's factorized mutation modeling is similar in spirit to our FeatureStats optimizer

---

## 6. Stanton et al. (2022) — *LaMBO: Latent Space BO*
**"Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders"** — ICML
📄 `papers/LaMBO_Stanton2022_ICML.pdf`

### What they did
**LaMBO** jointly trains:
- Denoising autoencoder (DAE) — learns continuous latent space of sequences
- Discriminative GP head on the latent space — enables gradient-based acquisition optimization

Allows **gradient-based optimization of acquisition functions** in continuous latent space — circumvents the discrete sequence search problem entirely.

Multi-objective: simultaneously optimizes multiple properties (stability + fluorescence intensity for RFPs).

**Benchmarks:** Small molecules + fluorescent proteins (in silico + wet lab with red fluorescent proteins).

**Result:** Outperforms genetic algorithms, doesn't require large pretraining corpus. Demonstrated in vitro on RFPs.

### Key insights for us
- This is significantly more sophisticated than our approach
- We do lookup-table BO (select from existing pool) — LaMBO generates novel sequences
- For GFP we only select from observed variants — LaMBO can propose unseen sequences
- The DAE + GP combination is the principled version of what we're attempting
- Our work is closer to the MLDE tradition (select from pool) than latent-space BO

---

## 7. Yang et al. (2025) — *ALDE: Active Learning with Proper UQ*
**"Active Learning-Assisted Directed Evolution"** — Nature Communications
📄 `papers/ALDE_Yang2024_biorxiv.pdf`

### What they did (see `insights/LIT005_ALDE_deep_read.yaml` for full details)

Iterative active learning BO on **5 epistatic active-site residues** of ParPgb enzyme. Benchmarked exhaustively on GB1 + TrpB.

**Encodings tested:** AAIndex (4-dim/res), Georgiev (19-dim/res), One-hot (20-dim/res), ESM2 (1280-dim/res at targeted sites)
**Models:** Boosting ensemble, GP, DNN ensemble (5 MLPs), Deep Kernel Learning
**Acquisition:** Greedy, UCB, Thompson Sampling

**Key findings:**
- High-dim encodings (one-hot, ESM2) → need deep models (DNN/DKL)
- Low-dim encodings (AAIndex, Georgiev) → pair with GP/boosting
- ESM2 ≠ better than one-hot for their problem
- DNN ensemble most accurate + best calibrated
- **Frequentist uncertainty > Bayesian** in practice
- Chose **DNN ensemble + one-hot + Thompson sampling** for wet lab (12% → 93% yield in 3 rounds)

### Critical difference from our work
Their ESM2: per-residue embeddings at 5 targeted sites (1280-dim each)
Our ESM2: mean pool over all 238 residues (320-dim, smaller model)
Their dataset: 5-site combinatorial (3.2M variants, but focused)
Our dataset: full-protein GFP (58k measured variants, 1-15 mutations anywhere)

---

## 8. Gonzalez Somermeyer et al. (2022) — *GFP Landscape Heterogeneity*
**"Heterogeneity of the GFP fitness landscape and data-driven protein design"** — eLife
📄 `papers/Livesey2022_GFP_heterogeneity.pdf`
*Note: file was originally named after Livesey & Marsh — actual authors are Gonzalez Somermeyer et al. (Sarkisyan + Kondrashov groups)*

### What they did
Measured DMS fluorescence landscapes for 4 orthologous GFPs: avGFP (51,715 variants), amacGFP (35,500), cgreGFP (26,165), ppluGFP2 (32,260). ~4 mutations/variant on average. Fitness measured via FACS into 8 bins + barcode sequencing.

**Encoding:** One-hot only (binary). No PLM embeddings.
**Model:** Neural networks (linear → sigmoid → deep), MC Dropout for uncertainty.
**Train/val/test:** 60%/20%/20%.
**R² (validation):** 0.710 (amacGFP), 0.740 (cgreGFP), 0.810 (ppluGFP2).
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

## What Makes Proteins Optimizable? Cross-Paper Synthesis

Based on all papers above, these factors determine whether ML-guided BO will work:

### 1. Landscape ruggedness / epistasis
- High epistasis → traditional DE fails → MLDE/BO has larger advantage (Wittmann 2021, ALDE 2025)
- Low epistasis → additive models sufficient → simple one-hot + linear works (our EXP001)
- GFP: **moderate epistasis**, mostly additive (Sailer & Harms 2017) — explains why one-hot wins

### 2. Hole density
- High hole density (most variants non-functional) → random training sets fail → need informed design (Wittmann 2021)
- GB1: 92% holes. GFP: **lower hole density than GB1** — fitness distribution is bimodal (confirmed by Gonzalez Somermeyer 2022) but functional plateau is far wider than GB1. %% CHECK: "min=1.28" figure unverified; "all variants have some fitness" is incorrect — non-functional variants exist but are a smaller fraction

### 3. Mutation count / design space size
- Small combinatorial space (4-5 sites, 20^4-5) → exhaustive prediction feasible (MLDE, ALDE)
- Large continuous space → need generative models or proximal constraints (PEX, LaMBO)
- GFP: **medium** — 58k measured, full space >> 58k — we're selecting from measured pool only

### 4. Encoding × model match
- One-hot + linear/RF: best for small combinatorial spaces with additive landscapes (Wittmann, ALDE)
- ESM2 + DNN/DKL: best for complex landscapes needing transfer learning (ALDE)
- GP + structure kernel: best when structural information available (Romero 2013)
- Mean-pooled ESM2 + RF: **worst combination** — loses spatial info, RF can't exploit 320-dim well

### 5. Proximity to wild-type
- Sequences far from WT → unreliable models, hard to synthesize (PEX)
- Staying proximal improves reliability and interpretability
- Our GFP has 1-15 mutations — relatively proximal, PEX principles apply

---

## Where Our Work Sits

| Aspect | Our work | Field status |
|--------|----------|--------------|
| Core approach | Select from pool, RF+UCB | 2019 MLDE baseline |
| Representation | ESM2 mean pool | Wrong pairing per ALDE 2025 |
| Surrogate model | Random Forest | DNN ensemble is current best |
| Acquisition | UCB fixed β | Thompson sampling preferred |
| Benchmark | GFP only | GB1 is field standard |
| Baseline | Random only | Need AdaLead comparison |
| Training set design | Random initial | Should use zero-shot ESM |
| Hole handling | Not addressed | Critical per Wittmann 2021 |
| Proximity constraint | None | PEX shows value |

**What we do that's novel / not directly replicated:**
1. Full-protein GFP landscape (not combinatorial k-site) — different regime from GB1 MLDE papers
2. Mean-pooled ESM2 on pool-based MLDE — not tested in ALDE (per-residue, 5-site only) or Gonzalez Somermeyer (one-hot only)
3. Systematic encoding × surrogate comparison on avGFP pool-based setting — untested in any prior work
4. Precision@K as metric — underused in literature
5. Scale: 5,276 experimental replicates — high statistical robustness
6. One-hot mutations on full protein (2987-dim) — not tested in prior pool-based MLDE work on GFP

**Closest prior works and why they don't replicate us:**
- Wittmann 2021: GB1 only (4-site combinatorial), not full-protein GFP
- ALDE 2025: 5-site combinatorial, per-residue ESM2 (not mean-pool), not avGFP
- Gonzalez Somermeyer 2022: avGFP + full-protein, but generative oracle paradigm, one-hot only
- Biswas 2021 (eUniRep): avGFP, low-N generative hybrid, wet-lab validated — closest in spirit but different paradigm and embedding (eUniRep not ESM2)
