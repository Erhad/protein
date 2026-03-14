# Literature Review: Protein Fitness Datasets

## Overview

Benchmarking machine learning models for protein engineering requires datasets where fitness has been experimentally measured across a large number of variants. The ideal benchmark dataset is (1) exhaustively or near-exhaustively measured, (2) covers multi-mutant combinations so epistasis is observable, and (3) has a well-defined fitness proxy. Below are the primary datasets used in the literature for evaluating Bayesian optimization and machine learning-directed evolution (MLDE) methods.

---

## 2.1 GB1 — IgG-Binding Domain of Protein G

**Source**: Wu et al. (2016). *Adaptation in protein fitness landscapes is facilitated by indirect paths.* eLife, 5, e16965.

**Protein**: GB1 is a 56-residue domain of streptococcal protein G that binds the Fc region of IgG antibodies. It is a model system for studying protein stability and binding.

**Experimental design**: A combinatorial library was constructed at four positions (V39, D40, G41, V54), covering all 20⁴ = 160,000 possible amino acid combinations. Variants were subjected to selection for IgG binding via phage display, and fitness was measured as the enrichment ratio (post-selection reads / pre-selection reads).

**Raw dataset statistics**:
- Total theoretical variants: 160,000 (20⁴)
- Raw sequenced rows in full data: ~530,000 (multiple observations per variant)
- Variants with measurable fitness (input read count ≥ 10): **149,361**
- Variants with too few reads, excluded from standard benchmarks: **10,639**
- The 10,639 excluded variants have been imputed in some analyses (Wu et al. 2016 imputed them to complete the landscape for biological analysis), but **all ML benchmarks work exclusively with the 149,361 measured variants**

**What "filtered" means**: The standard quality filter (input read count ≥ 10) removes variants whose sequencing depth was too low to estimate a reliable enrichment ratio. These are not failed experiments — they were present in the library but happened to receive few sequencing reads. Their fitness is genuinely unknown. They are excluded from the candidate oracle pool in all simulation studies, meaning the algorithm can never query them.

**Epistasis**: High. Because all 160,000 combinations are constructed and nearly all are measured, epistasis can be directly quantified across the full combinatorial space. Wu et al. (2016) identified 30 local optima, demonstrating that the landscape is highly non-additive with indirect routes between peaks.

**Why used**: Near-exhaustive coverage at 4 sites makes it possible to simulate a perfect oracle (lookup table), enabling fair comparison of BO and MLDE algorithms without confounding from imputed or missing labels. The combinatorial structure (fixed positions, exhaustive amino acid coverage) is ideal for simulating iterative laboratory campaigns.

**Used in**: Wu et al. 2019 (original MLDE, PNAS), Wittmann et al. 2021 (ftMLDE, Cell Systems), ALDE (Yang et al. 2024). ⚠️ Note: AdaLead (Sinai et al. 2020) and PEX (Ren et al. 2022) do **not** benchmark on GB1 — see Section 2.6 for details.

---

### 2.1.1 Per-Paper Dataset Version Analysis: GB1

**Wu et al. 2019 (MLDE, PNAS) — ref [11] in ALDE**
- Uses the **149,361 filtered variant set** as the complete benchmark landscape
- Wittmann et al. (2021) explicitly confirms: "Originally reported by Wu et al., this landscape consists of 149,361 experimentally determined fitness measurements for 160,000 possible variants"
- This is the paper that introduced the MLDE paradigm and established the 149,361 filtered set as the standard. Training set sizes used: 96–480 variants drawn from the full 149,361

**Wittmann et al. 2021 (ftMLDE, Cell Systems)**
- Uses the **149,361 filtered variant set** — stated explicitly in the paper: "consists of 149,361 experimentally determined fitness measurements for 160,000 possible variants"
- The 10,639 missing variants with imputed fitness are **explicitly excluded**: "to ensure that our simulations match what would have been observed experimentally had our simulated experiments actually been performed, we do not use the variants with imputed fitness in this study"
- Training sets of **24, 48, or 384 variants** are drawn at random from the 149,361-variant pool; predictions tested on the unlabeled remainder (148,977, 149,313, or 148,977 variants depending on training size)
- The 149,361 serves as both the candidate space and the oracle

**ALDE (Yang et al. 2024)**
- States the landscape has "fitnesses measured experimentally for nearly all of the 20⁴ = 160,000 variants" (p. 11 of preprint)
- Does not explicitly state the 149,361 cutoff in the paper, but this phrasing and universal convention confirms the **149,361 filtered set** is used
- Simulations draw an initial random batch of **96 variants**, followed by four active learning rounds of **96 variants each** (480 total), all queried from the 149,361 oracle
- Performance metric: maximum fitness achieved by end of campaign, normalized to the global maximum in the dataset

**AdaLead (Sinai et al. 2020) — does NOT use GB1**
- AdaLead benchmarks on TF binding landscapes (transcription factor DNA-binding), RNA secondary structure folding landscapes (ViennaRNA), and protein design using PyRosetta (66 amino acid antifreeze protein 3MSI)
- GB1 is not used as a benchmark in this paper
- The thesis citation "AdaLead (Sinai et al. 2020)" under GB1 usage is **incorrect**

**PEX (Ren et al. 2022, ICML) — does NOT use GB1**
- PEX benchmarks on avGFP (Sarkisyan et al. 2016), AAV capsid protein, TEM-1 β-Lactamase, E4B (ubiquitination factor), AMIE (aliphatic amide hydrolase), LGK (levoglucosan kinase), Pab1 (poly(A)-binding protein), and UBE2I (SUMO E2 conjugase)
- GB1 is not among the benchmark tasks
- The thesis citation "PEX (Gruver et al. 2023)" is also a name error — the PEX paper is Ren et al. 2022, ICML

---

## 2.2 TrpB — Tryptophan Synthase Beta Subunit

**Source**: Johnston et al. (2024). *A combinatorially complete epistatic fitness landscape in an enzyme active site.* Proc. Natl. Acad. Sci., 121, e2400439121.

⚠️ **Correction from earlier draft**: The TrpB 4-site combinatorial dataset used in ALDE is from Johnston et al. 2024, not Wittmann et al. 2021. Wittmann et al. (2021) is an MLDE methodology paper that works exclusively on GB1. Johnston et al. (2024) is the primary source of the TrpB combinatorial landscape.

**Protein**: TrpB is the β-subunit of tryptophan synthase, an enzyme involved in the biosynthesis of tryptophan. It has been engineered for non-natural amino acid synthesis, making it a biotechnologically relevant target.

**Experimental design**: A 4-site combinatorial library covering all 20⁴ = 160,000 variants at selected active site positions (V183, F184, S228, V227 in ALDE's Fig. 4 inset). Fitness is coupled growth rate measuring rate of tryptophan formation (enzymatic activity toward a non-natural substrate). Johnston et al. describe this as "a combinatorially complete epistatic fitness landscape."

**Dataset statistics**: ~160,000 variants, near-exhaustive combinatorial coverage at 4 positions. ALDE states that "fitnesses have been measured experimentally for nearly all of the 20⁴ = 160,000 variants."

**Epistasis**: High, directly characterizable from the combinatorial structure, similar to GB1.

**Why used**: Provides a second near-exhaustive combinatorial benchmark that generalizes GB1 findings to a different protein family and fitness type (enzymatic activity vs. antibody binding). ALDE used both GB1 and TrpB to validate their DNN Ensemble + One-hot recommendation (Fig. 4D) and to show that conclusions hold across datasets.

**Key difference from GB1**: Fitness type is enzymatic activity rather than binding affinity. This probes a different functional mechanism and landscape topology.

**Used in**: ALDE (Yang et al. 2024) as a computational simulation benchmark alongside GB1.

---

### 2.2.1 Per-Paper Dataset Version Analysis: TrpB

**ALDE (Yang et al. 2024)**
- Uses the **Johnston et al. 2024** TrpB landscape
- Described as "combinatorially complete" — approximately all 160,000 variants are measured
- Same simulation protocol as GB1: 96 initial samples + 4 × 96 rounds
- ALDE Figure 4D shows side-by-side results on GB1 and TrpB with consistent findings across both

---

## 2.3 GFP — Green Fluorescent Protein (avGFP)

**Source**: Sarkisyan et al. (2016). *Local fitness landscape of the green fluorescent protein.* Nature, 533(7603), 397–401.

**Protein**: avGFP is a 239-residue fluorescent protein from *Aequorea victoria*, widely used as a reporter in cell biology. Fitness is measured as log-fluorescence intensity — a continuous, quantitative readout with clear biological interpretation.

**Experimental design**: Random mutagenesis across the full protein sequence. Variants were sorted by fluorescence into four bins via FACS and sequenced by deep sequencing. This produces a sparse sample of the full sequence space, not a combinatorial sweep at fixed positions. Mutations occur at variable positions across the whole 239-residue protein.

**Dataset statistics**:
- Total measured variants: **~52,000 genotypes** (PEX reports "nearly 52,000 genotypes"; the original Sarkisyan paper reports "~54,000" in some summaries — the discrepancy reflects different counting methods for unique vs. observed sequences)
- Single-mutant space: 239 × 19 ≈ 4,500 possible single mutants
- Multi-mutant combinations: the ~52k variants span many combinations of up to 15 mutations from wild type
- Coverage: infinitesimally sparse relative to the full 20^239 sequence space; even relative to multi-mutant combinations the coverage is very low

**Epistasis**: Present but not precisely characterizable due to the sparse, unstructured nature of the data. The chromophore formation mechanism requires correct folding across the whole protein, making long-range epistatic interactions structurally important.

**Why used**: Represents a fundamentally different benchmarking regime from GB1/TrpB. The search space is enormous and sparsely sampled, requiring models to generalize across sequence space rather than interpolate within a near-exhaustive library. It tests whether models can leverage protein-level structural and evolutionary priors (e.g. ESM2 embeddings) to navigate regions with no direct coverage.

**Oracle type**: Lookup table with gaps. BO algorithms can propose sequences not in the measured set, which creates the "sparse oracle problem" — proposed variants may land outside the 52k measured variants and receive no oracle score.

**Used in**: PEX (Ren et al. 2022) as avGFP. Widely used in MLDE and generative model literature. Less common in BO-specific benchmarks due to the sparse oracle gap problem.

---

### 2.3.1 Per-Paper Dataset Version Analysis: GFP

**PEX (Ren et al. 2022, ICML)**
- Uses the **Sarkisyan et al. 2016 avGFP dataset** with "nearly 52,000 genotypes" (their language, p. 5)
- The optimization goal is to maximize log-fluorescence intensity
- Search space defined as the full 239-residue avGFP sequence space: 20^239 possible sequences
- Fitness values of the ~52k genotypes serve as the oracle; proposals outside this set receive no score in the simulation
- PEX's GFP results (Figure 4 and Table 1) show maximum fitness reaching ~3.1–3.2 (log-scale), compared to wild-type ~1.0

**AdaLead (Sinai et al. 2020) — does NOT use GFP**
- AdaLead does not benchmark on GFP in its paper

---

## 2.4 Summary Comparison

| Dataset | Protein Function | Space Type | Total Theoretical | Measured Variants (Used in ML) | Coverage | Epistasis | Oracle Type | Key Papers |
|---|---|---|---|---|---|---|---|---|
| GB1 | IgG binding | 4-site combinatorial | 160,000 | **149,361** (filtered, explicit) | ~93% | High, directly measurable | Lookup table (no gaps) | Wu 2019, Wittmann 2021, ALDE 2024 |
| TrpB | Enzymatic activity | 4-site combinatorial | 160,000 | ~160,000 (near-exhaustive) | ~100% | High, directly measurable | Lookup table (no gaps) | ALDE 2024 (Johnston et al. 2024 source) |
| GFP | Log-fluorescence | Full protein, sparse | 20^239 | ~52,000 (sparse sample) | ≈ 0% | Present, not well characterized | Lookup table (with gaps) | PEX 2022 |

---

## 2.5 Datasets Not Used and Why

**ProteinGym (Notin et al. 2022)**: A large benchmark collection of ~250 proteins with DMS data, primarily single-site substitutions. ProteinGym is the standard for evaluating **zero-shot** fitness predictors (models that predict fitness without any protein-specific training data). It is not appropriate for BO benchmarking because: (1) single-site data cannot reveal epistasis, (2) the optimization problem is trivial — simply pick the best single mutant, and (3) the benchmark was designed for a different task (zero-shot prediction, not iterative design).

**Olson et al. 2014 GB1**: Single and double mutants across the full 55-residue GB1 sequence (~3,135 variants). Limited combinatorial depth makes it unsuitable for studying epistasis or BO across a large combinatorial space.

---

## 2.6 Clarifications on Literature Claims

Several common attributions in the MLDE/BO literature require clarification:

**AdaLead (Sinai et al. 2020) does not use GB1, TrpB, or GFP**. AdaLead benchmarks on: (1) transcription factor (TF) DNA-binding landscapes, (2) RNA secondary structure folding landscapes via ViennaRNA, and (3) protein design using PyRosetta on a 66-residue antifreeze protein. It is a sequence design algorithm paper, not a combinatorial protein fitness benchmark study. Claims that "AdaLead used GB1" in survey papers appear to conflate AdaLead with papers that cite it.

**PEX (Ren et al. 2022, ICML) does not use GB1 or TrpB**. PEX uses avGFP (~52k variants) plus seven other DMS datasets (AAV, TEM-1, E4B, AMIE, LGK, Pab1, UBE2I). The PEX paper is by Ren et al. (2022) from ICML — not "Gruver et al. 2023." Gruver et al. (2021) is a separate ICML workshop paper on effective surrogate models with Bayesian optimization, which is a different work.

**TrpB source is Johnston et al. 2024, not Wittmann et al. 2021**. Wittmann et al. (2021) is an MLDE methodology paper focused entirely on GB1. The TrpB combinatorial landscape used in ALDE comes from Johnston et al. (2024), which explicitly describes "a combinatorially complete epistatic fitness landscape in an enzyme active site."

**The "MLDE" paper is Wu et al. 2019 (PNAS)**. References to "MLDE (Greenberg et al. 2022)" conflate two papers. The original MLDE protocol paper is Wu, Z. et al. (2019) in PNAS (Arnold lab). Greenberg et al. (2023, bioRxiv) is a later paper on benchmarking uncertainty quantification for protein engineering, which uses GB1 as an evaluation dataset but did not introduce MLDE.

---

## References

- Wu, N.C. et al. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. *eLife*, 5, e16965.
- Wu, Z. et al. (2019). Machine learning-assisted directed protein evolution with combinatorial libraries. *Proc. Natl. Acad. Sci.*, 116(18), 8852–8858.
- Wittmann, B.J. et al. (2021). Informed training set design enables efficient machine learning-assisted directed protein evolution. *Cell Systems*, 12(11), 1026–1045.
- Johnston, K.E. et al. (2024). A combinatorially complete epistatic fitness landscape in an enzyme active site. *Proc. Natl. Acad. Sci.*, 121, e2400439121.
- Sarkisyan, K.S. et al. (2016). Local fitness landscape of the green fluorescent protein. *Nature*, 533(7603), 397–401.
- Yang, K.K., Lal, R.G., Bowden, J.C. et al. (2024). Active Learning-Assisted Directed Evolution (ALDE). *bioRxiv*, https://doi.org/10.1101/2024.07.27.605457.
- Sinai, S. et al. (2020). AdaLead: A simple and robust adaptive greedy search algorithm for sequence design. *arXiv:2010.02141*.
- Ren, Z. et al. (2022). Proximal Exploration for Model-guided Protein Sequence Design. *ICML 2022*, PMLR 162.
- Notin, P. et al. (2022). Tranception: protein fitness prediction with autoregressive transformers and inference-time retrieval. *ICML 2022*.
- Greenberg, K.P., Amini, A.P. & Yang, K.K. (2023). Benchmarking Uncertainty Quantification for Protein Engineering. *bioRxiv*.
