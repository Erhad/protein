# Literature Review: Protein Fitness Optimization — Field Overview

## How the Field Has Evolved

---

## 1. Historical Foundations: Wet Lab Directed Evolution (1990s–2000s)

Protein engineering before machine learning was entirely a wet-lab discipline. The dominant paradigm was **directed evolution** — a process explicitly inspired by natural selection but compressed into laboratory timescales. The central insight, articulated most clearly by Frances Arnold, is that useful proteins can be discovered without understanding the underlying physics: if you can screen for a desired function, you can evolve toward it.

**Stemmer (1994)** introduced DNA shuffling as the first systematic method for recombining beneficial mutations across homologous sequences. Rather than making single mutations and testing each one, shuffling breaks parental genes into fragments, randomly reassembles them, and thereby explores combinations of mutations that no single-parent lineage would reach. This was a conceptual shift: epistasis — the non-additive interaction between mutations — was acknowledged as a fundamental property of fitness landscapes, not a nuisance. Shuffling implicitly addresses epistasis by allowing many mutations to combine simultaneously.

**Arnold (1998)** provided the theoretical and practical framing for directed evolution as an engineering strategy. The paper argues that evolution, not rational design, is the most reliable route to proteins with desired functions, because rational design requires complete mechanistic understanding that rarely exists. The iterative loop of mutation, selection, and amplification is presented as sufficient — the engineer need not understand why a mutation helps, only that it does. This paper established the intellectual framework that still underlies ML-assisted directed evolution today: the protein fitness landscape exists, it is navigable by search, and the key constraint is the number of variants you can screen per round.

**What these papers share**: Both are entirely wet-lab. There is no computational fitness model, no surrogate, and no oracle beyond the physical assay. The search strategy is essentially random with selection pressure — effective but expensive in screening burden, and prone to getting stuck at local optima because the search is inherently local (one or a few mutations per round).

---

## 2. Fitness Landscapes as Datasets (2009–2016)

Before ML methods could be developed, the field needed **maps** of fitness landscapes — experimental datasets where fitness had been measured for many variants simultaneously. Two conceptual frameworks helped crystallize thinking about landscape structure.

**Romero & Arnold (2009)** provided an influential review of directed evolution framed explicitly in terms of fitness landscape topology. A smooth landscape (single peak, no epistasis) is trivially navigable by greedy hill-climbing. A rugged landscape (many peaks, strong epistasis) is where directed evolution fails and where more sophisticated search is needed. The paper catalogs the ways epistasis appears in practice: sign epistasis (a mutation is beneficial alone but deleterious in combination), magnitude epistasis (synergistic or antagonistic interactions), and the resulting indirect paths between local optima. This review is the conceptual foundation for why the BO and MLDE literature focuses so heavily on epistatic landscapes.

**Wu et al. (2016) — GB1 dataset**: The first near-exhaustive experimental map of a 4-site combinatorial protein fitness landscape. By constructing all 20⁴ = 160,000 variants of the IgG-binding domain of protein G at four positions and measuring binding affinity via phage display enrichment, Wu et al. produced a ground-truth landscape against which computational methods could be evaluated. Key findings: 92% of variants have fitness below 1% of the global maximum, the landscape has 30 local optima, and routes to the optimum are often indirect (going through lower-fitness intermediates). The dataset is **non-additive / highly epistatic**: a simple additive model predicts only ~30% of fitness variance. This dataset became the standard benchmark for MLDE and BO simulation studies. The filtered version — 149,361 variants with sufficient sequencing depth — is the industry standard used in all ML benchmarks.

**Sarkisyan et al. (2016) — GFP dataset**: A complementary large-scale fitness map, but in a fundamentally different regime. Rather than exhaustive combinatorial coverage at 4 sites, Sarkisyan et al. used random mutagenesis across the full 239-residue avGFP sequence, measuring fluorescence for ~52,000 variants by FACS sorting into 4 fluorescence bins. The landscape is **sparse**: the 52k variants represent an infinitesimal fraction of the full sequence space. Epistasis is present but difficult to characterize precisely from such sparse data. The GFP dataset tests a different capability than GB1 — a model must generalize across sequence space, not just interpolate within a near-exhaustive combinatorial library. Used heavily in generative model and protein language model benchmarks.

**Johnston et al. (2024) — TrpB dataset**: A second near-exhaustive 4-site combinatorial landscape, this time for tryptophan synthase β-subunit (TrpB). Like GB1, all 20⁴ = 160,000 variants are measured, with fitness defined as enzymatic activity (growth coupling assay for tryptophan synthesis). The landscape is similarly rugged and epistatic, providing a second ground-truth oracle for BO simulation studies. ALDE (Yang et al. 2024) used both GB1 and TrpB to validate their methodological recommendations.

---

## 3. Bayesian Optimization Foundations (2012–2013)

Before BO was applied to proteins specifically, the general framework was established in the machine learning community.

**Snoek, Larochelle & Adams (2012)** demonstrated that Gaussian process-based Bayesian optimization, with the Expected Improvement (EI) acquisition function, could outperform manual tuning and grid search for hyperparameter optimization of machine learning algorithms. This paper popularized practical BO in the ML community and introduced the infrastructure (surrogate model + acquisition function + iterative querying) that all subsequent protein BO work inherits. The key insight is that EI explicitly models **uncertainty** — it does not just exploit the current best prediction, but actively queries regions where the model is uncertain yet the upper confidence bound is high. This exploration-exploitation tradeoff is the core of all BO approaches.

**Frazier (2018)** wrote the definitive tutorial on Bayesian optimization, covering Gaussian process surrogates, acquisition functions (EI, UCB, Thompson Sampling, Probability of Improvement), and the theoretical guarantees of BO. This tutorial is the standard reference for any new BO application and is cited universally in the protein engineering BO literature.

**Romero et al. (2013)** was the first application of Gaussian process BO to protein engineering. Using a small dataset of cytochrome P450 variants with measured thermostability, they trained a GP surrogate on one-hot encoded sequences and used it to recommend new variants to screen. This paper demonstrated that a GP could learn a useful fitness model from modest amounts of data (~100 variants) and that the resulting recommendations outperformed random sampling. It established the template — train surrogate, acquire next batch, update, repeat — that all subsequent BO papers follow. The landscape used was **small and not exhaustive** (no ground-truth oracle), which limits the ability to rigorously evaluate performance.

---

## 4. Machine Learning-Assisted Directed Evolution (2019–2021)

The MLDE paradigm — using ML to predict fitness across an unsampled combinatorial library and then experimentally testing the top predictions — was formalized in this period.

**Yang et al. (2019) — Review, Nature Methods**: This review articulated the landscape of ML-guided directed evolution, categorizing approaches as: (1) supervised regression on sequence-fitness data, (2) unsupervised/generative models, and (3) reinforcement learning. The paper was influential in framing the space of methods and in drawing attention to the limits of existing approaches: most methods had been tested only on small datasets or without rigorous comparison to DE baselines. It also highlighted the practical constraints — small training sets (tens to hundreds of variants), high-dimensional sequence spaces, and the difficulty of generalizing from sparse data.

**Wu et al. (2019) — MLDE, PNAS**: The foundational MLDE paper from the Arnold lab. Starting from a combinatorial library, a random subsample (~96 variants) is characterized experimentally and used to train an ensemble of regressors. The ensemble predicts fitness for all unsampled combinations, the top predictions are experimentally tested, and the cycle repeats. Applied to the GB1 landscape (149,361 filtered variants as oracle), MLDE significantly outperformed greedy single-step directed evolution at identifying high-fitness variants within a fixed screening budget. The original encoding was one-hot; training set size was 96–480 variants. This is the paper that established GB1 as the standard MLDE benchmark.

**Wittmann et al. (2021) — ftMLDE, Cell Systems**: A systematic evaluation of design choices within the MLDE framework, using GB1 as the sole benchmark. Key findings: (1) More informative protein encodings (Georgiev physicochemical parameters, MSA Transformer embeddings) generally improve MLDE outcome, but the improvement is inconsistent across metrics and training set sizes. (2) The most important design choice is **training set composition**: combinatorial libraries are dominated by zero- and very-low-fitness variants ("holes"), and training on such data biases the ML model toward predicting low fitness. Focused MLDE (ftMLDE) uses zero-shot predictors (EVmutation, MSA Transformer, Rosetta ΔΔG) to enrich the training set with functional variants, dramatically improving outcome. Explicitly uses the **149,361 filtered GB1 variants** and explicitly excludes imputed variants. This paper is the most rigorous head-to-head comparison of encoding and training set strategies on a combinatorial landscape.

**Biswas et al. (2021) — Low-N protein engineering, Nature Methods**: Demonstrates that combining a protein language model (ESM1b) with a small experimental dataset enables effective protein engineering with very few labeled examples. Using ESM1b embeddings as sequence features, a simple linear regression trained on as few as 24 variants can outperform methods that use more data but weaker encodings. Applied to several proteins including GB1 and fluorescent proteins. The key insight is that language model embeddings implicitly encode evolutionary context — they are pre-trained on millions of sequences and capture which mutations are likely to preserve function. This is the "Low-N" regime that is practically important because most real engineering campaigns can only screen tens to hundreds of variants.

**Hie et al. (2020) — Uncertainty, Cell Systems**: Demonstrates that uncertainty-aware machine learning (specifically GP models) dramatically accelerates biological discovery by enabling efficient exploration. Applied to drug discovery (kinase inhibitor identification and tuberculosis growth inhibition) as well as protein engineering. Shows that training on just 72 compounds and using uncertainty-based acquisition finds nanomolar-affinity compounds. This paper introduced the principle of **calibrated uncertainty** as a design criterion — a model's predicted uncertainty should correlate with its actual error, and this calibration is what makes active learning effective. Published directly in Cell Systems with no bioRxiv preprint.

---

## 5. Generative Models and Reinforcement Learning (2018–2020)

A parallel research thread attempted to move beyond supervised regression to fully generative approaches — models that can propose novel sequences rather than just ranking a predefined library.

**Brookes & Listgarten (2018) — DbAS**: Design by Adaptive Sampling uses a variational autoencoder (VAE) as a generative model, trained on high-fitness sequences from the previous round. The generator learns a latent space over sequences; sequences sampled from this space and filtered by a fixed oracle constitute the next batch. The key limitation identified by the authors is that the oracle is unreliable far from its training domain — a problem they term "adversarial exploitation." DbAS treats this as a data-weighting problem: samples that the oracle would have assigned low probability under the original data distribution are downweighted.

**Brookes et al. (2019) — CbAS, ICML**: Conditioning by Adaptive Sampling directly addresses the oracle trust-region problem. CbAS trains the generative model to sample sequences that balance high predicted fitness and proximity to the training distribution, using importance weighting to handle the mismatch between the generative model's distribution and the oracle's training distribution. CbAS is one of the stronger generative baselines in the literature and is frequently used as a comparison point for BO methods on protein benchmarks.

**Angermueller et al. (2020) — DyNA-PPO, ICLR**: Model-based reinforcement learning for biological sequence design. A policy network (implemented with PPO) is trained to propose sequences, using an ensemble of surrogate models as an environment simulator (hence "DyNA" — dynamic surrogate). The ensemble provides both mean predictions and uncertainty estimates, which the RL agent uses for exploration. Applied to DNA and protein design tasks. DyNA-PPO is a strong representative of the RL class of sequence design algorithms and demonstrates that RL-based exploration can be more effective than simple BO when the sequence space is very large.

---

## 6. Active Learning and Bayesian Optimization for Proteins (2020–2024)

This is the core period for methods most directly comparable to the work in this thesis.

**Sinai et al. (2020) — AdaLead**: A deliberately simple adaptive greedy search algorithm. Starting from a set of high-fitness seed sequences, AdaLead iteratively recombines and mutates them, using a surrogate model to filter candidates before oracle queries. Despite its simplicity, AdaLead is competitive with or outperforms more complex approaches (DyNA-PPO, CbAS) across a wide range of landscapes (TF binding, RNA folding, protein design). AdaLead's benchmark is NOT GB1 or GFP — it uses TF binding, RNA secondary structure, and PyRosetta protein design, which is a different evaluation regime from the combinatorial lookup-table benchmarks. The paper's main contribution is showing that a simple hill-climbing evolutionary algorithm with adaptive thresholding is a surprisingly strong baseline that should be used for comparison before pursuing more complex methods.

**Stanton et al. (2022) — LaMBO, ICML**: Latent-space Bayesian Optimization for proteins. LaMBO trains a masked language model on sequence data to define a latent space, then performs BO directly in that latent space using a GP surrogate with the Matern kernel. The GP is conditioned on the latent representation, and acquisition is performed by gradient ascent in latent space followed by decoding. Key finding: doing BO in a pre-trained latent space is substantially more effective than doing BO in raw sequence space, because the latent space provides a smooth prior over functional sequences. Applied to several protein engineering tasks including fluorescence and drug resistance.

**Ren et al. (2022) — PEX, ICML**: Proximal Exploration for model-guided protein design. PEX introduces a proximity constraint into the acquisition function: rather than proposing the globally highest-scoring sequence according to the surrogate, PEX only proposes sequences within a defined edit distance (Hamming distance) of the current best-known variant. This "proximal" acquisition prevents the surrogate from being fooled by out-of-distribution sequences that appear high-fitness but are far from any training example. Applied to avGFP (~52k variants, Sarkisyan 2016), AAV, TEM-1 β-lactamase, and five other DMS datasets. PEX consistently outperforms standard EI-based BO and CbAS on these tasks.

**Jiang et al. (2024) — EVOLVEpro, Science**: EVOLVEpro is a few-shot active learning framework that combines a protein language model with a lightweight regression head (random forest or ridge regression) and iterates in small experimental rounds of ~10–16 variants per cycle. The key distinction from most prior work is the emphasis on **extreme data efficiency**: EVOLVEpro is explicitly designed for the regime where only a handful of labeled measurements are available at each round, and the PLM embedding provides the necessary prior over functional sequence space. Applied to six engineering targets spanning RNA polymerase, CRISPR prime editing, genome integration (serine integrase), and antibody binding, EVOLVEpro achieved up to **100-fold improvements** in target activity within 3–4 rounds. Notably, this was achieved with as few as 16 labeled variants per round — far below the 96–384 typically used in MLDE. EVOLVEpro's success on diverse protein families using a fixed PLM (ESM2) + regression architecture suggests that the combination of strong evolutionary priors and simple models is broadly generalizable, echoing the findings of Biswas et al. (2021) on low-N engineering. Published in *Science* 2024, this paper directly competes with ALDE in the active learning for proteins space, but emphasizes the wet-lab side (fewer variants per round, more diverse protein targets) rather than the algorithmic benchmarking side.

**Yang et al. (2024) — ALDE**: Active Learning-Assisted Directed Evolution. The paper that most directly motivates this thesis. ALDE frames protein engineering as batch Bayesian optimization on a combinatorial landscape, benchmarking on GB1 (Wu et al. 2016, 149,361 filtered variants) and TrpB (Johnston et al. 2024, ~160k variants). Key methodological findings from computational simulations: (1) DNN ensembles with one-hot encoding perform best overall — frequentist uncertainty quantification (ensemble variance) outperforms Bayesian approaches (GP) for active learning in this regime; (2) ESM2 embeddings do not improve over one-hot on combinatorial landscapes; (3) Thompson Sampling is the most effective acquisition function. The wet-lab component applies ALDE to a 5-site combinatorial space in a protoglobin enzyme, using only 480 variants over 3 rounds to reach 99% cyclopropanation yield and 14:1 selectivity. This demonstrates that BO can navigate a 20⁵ = 3.2M variant space with a screening budget comparable to 2 rounds of standard DE.

**Kirjner et al. (2023) — Smoothed Fitness Landscapes, ICLR 2024**: Addresses a fundamental challenge in using surrogate models for protein optimization: the surrogate fitness landscape is typically much smoother than the true landscape, which can cause the optimizer to get stuck. Kirjner et al. propose **fitness landscape smoothing** — using diffusion-based or kernel-based smoothing to create an intermediate surrogate that is easier to optimize than the true landscape but still predictive. The smoothed landscape provides better gradient signal for sequence optimization. Applied to GFP, TEM, and other protein benchmarks, showing consistent improvements over standard BO.

**Greenberg et al. (2023) — Benchmarking UQ, bioRxiv**: A systematic study of uncertainty quantification methods in the context of protein engineering, using GB1 as a primary benchmark. Tests GP, MC-Dropout, ensemble, and conformal prediction approaches. Key findings: (1) well-calibrated uncertainty matters for active learning performance, but no single UQ method dominates; (2) ensemble-based UQ is consistently competitive; (3) calibration measured on the full landscape (which is only possible in simulation) does not always predict active learning performance, suggesting that calibration in-distribution and calibration at the frontier (acquisition decisions) are different problems. This paper provides the benchmarking infrastructure that ALDE's UQ analysis builds on.

---

## 7. Protein Language Models (2021–2023)

A major parallel development is the emergence of large-scale protein language models, pre-trained on evolutionary sequence databases without any fitness labels. These models provide **zero-shot fitness predictors** and high-dimensional sequence representations.

**Rives et al. (2021) — ESM1b, PNAS**: The first large-scale protein language model from Meta. ESM1b is a 650M-parameter transformer trained on 250 million UniRef50 protein sequences using masked language modeling. The key finding is that structural and functional information emerges implicitly from sequence co-evolutionary statistics learned at scale: attention patterns correspond to contact maps, and per-position likelihoods predict mutational effects. ESM1b was the first model to demonstrate that unsupervised pre-training on protein sequences could produce embeddings useful for supervised fitness prediction without any protein-specific training data.

**Lin et al. (2023) — ESM2 + ESMFold, Science**: ESM2 scales the ESM architecture to 15 billion parameters and introduces ESMFold — direct protein structure prediction from the ESM2 trunk in a single forward pass, without multiple sequence alignments. ESM2 embeddings are the most commonly used protein language model features in ML-guided protein engineering as of 2024. For protein property prediction on combinatorial landscapes like GB1, ESM2 (1280-dimensional per-residue embeddings) does not consistently outperform one-hot encoding when training sets are small (<384 variants), as shown by both Wittmann (2021) and ALDE (2024). ESM2 is more useful when the task requires generalization across diverse sequence space (GFP, DMS tasks) rather than interpolation within a small combinatorial library.

**Elnaggar et al. (2022) — ProtTrans, TPAMI**: A family of protein language models (ProtBERT, ProtT5, etc.) trained on BFD (2.1 billion sequences) and UniRef. ProtTrans established that transformer architectures adapted from NLP (BERT, T5, XLNet) could be directly applied to protein sequences with strong results on secondary structure prediction, contact prediction, and subcellular localization. ProtBERT-BFD is a common baseline in embedding comparisons. ProtTrans models are generally competitive with ESM1b and sometimes superior depending on the task, though ESM2 has largely superseded both for engineering applications.

**Notin et al. (2022) — Tranception + ProteinGym, ICML**: Tranception is an autoregressive transformer for protein fitness prediction that incorporates retrieval of homologous sequences at inference time. More importantly, this paper introduced **ProteinGym** — a benchmark of ~250 deep mutational scanning (DMS) datasets covering diverse proteins and fitness types. ProteinGym is the standard benchmark for **zero-shot** fitness predictors (models that predict variant fitness without any protein-specific labeled data). On ProteinGym, Tranception with retrieval outperforms EVmutation, ESM1b masked marginals, and other zero-shot baselines. ProteinGym is not appropriate for BO benchmarking (the optimization problem is trivial — just pick the best single mutant), but it is the definitive resource for evaluating the zero-shot fitness prediction capabilities of language models.

**Madani et al. (2023) — ProGen2, Nature Biotechnology**: Large language model for protein generation. ProGen2 is an autoregressive model (up to 6.4B parameters) trained on ~280 million protein sequences, capable of generating novel protein sequences conditioned on protein family tokens. The key result is wet-lab validation: ProGen2-generated lysozyme sequences have measurable enzymatic activity, including sequences with only 31.4% identity to any known lysozyme. This establishes that generative protein language models can produce functional sequences in regions of sequence space not covered by known proteins — a capability that purely discriminative models (like ESM2 used as a fitness predictor) cannot provide.

**Hayes et al. (2025) — ESM3, Science**: ESM3 is EvolutionaryScale's third-generation protein language model and the first to be natively **multimodal**: it jointly reasons over sequence, structure, and function, all three tokenized and trained together. At 98B parameters trained on 2.78 billion proteins and 771 billion tokens, ESM3 is roughly an order of magnitude larger than ESM2. The key demonstration is **esmGFP** — a novel fluorescent protein generated by ESM3 that is only 58% identical to the closest known fluorescent protein, with an estimated evolutionary distance equivalent to 500 million years of natural selection. This is qualitatively different from ESM2-based design: rather than using language model embeddings as features for a downstream regression model, ESM3 directly generates sequences conditioned on structural and functional prompts. **Practical note**: the full 98B model is accessible only via EvolutionaryScale's Forge API. A smaller open-weight version (`esm3-sm-open-v1`, 1.4B params) is available on HuggingFace under a non-commercial research license. EvolutionaryScale also released **ESM Cambrian / ESM C** as a direct successor to ESM2 for embedding tasks: ESM C 300M matches ESM2-650M performance with substantially lower compute cost. For the BO/MLDE paradigm, ESM3 introduces a different usage pattern — it is more naturally applied as a generative oracle or structural predictor than as a fixed feature extractor, and its utility for the combinatorial landscape (GB1, TrpB) BO setting remains an open question.

---

## 8. Structure-Based Computational Design (2021–2023)

A third major thread, largely orthogonal to the fitness-landscape BO literature, uses protein structure as the design target rather than a measured fitness proxy.

**Jumper et al. (2021) — AlphaFold2, Nature**: AlphaFold2 solved the protein structure prediction problem that had been open for 50 years, achieving near-experimental accuracy on CASP14 benchmarks. The model uses multiple sequence alignments, pairwise residue representations, and iterative structure module updates. AlphaFold2's impact on protein engineering is primarily indirect: it provides accurate structure models for proteins where experimental structures are unavailable, enabling structure-based rational design, docking, and engineering campaigns that previously required expensive crystallography. AlphaFold2 is not a fitness predictor and is not used directly in BO pipelines, but it has transformed the ability to design starting points for engineering campaigns.

**Dauparas et al. (2022) — ProteinMPNN, Science**: A graph neural network for protein sequence design given a fixed backbone structure. ProteinMPNN takes a backbone (coordinates only) as input and outputs a probability distribution over amino acids at each position, trained to maximize sequence recovery on known protein structures. ProteinMPNN dramatically outperformed prior structure-based sequence design methods and enabled efficient generation of diverse sequences for a given structural scaffold. It is widely used in the protein design pipeline: RFdiffusion generates a backbone, ProteinMPNN designs sequences for it, AlphaFold2 validates the fold. ProteinMPNN is relevant to the fitness optimization literature because it defines a probabilistic model over sequences conditioned on structure — a form of zero-shot fitness predictor for structural tasks.

**Watson et al. (2023) — RFdiffusion, Nature**: Diffusion model for de novo protein structure generation. RFdiffusion adapts the denoising diffusion probabilistic model framework to protein backbone generation, enabling design of novel folds, protein-protein interfaces, enzyme active sites, and constrained scaffolds. RFdiffusion generated binders to four therapeutic targets with experimental hit rates up to 20%, orders of magnitude above typical computational design pipelines. Together with ProteinMPNN and AlphaFold2, RFdiffusion completes the fully computational protein design pipeline that does not require any experimental measurement during the design stage. This represents a fundamentally different paradigm from BO/MLDE: rather than navigating a measured fitness landscape, these methods design proteins from structural and physical principles.

---

## 9. The BO in Embedding Space (2024)

**BO in Embedding Space (BIBM 2024)**: The most closely related work to this thesis. Directly tests whether Bayesian optimization performed in the embedding space of a protein language model (ESM2) outperforms BO in raw sequence space for combinatorial protein fitness optimization. Uses GB1 as the primary benchmark. Finding: BO in ESM2 embedding space with a GP surrogate achieves competitive but not consistently superior results compared to one-hot BO, consistent with ALDE's finding that ESM2 embeddings do not add value on combinatorial landscapes. The paper highlights the computational cost of embedding-space BO (ESM2 forward passes required for all candidates) as a practical limitation.

---

## 10. Cross-Cutting Themes

### 10.1 Wet Lab vs. Non-Wet Lab

The field divides cleanly into two types of work:

**Fully wet-lab** (Stemmer 1994, Arnold 1998): No computational model. Mutation, screening, and selection are all physical. Performance is measured by the actual engineered protein. These papers define the baseline that computational methods must beat.

**Simulation studies on fixed oracles** (Wittmann 2021, ALDE simulations, LaMBO, PEX, AdaLead on RNA/TF): The protein fitness landscape is treated as a fixed lookup table. An algorithm proposes sequences, the oracle returns fitness values, and performance is measured by the maximum fitness discovered within a fixed query budget. This regime allows rigorous algorithmic comparison but assumes the oracle is perfect and that proposed sequences are always in the measured set.

**Hybrid: computational recommendations + wet-lab validation** (Romero 2013, Wu 2019, ALDE wet-lab, Biswas 2021): The computational method makes recommendations, which are actually synthesized and measured in the lab. This is the most practically relevant regime but the hardest to evaluate rigorously, because the experimental noise, synthesis failures, and oracle gaps all affect results.

**Fully computational design** (AlphaFold2, ProteinMPNN, RFdiffusion, ProGen2): No experimental measurement in the loop. These methods design sequences from structural or evolutionary principles and then validate in the lab. Performance metrics are structural accuracy and experimental hit rates, not fitness landscape navigation.

### 10.2 The Additivity Question

A central empirical question across datasets is whether mutations combine **additively** (the effect of a double mutant is the sum of the individual effects) or **epistatically** (the combined effect is different from the sum).

| Dataset | Additivity | Notes |
|---|---|---|
| GB1 (Wu 2016) | **Low** — highly epistatic | 92% of variants have near-zero fitness; 30 local optima; indirect paths dominate |
| TrpB (Johnston 2024) | **Low** — highly epistatic | Similar ruggedness to GB1; many indirect paths |
| GFP (Sarkisyan 2016) | **Moderate** — partially additive | Many single mutants are functional; multi-mutant effects partially predictable from singles, but not reliably |
| ProteinGym (Notin 2022) | **Moderate** — mostly additive | Single-site only; epistasis between distant sites not measured |

The degree of epistasis determines which methods work. On highly epistatic landscapes (GB1, TrpB), methods that can capture non-additive interactions (ensemble regressors, GP with appropriate kernels, DNN ensembles) significantly outperform additive models. On landscapes with moderate additivity (GFP, ProteinGym), simpler methods and zero-shot predictors perform well.

### 10.3 Methodology Evolution

| Era | Methods | Key Papers |
|---|---|---|
| 1990s–2000s | Random mutagenesis, DNA shuffling, phage display | Stemmer 1994, Arnold 1998 |
| 2009–2016 | Fitness landscape characterization, GP foundations | Romero & Arnold 2009, Romero et al. 2013, Wu 2016, Sarkisyan 2016 |
| 2019–2021 | MLDE (supervised regression on combinatorial libraries) | Wu 2019, Wittmann 2021, Yang 2019, Biswas 2021 |
| 2020–2022 | Active learning, generative models, RL | AdaLead, CbAS, DbAS, DyNA-PPO, LaMBO |
| 2021–2023 | Protein language models as feature extractors | ESM1b, ESM2, ProtTrans, Tranception |
| 2021–2023 | Structure-based computational design | AlphaFold2, ProteinMPNN, RFdiffusion |
| 2022–2024 | BO with uncertainty quantification, active learning for proteins | PEX, ALDE, Kirjner, Greenberg, BO in embedding space |
| 2024–2025 | Few-shot active learning + multimodal generative PLMs | EVOLVEpro, ESM3 |

---

## 11. Summary Table

| # | Paper | Year | Type | Method | Dataset(s) | Wet Lab? | Dataset Size | Additive? | Key Finding |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Stemmer 1994 | 1994 | Wet lab | DNA shuffling | N/A | ✅ Yes | N/A | N/A | Recombination enables combinatorial exploration; first practical DE method |
| 2 | Arnold 1998 | 1998 | Wet lab | Directed evolution | N/A | ✅ Yes | N/A | N/A | DE as engineering strategy; no mechanistic understanding needed |
| 3 | Romero & Arnold 2009 | 2009 | Review | Fitness landscape theory | N/A | ✅/📚 Review | N/A | N/A | Ruggedness and epistasis explain DE failures; motivates search beyond local |
| 4 | Snoek et al. 2012 | 2012 | ML / BO | GP + EI acquisition | ML hyperparameters | ❌ No | N/A | N/A | Practical BO with GP surrogate; EI acquisition function; foundational BO paper |
| 5 | Romero et al. 2013 | 2013 | Hybrid | GP-BO | P450 thermostability (~100 variants) | ✅ Hybrid | ~100 | Unknown | First BO application to proteins; GP surrogate works with small data |
| 6 | Frazier 2018 | 2018 | Tutorial | BO theory | N/A | ❌ No | N/A | N/A | EI, UCB, TS acquisition functions; definitive BO tutorial |
| 7 | Brookes & Listgarten 2018 | 2018 | In silico | VAE generative model (DbAS) | GFP, TF binding | ❌ No | ~50k / small | Moderate | Generative model + oracle; adversarial exploitation is a key failure mode |
| 8 | Wu et al. 2016 | 2016 | Wet lab | Phage display DMS | GB1 (160,000 variants) | ✅ Yes | **149,361** filtered | **No — highly epistatic** | Near-exhaustive 4-site landscape; 30 local optima; indirect paths dominate |
| 9 | Sarkisyan et al. 2016 | 2016 | Wet lab | FACS + deep seq | avGFP (~52,000 variants) | ✅ Yes | **~52,000** | Moderate | Sparse full-protein landscape; threshold epistasis; chromophore folding required |
| 10 | Yang et al. 2019 | 2019 | Review | Survey of ML methods | Various | 📚 Review | Various | Various | ML-guided DE taxonomy; identifies key open problems in the field |
| 11 | Wu et al. 2019 | 2019 | Hybrid | MLDE (ensemble regression) | GB1 149,361 | ✅ Hybrid | **149,361** | **No** | MLDE outperforms DE; one-hot encoding; 96–480 variant budget |
| 12 | Brookes et al. 2019 | 2019 | In silico | VAE generative model (CbAS) | GFP, TF binding | ❌ No | ~50k / small | Moderate | Importance weighting fixes oracle trust-region problem |
| 13 | Angermueller et al. 2020 | 2020 | In silico | RL (DyNA-PPO) | DNA/protein design | ❌ No | N/A | N/A | RL with ensemble simulator effective for large sequence spaces |
| 14 | Hie et al. 2020 | 2020 | Hybrid | GP uncertainty + pretrained features | Drug discovery, proteins | ✅ Hybrid | Small (~72–200) | N/A | Calibrated uncertainty critical; low-N learning with strong priors |
| 15 | Sinai et al. (AdaLead) 2020 | 2020 | In silico | Adaptive greedy search | TF binding, RNA, Rosetta | ❌ No | Various | N/A | Simple greedy + adaptive threshold competitive with complex methods |
| 16 | Biswas et al. 2021 | 2021 | Hybrid | ESM1b embeddings + linear | GB1, others | ✅ Hybrid | **149,361** oracle | **No** | Language model embeddings enable low-N (~24 samples) engineering |
| 17 | Rives et al. (ESM1b) 2021 | 2021 | In silico | Masked LM (650M params) | UniRef50 pretraining | ❌ No | 250M sequences | N/A | Structural/functional info emerges from seq co-evolution at scale |
| 18 | Wittmann et al. 2021 | 2021 | In silico | MLDE + ftMLDE | GB1 **149,361** | ❌ Simulation | **149,361** | **No** | Training set design (holes) most important MLDE factor; ftMLDE 81x better than DE |
| 19 | Stanton et al. (LaMBO) 2022 | 2022 | In silico | BO in latent space (GP + masked LM) | Fluorescence, drug resistance | ❌ No | Medium | Moderate | BO in language model latent space superior to raw sequence BO |
| 20 | Ren et al. (PEX) 2022 | 2022 | In silico | Proximal BO | avGFP 52k + 7 others | ❌ No | **~52,000** | Moderate | Proximity constraint prevents surrogate exploitation; PEX > EI, CbAS |
| 21 | Notin et al. (Tranception) 2022 | 2022 | In silico | Autoregressive LM + retrieval | ProteinGym ~250 DMS | ❌ No | ~250 datasets | Mostly single-site | Zero-shot fitness prediction benchmark; Tranception best zero-shot predictor |
| 22 | Elnaggar et al. (ProtTrans) 2022 | 2022 | In silico | BERT/T5 on proteins | BFD 2.1B sequences | ❌ No | 2.1B sequences | N/A | ProtBERT/ProtT5 competitive with ESM1b; strong NLP-to-protein transfer |
| 23 | Jumper et al. (AlphaFold2) 2021 | 2021 | In silico | Structure prediction (transformer) | PDB + MSA | ❌ No | PDB ~170k | N/A | Near-experimental structure prediction; enables structure-based engineering |
| 24 | Dauparas et al. (ProteinMPNN) 2022 | 2022 | Hybrid | GNN sequence design | PDB structures | ✅ Hybrid | PDB ~170k | N/A | Structure-conditioned sequence design; enables efficient scaffold-based engineering |
| 25 | Watson et al. (RFdiffusion) 2023 | 2023 | Hybrid | Diffusion model (backbone gen) | PDB + AF2 | ✅ Hybrid | PDB + AF2 | N/A | De novo backbone generation; 20% experimental hit rates for binders |
| 26 | Madani et al. (ProGen2) 2023 | 2023 | Hybrid | Autoregressive LM (up to 6.4B) | UniRef + family tokens | ✅ Hybrid | 280M sequences | N/A | Generated lysozymes with 31.4% identity to known proteins retain function |
| 27 | Lin et al. (ESM2) 2023 | 2023 | In silico | Masked LM + structure (15B params) | UniRef pretraining | ❌ No | 250M sequences | N/A | ESM2 + ESMFold; 1280-dim embeddings; does not beat one-hot on small combinatorial sets |
| 28 | Greenberg et al. 2023 | 2023 | In silico | UQ benchmarking | GB1 149,361 | ❌ No | **149,361** | **No** | Ensemble UQ competitive; calibration in-distribution ≠ calibration at frontier |
| 29 | Kirjner et al. 2023 | 2023 | In silico | Smoothed BO | GFP, TEM, others | ❌ No | ~52k / medium | Moderate | Fitness landscape smoothing improves gradient signal for surrogate optimization |
| 30 | Johnston et al. 2024 | 2024 | Wet lab | DMS (growth coupling) | TrpB 160,000 variants | ✅ Yes | **~160,000** | **No — highly epistatic** | Second near-exhaustive 4-site combinatorial landscape; used by ALDE for validation |
| 31 | Yang et al. (ALDE) 2024 | 2024 | Hybrid | Batch BO (DNN ensemble + TS) | GB1 149,361 + TrpB 160k | ✅ Hybrid | **149,361 / ~160k** | **No** | DNN ensemble + one-hot + Thompson Sampling best; ESM2 no benefit; wet-lab: 480 variants → 99% yield |
| 32 | BO in Embedding Space 2024 | 2024 | In silico | BO in ESM2 latent space | GB1 | ❌ No | **149,361** | **No** | Embedding-space BO competitive but not better than one-hot; high computational cost |
| 33 | Jiang et al. (EVOLVEpro) 2024 | 2024 | Hybrid | Few-shot active learning (PLM + random forest) | RNA Pol, CRISPR, integrase, antibody | ✅ Hybrid | Small (~16–64/round) | N/A | Up to 100× improvement in 3–4 rounds with ≤16 variants/round; PLM prior enables extreme low-N |
| 34 | Hayes et al. (ESM3) 2025 | 2025 | Hybrid | Multimodal generative LM (98B params) | UniRef + PDB (2.78B proteins) | ✅ Hybrid | 2.78B sequences | N/A | First multimodal PLM (seq+struct+function); esmGFP only 58% identical to known FPs; API-only at full scale |

---

## References

1. Stemmer, W.P.C. (1994). Rapid evolution of a protein in vitro by DNA shuffling. *Nature*, 370, 389–391.
2. Arnold, F.H. (1998). Design by directed evolution. *Acc. Chem. Res.*, 31(3), 125–131.
3. Romero, P.A. & Arnold, F.H. (2009). Exploring protein fitness landscapes by directed evolution. *Nat Rev Mol Cell Biol*, 10, 866–876.
4. Snoek, J., Larochelle, H. & Adams, R.P. (2012). Practical Bayesian optimization of machine learning algorithms. *NeurIPS*. arXiv:1206.2944.
5. Romero, P.A. et al. (2013). Navigating the protein fitness landscape with Gaussian processes. *PNAS*, 110(3), E193–E201.
6. Wu, N.C. et al. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. *eLife*, 5, e16965.
7. Sarkisyan, K.S. et al. (2016). Local fitness landscape of the green fluorescent protein. *Nature*, 533(7603), 397–401.
8. Yang, K.K., Wu, Z. & Arnold, F.H. (2019). Machine-learning-guided directed evolution for protein engineering. *Nat Methods*, 16, 687–694. arXiv:1811.10775.
9. Wu, Z. et al. (2019). Machine learning-assisted directed protein evolution with combinatorial libraries. *PNAS*, 116(18), 8852–8858.
10. Brookes, D.H. & Listgarten, J. (2018). Design by adaptive sampling. arXiv:1810.03714.
11. Brookes, D.H. et al. (2019). Conditioning by adaptive sampling for robust design. *ICML*. arXiv:1901.10060.
12. Angermueller, C. et al. (2020). Model-based reinforcement learning for biological sequence design. *ICLR*. openreview:HklxbgBKvr.
13. Hie, B., Bryson, B.D. & Berger, B. (2020). Leveraging uncertainty in machine learning accelerates biological discovery and design. *Cell Systems*, 11, 461–477.
14. Sinai, S. et al. (2020). AdaLead: A simple and robust adaptive greedy search algorithm for sequence design. arXiv:2010.02141.
15. Biswas, S. et al. (2021). Low-N protein engineering with data-efficient deep learning. *Nat Methods*, 18, 389–396. bioRxiv:2020.01.23.917682.
16. Rives, A. et al. (2021). Biological structure and function emerging from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15), e2016239118. bioRxiv:622803.
17. Wittmann, B.J. et al. (2021). Informed training set design enables efficient machine learning-assisted directed protein evolution. *Cell Systems*, 12(11), 1026–1045.
18. Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.
19. Frazier, P.I. (2018). A tutorial on Bayesian optimization. arXiv:1807.02811.
20. Dauparas, J. et al. (2022). Robust deep learning-based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49–56. bioRxiv:2022.06.03.494563.
21. Elnaggar, A. et al. (2022). ProtTrans: Toward understanding the language of life through self-supervised learning. *TPAMI*, 44(10), 7112–7127. arXiv:2007.06225.
22. Notin, P. et al. (2022). Tranception: Protein fitness prediction with autoregressive transformers and inference-time retrieval. *ICML*. arXiv:2205.13760.
23. Stanton, S. et al. (2022). Accelerating Bayesian optimization for biological sequence design with denoising autoencoders. *ICML*. arXiv:2203.12597.
24. Ren, Z. et al. (2022). Proximal exploration for model-guided protein sequence design. *ICML*. PMLR 162.
25. Watson, J.L. et al. (2023). De novo design of protein structure and function with RFdiffusion. *Nature*, 620, 1089–1100. bioRxiv:2022.12.09.519842.
26. Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130. bioRxiv:2022.07.20.500902.
27. Madani, A. et al. (2023). Large language models generate functional protein sequences across diverse families. *Nat Biotechnol*, 41, 1099–1106. bioRxiv:2022.12.21.521526.
28. Greenberg, K.P., Amini, A.P. & Yang, K.K. (2023). Benchmarking uncertainty quantification for protein engineering. *bioRxiv*:2023.09.05.556426.
29. Kirjner, A. et al. (2023). Improving protein optimization with smoothed fitness landscapes. arXiv:2307.00494. *ICLR 2024*.
30. Johnston, K.E. et al. (2024). A combinatorially complete epistatic fitness landscape in an enzyme active site. *PNAS*, 121, e2400439121. bioRxiv:2024.06.23.600144.
31. Yang, K.K. et al. (2024). Active learning-assisted directed evolution (ALDE). *bioRxiv*:2024.07.27.605457.
32. [BO in Embedding Space] (2024). Bayesian optimization in protein language model embedding space. *BIBM 2024*.
33. Jiang, K. et al. (2024). Rapid in silico directed evolution by a protein language model with EVOLVEpro. *Science*, 386(6722). doi:10.1126/science.adr6006.
34. Hayes, E.L. et al. (2025). Simulating 500 million years of evolution with a language model. *Science*, 387(6730). doi:10.1126/science.ads0018. bioRxiv:2024.07.01.600583.
