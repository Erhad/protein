# Literature Pivot Tables
## Three views: Per-paper · Per-dataset · Per-model

---

# PARADIGM CLASSIFICATION — ALL PAPERS

| Paradigm | Papers |
|---|---|
| 🟢 **Pool-based MLDE** (train on n variants from fixed pool; ground truth = real measurements) | Wu 2019, Wittmann 2021, Romero 2013 (GP), ALDE 2025, BOES 2025, Hie 2020 |
| 🔴 **Generative oracle search** (train oracle on ALL data; propose novel sequences; oracle = ground truth) | DbAS 2018, CbAS 2019, DynaPPO 2020, AdaLead 2020, PEX 2022, LaMBO 2022, GGS 2023 |
| 🟡 **Hybrid** (low-N training + generative search + wet-lab validation) | Biswas 2021, EVOLVEpro 2024, LaMBO 2022 (partial) |
| 🔵 **Zero-shot** (no training labels; PLM log-likelihood as fitness proxy) | Notin/Tranception 2022, ESM1b 2021, ESM2 2023, ESM3 2025, ProtTrans 2022, ProGen2 2023 |
| ⚪ **Dataset / landscape characterisation** (wet lab, no ML) | Stemmer 1994, Arnold 1998, Sarkisyan 2016, Wu 2016 (GB1), Johnston 2024 (TrpB) |
| 📚 **Review / tutorial** | Romero & Arnold 2009, Yang 2019, Frazier 2018, Snoek 2012, Minding the Gaps 2021 |
| 🏗️ **Structure-based design** (orthogonal paradigm) | AlphaFold2, ProteinMPNN, RFdiffusion |

**GFP dataset usage by paradigm:**
- Generative oracle: DbAS, CbAS, DynaPPO, AdaLead, PEX, GGS, LaMBO — oracle = CNN on full 52k
- Pool-based MLDE: **NONE in current collection** (gap your thesis fills)
- Low-N hybrid: Biswas 2021 — n=24 training, generative MCMC, wet lab validation

**GB1 dataset usage by paradigm:**
- Pool-based MLDE: Wu 2019, Wittmann 2021, ALDE 2025, BOES 2025 — systematic benchmark
- Generative oracle: not commonly used (all variants already measured → no novel sequence generation possible)
- ⚠️ **Note**: GB1's complete combinatorial coverage makes it structurally unsuitable for generative methods — you cannot propose a novel sequence at those 4 sites because it's already in the table

---

# SECTION 1: Per-Paper Summaries (34 papers)

---

### 1. Stemmer 1994 — DNA Shuffling
| Field | Detail |
|---|---|
| **Venue** | Nature |
| **Wet lab?** | ✅ Fully wet lab |
| **Dataset** | No fixed dataset — β-lactamase and other enzymes, random combinatorial libraries |
| **Models/Encodings** | None — purely experimental |
| **Key finding** | DNA shuffling enables recombination of beneficial mutations across homologous sequences; first systematic combinatorial DE method |
| **Why cited** | Historical baseline — defines what computational methods must beat |

---

### 2. Arnold 1998 — Directed Evolution as Engineering Strategy
| Field | Detail |
|---|---|
| **Venue** | Acc. Chem. Res. |
| **Wet lab?** | ✅ Fully wet lab |
| **Dataset** | Various enzymes (subtilisin, p-nitrobenzyl esterase) |
| **Models/Encodings** | None |
| **Key finding** | Articulates DE as a search strategy requiring no mechanistic understanding; mutation + selection is sufficient; establishes the iterative loop framework |
| **Why cited** | Foundational framing: the fitness landscape exists and is navigable |

---

### 3. Romero & Arnold 2009 — Fitness Landscape Theory
| Field | Detail |
|---|---|
| **Venue** | Nat Rev Mol Cell Biol |
| **Wet lab?** | 📚 Review |
| **Dataset** | Surveys multiple landscape studies |
| **Models/Encodings** | N/A |
| **Key finding** | Classifies landscape ruggedness: smooth (greedy hill-climbing works), rugged (local optima trap DE); defines epistasis types (sign, magnitude); explains why DE fails on epistatic landscapes |
| **Why cited** | Conceptual foundation for why ML/BO is needed |

---

### 4. Snoek, Larochelle & Adams 2012 — Practical Bayesian Optimization
| Field | Detail |
|---|---|
| **Venue** | NeurIPS |
| **Wet lab?** | ❌ Purely computational (ML hyperparameter tuning) |
| **Dataset** | ML benchmark hyperparameter spaces |
| **Models/Encodings** | GP with Matérn kernel; Expected Improvement (EI) acquisition |
| **Key finding** | GP + EI outperforms manual tuning and grid search for ML hyperparameters; popularizes BO in ML community; introduces the standard surrogate+acquisition framework |
| **Why cited** | The BO paper all protein work inherits from |

---

### 5. Romero, Krause & Arnold 2013 — First Protein BO
| Field | Detail |
|---|---|
| **Venue** | PNAS |
| **Wet lab?** | ✅ Hybrid (computational recommendations + real wet lab validation) |
| **Dataset** | Cytochrome P450 chimeras (~100 variants, thermostability); not a lookup table — real measurements |
| **Dataset size** | ~100 measured variants |
| **Models/Encodings** | GP with **structure-based contact map kernel** (not sequence Hamming); UCB acquisition |
| **Key finding** | First BO for proteins; GP can model fitness from ~100 variants; structure kernel > sequence kernel; achieved more thermostable P450s than any prior DE campaign |
| **Limitation** | No exhaustive oracle; can't evaluate how close to global optimum |

---

### 6. Wu et al. 2016 — GB1 Fitness Landscape
| Field | Detail |
|---|---|
| **Venue** | eLife |
| **Wet lab?** | ✅ Fully wet lab (landscape characterization) |
| **Dataset** | **GB1** — IgG-binding domain B1 of protein G, 4 positions (V39, D40, G41, V54), all 20⁴ = 160,000 variants |
| **Dataset size** | **149,361** variants after filtering (input count ≥ 10); 10,639 imputed variants excluded |
| **Dataset type** | Near-exhaustive combinatorial; phage display enrichment |
| **Fitness** | Log-enrichment ratio (binding affinity proxy) |
| **Additivity** | **Highly non-additive / epistatic**: additive model explains ~30% variance; 30 local optima; 92% variants below 1% of global max |
| **Models** | None — landscape characterization only |
| **Key finding** | Routes to optimum are often indirect (fitness decreases on the way); most of landscape has near-zero fitness |

---

### 7. Sarkisyan et al. 2016 — GFP Fitness Landscape
| Field | Detail |
|---|---|
| **Venue** | Nature |
| **Wet lab?** | ✅ Fully wet lab (landscape characterization) |
| **Dataset** | **avGFP** — full 239-residue protein, random mutagenesis (1–15 mutations) |
| **Dataset size** | **~52,000 variants** (MaveDB: 54,025 total; ~52,715 with fitness; commonly cited as ~52k) |
| **Dataset type** | Sparse — 1-15 mutations anywhere in full protein; FACS-sorted into 4 fluorescence bins |
| **Fitness** | Mean fluorescence intensity (log scale) |
| **Additivity** | **Moderate**: threshold epistasis — chromophore must fold, so many mutations are neutral until key residues broken |
| **Hole density** | LOW — most variants have some fluorescence; distribution not as skewed as GB1 |
| **Models** | None — landscape characterization only |
| **Key finding** | First full-protein fitness landscape; fluorescence threshold behavior; sparse coverage of a massive sequence space |

---

### 8. Frazier 2018 — BO Tutorial
| Field | Detail |
|---|---|
| **Venue** | arXiv tutorial |
| **Wet lab?** | ❌ Theory/tutorial |
| **Dataset** | N/A |
| **Models/Encodings** | GP theory; EI, UCB, TS, PI acquisition functions; batch BO |
| **Key finding** | Definitive reference for BO; derives acquisition functions; proves convergence; covers batch settings; standard citation for any protein BO paper |

---

### 9. Brookes & Listgarten 2018 — DbAS
| Field | Detail |
|---|---|
| **Venue** | arXiv preprint |
| **Wet lab?** | ❌ In silico |
| **Dataset** | **GFP** (~50k, Sarkisyan); TF binding sequences (short DNA) |
| **Models/Encodings** | VAE (generative model); fixed oracle (separate regression model); data weighting by oracle likelihood |
| **Key finding** | Design by Adaptive Sampling — generative model trains on high-fitness sequences; key problem identified: oracle unreliable far from training data ("adversarial exploitation") |
| **Limitation** | Oracle trust-region problem not fully solved here (solved in follow-up CbAS) |

---

### 10. Yang, Wu & Arnold 2019 — MLDE Review
| Field | Detail |
|---|---|
| **Venue** | Nature Methods |
| **Wet lab?** | 📚 Review |
| **Dataset** | N/A — surveys existing datasets |
| **Models/Encodings** | Survey: supervised regression, generative models, RL |
| **Key finding** | First taxonomy of ML-guided DE: supervised (MLDE), unsupervised/generative, RL; identifies limits of existing approaches: small data, high-dimensional spaces, lack of rigorous DE comparison |

---

### 11. Wu et al. 2019 — MLDE (PNAS)
| Field | Detail |
|---|---|
| **Venue** | PNAS |
| **Paradigm** | 🟢 **Pool-based MLDE** — samples n=96 from GB1 pool; surrogate predicts all remaining; top selected |
| **Wet lab?** | ✅ Hybrid (GB1 simulation + real Si–C bond enzyme) |
| **Dataset** | **GB1** (149,361 filtered variants, lookup table oracle); also real enzyme (Si–C bond formation, ~384 wet-lab measurements) |
| **Dataset size** | GB1: 149,361 oracle; Wet lab: 96–480 variants |
| **Models/Encodings** | One-hot encoding; linear regression, ridge, lasso, SVR, random forest, ensemble |
| **Acquisition** | Top-K greedy selection (no explicit BO) |
| **Key finding** | MLDE workflow: train on random subsample → predict all → test top → repeat; significantly outperforms greedy DE on GB1 for finding global optimum within budget; one-hot + simple ensemble works well |
| **Limitation** | No iterative active learning — one-shot prediction then test; no UCB/exploration |

---

### 12. Brookes, Park & Listgarten 2019 — CbAS (ICML)
| Field | Detail |
|---|---|
| **Venue** | ICML |
| **Wet lab?** | ❌ In silico |
| **Dataset** | **GFP** (~50k, Sarkisyan); TF binding sequences |
| **Models/Encodings** | VAE generative model; importance-weighted retraining; fixed oracle |
| **Key finding** | Conditioning by Adaptive Sampling — importance weighting fixes oracle trust-region problem; CbAS is the standard generative model baseline in protein design benchmarks |

---

### 13. Angermueller et al. 2020 — DyNA-PPO (ICLR)
| Field | Detail |
|---|---|
| **Venue** | ICLR |
| **Paradigm** | 🔴 **Generative oracle search** — RL policy trained against pre-trained oracle; proposes novel sequences outside dataset |
| **Wet lab?** | ❌ In silico |
| **Dataset** | **avGFP** (Sarkisyan oracle); DNA regulatory sequences; TF binding; also PyRosetta tasks |
| **GFP usage** | Oracle = CNN trained on full 52k Sarkisyan data; RL agent generates novel sequences and queries oracle |
| **Models/Encodings** | Policy network (PPO); ensemble of surrogate models as environment simulator; uncertainty = ensemble variance |
| **Key finding** | RL with model-based environment (DyNA = dynamic surrogate); ensemble provides both mean prediction and uncertainty for RL agent to explore; competitive with AdaLead |
| **Relevance to thesis** | Uses GFP but in generative oracle paradigm — does not address encoding/surrogate quality under limited labels |

---

### 14. Hie, Bryson & Berger 2020 — Uncertainty, Cell Systems
| Field | Detail |
|---|---|
| **Venue** | Cell Systems |
| **Wet lab?** | ✅ Hybrid (computational selection + wet-lab validation) |
| **Dataset** | Drug discovery (kinase inhibitors, TB growth inhibition); protein engineering (limited); NOT a fixed protein fitness landscape |
| **Dataset size** | ~72–200 compounds per campaign |
| **Models/Encodings** | GP models with pre-trained feature representations; uncertainty = GP posterior variance |
| **Key finding** | Calibrated uncertainty is critical — GP uncertainty should correlate with actual error; low-N learning with strong priors effective; introduced the concept of uncertainty calibration for bio applications |
| **Note** | Published directly in Cell Systems — no bioRxiv preprint |

---

### 15. Sinai et al. 2020 — AdaLead
| Field | Detail |
|---|---|
| **Venue** | arXiv |
| **Paradigm** | 🔴 **Generative oracle search** — greedy hill-climbing proposes novel sequences; oracle = pre-trained model on full dataset |
| **Wet lab?** | ❌ In silico (FLEXS benchmarking framework) |
| **Dataset** | **avGFP** (Sarkisyan oracle); TF binding sequences; RNA landscapes; also PyRosetta tasks |
| **GFP usage** | Oracle = CNN/regression model trained on full 52k Sarkisyan; AdaLead generates novel sequences via mutation+recombination and queries oracle |
| **Models/Encodings** | Adaptive greedy search; surrogate model for filtering (any); recombination + mutation; κ threshold controls exploit/explore |
| **Key finding** | Beats DyNA-PPO, CbAS, DbAS, BO on most benchmarks despite simplicity; recombination is crucial mechanism; introduced FLEXS sandbox framework |
| **Critical warning** | BO does not always justify its complexity; simple greedy is a strong baseline |
| **Relevance to thesis** | Uses GFP but generative oracle paradigm — not pool-based selection under limited labels |

---

### 16. Biswas et al. 2021 — Low-N Protein Engineering
| Field | Detail |
|---|---|
| **Venue** | Nature Methods |
| **Paradigm** | 🟡 **Hybrid** — low-N training (n=24–96) + generative MCMC search over novel sequences + real wet-lab validation. NOT pure oracle search (limited labels) but NOT pool-based selection either |
| **Wet lab?** | ✅ Hybrid — trains surrogate on n=24–96 measured variants; generates novel sequences; validates top candidates in wet lab |
| **Dataset** | **avGFP** (n=24–96 training variants from Sarkisyan; generates sequences OUTSIDE the 52k pool; retrospective eval on Sarkisyan); TEM-1 β-lactamase (wet lab); others retrospective |
| **Training set** | As few as **24 randomly selected variants** |
| **Models/Encodings** | **eUniRep** (NOT ESM1b — this is UniRep with evolutionary fine-tuning, precursor to ESM); linear regression (ridge) on top; one-hot (Full AA) as baseline |
| **Key finding** | eUniRep + ridge on n=24 variants outperforms one-hot + stronger models requiring more data; demonstrates low-N feasibility; generated avGFP variants rivalling sfGFP brightness; epistatically non-trivial designs (25% novel mutations) |
| **Critical encoding note** | Compares eUniRep vs one-hot on GFP in limited-label regime — most similar to your thesis question, BUT generative (novel sequences) not pool-based (within Sarkisyan pool) |
| **Relevance to thesis** | ⭐ Closest prior work using GFP with limited labels + encoding comparison; key distinction: they propose novel sequences while this thesis does pool-based prediction |

---

### 17. Rives et al. 2021 — ESM1b (PNAS)
| Field | Detail |
|---|---|
| **Venue** | PNAS |
| **Wet lab?** | ❌ In silico (language model training + evaluation) |
| **Dataset** | Pre-training: UniRef50 (250M sequences); evaluation: structure prediction, contact maps, mutational effect benchmarks |
| **Models/Encodings** | ESM1b — 650M parameter transformer; masked language modeling (MLM); 33-layer; 1280-dim per-residue embeddings |
| **Key finding** | Structural and functional information emerges from sequence co-evolution at scale without structural supervision; attention maps ≈ contact maps; per-position log-likelihoods predict mutational effects (zero-shot fitness) |

---

### 18. Wittmann, Yue & Arnold 2021 — ftMLDE (Cell Systems)
| Field | Detail |
|---|---|
| **Venue** | Cell Systems |
| **Paradigm** | 🟢 **Pool-based MLDE** — n=96–384 from GB1 pool; systematic encoding comparison; one-shot (not iterative) |
| **Wet lab?** | ❌ In silico simulation |
| **Dataset** | **GB1 exclusively** — 149,361 filtered variants; explicitly excludes 10,639 imputed variants |
| **Dataset size** | 149,361 oracle; training set: 96–384 variants |
| **Models/Encodings** | One-hot; physicochemical (AAIndex, Georgiev, 4–19 dim/residue); **ESM1b** (1280-dim/residue); MSA Transformer; EVmutation; ridge regression; XGBoost; ensemble |
| **Acquisition** | Top-K greedy (one-shot MLDE, not iterative) |
| **Key finding** | **The hole problem**: 92% of GB1 variants near-zero fitness → random training biases model; **ftMLDE** uses zero-shot predictors (ESM1b log-likelihood, EVmutation) to enrich training set; up to 81× more frequent global optimum recovery vs DE; **no encoding dominates across all conditions** |
| **Critical finding** | One-hot competitive with ESM1b for non-deep models; ESM1b helps when paired with appropriate models |

---

### 19. Jumper et al. 2021 — AlphaFold2 (Nature)
| Field | Detail |
|---|---|
| **Venue** | Nature |
| **Wet lab?** | ❌ Computational (structure prediction) |
| **Dataset** | Training: PDB (~170k structures) + MSA from genetic databases; Evaluation: CASP14 targets |
| **Models/Encodings** | Transformer-based; pairwise residue representation (Evoformer); iterative structure module; Invariant Point Attention (IPA) |
| **Key finding** | Solved protein structure prediction at near-experimental accuracy; indirect impact on protein engineering — enables structure-based design for proteins without experimental structures |
| **Relevance to BO** | Not a fitness predictor; not directly used in BO pipelines; provides starting points for design |

---

### 20. Stanton et al. 2022 — LaMBO (ICML)
| Field | Detail |
|---|---|
| **Venue** | ICML |
| **Wet lab?** | ✅ Hybrid (in silico + wet-lab validation on red fluorescent proteins) |
| **Dataset** | **GFP/RFP** fluorescent proteins (in silico + wet lab); drug resistance tasks |
| **Dataset size** | In silico: Sarkisyan-scale; Wet lab: small RFP panel |
| **Models/Encodings** | Denoising autoencoder (DAE) learns continuous latent space; GP with Matérn kernel on latent space; gradient-based acquisition optimization in latent space |
| **Key finding** | BO in language model latent space outperforms BO in raw sequence space; gradient-based acquisition enables efficient search without enumeration; multi-objective simultaneous optimization (stability + fluorescence); first to demonstrate BO in pre-trained latent space concept |

---

### 21. Ren et al. 2022 — PEX (ICML)
| Field | Detail |
|---|---|
| **Venue** | ICML |
| **Paradigm** | 🔴 **Generative oracle search** — proposes novel sequences via constrained surrogate; oracle = TAPE model trained on full dataset; queries novel sequences outside pool |
| **Wet lab?** | ❌ In silico |
| **Dataset** | **avGFP** (~52,000 Sarkisyan; oracle = TAPE); AAV; TEM-1; 5 other DMS datasets from ProteinGym |
| **GFP usage** | Oracle = TAPE regression model trained on full Sarkisyan dataset; PEX generates new sequences and queries oracle |
| **Models/Encodings** | MuFacNet (factorized mutation neural network); TAPE features; Hamming distance penalty (proximal constraint); EI, UCB baselines |
| **Key finding** | Proximity constraint prevents surrogate exploitation by out-of-distribution sequences; proximal frontier concept; outperforms EI-BO and CbAS on GFP |
| **Relevance to thesis** | Uses GFP extensively but generative oracle paradigm — does not evaluate encoding quality under limited labels |

---

### 22. Notin et al. 2022 — Tranception + ProteinGym (ICML)
| Field | Detail |
|---|---|
| **Venue** | ICML |
| **Wet lab?** | ❌ In silico evaluation |
| **Dataset** | **ProteinGym**: ~250 DMS assays covering diverse proteins (~2.5M variant measurements total); mostly single-site mutations |
| **Dataset type** | Sparse single-site; largely additive; broad protein diversity |
| **Models/Encodings** | Tranception — autoregressive transformer with MSA retrieval at inference; zero-shot fitness prediction (no training labels needed) |
| **Key finding** | ProteinGym is the standard zero-shot fitness benchmark; Tranception with retrieval beats ESM1b, EVmutation on ProteinGym; zero-shot performance not predictive of supervised optimization performance |
| **Note for BO** | ProteinGym is NOT suitable for BO benchmarking — single-site landscape is trivially optimizable (just pick best single mutant) |

---

### 23. Elnaggar et al. 2022 — ProtTrans (TPAMI)
| Field | Detail |
|---|---|
| **Venue** | IEEE TPAMI |
| **Wet lab?** | ❌ In silico |
| **Dataset** | Pre-training: **BFD** (2.1B sequences) + UniRef; evaluation: secondary structure, contact prediction, localization |
| **Models/Encodings** | ProtBERT (BERT on proteins, 420M params); ProtT5 (T5 architecture, 3B params); XLNet variant; 1024-dim per-residue embeddings |
| **Key finding** | NLP architectures transfer directly to protein sequences; ProtBERT-BFD competitive with ESM1b on structure tasks; ProtT5 often best for secondary structure; largely superseded by ESM2 for engineering applications |

---

### 24. Dauparas et al. 2022 — ProteinMPNN (Science)
| Field | Detail |
|---|---|
| **Venue** | Science |
| **Wet lab?** | ✅ Hybrid (computational + wet-lab validation of designed proteins) |
| **Dataset** | Training: PDB backbone structures (~170k); evaluation: designed proteins, hit rates |
| **Models/Encodings** | Graph neural network on backbone coordinates; message passing over residue–residue edges; trained to maximize sequence recovery |
| **Key finding** | Takes backbone coordinates → designs amino acid sequences; dramatically outperforms prior structure-based design; part of the AF2 + ProteinMPNN + RFdiffusion pipeline |
| **Relevance to BO** | Structure-conditioned sequence prior — alternative paradigm to fitness landscape BO |

---

### 25. Watson et al. 2023 — RFdiffusion (Nature)
| Field | Detail |
|---|---|
| **Venue** | Nature |
| **Wet lab?** | ✅ Hybrid (computational design + extensive wet-lab validation) |
| **Dataset** | Training: PDB (~170k) + AF2-predicted structures; Evaluation: designed binders, enzymes, novel folds |
| **Models/Encodings** | Denoising diffusion on protein backbone coordinates; RoseTTAFold as the diffusion network; conditional prompting by functional motifs |
| **Key finding** | De novo protein design with experimental hit rates up to 20% for binders (vs <<1% for prior methods); enables enzyme active site design, novel folds, protein-protein interfaces |
| **Relevance to BO** | Fully orthogonal paradigm — design from structure, not fitness landscape navigation |

---

### 26. Madani et al. 2023 — ProGen2 (Nature Biotechnology)
| Field | Detail |
|---|---|
| **Venue** | Nature Biotechnology |
| **Wet lab?** | ✅ Hybrid (generated sequences + wet-lab validation) |
| **Dataset** | Pre-training: ~280M protein sequences (UniRef + other); Evaluation: lysozyme variants tested in lab |
| **Models/Encodings** | Autoregressive LM (GPT-style); up to 6.4B parameters; conditioned on protein family tokens |
| **Key finding** | Generates functional lysozyme sequences with only 31.4% identity to known proteins; first large-scale generative LM with wet-lab validation across diverse families |

---

### 27. Lin et al. 2023 — ESM2 + ESMFold (Science)
| Field | Detail |
|---|---|
| **Venue** | Science |
| **Wet lab?** | ❌ In silico |
| **Dataset** | Pre-training: UniRef (250M sequences); Evaluation: CASP15, structure prediction benchmarks, DMS tasks |
| **Models/Encodings** | ESM2 — up to **15B parameters** (also 650M, 3B, 8B, 150M, 8M sizes); 33/36/48-layer transformer; 1280-dim per-residue; **ESMFold** adds structural head to ESM2 trunk (no MSA needed) |
| **Key finding** | ESM2 + ESMFold enables atomic-level structure prediction in single forward pass without MSA; ESM2 embeddings are the standard for supervised protein engineering as of 2024; **on combinatorial landscapes (GB1, TrpB), ESM2 does NOT consistently beat one-hot** with small training sets |

---

### 28. Greenberg, Amini & Yang 2023 — UQ Benchmarking
| Field | Detail |
|---|---|
| **Venue** | bioRxiv |
| **⚠️ FILE WARNING** | **The file `Greenberg2023_benchmarking_UQ_real.pdf` on disk is WRONG** — it is an unrelated antibody-drug conjugate cancer paper (Panikar et al., Washington University). The entry below is based on the actual Greenberg 2023 paper description, not the file. Need to re-download. |
| **Wet lab?** | ❌ In silico |
| **Dataset** | **GB1** (149,361 filtered variants) as primary |
| **Models/Encodings** | GP; MC-Dropout; ensemble; conformal prediction; various encodings |
| **Key finding** | Well-calibrated uncertainty matters for active learning but no single UQ method dominates; ensemble-based UQ is consistently competitive; calibration on full landscape ≠ calibration at acquisition frontier; ALDE's UQ analysis builds directly on this work |

---

### 29. Kirjner et al. 2023 — GGS: Smoothed Fitness Landscapes (ICLR 2024)
| Field | Detail |
|---|---|
| **Venue** | ICLR 2024 |
| **Paradigm** | 🔴 **Generative oracle search** — Gibbs sampling proposes novel sequences; oracle (CNN) trained on full dataset is ground truth |
| **Wet lab?** | ❌ In silico |
| **Dataset** | **avGFP** (Sarkisyan oracle, CNN trained on full 52k); AAV capsid proteins; also GFP "Hard" variant (starting from lower-fitness sequences) |
| **GFP usage** | Oracle = CNN trained on full Sarkisyan data; GGS generates novel sequences via Gibbs sampling and queries oracle; does NOT operate under limited labelling budget |
| **Models/Encodings** | Graph-based smoothing of oracle fitness scores; Gibbs sampling in sequence space; CNN oracle as ground truth |
| **Key finding** | Real protein fitness landscapes are **rugged** (not smooth); smoothing the oracle's predictions via graph-based averaging of neighbours reduces ruggedness, enabling more effective Gibbs sampling; consistently outperforms CbAS, AdaLead, BO on GFP and AAV |
| **Correction note** | Description was previously wrong: GGS smoothes the landscape TO HELP search (landscape is too rugged, not too smooth) |
| **Relevance to thesis** | Uses GFP but generative oracle paradigm; does not address encoding quality under limited labels |

---

### 30. Johnston et al. 2024 — TrpB Landscape (PNAS)
| Field | Detail |
|---|---|
| **Venue** | PNAS |
| **Wet lab?** | ✅ Fully wet lab (landscape characterization) |
| **Dataset** | **TrpB** — tryptophan synthase β-subunit, 4 positions (V183, F184, S228, V227), all 20⁴ = 160,000 variants |
| **Dataset size** | ~160,000 measured variants |
| **Fitness** | Growth-coupling assay — enzymatic activity drives cell growth |
| **Additivity** | **Highly epistatic** — rugged, many indirect paths, similar structure to GB1 |
| **Models** | Landscape characterization only |
| **Key finding** | Second near-exhaustive 4-site combinatorial landscape; provides independent ground-truth oracle for BO simulation studies; used by ALDE for validation alongside GB1 |

---

### 31. Yang et al. 2024 — ALDE
| Field | Detail |
|---|---|
| **Venue** | bioRxiv (published Nature Communications) |
| **Paradigm** | 🟢 **Pool-based MLDE** (simulation on GB1/TrpB) + 🟡 **Hybrid wet-lab** (ParPgb enzyme campaign) |
| **Wet lab?** | ✅ Hybrid — full wet-lab campaign + computational simulations |
| **Dataset (simulation)** | **GB1** (149,361 filtered) + **TrpB** (~160k) |
| **Dataset (wet lab)** | ParPgb enzyme — 5-site combinatorial space (20⁵ = 3.2M variants); 384 initial random variants + 2 × 90 directed variants |
| **Models/Encodings** | AAIndex (4-dim/res); Georgiev (19-dim/res); One-hot (20-dim/res); **ESM2-650M** (1280-dim/res at **5 targeted sites only** — NOT mean pooled); see Table 1 |
| **Surrogates** | Boosting ensemble; **GP**; **DNN ensemble** (5 MLPs); **Deep Kernel Learning** (DKL = GP on DNN last layer) |
| **Acquisition** | Greedy; UCB; **Thompson Sampling (TS)** |
| **Simulation setup** | 96 initial random samples; 4 rounds × 96 samples; 70 random seeds each |
| **Key findings** | (1) DNN ensemble most accurate + best calibrated overall; (2) **Frequentist > Bayesian** in practice; (3) One-hot + DNN ensemble best combination; (4) **ESM2 provides no benefit over one-hot** on combinatorial landscapes; (5) ESM2 incompatible with GP (too high-dimensional); (6) Thompson Sampling best acquisition function; (7) Wet lab: 480 variants over 3 rounds → 12% → 93% cyclopropanation yield, 14:1 selectivity; (8) Final wet-lab variant (MPFDY) contains none of the single best mutations from SSM screen — pure epistasis |
| **Critical note** | ESM2 was used **site-specifically** (5 sites × 1280 = 6400-dim vector), NOT as global mean pool |

---

### 32. Soldát & Kléma 2025 — BOES: BO in Embedding Space
| Field | Detail |
|---|---|
| **Venue** | arXiv:2509.04998 (submitted 2025) |
| **Paradigm** | 🟢 **Pool-based MLDE** — BO on fixed pool of measured variants, ground truth = experimental measurements |
| **Wet lab?** | ❌ In silico simulation |
| **Dataset** | **GB1** (149,361 filtered variants); **PhoQ** (protein kinase fitness dataset) |
| **Models/Encodings** | **ESM2 mean-pooled embeddings** as input space; **GP** surrogate with RBF kernel in embedding space; UCB acquisition; compared against one-hot BO baselines |
| **Acquisition** | UCB; also DE (directed evolution) baselines |
| **Key finding** | ESM2 embedding space is better structured for GP-BO than raw one-hot space on GB1; BO in PLM latent space outperforms standard MLDE methods on GB1; corroborates that PLM embeddings can add value when paired with GP (not just DNN) in certain regimes |
| **Key contrast with your work** | Tests ESM2 mean pool + GP on GB1 only; does NOT test on GFP; does NOT compare multiple encodings systematically; does NOT report Precision@K |
| **Relevance to thesis** | ⭐ Very relevant — closest prior work to your BOES question on GB1; one of the papers that makes the GFP gap more glaring |

---

### 33. Jiang et al. 2024 — EVOLVEpro (Science)
| Field | Detail |
|---|---|
| **Venue** | Science |
| **Wet lab?** | ✅ Fully wet lab experimental campaigns |
| **Dataset (benchmarking)** | 12 DMS datasets for grid search: MAPK1/ERK2, HIV envelope, Influenza hemagglutinin, RNA structural determinants, TEM-1, TP53, GPCR, codon usage, spike proteins, kinases, lactases — all from ProteinGym or DMS literature; **NOT GB1 or GFP** |
| **Dataset (wet lab)** | 5 proteins × multiple rounds × 11–16 variants/round: REGN10987 antibody, PsaCas12f nuclease, PE2 prime editor (M-MLV RT domain), Bxb1 integrase, T7 RNA polymerase |
| **Variants measured** | ~11–16 per round × 4–8 rounds per protein; total ~50–130 per protein |
| **Models/Encodings** | **ESM2-15B** (best); residue-pooled **mean** embeddings; grid-searched against: ESM2-650M, ESM1b-650M, UniRep, ProtT5, ProteinBERT, Ankh, one-hot, integer |
| **Surrogate** | **Random forest regressor** (best in grid search over linear, ridge, SVM, neural net, RF) |
| **Acquisition** | **Top-N selection** (greedy, no uncertainty) |
| **Key findings** | (1) ESM2-15B beats all other representations; (2) Random forest + ESM2-15B mean pool is the best combination for low-N active learning on diverse protein families; (3) PLM fitness (pMMS) negatively correlated with actual activity (−0.22 to −0.13 across proteins); (4) 5 rounds × 16 variants = equivalent to pre-training on 500 labeled variants; (5) Multi-objective BO achievable (T7 RNAP: 57× more translation + 515× less immunogenicity); (6) In vivo validation for two proteins (PsaCas12f PCSK9 editing in mice, T7 mRNA in LNPs) |
| **Key contrast vs ALDE** | EVOLVEpro: diverse real therapeutic proteins, individual assays, 16/round, random forest; ALDE: combinatorial benchmark landscapes, ensemble uncertainty, 96/round, DNN ensemble |
| **Critical finding** | PLM fitness ≠ protein activity — the regression layer is ESSENTIAL |

---

### 34. Hayes et al. 2025 — ESM3 (Science)
| Field | Detail |
|---|---|
| **Venue** | Science |
| **Wet lab?** | ✅ Hybrid (generated esmGFP + wet-lab validation of fluorescence) |
| **Dataset** | Pre-training: 2.78B proteins + 771B tokens (sequence + structure + function); Evaluation: structure prediction benchmarks, esmGFP fluorescence |
| **Models/Encodings** | ESM3 — 98B parameters; multimodal: sequence tokens, structure tokens, function tokens, trained jointly; generative (not just discriminative) |
| **Key finding** | First multimodal generative PLM; esmGFP with only 58% identity to closest known FP; equivalent to 500M years of simulated evolution; demonstrates that multimodal generation outperforms sequence-only models for truly novel protein design |
| **Practical accessibility** | Full 98B model: API only (Forge); Open weights: esm3-sm-open-v1 (1.4B, non-commercial Cambrian license); ESM C 300M/600M: practical drop-in for ESM2; NOT as accessible as ESM2 (MIT license) |
| **Relevance to BO** | As a feature extractor for fitness BO: unclear benefit over ESM2; more naturally used as generative oracle; ESM C is the practical ESM2 replacement |

---
---

# SECTION 2: Per-Dataset Descriptions

---

## Datasets With Papers in This Collection

### GB1 (Wu et al. 2016)
| Property | Value |
|---|---|
| **Protein** | B1 domain of protein G (IgG-binding, 56 residues) |
| **Fitness measure** | Phage display enrichment ratio (binding affinity proxy) |
| **Positions mutated** | 4 fixed sites: V39, D40, G41, V54 |
| **Total variants** | 20⁴ = 160,000 |
| **Filtered dataset** | **149,361** (input read count ≥ 10; used in all ML benchmarks) |
| **Excluded** | 10,639 imputed variants — explicitly excluded by Wittmann 2021, ALDE |
| **Coverage** | Near-exhaustive (~93% of all possible 4-site combinations) |
| **Dataset type** | Dense combinatorial k-site |
| **Epistasis** | **Highly epistatic** — additive model explains ~30% variance |
| **Hole density** | **Very high** — 92% of variants below 1% of global max fitness |
| **Fitness distribution** | Extremely skewed: most variants ~0, long tail |
| **Local optima** | ~30 local optima |
| **Global max** | VDGV → FWLM mutation combination |
| **Used by** | Wu 2019, Wittmann 2021, Biswas 2021, Greenberg 2023, ALDE 2024, BO-in-embedding 2024 |
| **Field status** | THE standard benchmark for MLDE/BO simulation studies |

---

### TrpB (Johnston et al. 2024)
| Property | Value |
|---|---|
| **Protein** | Tryptophan synthase β-subunit, *Pyrococcus furiosus* (391 residues) |
| **Fitness measure** | Growth coupling — enzymatic tryptophan production drives cell growth |
| **Positions mutated** | 4 fixed sites: V183, F184, S228, V227 |
| **Total variants** | 20⁴ = 160,000 |
| **Measured variants** | ~160,000 (near-exhaustive) |
| **Dataset type** | Dense combinatorial k-site |
| **Epistasis** | **Highly epistatic** — similar ruggedness to GB1 |
| **Hole density** | High (combinatorial landscape) |
| **Used by** | Johnston 2024, ALDE 2024 |
| **Field status** | Second standard benchmark; more biologically complex than GB1 (enzymatic activity vs binding) |
| **Note** | Correctly attributed to Johnston et al. 2024, NOT Wittmann 2021 — common error in older lit |

---

### avGFP (Sarkisyan et al. 2016)
| Property | Value |
|---|---|
| **Protein** | Aequorea victoria green fluorescent protein (239 residues) |
| **Fitness measure** | Fluorescence intensity (FACS-sorted into 4 bins, continuous brightness score) |
| **Positions mutated** | Random across full 239-residue sequence (1–15 mutations) |
| **Total variants** | ~52,715 (commonly cited as ~52k) |
| **Coverage** | Sparse — tiny fraction of full sequence space |
| **Dataset type** | Sparse full-protein (not k-site combinatorial) |
| **Epistasis** | **Moderate** — threshold epistasis (chromophore must form); partially additive |
| **Hole density** | **Low** — most variants have measurable fluorescence; less skewed than GB1 |
| **Fitness distribution** | More bell-shaped than GB1; minimum brightness ~1.28 |
| **Used by (generative oracle)** | DbAS 2018, CbAS 2019, DynaPPO 2020, AdaLead 2020, PEX 2022, LaMBO 2022, GGS/Kirjner 2023 — oracle = CNN on full 52k |
| **Used by (low-N hybrid)** | Biswas 2021 — n=24 training, MCMC generative search, wet lab validation |
| **Used by (zero-shot)** | ProteinGym (Notin 2022); FLIP (Dallago 2021) |
| **Used by (pool-based MLDE)** | **NONE in existing literature** — this is the gap this thesis fills |
| **Key difference from GB1** | Full-protein sparse vs 4-site exhaustive; low vs high hole density; moderate vs high epistasis |
| **Why GB1 ≠ generative** | GB1 covers ALL variants at 4 sites — you cannot propose a "novel" sequence at those positions; it already exists in the table. This is why generative papers always use GFP (open space) not GB1 (closed space). |

---

### ParPgb — ALDE Wet Lab Dataset
| Property | Value |
|---|---|
| **Protein** | Protoglobin from *Pyrobaculum arsenaticum* (ParPgb), ~200 residues |
| **Fitness measure** | *cis–trans* yield difference in cyclopropanation reaction (gas chromatography) |
| **Positions mutated** | 5 active-site residues: W56, Y57, L59, Q60, F89 (WYLQF) |
| **Total design space** | 20⁵ = 3.2M variants |
| **Measured (ALDE)** | ~384 initial (random) + 90 × 2 = 564 directed; total ~948 wet-lab measurements |
| **Dataset type** | Sparse subset of combinatorial space — NOT exhaustive |
| **Epistasis** | **Highly epistatic** — single-site screen found no good variants; optimal MPFDY undetectable from SSM |
| **Starting point** | ParLQ (W59L Y60Q variant) — only moderate cyclopropanation yield (~40%), 3:1 trans preference |
| **Used by** | ALDE 2024 (wet lab only) |
| **Note** | No lookup table oracle exists; real individual biochemical assays required |

---

### ProteinGym (~250 DMS assays)
| Property | Value |
|---|---|
| **What it is** | Benchmark collection of deep mutational scanning datasets |
| **Version** | v1.0 (~250 assays); v1.3 (217 assays, ~2.7M variants) |
| **Proteins covered** | Diverse: antibodies, enzymes, viral proteins, transcription factors, structural proteins |
| **Mutation type** | Predominantly **single-site** substitutions (some double mutants) |
| **Total variants** | ~2.7M+ across all assays |
| **Fitness measures** | Activity, binding affinity, stability, fitness, growth — varies per assay |
| **Additivity** | **Mostly additive** — single-site, so by definition no multi-site epistasis measured |
| **Used by** | Tranception/ProteinGym 2022, EVOLVEpro (12 datasets from here), PEX (7 datasets from here) |
| **NOT suitable for** | BO benchmarking — trivial to optimize (just pick best single mutant) |
| **Suitable for** | Zero-shot fitness prediction benchmarking |
| **Source** | [proteingym.org](https://proteingym.org) |

---

### FLIP — Fitness Landscape Inference for Proteins
| Property | Value |
|---|---|
| **What it is** | Benchmark with structured train/test splits for fitness landscape problems |
| **Proteins** | GB1, AAV, GFP, SCR mutations, TEM-1, AMIE, others |
| **Key feature** | Provides biologically meaningful train/test splits (e.g., train on 1-2 mutations, test on 3+) |
| **Used by** | Not explicitly in this collection; referenced by several papers |
| **FLIP2 (2026)** | Extends with 7 new datasets: alpha amylases, TrpB sub-landscapes, PDZ domains, protein core stability |
| **Source** | [flip.protein.properties](https://flip.protein.properties) |

---

### P450 Chimeras (Romero 2013)
| Property | Value |
|---|---|
| **Protein** | Cytochrome P450 chimeras (engineered from 3 parent P450s) |
| **Fitness measure** | Thermostability (T50) |
| **Variants** | ~100 measured (not exhaustive) |
| **Dataset type** | Small, non-exhaustive; real wet-lab measurements |
| **Used by** | Romero 2013 |
| **Note** | No lookup-table oracle; GP trained directly on wet-lab data |

---

### ML Hyperparameter Spaces (Snoek 2012)
| Property | Value |
|---|---|
| **Domain** | ML algorithm hyperparameter optimization (not proteins) |
| **Used by** | Snoek 2012 |
| **Relevance** | Established GP+EI BO framework later applied to proteins |

---

### Red Fluorescent Proteins (LaMBO 2022)
| Property | Value |
|---|---|
| **Protein** | Red fluorescent proteins (mCherry variants, diverse) |
| **Fitness measure** | Fluorescence intensity + stability |
| **Used by** | LaMBO 2022 (wet-lab component) |
| **Dataset type** | Small panels; multi-objective optimization target |

---

### Other Important Datasets NOT in This Collection

| Dataset | Protein | Size | Key papers | Source |
|---|---|---|---|---|
| **TEM-1 β-lactamase** | Antibiotic resistance enzyme | ~5k variants (Stiffler 2015) | PEX 2022, Kirjner 2023 | Various |
| **AAV capsid** | Adeno-associated virus, packaging efficiency | ~280k variants | PEX 2022 | FLIP |
| **DHFR** | Dihydrofolate reductase | ~125k variants | Multiple | ProteinGym |
| **P53** | Tumor suppressor (cancer mutations) | ~9k variants | Multiple | ProteinGym |
| **Ubiquitin** | Cellular signaling | ~1k variants | Multiple | FLIP |
| **HIV Env** | HIV envelope protein | ~1k variants | EVOLVEpro benchmark | ProteinGym |
| **Influenza HA** | Hemagglutinin, viral escape | ~50k variants | EVOLVEpro benchmark | ProteinGym |
| **BLAT (TEM variant)** | Beta-lactamase | Multiple | Biswas 2021 | Various |
| **Kinase panels** | Various kinases | EVOLVEpro benchmark | — | ProteinGym |
| **REGN10987 antibody** | COVID-19 antibody (heavy chain) | ~16k single mutants | EVOLVEpro wet lab | — |
| **PsaCas12f** | Miniature CRISPR nuclease | ~4 rounds × 12 = 48 wet lab | EVOLVEpro wet lab | — |
| **T7 RNAP** | RNA polymerase for mRNA manufacturing | ~4 rounds × 16 = 64 wet lab | EVOLVEpro wet lab | — |
| **M-MLV RT (PE2)** | Prime editor reverse transcriptase | ~4 rounds × 16 = 64 wet lab | EVOLVEpro wet lab | — |
| **Bxb1 integrase** | Serine recombinase | ~8 rounds × 11 = 88 wet lab | EVOLVEpro wet lab | — |

---

## Dataset Cross-Reference: Which Papers Use Which Datasets

| Dataset | Papers |
|---|---|
| **GB1** | Wu 2019, Wittmann 2021, Biswas 2021, Greenberg 2023, ALDE 2024, BO-in-embedding 2024 |
| **TrpB** | Johnston 2024, ALDE 2024 |
| **avGFP** | DbAS 2018, CbAS 2019, LaMBO 2022, PEX 2022, Biswas 2021, Kirjner 2023, EVOLVEpro (benchmark) |
| **ParPgb (ALDE wet lab)** | ALDE 2024 only |
| **ProteinGym (any subset)** | Tranception 2022, EVOLVEpro 2024, PEX 2022 |
| **FLIP** | PEX 2022 (partial), Kirjner 2023 (partial) |
| **TF binding (DNA)** | AdaLead 2020, DbAS 2018, CbAS 2019, DyNA-PPO 2020 |
| **RNA landscapes** | AdaLead 2020, DyNA-PPO 2020 |
| **P450 chimeras** | Romero 2013 |
| **PDB structures** | AlphaFold2 2021, ProteinMPNN 2022, RFdiffusion 2023 |
| **UniRef/BFD (PLM training)** | ESM1b 2021, ESM2 2023, ProtTrans 2022, ESM3 2025, ProGen2 2023 |

---
---

# SECTION 3: Per-Model/Encoding Pivot

*"Who tested what, on which dataset, with what result?"*

---

## 3.1 Encodings

### One-hot / Amino acid identity
| Paper | Dataset | Model paired with | Result vs. other encodings |
|---|---|---|---|
| Wu 2019 | GB1 (149k) | Ensemble regression | ✅ Best or competitive at small N |
| Wittmann 2021 | GB1 (149k) | Ridge, XGBoost, ensemble | ✅ Competitive with LLM embeddings for non-deep models |
| ALDE 2024 | GB1 + TrpB | DNN ensemble, GP, Boosting, DKL | ✅ **Best combination = one-hot + DNN ensemble** |
| ALDE 2024 | GB1 + TrpB | GP | ⚠️ Limited by dimensionality; GP works best with low-dim |
| BO-in-embedding 2024 | GB1 | GP | ✅ Competitive with ESM2 mean pool |
| EVOLVEpro 2024 | 12 DMS datasets | Random forest | ❌ Worse than ESM2-15B on diverse protein families |

**Verdict:** One-hot wins on combinatorial k-site landscapes (GB1, TrpB) paired with non-deep models. ESM2 mean pool wins on diverse full-protein tasks paired with large models.

---

### AAIndex (4 physicochemical descriptors/residue)
| Paper | Dataset | Model | Result |
|---|---|---|---|
| Wittmann 2021 | GB1 | Ridge, ensemble | ⚠️ Worse than one-hot for most conditions |
| ALDE 2024 | GB1 + TrpB | GP, Boosting | ✅ Best pairing for GP — low-dim encodings suit GP |
| ALDE 2024 | GB1 + TrpB | DNN | ❌ Deep models benefit more from high-dim |

---

### Georgiev (19 physicochemical descriptors/residue)
| Paper | Dataset | Model | Result |
|---|---|---|---|
| Wittmann 2021 | GB1 | Ensemble | ✅ Better than one-hot for some conditions |
| ALDE 2024 | GB1 + TrpB | GP, Boosting | ✅ Best low-dim encoding; better than AAIndex |
| ALDE 2024 | GB1 + TrpB | DNN | ⚠️ Less benefit vs one-hot |

---

### ESM1b (650M params, 1280-dim/residue)
| Paper | Dataset | Pooling | Model | Result |
|---|---|---|---|---|
| Biswas 2021 | GB1, GFP, others | Mean pool | Linear (ridge) | ✅ Outperforms one-hot at small N (<100) |
| Wittmann 2021 | GB1 | Mean pool | Ridge, ensemble | ⚠️ No consistent advantage over one-hot |
| Rives 2021 | Various | Per-residue | Structure/function tasks | ✅ Excellent zero-shot; contact map prediction |

---

### ESM2 (various sizes; 650M most common)
| Paper | Dataset | Pooling strategy | Model | Result |
|---|---|---|---|---|
| ALDE 2024 | GB1 + TrpB | **Site-specific** (5 sites × 1280 = 6400-dim) | DNN, DKL | ⚠️ No benefit over one-hot; DKL best for ESM2 |
| ALDE 2024 | GB1 + TrpB | Site-specific | GP | ❌ Too high-dimensional for GP |
| BO-in-embedding 2024 | GB1 | **Mean pool** | GP | ⚠️ Competitive but not better than one-hot |
| EVOLVEpro 2024 | 12 DMS (diverse) | **Mean pool** | Random forest | ✅ Best encoding across 12 diverse datasets |
| Lin 2023 | Various | Per-residue | Structure tasks | ✅ Best PLM for structure prediction |
| **Your work (GFP)** | GFP (52k) | **Mean pool** (320-dim) | Random forest | ⚠️ R²=0.469 ≈ mutation features (0.466); crossover at N~500 |

**Critical nuance:**
- ESM2 **site-specific** (targeted positions) + DNN = ALDE's tested approach
- ESM2 **mean pool** + RF = your approach, EVOLVEpro's approach, BO-in-embedding's approach
- These are fundamentally different uses of ESM2

---

### ESM2-15B (largest variant)
| Paper | Dataset | Pooling | Model | Result |
|---|---|---|---|---|
| EVOLVEpro 2024 | 12 DMS (diverse) | Mean pool | Random forest | ✅ Best PLM; outperforms 650M by significant margin on diverse tasks |

---

### Sparse Mutation Features (your work)
| Paper | Dataset | Encoding | Model | Result |
|---|---|---|---|---|
| Your work | GFP (52k) | Sparse index (15 unique mutation identities) | Random forest | R²=0.466; beats ESM2 at N<200-500 |

---

### ESM2 Log-Likelihood Ratios (zero-shot)
| Paper | Dataset | Method | Result |
|---|---|---|---|
| Wittmann 2021 | GB1 | EVmutation / ESM1b marginals | ✅ Useful for training set enrichment (ftMLDE) |
| Notin 2022 | ProteinGym (~250 DMS) | Tranception with retrieval | ✅ Best zero-shot predictor on diverse DMS |
| Not tested | GFP | Not yet benchmarked | — (opportunity for your thesis) |

---

### ProtBERT / ProtT5 (Elnaggar 2022)
| Paper | Dataset | Pooling | Model | Result |
|---|---|---|---|---|
| Elnaggar 2022 | BFD/UniRef | Per-residue | Structure tasks | ✅ Competitive with ESM1b |
| EVOLVEpro 2024 | 12 DMS | Mean pool | Random forest | ❌ Worse than ESM2-15B |

---

## 3.2 Surrogate Models

### Random Forest (RF)
| Paper | Dataset | Encoding | Acquisition | Result |
|---|---|---|---|---|
| Wu 2019 | GB1 | One-hot | Top-K greedy | ✅ Works well for MLDE (one-shot) |
| Wittmann 2021 | GB1 | Multiple | Top-K | ✅ Competitive baseline |
| EVOLVEpro 2024 | 12 DMS (wet lab) | ESM2-15B mean | Top-N | ✅ Grid-search winner for low-N active learning |
| **Your work** | GFP | ESM2-650M mean / mutations | UCB | R²~0.47, Precision@100 3–9% |
| ALDE 2024 note | — | — | — | ❌ ALDE chose DNN ensemble over RF |

**Verdict:** RF is practical and good for MLDE. For active learning with uncertainty, DNN ensemble outperforms RF in ALDE simulations. BUT EVOLVEpro shows RF + ESM2-15B is best for diverse real proteins at small N. Context matters.

---

### Gaussian Process (GP)
| Paper | Dataset | Encoding | Kernel | Acquisition | Result |
|---|---|---|---|---|---|
| Romero 2013 | P450 chimeras | Structure contact kernel | Structure-based | UCB | ✅ Pioneer; outperforms DE |
| Snoek 2012 | ML hyperparameters | N/A | Matérn 5/2 | EI | ✅ Foundational |
| ALDE 2024 | GB1 + TrpB | AAIndex, Georgiev | RBF | UCB, TS | ✅ Good with low-dim encodings |
| ALDE 2024 | GB1 + TrpB | ESM2 (6400-dim) | RBF | — | ❌ Too high-dimensional |
| BO-in-embedding 2024 | GB1 | ESM2 mean (320-dim) | RBF | UCB | ⚠️ Competitive but not better than one-hot |
| Hie 2020 | Drug discovery | Pre-trained features | RBF | UCB | ✅ Calibrated uncertainty key finding |

---

### DNN Ensemble (5 MLPs with ensemble variance for UQ)
| Paper | Dataset | Encoding | Acquisition | Result |
|---|---|---|---|---|
| ALDE 2024 | GB1 + TrpB | One-hot, ESM2 | Thompson Sampling | ✅ **Best combination overall; most accurate + best calibrated** |
| ALDE 2024 wet lab | ParPgb | One-hot (5 sites) | Thompson Sampling | ✅ 3 rounds, 480 variants → 12% → 93% yield |

---

### Deep Kernel Learning (DKL = GP on DNN last layer)
| Paper | Dataset | Encoding | Result |
|---|---|---|---|
| ALDE 2024 | GB1 + TrpB | ESM2 | ✅ Best pairing for ESM2 encoding; but worst UQ calibration despite good performance |

---

### Boosting Ensemble (5 gradient boosted trees)
| Paper | Dataset | Encoding | Result |
|---|---|---|---|
| ALDE 2024 | GB1 + TrpB | AAIndex, Georgiev | ✅ Good for low-dim encodings; lower MAE than GP |

---

### VAE + Oracle (DbAS / CbAS)
| Paper | Dataset | Result |
|---|---|---|
| DbAS 2018 | GFP, TF binding | ✅ Generative; but oracle exploitation problem |
| CbAS 2019 | GFP, TF binding | ✅ Fixes exploitation; standard generative baseline |

---

### RL Policy (DyNA-PPO)
| Paper | Dataset | Result |
|---|---|---|
| Angermueller 2020 | DNA, RNA, PyRosetta proteins | ✅ Competitive with AdaLead; RL with ensemble simulator |

---

### Denoising Autoencoder + GP (LaMBO)
| Paper | Dataset | Result |
|---|---|---|
| Stanton 2022 | GFP, RFP, drug resistance | ✅ First to show BO in PLM latent space; enables gradient-based acquisition; multi-objective |

---

## 3.3 Acquisition Functions

| Acquisition | Papers tested | Finding |
|---|---|---|
| **Greedy (top-K)** | Wu 2019, Wittmann 2021, ALDE 2024 | ✅ Fast; good baseline; over-exploits |
| **UCB (fixed β)** | Romero 2013, Snoek 2012, ALDE 2024, BO-in-embedding 2024, your work | ✅ Standard; β is hard to tune; exploration-exploitation tradeoff |
| **Expected Improvement (EI)** | Snoek 2012, PEX 2022 | ✅ Theoretically motivated; slower than TS in practice |
| **Thompson Sampling (TS)** | ALDE 2024 | ✅ **Best acquisition function in ALDE; diverse batch selection naturally** |
| **Proximal (PEX)** | PEX 2022 | ✅ Hamming distance penalty prevents out-of-distribution exploitation |

---

## 3.4 Summary: The Encoding × Model Interaction (Key Matrix)

Based on ALDE 2024 (the most rigorous benchmarking):

| | Boosting | GP | DNN Ensemble | DKL |
|---|---|---|---|---|
| **AAIndex (4-dim)** | ✅ Good | ✅ **Best for GP** | ⚠️ Underuses capacity | ⚠️ |
| **Georgiev (19-dim)** | ✅ Good | ✅ Good | ⚠️ | ⚠️ |
| **One-hot (80-dim for 4 sites)** | ✅ Good | ✅ OK | ✅ **Overall best combo** | ✅ Good |
| **ESM2 (6400-dim, site-specific)** | ⚠️ | ❌ Too high-dim | ✅ Good | ✅ **Best for ESM2** |

**Your setup** (ESM2 mean-pool 320-dim + RF) = between "one-hot + GP" and "ESM2 + Boosting" territory — not tested by ALDE because ALDE used site-specific ESM2, not mean pooling.

---

## 3.5 The "No Free Lunch" Summary

| Claim | Supported by | Counterexample |
|---|---|---|
| "ESM2 always better than one-hot" | ❌ False | Wittmann 2021, ALDE 2024: one-hot + DNN ensemble wins on GB1/TrpB |
| "One-hot always better" | ❌ False | EVOLVEpro 2024: ESM2-15B better for diverse protein families |
| "GP is the right surrogate for BO" | ❌ Oversimplified | ALDE: DNN ensemble better calibrated; RF wins for small-N diverse proteins |
| "More parameters = better PLM" | Mostly true | EVOLVEpro: ESM2-15B > 650M; but ALDE: ESM2 doesn't help over one-hot on k-site landscapes |
| "Frequentist UQ works better than Bayesian" | ✅ Supported | ALDE 2024: DNN ensemble (frequentist) beats GP (Bayesian) consistently |
| "PLM fitness = protein activity" | ❌ False | EVOLVEpro: −0.22 to −0.13 correlation in all 5 proteins |
| "BO always beats greedy" | ❌ Not proven | AdaLead (greedy) beats BO on most AdaLead benchmarks |

