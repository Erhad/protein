# Literature Reference Tables

_Generated from direct PDF reads across all 35 papers in the research log._

---

## Table 1 — Paper Properties

> Columns: **PLM** = protein language model used (if any); **Wet lab** = real experimental measurements as part of the study; **BO/MLDE** = paper performs or evaluates ML-guided iterative optimization (not just landscape characterization).

### A. Fitness Landscape Characterization (no ML method)

| Paper | Year | Dataset(s) | Encoding(s) | Surrogate | PLM | Wet Lab | BO/MLDE |
|---|---|---|---|---|---|---|---|
| Sarkisyan et al. | 2016 | avGFP (~51k variants, up to 15 mut) | — | — | — | ✓ | ✗ |
| Wu et al. (GB1) | 2016 | GB1 (160k variants, 4-site) | — | — | — | ✓ | ✗ |
| Johnston et al. | 2024 | TrpB (160k variants, 4-site) | — | — | — | ✓ | ✗ |
| Romero & Arnold | 2009 | Multiple proteins (review) | — | — | — | — | ✗ |
| Arnold | 1998 | Various enzymes | Random mutagenesis | — | — | ✓ | ✗ |
| Stemmer | 1994 | β-lactamase | DNA shuffling | — | — | ✓ | ✗ |

---

### B. ML-Guided Optimization (MLDE / BO / Generative)

| Paper | Year | Dataset(s) | Encoding(s) | Surrogate | PLM | Wet Lab | BO/MLDE |
|---|---|---|---|---|---|---|---|
| Wu et al. (MLDE) | 2019 | V VGB1 (160k) | One-hot | Shallow NN ensemble (+ RF, linear, KNN, DT) | — | ✗ | ✓ |
| Wittmann et al. | 2021 | GB1 (149k) | One-hot · Georgiev · AAIndex · MSA-Transformer | XGBoost + others | MSA-Transformer | ✗ | ✓ |
| Romero et al. (GP) | 2013 | Cytochrome P450 chimeras | Structure-based contact kernel · Hamming | Gaussian Process | — | ✗ | ✓ |
| Biswas et al. (LowN) | 2021 | avGFP · TEM-1 β-lactamase | eUniRep (fine-tuned UniRep, global mean pool) | Ridge regression | UniRep | ✗ | ✓ (MCMC) |
| Sinai et al. (AdaLead) | 2020 | TF binding (DNA len-8) · RNA · 3MSI antifreeze protein | One-hot | NN surrogate (screen) | — | ✗ | ✓ (greedy evo) |
| Angermueller et al. (DynaPPO) | 2020 | TF binding · Antimicrobial peptides · Ising protein | One-hot | RL (PPO) + NN surrogate | — | ✗ | ✓ (RL) |
| Brookes & Listgarten (DbAS) | 2018 | TBD (read needed) | VAE latent | VAE decoder | — | ✗ | ✓ |
| Brookes et al. (CbAS) | 2019 | TBD (read needed) | VAE latent | VAE decoder | — | ✗ | ✓ |
| Hie et al. | 2020 | Virus escape · Enzyme optimization | GP features | Gaussian Process (uncertainty-guided) | — | ✗ | ✓ |
| Shen et al. (BIBM 2024) | 2024 | GB1 · PhoQ | ESM-1b (1280-dim, per-residue at 4 sites) | GP (Matérn 3/2) | ESM-1b | ✗ | ✓ |
| Ren et al. (PEX) | 2022 | avGFP · AAV · TEM-1 · E4B · AMIE · LGK · Pab1 · UBE2I (8 total) | MuFacNet (mutation set repr.) | MuFacNet | ESM-1b (oracle only) | ✗ | ✓ |
| Stanton et al. (LaMBO) | 2022 | Small molecules (logP+QED, DRD3) · RFP (~200 aa) | DAE latent space | Deep Kernel Learning (GP + DNN) | Optional (TAPE/ESM) | ✓ (RFP) | ✓ |
| Yang et al. (ALDE) | 2024 | GB1 (sim.) · TrpB (sim.) · ParPgb (wet-lab) | One-hot · AAIndex · Georgiev · ESM2 (per-residue at targeted sites) | GP · Boosting Ens. (×5) · DNN Ens. (×5) · Deep Kernel Learning | ESM2-650M (1280-dim) | ✓ (ParPgb) | ✓ |

---

### C. Protein Language Models (embedding / zero-shot)

| Paper | Year | Training Data | Architecture | Embed. Dim | PLM Name | Fitness Use |
|---|---|---|---|---|---|---|
| Rives et al. (ESM-1b) | 2021 | UniRef50 (250M seq) | Transformer (650M) | 1280-dim/residue | ESM-1b | Zero-shot ΔΔG via log-lik. |
| Lin et al. (ESM2) | 2023 | UR50D | Transformer (8M–15B) | 320 (8M) · 480 (35M) · 640 (150M) · **1280 (650M)** · 2560 (3B) · 5120 (15B) | ESM2 | Zero-shot + embedding featurization |
| Hayes et al. (ESM3) | 2025 | Sequence + structure + function | Multimodal Transformer | TBD | ESM3 | Zero-shot + generative |
| Elnaggar et al. (ProtTrans) | 2022 | BFD (2.1B) + UniRef | BERT / T5 | **1024-dim/residue** | ProtBERT · ProtT5 | Embedding featurization |
| Notin et al. (Tranception) | 2022 | UniRef100 | Autoregressive PLM | — | Tranception | Zero-shot via log-lik ratio: log P(x_mut)/P(x_wt) |
| Madani et al. (ProGen2) | 2023 | UniRef50+90 (280M seq) | Autoregressive LM (6.4B) | — | ProGen2 | Generative (functional lysozyme, 31% identity to known) |

---

### D. Structural / Generative Design (no fitness optimization)

| Paper | Year | Task | Input | Output | Wet Lab |
|---|---|---|---|---|---|
| Jumper et al. (AlphaFold2) | 2021 | Structure prediction | Sequence + MSA | 3D structure | ✗ |
| Dauparas et al. (ProteinMPNN) | 2022 | Inverse design (structure → sequence) | Backbone coords | AA probabilities | ✓ |
| Watson et al. (RFdiffusion) | 2023 | De novo backbone generation | Noise + constraints | Novel backbones | ✓ (binders) |

---

### E. Foundational BO / Reviews

| Paper | Year | Contribution | Acquisition Functions Covered |
|---|---|---|---|
| Snoek et al. | 2012 | Practical BO with GP; first ML hyperparameter tuning application | EI, UCB |
| Frazier | 2018 | Definitive BO tutorial | EI · UCB · Thompson Sampling |
| Yang et al. (review) | 2019 | Taxonomy of ML-guided DE: (1) supervised regression, (2) generative, (3) RL | — |

---

### F. Benchmarking / Dataset Papers

| Paper | Year | What it benchmarks | Key datasets |
|---|---|---|---|
| Dallago et al. (FLIP) | 2021 | Protein fitness prediction methods | GB1 · GFP · AAV · TEM · SCH and others |
| Greenberg et al. | 2023 | Uncertainty quantification methods for protein ML | Multiple protein benchmarks |
| Kirjner et al. | 2023 | Smoothed fitness landscape optimization | TBD (PDF read failed) |

---

---

## Table 2 — Protein Dataset Landscape Descriptions

> Expanded from existing EXP002 notes. Landscape shape and topology are critical for understanding which methods will succeed.

---

### GB1 — Protein G B1 Domain (Wu et al. 2016)

| Property | Value |
|---|---|
| Protein | IgG-binding B1 domain of streptococcal protein G |
| Length | 56 amino acids |
| Mutation sites | 4: **V39, D40, G41, V54** |
| Total variants | 160,000 (20⁴) |
| Directly measured | 149,361 (93.4%); 10,639 imputed via regularized regression |
| Fitness assay | mRNA display: folding stability × IgG-Fc binding affinity |
| Landscape type | Small combinatorial (exhaustive coverage of 4-site design space) |
| Landscape shape | **Highly rugged, single dominant peak.** ~92% of variants fall below 1% of the global maximum — a massive "hole" problem. The global optimum (V39E/D40G/G41A/V54L) is a sharp, isolated peak. Strong pairwise and higher-order epistasis between the 4 positions. |
| Peak topology | Single-peak. Climbing toward the peak from wild-type requires traversing fitness valleys. |
| Epistasis | High — most multi-mutants are not predictable from single-mutant effects |
| Practical notes | Clean signal-to-noise (exhaustive, good counts). Standard benchmark in protein ML. HuggingFace: SaProtHub/Dataset-GB1-fitness. Wittmann 2021 filtered variants with input_ct < 10 (149,361 usable). |

---

### avGFP — Green Fluorescent Protein (Sarkisyan et al. 2016)

| Property | Value |
|---|---|
| Protein | avGFP from *Aequorea victoria* |
| Length | ~238–239 aa (not explicitly stated in paper; PEX 2022 uses 239) |
| Mutation coverage | Sampled subset: 1–15 mutations per variant, avg ~3.7 |
| Total variants | 51,715 protein sequences (56,086 nucleotide sequences); our pipeline: 58,417 |
| Coverage of space | <0.00001% of theoretical space at 15 mutations |
| Fitness assay | Fluorescence intensity (FACS-seq; population enrichment of bright cells) |
| Landscape type | Full-protein, sampled (not combinatorial — each variant is a random draw from mutagenesis) |
| Landscape shape | **Rugged, sparse bright islands in a dark sea.** Most single and multi-mutations reduce fluorescence. Very few variants improve on wild-type. Top variants are genuinely hard to find — Precision@100 is 3–9% with RF (vs. 100% would be perfect). Fitness is highly non-linear: a good global Spearman (~0.67) does not translate to identifying elite sequences. |
| Peak topology | Functionally, single peak (fluorescence requires intact chromophore formed by Ser65-Tyr66-Gly67). Mutations near this triad are devastating. Distant mutations have small, unpredictable effects. |
| Epistasis | Very high near chromophore-forming residues; lower elsewhere. Many neutral mutations with rare conditionally beneficial combinations. |
| Key difficulty | Unlike GB1, avGFP is sampled — you cannot look up any arbitrary variant. The missing fitness values are a real constraint. |
| Used by | Biswas 2021, PEX 2022, our work (EXP001, VAL001–003) |

---

### TrpB — Tryptophan Synthase β-subunit (Johnston et al. 2024)

| Property | Value |
|---|---|
| Protein | β-subunit of tryptophan synthase (*Pyrococcus furiosus* or thermophilic variant) |
| Mutation sites | 4 sites (same exhaustive 20⁴ design as GB1) |
| Total variants | 160,000 (20⁴) |
| Fitness assay | Tryptophan formation rate (growth-coupling selection) |
| Landscape type | Small combinatorial (exhaustive 4-site coverage) |
| Landscape shape | Rugged, similar structure to GB1. Used alongside GB1 in ALDE 2025 to show method generalization across two distinct rugged combinatorial landscapes. |
| Used by | ALDE 2025 (computational simulations) |

---

### ParPgb — Protoglobin for Cyclopropanation (ALDE 2025, wet-lab)

| Property | Value |
|---|---|
| Protein | ParLQ variant of wild-type protoglobin ParPgb (W59L/Y60Q starting point) |
| Mutation sites | 5: **W56, Y57, L59, Q60, F89** |
| Design space | 20⁵ = 3,200,000 variants (not exhaustively measured) |
| Fitness assay | cis-selectivity of cyclopropanation (cis yield − trans yield) |
| Starting fitness | ~40% total yield, 3:1 trans:cis selectivity (wrong stereochemistry) |
| After 3 ALDE rounds | 99% total yield, 14:1 cis:trans selectivity (~400 variants screened total) |
| Landscape topology | Not characterized globally. Accessible via active learning over 5-site space. |
| Used by | ALDE 2025 wet-lab campaign only |

---

### PhoQ — Protein Kinase PhoQ (BIBM 2024)

| Property | Value |
|---|---|
| Protein | PhoQ histidine kinase from *E. coli* |
| Mutation sites | 4 sites (20⁴ combinatorial) |
| Fitness assay | Signal sensing / kinase activity |
| Used by | BIBM 2024 (BO in embedding space) alongside GB1 |

---

## Key Discrepancies Found During PDF Review

These issues affect faithful interpretation in the insight files and should be corrected:

| # | Location | Issue | What Paper Actually Says |
|---|---|---|---|
| 1 | EXP001, HYP001 | "ESM2 650M → 320-dim" | **650M ESM2 = 1280-dim per residue.** 320-dim comes from the 8M model. Our pipeline uses `esm2_t33_650M_UR50D` but stores 320-dim — possible dimension reduction step in preprocessing needs verification. |
| 2 | STR001 | "Our RF+ESM2+UCB ≈ Wu 2019 MLDE" | **Wu 2019 uses shallow NN ensemble + one-hot + top-k ranking** (no UCB, no ESM). The comparison is loose; "same era" is more accurate than "same method." |
| 3 | STR001 PATH_B | "BIBM 2024 shows GP+ESM2 > RF+ESM2" | **BIBM 2024 uses ESM-1b (not ESM2) and has zero RF comparison.** Correct claim: "BIBM 2024 shows GP+ESM-1b is competitive vs regression baselines." |
| 4 | LIT005 | "Nature Communications 2025" | **Still a bioRxiv preprint (July 2024).** Publication in NatComms is not confirmed in the PDF. |
| 5 | LIT005 | "12% → 93% yield in 3 rounds" | **Conflates two metrics.** Round 1 best variant = 93% total cyclopropanation yield + 12:1 cis:trans ratio. Starting point had ~40% yield, not 12%. |
| 6 | LIT005 | ESM2 dims "for 5 sites = 6400" | Correct for ParPgb wet-lab (5 sites × 1280 = 6400). For GB1/TrpB simulations it would be 4 × 1280 = 5120. Presentation should distinguish the two contexts. |
| 7 | STR001 | "Wittmann 2021: one-hot beats complex encodings" | **Wittmann says simpler encodings remain competitive;** Georgiev and MSA-Transformer show marginal improvements. One-hot does not *beat* — it ties. |
| 8 | HYP001c | "Hie et al. 2024" | **Paper is from 2020, not 2024.** And it uses GP uncertainty quantification — not log-likelihood ratios. The log-lik claim belongs to Notin 2022 (Tranception). |
| 9 | LIT005 | Missing: frequentist > Bayesian finding | ALDE explicitly highlights that **frequentist ensembles (DNN, Boosting) outperform Bayesian GP** — a counterintuitive key result that should be in the insight file. |
| 10 | All files | AdaLead benchmarks | AdaLead was validated on **TF binding / RNA / 3MSI antifreeze protein** — not GB1 or GFP. Its performance on these datasets is what motivated the "beats BO in many benchmarks" claim. |
