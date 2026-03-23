# BOES — Theoretical Foundations

Shen et al., BIBM 2024. Bayesian Optimisation in Embedding Space.

---

## 1. The GP Surrogate

We model fitness as a Gaussian Process:

    f ~ GP(0, k(x, x'))

**Kernel:** Matérn 3/2 operating on ESM2 embeddings.

    k(x, x') = α² (1 + √3·d/ℓ) exp(−√3·d/ℓ)
    d = ||x − x'||₂

where α² is the output scale and ℓ is the length scale. Both are learned by
maximising the log marginal likelihood on the labeled set.

**Likelihood:** Gaussian observation noise σ_noise².

After observing labeled set (X, y), the posterior is:

    f | y ~ GP(μ_post, k_post)
    μ_post(x*) = K(x*, X) [K(X,X) + σ²I]⁻¹ y
    k_post(x*, x*') = k(x*, x*') − K(x*, X) [K(X,X) + σ²I]⁻¹ K(X, x*')

Fitting: maximise log p(y | X, θ) w.r.t. θ = {ℓ, α, σ_noise}.
Up to 5 random restarts to escape local optima. Cost: O(n³) per fit.

---

## 2. Acquisition: Expected Improvement (EI)

**When to use:** BOES-EI (faithful replication of the paper).

EI measures the expected gain over the current best fitness f* = max(y):

    EI(x) = (μ(x) − f*) Φ(Z) + σ(x) φ(Z)
    Z = (μ(x) − f*) / σ(x)

where Φ is the standard normal CDF and φ is the PDF.

**Behaviour:** EI is zero for x predicted worse than f*. It balances
exploitation (high μ) vs exploration (high σ), but implicitly uses f* as a
threshold, which can cause it to cluster selections near the current best.

**Batch selection:** Score all pool candidates by EI, return top-B.
This is greedy and produces correlated batches — fine for small B, degrades at
large B because near-duplicate selections waste experimental budget.

---

## 3. Acquisition: Thompson Sampling (TS) via Random Fourier Features

**When to use:** BOES-TS (thesis addition). TS naturally produces diverse
batches because each selected variant is optimal under a *different* plausible
fitness function.

### 3a. Why TS over EI at large batch sizes

EI is a deterministic function of (μ, σ). Picking top-B by EI selects B
variants that are all optimal under the *same* estimate. TS draws B independent
function samples; each variant is the best under a different "what if the
world looks like this" scenario. This matches the batch parallel setting.

### 3b. Thompson Sampling from a GP

Ideal TS: draw f* ~ p(f | y), return argmax_x f*(x).

Direct sampling requires O(N²) memory for the posterior covariance over N pool
candidates — infeasible at N=149k.

### 3c. Matheron's Rule (exact posterior sample, tractable)

Any GP posterior sample can be written as:

    f*(x) = f_prior(x)  +  K(x, X) [K(X,X) + σ²I]⁻¹ (y − f_prior(X))
              ↑ prior sample         ↑ deterministic update given prior draw

The update term is O(N × n_train) — cheap. The only non-trivial part is
drawing f_prior, a sample from the prior GP.

### 3d. Sampling the Prior via Random Fourier Features (RFF)

For a stationary kernel k, Bochner's theorem gives a spectral density S(ω):

    k(x, x') = E_ω[φ_ω(x) φ_ω(x')ᵀ]

We approximate with D random features:

    φ(x) = sqrt(2/D) · cos(Wᵀx + b)     shape: (D,)

where each row of W is sampled from S(ω) and b ~ Uniform(0, 2π).

**Sampling W for Matérn 3/2** (Student-t trick):

    z ~ N(0, I_d)
    u ~ χ²(3)               ← 2ν = 2 × 3/2 = 3
    ω = z · sqrt(3/u) / ℓ

Then a prior function sample is:

    f_prior(x) = α · θᵀ φ(x),    θ ~ N(0, I_D)

### 3e. Full TS Procedure (one batch draw)

    1. Sample W (D × d), b (D,), θ (D,) as above — all fresh per TS draw
    2. Compute Φ_train = φ(X_train)           shape: (n_train, D)
    3. Compute Φ_pool  = φ(X_pool)            shape: (N_pool,  D)
    4. Compute residual: r = y − α · Φ_train @ θ
    5. Solve: v = [K(X,X) + σ²I]⁻¹ r         reuse Cholesky from GP fit
    6. Score: f*(x) = α · θᵀ φ(x) + K(x, X) v    (Matheron update)
    7. Return argmax_x f*(x)

For a batch of size B: repeat steps 1–7 independently B times, collect
argmaxes. Duplicate argmaxes are broken by taking the next-best from that draw.

**Complexity per batch:**
  - RFF features for pool: O(D × d × N_pool) ~ 512 × 5120 × 149k ≈ 390B ops → ~0.5s on CPU
  - Matérn update: reuses GP Cholesky, O(N_pool × n_train) ~ 149k × 208 ≈ 31M → negligible

D=512 features gives a good kernel approximation for this problem scale.

---

## 4. Implementation Notes

- Inputs are StandardScaler-normalised before GP fitting (fit on train, apply to pool).
- y values are also standardised (zero mean, unit variance) before GP fitting;
  EI incumbent f* is in standardised space; final metrics use original scale.
- GP hyperparameters are re-optimised from scratch each round (no warm-starting).
- At batch=1 with 192 rounds, the GP refitting cost dominates. Cap restarts at 5.
- The same Cholesky factorisation used for GP inference is reused for TS.

---

## 5. References

- Shen et al. (2024). BOES. BIBM 2024.
- Rasmussen & Williams (2006). Gaussian Processes for Machine Learning. MIT Press.
- Wilson et al. (2020). Efficiently Sampling Functions from Gaussian Process Posteriors. ICML.
- Rahimi & Recht (2007). Random Features for Large-Scale Kernel Machines. NeurIPS.
