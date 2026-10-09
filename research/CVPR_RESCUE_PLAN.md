# CVPR research rescue: QFS-DMAE -> certified frequency-aware representation learning

> Status: **research branch / NOT a validated CVPR submission or a reproduced SOTA result**.
> Core aim: preserve the useful frequency-domain representation hypothesis while
> removing any unsupported claim of an L2 certificate.

## 0. Paper / implementation audit

Paper: QFS-DMAE (KDD 2027 draft PDF), particularly Section 3.3, Eq. (13)-(15),
and Tables 1-4.

### P0: the current certificate is invalid for the implemented noise

* Eq. (13)-(15) concern (at most) an averaged covariance matrix, not equality
  of distributions. An isotropic covariance does NOT mean the noise is an
  isotropic Gaussian. Standard Cohen et al. certification must not apply
  `sigma_pix * Phi^{-1}(p_A)` to arbitrary Gaussian mixtures.
* The paper asserts Haar-uniform T over O(d); `util/quadatasetgpu.py` actually
  samples 32 finite combinations of rotations/flips/1-pixel translations.
  Those are pixel permutations, not a Haar-uniform orthogonal 2-design.
* The old inverse QWT used flipped highpass filters in transposed convolution
  (which already implements the adjoint), so zero-noise reconstruction failed.
  The old inverse finite transform also applied the wrong shift and operation
  order; no longer was the pre-noise map exactly the identity.
* QWT code divides imaginary-component coefficient noise by sqrt(3), so the
  old `sigma_total_from_pixel` corresponded to approximately
  `sigma_pix / sqrt(3)` per-channel pixel standard deviation, rather than
  `sigma_pix`. This branch corrects the training-noise calibration; **old
  checkpoints are NOT retroactively fixed or equivalent to retraining**.
* `engine_finetune.certify_evaluate_dist` previously computed
  `restorer(x)` and added Gaussian noise afterwards. That is a certificate
  in restored-image space, not necessarily input-image space. This branch
  fails closed until restoration is moved *inside* the classifier evaluated
  at each independently Gaussian-perturbed input.
* The implemented RGB \"quaternion\" wavelet acts as real separable Haar
  wavelets **independently on R/G/B** (with a zero real component). It is not
  yet evidence of genuinely quaternion-coupled color processing.
* `certify_cifar10.py` defaulted to 1,000 Monte Carlo certification samples
  while the manuscript says 10,000. Updated default: 10,000. Other experiment
  choices must be matched explicitly, including evaluation subset and alpha.

Consequently, **all previous colored-noise \"certified accuracy\" tables
must be labelled UNVERIFIED and rerun from checkpoints under a justified
noise law**. Do not silently reinterpret the published numbers.

## 1. Baseline switch (2026 top-conference work)

**Primary stronger/newer baseline: Dual Randomized Smoothing (Dual RS)**
Sun, Mao, Vechev, ICLR 2026.
- Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/cd9664c7094d90e512ce27f2fd58198b-Abstract-Conference.html
- Official implementation: https://github.com/eth-sri/Dual-Randomized-Smoothing
- Its input-dependent sigma is rigorously certified using a separately
  smoothed sigma estimator and a local-constancy argument. **Do not replace
  this with an unverified `sigma(x)` heuristic.**
- Its ImageNet and CIFAR-10 default pipelines use different input models,
  sigma candidates, validation subsets, and alpha budgets in places; an
  apples-to-apples comparison REQUIRES unified data indices, class models,
  N/N0/alpha, sigma/radius grid, normalization and measured wall-clock cost.

**Strong representation-learning baseline: rRCM (ICLR 2025)**
- Paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/eb9b5c5d2e6fad9922198168a7786874-Abstract-Conference.html
- Code: https://github.com/jiachenlei/rRCM
- Relevant because it learns denoising-invariant representations, rather
  than relying on large diffusion preprocessing at inference.

Other mandatory comparisons: original DMAE, MAE / ViT and the Gaussian
version of our continued-pretraining pipeline; vanilla orthogonal Haar;
frequency-selective Gaussian corruption without quaternion packaging;
learned frequency regularization without changing inference noise;
Dual RS with and without proposed representation training.

## 2. Research question / defensible direction

**Proposed working title**
Frequency-Selective Representation Learning for Certifiably Robust Vision Models

Hypothesis: under rigorously Gaussian randomized-smoothing certification,
frequency-selective consistency learning improves the classifier's noisy
class-probability margin, increasing certified accuracy across radii.
This would be a *representation-training* contribution and not require
claiming an invalid certificate for colored Gaussian mixtures.

### Safe inference option A (simplest)

During representation learning, use properly reconstructed frequency-aware
corruption, plus paired pixel-Gaussian augmentation and spectral-consistency
loss. During classifier tuning and certification, use independent
`epsilon ~ N(0, sigma^2 I)` in the raw pixel coordinate system.

Then for independently selected class A and binomial lower confidence bound
`p_lower > 1/2`:

    R = sigma * Phi^{-1}(p_lower)

is a standard Cohen-style valid L2 certificate of the Gaussian-smoothed
classifier for a fixed noise sigma (subject to exact MC statistical procedure).

### Research option B (non-Gaussian mixture, requires a new proof)

Explore factoring a fixed isotropic Gaussian floor out of every conditional
colored Gaussian covariance. If the restored transform is *exactly* additive,
and for every random transformation T:

    Sigma_T >= sigma_floor^2 * I,

the noise distribution can be expressed as
`N(0, sigma_floor^2 I) * Q` for an input-independent Q. A Gaussian
certificate at **sigma_floor**, NOT the trace-matched pixel sigma, may then
be justified using a randomized base classifier. Prove the precise
conditions, include clipping/normalization and transform boundaries, and
unit-test the distribution before implementing or publishing this option.
**This research branch deliberately does not claim or enable this theorem.**

### Dual RS integration

Use the **official Dual RS implementation** as a separate baseline, then
study whether spectral-consistency training can improve the base experts.
When using its full dual estimator, keep the official certified-local-
constancy procedure. Never compare its input-dependent sigma certificates
to an oracle chosen using ground-truth labels at test time.

## 3. Minimum ablation grid

| ID | Backbone/training | Inference noise | Claimed certificate |
|---|---|---|---|
| A0 | Vanilla DMAE ViT-B | Pixel Gaussian | Cohen |
| A1 | DMAE + matched additional pretraining | Pixel Gaussian | Cohen |
| A2 | A1 + ordinary Haar frequency corruption at training | Pixel Gaussian | Cohen |
| A3 | A2 + feature spectral consistency | Pixel Gaussian | Cohen |
| A4 | A3 + truly quaternion-coupled component (if built) | Pixel Gaussian | Cohen |
| B0 | Official Dual RS | Official dual pipeline | Original Dual RS proof |
| B1 | B0 with A3 representation model, jointly verified | Official dual pipeline | Original Dual RS proof, only if its assumptions still hold |
| C0 | Original QWT-mixture evaluation | QWT mixture | UNVERIFIED; no certified result |

Also ablate high/low ratio r = 1, 2, 3, 4; sigma choice; feature loss weight;
number of seeds; training data/compute; `sigma_train != sigma_eval`;
frequency masking order; random orthogonal mixing; mixed channel / ordinary
real Haar / learned channel coupling.

## 4. Required CVPR evidence before writing SOTA claims

1. Code-level identity assertions: `IQWT(QWT(x)) == x`,
   `T^{-1}T(x) == x`, and `noise(x, sigma=0) == x`.
2. Noise calibration: per-channel empirical mean/covariance and
   4th-order/non-Gaussian tests. **A variance test alone is not a proof.**
3. Unify ImageNet-1K evaluation indices; report official protocol separately
   when original papers use 500/1,000 subsets. CIFAR-10 complete test set.
4. N0=100, N=10,000 minimum, alpha=0.001, exact class selection and
   Clopper-Pearson intervals; stronger N=100,000 as a sensitivity check.
5. Reproduce baseline outputs independently before altering training or
   cherry-picking sigma/thresholds.
6. Certified accuracy-versus-radius curves, fixed-radius robust accuracy,
   mean noisy class probability / pA distribution, clean accuracy, number
   of parameters, training and inference GPU time, memory.
7. Mean, standard deviation, independent 3+ seeds and paired confidence
   intervals for the method difference, especially on ImageNet subsets.
8. Code/checkpoint provenance and authorship. Current README implies this is
   the official DMAE implementation; update attribution and licenses before
   releasing the project paper as original work.

ICML 2025 cautions against using average certified radius (ACR) as the
headline metric: https://proceedings.mlr.press/v267/sun25o.html
Use certified-accuracy curves and per-sample statistics instead.

## 5. Submission gating

- P0: reconstruct + clean Gaussian certificate + shared protocol tests.
- P1: reproduce original DMAE and official Dual RS/rRCM on matching sets.
- P2: show compelling gains over *ordinary real wavelet* and *equal-budget*
  Gaussian consistency baselines, not just DMAE.
- P3: if real quaternion coupling is not superior, **remove the quaternion
  novelty claim** rather than inventing a difference.
- P4: revise CVPR narrative/results after the experiments, not before.

No test accuracy, training improvement, acceptance probability or GPU run
has been fabricated here.
