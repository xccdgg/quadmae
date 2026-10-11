# One RTX 4090: staged, auditable CVPR feasibility plan

**CVPR 2027 hard deadlines (AOE):** Registration **Nov 10, 2026**;
Main-paper submission **Nov 16, 2026**; Supplementary **Nov 23, 2026**.
Official CFP: https://cvpr.thecvf.com/Conferences/2027/CallForPapers
As of Oct 10, 2026, the main-paper deadline is only ~37 days away.
Do not assume full-scale multiple-GPU replications fit this schedule.

**First 7-day stop/go condition:** Gaussian certificate code and transforms
pass regression tests; at least one baseline checkpoint is reproduced under
matched validation sampling; a 5-epoch frequency-consistency pilot has a
measurable and reproducible advantage over the equal-budget consistency
control. If not, narrow the claim or pivot to a more feasible venue instead
of writing unsupported SOTA assertions.

Hardware assumption: **one desktop RTX 4090, typically 24 GB VRAM**.
If this is a laptop or 4090D, run the profiler to replace all assumptions.
No benchmark has been run on the user's GPU.

## Executive decision

- **Do not** attempt another 100-epoch ImageNet-1K ViT-B full-model
  continued-pretraining run on a single 4090.
- **Do not** claim we trained official Dual RS (ICLR 2026) or rRCM (ICLR
  2025) just because we cite them.
- Primary novel *training* component: a **frequency-conditioned consistency
  objective** (initially real orthogonal Haar) attached to an existing
  pretrained robust classifier. At inference, retain pixel Gaussian RS.
- Primary *new published* benchmark: **official ICLR 2026 Dual RS**. Keep
  its sigma-estimator / local-constancy certificate intact, and treat its
  official checkpoint experiments as a separate protocol until matched.
- Important strong representation benchmark: official **ICLR 2025 rRCM**.
  Its official README uses 8 GPUs for CIFAR-10 pretraining and 16-32 GPUs
  for ImageNet pretraining, so on one 4090 **reuse released models or
  published figures with protocol caveats; do not promise full retraining**.
- Current DMAE/QWT implementation is a hypothesis generator and historical
  control, not a valid new robustness certificate in QWT-inference mode.

## Why full evaluation can dominate budget

- One Gaussian classifier certified with `n0=100, n=10,000` over 10,000
  CIFAR-10 test images uses **101,000,000 classifier forwards**.
- Two models × three sigma levels × three seeds scales to **1.818 billion
  forwards** before sigma-estimator computations; this may require multiple
  weeks on one GPU. These are *counts*, not measured wall-clock times.
- One 1,000-image subset with n=1,000 uses **1.1 million forwards**, so
  early screening is approx 91.8x cheaper per classifier/sigma than a
  10,000-image full evaluation with n=10,000.
- A sigma estimator in Dual RS adds another certification workload.
  Its 2026 official README itself notes multi-GPU ImageNet usage.
- n=1,000 gives coarser confidence intervals and cannot replace the
  final `n=10,000` tests, especially for large radii.

## Stage gates, not fixed GPU-hour promises

### Gate 0: theory/code safety (before any training)

- Run `python -m unittest discover -s tests -p 'test_rescue_safety.py' -v`.
- Inspect remaining normalizations, `input_size`, sigma units, image
  preprocessing, statistical confidence interval correctness and noisy
  inference stochasticity. Fix failures.
- Only use `use_quaternion_noise=False` when reporting Cohen-certified
  `L2` radii. Train-time QWT corruption is allowed.
- Check tiny CPU fixtures before GPU tests.
- Profile `python scripts/profile_single_gpu_certification.py
  --eval-images 1000 --n 1000 --batch-sizes 4 8 16 32`.
- If `models_vit` timm APIs are incompatible, pin compatible dependencies
  before benchmarking. Benchmark is forward-only and is a LOWER BOUND.

### Gate 1: establish strong, fair, feasible baselines

For the 224x224 ViT-B pipeline:
1. Fixed off-the-shelf DMAE encoder/checkpoints already owned and auditable.
2. Additional-pretrain / fine-tune Gaussian-only matching compute (control).
3. Ordinary separable Haar train corruption (no quaternion terminology).
4. Frequency-conditioned consistency training (candidate innovation).
5. When possible, official Dual RS with its *own* official weights and
   sigma estimator; do not report it as matched to our ViT unless same
   model, data, sigma, and confidence budget are used.

For full CIFAR-10, use a much cheaper 32x32 backbone (e.g., ResNet-18/110)
as an additional small-backbone setting, not as proof of beating ViT-B.
New weights, sigma ranges and no 224x224 resize need separate baselines.

**Training suggestion** (starting knobs for profiling, NOT proven defaults):
- ViT-B frozen backbone + trainable small adapter/head (or LoRA rank 8).
- AMP BF16, micro-batch 8, gradient accumulation 8, 5 epochs smoke run.
- Increase to 20-30 epochs only if gains are visible relative to an
  exactly matched training control. Unlock last transformer blocks only
  if loss plateaus / resources permit.
- No decoder unless the experiment specifically requires reconstruction.
  Spectral consistency needs two views of the *same image*, one regular
  Gaussian and one frequency-perturbed, and trains representations to
  agree. The final certified pipeline MUST still sample iid pixel
  Gaussian at inference.
- Implement and compare ablations: r=1,2,3,4; spectral loss weight
  0 vs >0; real Haar vs no wavelet; same-view vs cross-view; and a
  comparably parameterized Gaussian consistency regularizer.

### Gate 2: efficient exploratory certification

- During debugging: 100 images, n=100 to catch correctness errors.
  Such results are NOT publications or evidence of SOTA.
- During screening: fixed 500 or 1000 ImageNet/CIFAR10 validation
  images, `n0=100`, `n=1000` for shape of accuracy-radius curve.
- Keep exact sample IDs and seed fixed across all models.
- Do not select sigma, loss weight, epochs or noise ratio on the
  held-out test subset that will appear in the paper.

### Gate 3: high-confidence final certification

- Lock hyperparameters from validation, freeze weights.
- Per dataset and model: fixed, predeclared `sigma` and grid of radii.
- `n0=100, n=10,000, alpha=0.001` for ordinary one-component Gaussian
  smoothing; Dual RS needs its own *joint* confidence budget (often
  0.0005 + 0.0005) and official local-constancy procedure.
- Use all 10,000 CIFAR10 test examples if computationally feasible and
  present a 1000-image interim only as an interim subset with its CI.
- For ImageNet-1K, match IDs exactly: current DMAE work uses 1000
  images, while official Dual RS examples use 500 with stride 100.
- Training seeds >= 3; paired uncertainty across examples for fixed
  checkpoints; report both forms of variability and avoid inflated
  independent-example confidence claims.
- Wall-clock speed, number of forwards, energy/memory/VRAM and
  parameters are part of the experimental evidence.

## Example forward-only workload: use real measured throughput

With batch throughput **T measured images per second**:

    lower_bound_hours = n_images * (n + n0) / T / 3600

For illustration ONLY, not a claimed 4090 measurement:
- At T=200 img/s, 10,000 × (10,000 + 100) ~ 140.3 hours of
  classifier-only time, PER model and sigma.
- Actual certified evaluation is SLOWER due to noise generation,
  data transforms, IO, and, for Dual RS, extra estimator passes.
- Reduce redundant settings (no post-hoc cherry-picking): prioritize
  a single sigma where a meaningful improvement exists and a
  pre-registered secondary sigma.

## Timeline as ordered checkpoints, not promises

Week 1: correctness + pretrained Gaussian baseline + profiler.
Week 2: 5-epoch spectral-consistency pilot + matched training control.
Week 3: sigma/radius ablation and partial validation certification.
Week 4+: only if significant: final matched comparison on CIFAR-10 and
selected ImageNet protocol. Full confirmation time determined by profiler.

## Do not claim CVPR acceptance

Acceptance depends on research novelty, matched stronger competitors,
confirmed `L2` guarantee and final experimental evidence. If the
spectral objective does not improve certified accuracy after correct
calibration against ordinary Haar/consistency controls, pivot rather
than overstate or cherry-pick results.
