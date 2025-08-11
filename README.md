Working title: EO-PCN + Δμ Memory
Vision: predictive coding with error optimization (EO) for stable inference, augmented by an associative memory that stores successful state deltas (Δμ) keyed by local context. Memory proposals nudge μ toward regions that historically reduced error → faster convergence, better recon — without losing the original PCN inspiration.

Core loop (current):

Bottom-up init → μ₀…μ_L

EO iterations: for each layer l, compute error ε_l = μ_l − decode(μ_{l+1}); update μ_{l+1} with correct gradient sign and damping.

Δμ memory: per layer, query top-K with context [μ_l, μ_{l+1}] and add gated Δμ proposals.

Top μ final refine with memory + gate → decode, classify.

Loss = recon(final + init) + CE(final + init).

Recent run (MNIST-ish):

Recon improved with memory: MSE 0.0376 → 0.0343

Acc dipped in sample eval: 0.922 → 0.898

Problems to solve
P1: Memory can improve recon but harm decision boundary (classification).

P2: Need clearer ablation + metrics to quantify where memory helps/hurts.

P3: Hyperparams (top-K, temperature, strengths) not tuned systematically.

P4: Δμ proposals sometimes too strong/weak across layers; per-layer gating could learn more.

P5: Code needs tests (shape, grad safety, no in-place), and clean experiment scaffolding.

Milestones
M1 — Stabilize performance (classification-safe memory)

Deliver a “confidence-safe classifier” path so memory never reduces margin.

Add small margin regularizer.

Quick hyperparam sweep script.

M2 — Measurement & ablations

Reproducible eval: recon/acc with & without memory; per-layer contribution; gates and Δμ norms.

Plots saved per epoch.

M3 — Quality & speed

Learn per-layer temps/strengths; optional kNN (faiss) backend for big memories; conv encoder/decoder variant.

Task breakdown (create GitHub issues from these)
Confidence-safe classification gate

Implement “keep refined μ for decoder, but choose μ for classifier per-sample based on logit margin”.

Add optional margin penalty: max(0, margin_before - margin_after) with weight 0.1.

DoD: On a fixed seed run, acc_with_mem ≥ acc_no_mem (±0.003) while recon_with_mem ≤ recon_no_mem by ≥5%.

Hyperparam sweep script

Grid over {topk ∈ [8,16,32], temperature ∈ [0.2,0.35,0.5], strength_internal ∈ [0.2,0.4,0.6], strength_final ∈ [0.3,0.5,0.8]}.

Save CSV with metrics (train loss, eval recon/acc, per-layer mean |Δμ_prop|, mean gate).

DoD: sweeps/results.csv + best row highlighted in stdout.

Per-layer learned strengths & temperatures

Replace global retrieval_strength_* with learnable per-layer scalars α_l, τ_l (τ via softplus).

Clip/regularize to avoid collapse; log values.

DoD: Training is stable; logged α_l, τ_l are finite; metrics ≥ baseline.

Per-layer gate diagnostics

Log histogram/mean of gate output per layer per epoch; save as JSON.

Visual annotate first N samples with gate values on figure.

DoD: logs/gates_epoch_*.json + figure with gate overlays.

Ablations

(a) EO only (memory off)

(b) Memory only at top layer

(c) Memory on all layers

(d) Δe vs Δμ (sanity confirmation; expect Δμ > Δe)

DoD: Table in reports/ablations.md with MSE/ACC deltas.

Unit tests (grad & shape)

Test no in-place ops break autograd; forward/backward on small batch passes; shapes match dims; memory insert/query works with top-K and full-softmax paths.

DoD: pytest green; CI job runs tests.

Reproducible evaluation script

eval.py --ckpt path --seed N --n-batches M prints averaged recon/acc with/without memory, saves comparison figure.

DoD: Deterministic numbers across repeated runs with same seed.

Logging & experiment structure

Save args.json, metrics.csv, and figures to a unique run dir; include git commit hash.

DoD: runs/<timestamp>-<hash>/... created every train.

Conv encoder/decoder option

Swap MLP with small CNN (Conv-ReLU-Conv …) + flatten to top μ; keep Δμ memory unchanged.

DoD: Training stable; recon visuals noticeably sharper; report metrics.

(Optional) Confidence-aware gate training

Add auxiliary loss that encourages gate↑ when it improves margin and gate↓ otherwise.

DoD: Gate calibration curve (gate vs margin improvement) shows positive correlation.

Repo scaffolding (suggested)
bash
Copy
/eo_pcn
  pcn_error_optimization_fixed.py        # current main
  eval.py                                # Task 7
  sweeps.py                              # Task 2
  models/
    eo_pcn.py                            # factor class here later
  memory/
    delta_mu_memory.py
  utils/
    viz.py  metrics.py  logging.py       # save CSV/JSON/figures
  tests/
    test_shapes.py  test_autograd.py     # Task 6
  runs/                                  # auto-created
  reports/
    ablations.md
Issue templates (paste into GitHub “New issue”)
Bug/Regression template

What happened:

Repro steps (cmd, seed, commit):

Expected vs actual:

Logs/figures:

Suspected area (file/line):

Checklist: [ ] seed fixed [ ] env logged [ ] dims printed

Experiment template

Goal (e.g., recover classification without losing recon gains)

Hypothesis:

Config:

Metrics to watch:

Stop criteria:

Result summary:

Next action:

Quick next steps (today)
Implement the confidence-safe classifier path + margin penalty.

Run a small sweep (top-K/temperature/strengths) to bring acc back without losing recon gains.

Add logging of mean |Δμ_prop| and gate means per layer so we can see where memory is active.

If you want, I can generate the exact code patch for Task #1 and a tiny sweeps.py that Codex/Copilot can expand.
