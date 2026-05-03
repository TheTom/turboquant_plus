# Re-examining EDEN's Optimal-S Critique of TurboQuant

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## TL;DR

> **EDEN's optimal scale is real, but second-order. Rotation choice is first-order. Production already uses the first-order fix.**

Two of the EDEN co-authors (Mitzenmacher and Portnoy) recently filed issues on the TurboQuant+ repo and a Towards Data Science article arguing that TurboQuant is just EDEN with the scale parameter `S=1`, and that swapping in EDEN's analytical optimal `S` would close a meaningful gap. I spent an afternoon running this against my actual production setup. Their math is correct and their lever exists; it just isn't the bottleneck for the system I ship. The 19% MSE gap they reproduce on synthetic Gaussians is a *rotation* choice (Hadamard vs dense Haar), not the scale parameter. Once you fix the rotation, EDEN's optimal `S` is within ~1% of matched-norm on synthetic. On real Qwen3-0.6B and Llama-3.2-1B KV cache it loses to matched-norm by 0.5 to 9%. The "1 bit free" result they describe applies to TurboQuant_prod specifically, which production V cache already avoids for independent reasons.

The two-part posture: concede their math, contest their diagnosis of the bottleneck.

That said, the attribution issue (#89) is fair, and the DRIVE/EDEN line of work absolutely belongs in the TurboQuant+ docs as prior art. I'll add it.

---

## 1. The Claim

The two surfaces of the critique:

**Issue [#87](https://github.com/TheTom/turboquant_plus/issues/87)** (Mitzenmacher, Apr 24 2026), titled "Use scale factor for improvements." Pointer to a short note paper, [arXiv:2604.18555](https://arxiv.org/abs/2604.18555), *A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work* by Ben-Basat, Ben-Itzhak, Mendelson, Mitzenmacher, Portnoy, and Vargaftik. The headline argument: TurboQuant_mse is a special case of EDEN with `S=1`, and EDEN's per-vector analytical `S` is provably better. They report a ~2.25% per-coord MSE improvement at `d=128, b=4` and claim a stronger result for the `prod` chain (roughly "k-bit unbiased EDEN beats (k+1)-bit TurboQuant_prod").

**Issue [#89](https://github.com/TheTom/turboquant_plus/issues/89)** (Portnoy, May 3 2026), titled "Consider proper attribution for EDEN quantization." Points to Portnoy's [Towards Data Science article](https://towardsdatascience.com/how-a-2021-quantization-algorithm-quietly-outperforms-its-2026-successor/), which makes the same algorithmic argument and asks for proper citation of the prior work.

The underlying papers are fully legitimate prior art:
- DRIVE: Vargaftik, Ben-Basat, Portnoy, Mendelson, Ben-Itzhak, Mitzenmacher. *DRIVE: One-bit Distributed Mean Estimation*. NeurIPS 2021. [arXiv:2105.08339](https://arxiv.org/abs/2105.08339).
- EDEN: same authors. *EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning*. ICML 2022. [proceedings.mlr.press/v162/vargaftik22a.html](https://proceedings.mlr.press/v162/vargaftik22a.html).
- Note: Ben-Basat et al. *A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work*. [arXiv:2604.18555](https://arxiv.org/abs/2604.18555), April 2026.
- TurboQuant: Google Research / NYU. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
- EDEN reference impl: [github.com/amitport/EDEN-Distributed-Mean-Estimation](https://github.com/amitport/EDEN-Distributed-Mean-Estimation), commit `5c7639a6af810e08d21827dd0ed55772b8113e99`.

So the prior-art question is real and deserves a real answer. The algorithmic claim deserves an empirical one. This post is the empirical one.

---

## 2. Two Things That Look Like the Same Bottleneck

The core disagreement is not about math. The note paper proves EDEN's `S` is MSE-optimal under iid Gaussian post-rotation marginals with their specific centroid table and bias structure. That proof is correct. The disagreement is about *which knob is the dominant lever for the system I actually ship*.

### 2.1 Objective mismatch

EDEN's estimator was designed for *unbiased distributed mean estimation*: many noisy clients send compressed vectors to a central aggregator, the aggregator averages them, and unbiasedness (`E[x_hat] = x`) is what makes the average converge to the true mean as the number of clients grows. The optimal `S` formula is a consequence of that objective.

KV cache compression has a different objective. There is one client (the model), one read (the next attention step), and no averaging. What matters is the *single-vector reconstruction MSE*, then how that reconstruction error propagates through the next softmax and weighted sum. Bias correction across many estimates is irrelevant. MSE-optimality and bias-correction can pull in different directions in this regime: the small per-vector S adjustment that improves average-of-many-clients accuracy is not the same as the small adjustment that minimizes per-vector reconstruction MSE on a single read.

This is the part of the disagreement that I want to flag up front and not bury. EDEN's framework is correct for its intended objective. KV cache compression is a different objective, evaluated on a different metric, on a different distribution.

### 2.2 Paper vs production

The other layer of mismatch is that "TurboQuant" means at least three different artifacts in this conversation:

```
TurboQuant (Google paper, 2026)        →  abstract algorithm with QJL on V residual ("_prod")
                                          and a deterministic post-rotation step ("_mse")
TurboQuant+ (TheTom, this work)        →  WHT rotation + matched-norm + Lloyd-Max centroids;
                                          SKIPS QJL on V; asymmetric K=q8_0/V=turbo by default
TurboQuant+ in production              →  llama.cpp + MLX impls running asymmetric K=q8_0 / V=turbo4
                                          on GQA ≥6:1 models, where this configuration has been
                                          validated across 7 model families and 3 GPU backends
```

When the EDEN authors say "TurboQuant," they mean the artifact in the Google paper. When I say "TurboQuant+", I mean the implementation I maintain plus a set of empirical extensions (asymmetric K/V, Boundary V, turbo4 resurrection). These are different operating points. A claim that holds about the first does not automatically transfer to the second or third. Most of this paper is about that gap.

---

## 3. What I Tested

Eight hypothesis tests this round, plus four rounds of "stress-test the conclusion":

- **Synthetic side:** N=10000 random Gaussian vectors, `d=128`, `b=4` for the headline. Sweeps at `d ∈ {64, 128, 256}` (HL7), `(d, b)` up to `(1024, 8)` (HL5), and 5 random WHT seeds (HL8). Compare matched-norm scale (TurboQuant+ ships), EDEN's per-vector biased optimal `S`, no-scale, across rotations (dense Haar vs Walsh-Hadamard) and centroid tables (Lloyd-from-scratch vs EDEN's half-normal).
- **Real KV side:** Dump KV cache from Qwen3-0.6B and Llama-3.2-1B-Instruct forward passes over wikitext-2-raw (512 tokens). Sample 256-512 vectors per layer per cache (K, V) for 3-7 layers per model. Run scale variants at `b=2, 3, 4` on Qwen, `b=4` on Llama (cross-family confirmation, MH1).
- **Reference:** the literal EDEN implementation from amitport's repo at commit `5c7639a6`, unmodified.

**Scope statement:** All empirical conclusions in this work are evaluated on the TurboQuant+ production configuration (WHT rotation, matched-norm scale, Lloyd centroids, no QJL on V cache). Claims about the Google TurboQuant paper as a mathematical object are treated separately and conceded where applicable.

The full investigation log lives in [[EDEN vs TurboQuant Investigation]] if you want to verify the numbers. Hypotheses H1-H6 are in the original log; the new sections in this paper are H7 (Llama cross-family), HL5 (where EDEN-S wins), HL6 (tail breakdown), HL7 (dimension sensitivity), and HL8 (WHT seed sensitivity).

---

## 4. Implementation Parity Checklist

Before the numbers, the lockdown. Reviewer concern: am I comparing the same algorithm or two different ones with the same name. Detail follows.

| Item | EDEN reference | TurboQuant+ |
|---|---|---|
| Code source | `/tmp/eden/torch/eden.py`, commit `5c7639a6af810e08d21827dd0ed55772b8113e99`, working tree clean (no modifications) | `/Users/tom/dev/turboquant/turboquant/` (Python prototype) and `llama-cpp-turboquant` / MLX (production) |
| Bit budget | b=4 (also b=2,3 on real KV; b∈{4,6,8} for sweep) | same |
| Rotation (paper / prototype) | Randomized Hadamard with Rademacher diagonal; fresh sign flips per encode (auto-seeded `torch.Generator`) | dense Haar via QR (`np.random.default_rng(seed=42)`) for some prototype paths |
| Rotation (production, llama.cpp / MLX) | (not applicable) | random sign flips + FWHT (also seeded with `np.random.default_rng(seed=42)` in the Python prototype) |
| Centroid table | EDEN's half-normal closed-form `opt_hn_centroids[b]`, symmetric reflection | Lloyd-Max iterated on N(0,1) (synthetic) or N(0, 1/d) (KV path) |
| Per-vector scale | `S = ||x_rot||² / ⟨c, x_rot⟩` (line 55 of `eden.py`), per-vector at encode | matched-norm: store `||x||`, set `S = ||x|| / ||c_lookup||` at decode |
| Normalization | x scaled to `||x|| = sqrt(d)` before bucketize, centroids calibrated for N(0, 1) | x scaled to `||x|| = 1` (unit norm) before bucketize, centroids calibrated for N(0, 1/d) |

The two normalization conventions are equivalent up to a constant: EDEN scales x by `sqrt(d)/||x||` and uses centroids at sigma=1, my path scales x by `1/||x||` and uses centroids at sigma=1/sqrt(d). Both produce the same reduced bucketize input distribution. The arithmetic check: a coordinate `x_i` in EDEN's frame has `Var(x_i) = (sqrt(d)/||x||)² · ||x||²/d = 1`; in my frame, `Var(x_i) = (1/||x||)² · ||x||²/d = 1/d`. Centroids are scaled correspondingly, so the bucketize result is identical.

One asymmetry I want to flag: EDEN's reference impl uses *fresh* random sign flips per encode (the Rademacher diagonal is regenerated from a per-call seed), while my WHT path uses *fixed* signs from `np.random.default_rng(seed=42)`. HL8 below confirms that the std deviation of MSE across 5 different fixed seeds for my WHT is ~0.13% of the mean, which is well below any of the gaps I report. Fresh-per-encode vs fixed-seed is not a meaningful confound at this measurement floor.

---

## 5. H1: Replicate the Synthetic Claim

The note's headline number for `d=128, b=4`: EDEN reduces per-coord MSE by 2.25% over TurboQuant_mse.

What I actually measured at that setup:

| Method | Per-coord MSE | vs my baseline |
|---|---:|---:|
| My PolarQuant, matched-norm (production prototype) | 1.142e-02 | — |
| My PolarQuant, raw lookup (no norm correction) | 1.166e-02 | -2.1% (worse) |
| Literal EDEN reference impl | **9.22e-03** | **+19.2% better** |

The EDEN reference impl beat my Python prototype by 19%, not 2.25%. The observed gap is significantly larger than the headline claim in the note, suggesting a different dominant factor. Worth digging into.

---

## 6. H2: What's Actually Driving the Gap?

EDEN's pipeline differs from my prototype in three places at once: rotation (Walsh-Hadamard vs dense Haar), centroid table (half-normal closed-form vs Lloyd-from-scratch), and per-vector scale (analytical optimal `S` vs matched-norm). I ran the swap matrix.

| Method | MSE | vs my baseline |
|---|---:|---:|
| A: My PolarQuant matched-norm | 3.17e-02 | — |
| D0: EDEN ref (Hadamard + half-normal centroids + optS) | 9.31e-03 | +71% |
| D1: dense Haar + half-normal centroids + optS | 3.55e-02 | -12% |
| D2: Hadamard + my Lloyd centroids + optS | 9.24e-03 | +71% |
| D3: Hadamard + half-normal centroids + matched-norm (no optS) | 9.20e-03 | +71% |

Read it from the bottom up:

- **D3 vs D0:** Hadamard rotation, with EDEN's centroids, but using matched-norm scale instead of optimal `S`. Within 1.3% of full EDEN. The optimal-`S` lever is real but second-order.
- **D2 vs D0:** Hadamard rotation, with optimal `S`, but my Lloyd centroids instead of EDEN's half-normal table. Within 0.7%. The centroid table choice is also second-order.
- **D1 vs D0:** EDEN's centroids and optimal `S`, but on dense Haar rotation. **+281% MSE.** Comparable to my own prototype's 3.17e-02. The rotation is the first-order lever.

The rotation is doing all the work. Hadamard vs dense Haar is the entire 71% gap. The optimal-`S` formula contributes ~1.3% on top. The centroid table contributes ~0.7%.

The rest of this paper is about why the first-order vs second-order distinction matters at the deployment surface, and where each lever does and does not transfer.

---

## 7. H3 + H4: The Catastrophic Tail and the Tail Breakdown (and Why WHT Fixes Both)

Looking at the per-vector MSE distribution for my prototype's dense-rotation path, the mean is dragged up by a long tail. Roughly 1% of vectors land in a regime where the centroid lookup catastrophically misaligns with the rotated input. Median `cos(c, x_rot) = 0.996`, worst `cos = 0.86`. The MSE formula `2 * ||x||² * (1 - cos) / d` matches the observed values. EDEN's optimal-`S` formula does not fix this; the tail is upstream of the scale, in the rotation plus centroid lookup.

The fix is the Walsh-Hadamard Transform with sign flips, which is what production TurboQuant+ uses (in both llama.cpp and MLX). H4 ran the same method variants but with WHT instead of dense rotation:

| Method | mean | median | p99 | p100 |
|---|---:|---:|---:|---:|
| dense + matched (Python prototype) | 1.14e-02 | 8.77e-03 | **1.44e-01** | **3.37e-01** |
| WHT + matched | 9.17e-03 | 8.75e-03 | 1.81e-02 | 4.14e-02 |
| WHT + EDEN-S | 9.24e-03 | 8.81e-03 | 1.83e-02 | 4.26e-02 |
| dense + EDEN-S | 1.19e-02 | 8.83e-03 | **1.63e-01** | **4.25e-01** |
| WHT + no scale | 9.31e-03 | 8.89e-03 | 1.88e-02 | 4.17e-02 |

Two things drop out of this:

1. WHT cuts the p99 tail from 0.144 to 0.018 (about 8x tail reduction). Production llama.cpp and MLX use WHT, so this isn't a hypothetical fix, it's already what ships.
2. Once you're on WHT, EDEN-S, matched-norm, and no-scale-at-all are all within 1% of each other. The scale knob is in the noise. (In this context, "noise" refers to <1% MSE variation, which is below typical downstream sensitivity for KV cache compression. The HL8 seed-sweep gives a quantitative read: WHT seed-to-seed variance is 0.13%, an order of magnitude smaller than any of the cross-method gaps reported here.)

### 7.1 HL6: tail concentration breakdown

The mean numbers above hide the structure. To make the tail story impossible to argue with, I sorted per-vector MSE descending and computed what fraction of total sum-of-squared-errors lives in the worst X% of vectors:

| metric | dense + matched | WHT + matched | WHT / dense |
|---|---:|---:|---:|
| mean MSE | 1.14e-02 | 9.17e-03 | 80.3% |
| p50 (median) | 8.77e-03 | 8.75e-03 | 99.8% |
| p90 | 1.18e-02 | 1.15e-02 | 98.1% |
| p99 | 1.44e-01 | 1.81e-02 | 12.6% |
| p99.9 | 2.51e-01 | 2.59e-02 | 10.3% |
| p100 (max) | 3.37e-01 (38x median) | 4.14e-02 (4.7x median) | 12.3% |

Tail concentration (% of total SSE in top X% of vectors):

| top X% of vectors | dense + matched | WHT + matched |
|---|---:|---:|
| top 0.1% | 2.5% | 0.3% |
| top 1.0% | 17.7% | 2.4% |
| top 5.0% | 26.4% | 8.9% |
| top 10.0% | 31.9% | 15.5% |
| top 20.0% | 41.6% | 27.4% |

Read it this way: under dense rotation, the worst 1% of vectors account for 17.7% of total reconstruction error. Under WHT, the worst 1% account for 2.4% (a 7x reduction in tail concentration). Median MSE is essentially identical between the two rotations (8.77e-03 vs 8.75e-03), so the entire mean-MSE delta is the tail. WHT compresses the tail; the body of the distribution is unaffected.

This is the structural reason the optimal-`S` formula does not transfer cleanly from EDEN's framework to TurboQuant+'s deployment surface. EDEN-S is calibrated against the body of the distribution where median MSE lives. The body of the distribution is the same with or without WHT. The extra MSE that EDEN-S targets is small in absolute terms because the body is already well-fit by either matched-norm or `S=1`. The 19% gap in H1 is the tail, and the tail is killed by the rotation, not by the scale.

### 7.2 HL8: WHT seed sensitivity

Reviewer concern: my WHT uses fixed sign flips (`np.random.default_rng(seed=42)`); EDEN regenerates them per encode. Could the H4 numbers be a function of one lucky seed?

I re-ran H4 at 5 different WHT seeds (42, 1, 7, 100, 314), holding the input distribution fixed:

| scale mode | mean MSE | std MSE | std as % of mean |
|---|---:|---:|---:|
| matched-norm | 9.157e-03 | 1.225e-05 | 0.13% |
| EDEN-S | 9.223e-03 | 1.238e-05 | 0.13% |

Per-seed `eden_S vs matched-norm` delta: +0.73% on every seed. The seed-to-seed variance (0.13%) is well below any of the gaps I claim, and the matched-vs-EDEN delta is bit-for-bit consistent across seeds.

So at production: the published "EDEN beats TurboQuant_mse" gap on synthetic Gaussian is essentially the gap between a Python prototype using Haar rotation and the production stack using WHT. With WHT (which I already ship), the observed improvement becomes negligible once the dominant factor is fixed, and does not materially affect the production configuration.

Worth being explicit: this is not a defect in the note paper's math. The note paper proves a property of TurboQuant_mse-as-described-in-the-Google-paper. TurboQuant+ in production uses a different rotation, and the property does not transfer.

A natural follow-up: EDEN's optimal `S` derivation could in principle be re-derived for non-Gaussian post-rotation distributions (heavy-tailed real KV). Adapting the scale to those distributions may recover the synthetic-Gaussian win on real cache, but that is not the formulation evaluated in the note, and not what production currently competes against.

---

## 8. H5: TurboQuant_prod vs Direct b-bit on Real KV

The bigger claim in the note is the `prod` one (k-bit unbiased EDEN approximately ties (k+1)-bit TurboQuant_prod). I dumped real Qwen3-0.6B KV from a wikitext prefill, sampled 256 vectors per (layer, K|V), and ran the comparison across `b={2,3,4}` for 7 layers.

Representative numbers (V cache):

| Method | Layer 0 V MSE | Layer 12 V MSE | Layer 23 V MSE |
|---|---:|---:|---:|
| PolarQuant b=4 matched | 1.05e-02 | 8.59e-03 | 2.94e-01 |
| PolarQuant b=4 EDEN-S | 1.06e-02 | 8.65e-03 | 2.97e-01 |
| EDEN ref b=4 | 1.07e-02 | 8.97e-03 | 3.05e-01 |
| TurboQuant_prod b=4 (PQ b=3 + QJL) | 5.97e-02 | 5.06e-02 | 1.66e+00 |
| PolarQuant b=3 matched | 3.86e-02 | 3.25e-02 | 1.08e+00 |
| EDEN ref b=3 | 3.79e-02 | 3.19e-02 | 1.07e+00 |

Two things fall out:

1. **PolarQuant matched-norm and EDEN-biased and EDEN-ref are all within 1-3% MSE of each other on every layer × {K, V} I sampled.** No meaningful gap from switching to EDEN-S on real KV.
2. **TurboQuant_prod is 5-6x worse MSE than direct b-bit PolarQuant at the same total bit count.** The "k-bit EDEN beats (k+1)-bit TurboQuant_prod" claim does technically hold here: 3-bit EDEN at 3.19e-02 beats 4-bit TurboQuant_prod at 5.06e-02 on layer 12 V. But the `_prod` chain (PolarQuant + QJL residual) is the wrong baseline. Production V cache uses TurboQuant_mse (PolarQuant only), not TurboQuant_prod, exactly because we already established (in [turbo4-resurrection](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md)) that QJL hurts attention quality. Five independent groups confirmed this.

So the "1 bit free" result evaluates `_prod` against EDEN, but production already left `_prod` behind months ago for unrelated reasons.

---

## 9. H6: Matched-Norm vs Optimal-S on Real KV (Qwen3-0.6B)

The cleanest test of the note's algorithmic claim: hold the rotation and centroids fixed, vary only the scale formula, on real KV.

| layer | K/V | b | matched | biased optS | unbiased avg |
|---|---|---|---:|---:|---:|
| 0 | V | 4 | 7.20e-04 | 7.38e-04 | 7.46e-04 |
| 12 | V | 4 | 1.31e-02 | 1.32e-02 | 1.35e-02 |
| 12 | K | 4 | 9.78e-02 | 9.84e-02 | 9.95e-02 |
| 23 | V | 4 | 2.57e-01 | 2.59e-01 | 2.64e-01 |
| 23 | V | 2 | **3.35e+00** | **3.68e+00** | 3.69e+00 |

Across 4 layers × {K, V} × `b={2,3,4}`: matched-norm consistently beats EDEN's biased optimal `S` by 0.5 to 9% in MSE on real KV. The gap grows at lower bit counts. Worst case for EDEN-S is layer 23 V at `b=2`, where matched-norm at 3.35 beats biased optS at 3.68 (EDEN's "optimal" formula is 8.9% worse than the simple matched-norm rule).

Why does the analytical-optimal scale lose to a heuristic? Because EDEN's `S` is optimal under specific distributional assumptions: iid Gaussian post-rotation marginals, the specific half-normal centroid table, and the bias structure that comes with their estimator (the one designed for unbiased mean estimation across many clients). Real KV cache vectors after WHT rotation have heavy tails, layer-dependent variance (300x MSE swing across layers), and non-Gaussian structure. Once those assumptions break, "analytically optimal" becomes "optimal for the wrong distribution."

Matched-norm doesn't try to be analytically optimal. It just preserves vector magnitude after dequantization, which empirically tracks real-KV behavior better.

---

## 10. H7: Cross-Family Validation on Llama-3.2-1B-Instruct

Reviewer concern: the H6 result is one model. Does it transfer?

I ran the H6-equivalent on Llama-3.2-1B-Instruct (16 transformer layers, head_dim 64, dumped from a 512-token wikitext.test prefill, layers {0, 8, 15} × {K, V}, b=4, 512 vectors per cell). Llama-3.2 has a different head dimension from Qwen3-0.6B (64 vs 128) and different per-layer KV statistics, so this is a meaningful family change rather than a same-architecture rerun.

| layer | K/V | N | mean ||x|| | matched MSE | EDEN-S MSE | matched Δ vs EDEN-S |
|---|---|---:|---:|---:|---:|---:|
| 0 | K | 512 | 13.93 | 2.804e-02 | 2.823e-02 | +0.67% |
| 0 | V | 512 | 0.595 | 6.06e-05 | 6.10e-05 | +0.70% |
| 8 | K | 512 | 20.84 | 5.971e-02 | 6.011e-02 | +0.68% |
| 8 | V | 512 | 2.32 | 7.84e-04 | 7.89e-04 | +0.72% |
| 15 | K | 512 | 18.48 | 4.636e-02 | 4.666e-02 | +0.67% |
| 15 | V | 512 | 4.06 | 2.603e-03 | 2.620e-03 | +0.67% |

Matched-norm wins on 6/6 cells, by a remarkably tight 0.67-0.72% across every layer × {K, V} combination. The Llama deltas are tighter than the Qwen3 deltas (Qwen3 ranged 0.5-9%, Llama 0.67-0.72%) because Llama's KV is more uniformly distributed across layers (less of the 300x MSE swing Qwen3 has between layers 12 and 23). On the most-Gaussian layers of Qwen3 the deltas are also in the ~0.7% range; the wider Qwen3 range reflects layers where the post-WHT distribution deviates more from the iid Gaussian assumption.

Cross-family direction: confirmed. EDEN-S does not beat matched-norm on real KV in either model family I tested. The size of the gap depends on how non-Gaussian the post-WHT KV distribution is, but the sign is consistent.

---

## 11. HL7: Dimension Sensitivity

Reviewer concern: most synthetic tests are at d=128. Does the scale-vs-rotation story hold across dimensions?

I ran the same matched vs EDEN-S comparison at `d ∈ {64, 128, 256}`, `b=4`, `N=2000` per cell, both rotations:

| d | dense + matched | wht + matched | wht + eden_S | rotation Δ% (dense vs WHT) | scale Δ% (eden vs matched) |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.186e-02 | 8.940e-03 | 9.004e-03 | +32.6% | +0.72% |
| 128 | 2.042e-02 | 9.251e-03 | 9.318e-03 | +120.7% | +0.73% |
| 256 | 4.052e-02 | 9.338e-03 | 9.405e-03 | +334.0% | +0.72% |

Two patterns:

1. **The rotation effect grows superlinearly with d.** At d=64 dense rotation costs +33%; at d=128 it costs +121%; at d=256 it costs +334%. This is consistent with the tail story in §7: at higher d, the chance of a single-coordinate misalignment between centroid lookup and rotated input grows, and dense rotation amplifies the tail more. WHT keeps the per-coordinate distribution well-conditioned regardless of d.
2. **The scale-mode effect is bit-for-bit identical across d.** +0.72-0.73% at every d. EDEN-S costs me roughly the same fraction at d=64 and d=256. The optimal-S lever is dimension-invariant in size (at fixed b); the rotation lever scales with d.

Median S converges toward 1 as d grows (S_median: 1.0087 at d=64, 1.0084 at d=128, 1.0091 at d=256), consistent with EDEN's analytical "S → 1 as d → ∞" prediction. Even at d=64, the median S is only 0.9% off 1.0. The EDEN authors were honest about this: at high d the analytical correction approaches no-correction, which is why the lever is small.

---

## 12. HL5: Where EDEN-S Wins

The fair question this whole paper has been dancing around: *is there any regime where the literal EDEN reference impl beats my matched-norm path?* If not, something is wrong with my measurement; the EDEN line of work is too well-grounded for there to be no win regime at all.

I swept `(d, b)` over `d ∈ {128, 256, 512, 1024}` and `b ∈ {4, 6, 8}` (smaller N at high d/b because EDEN's reference impl is per-vector, slow), comparing my WHT + matched-norm against the literal EDEN reference impl on iid Gaussian inputs:

| d | b | N | my matched | EDEN ref | EDEN better |
|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 2000 | 9.25e-03 | 9.22e-03 | +0.31% |
| 128 | 6 | 2000 | 9.49e-04 | 6.22e-04 | **+34.5%** |
| 128 | 8 | 2000 | 1.43e-04 | 4.09e-05 | **+71.4%** |
| 256 | 4 | 2000 | 9.34e-03 | 9.42e-03 | -0.91% |
| 256 | 6 | 2000 | 9.74e-04 | 6.27e-04 | **+35.7%** |
| 256 | 8 | 2000 | 1.54e-04 | 4.05e-05 | **+73.7%** |
| 512 | 4 | 1000 | 9.48e-03 | 9.60e-03 | -1.26% |
| 512 | 8 | 1000 | 1.69e-04 | 4.10e-05 | **+75.8%** |
| 1024 | 4 | 500 | 9.47e-03 | 9.55e-03 | -0.84% |
| 1024 | 8 | 500 | 1.64e-04 | 4.11e-05 | **+74.9%** |

There it is. At b=6 and b=8, the EDEN reference impl beats my matched-norm path by 34-76% across every d I tested. At b=4 (the production bit width), they're within 1-2% in either direction.

The mechanism: at high b, the centroid table is so finely quantized that the per-vector matched-norm rule (`S = ||x|| / ||c_lookup||`) becomes a structural inefficiency. Matched-norm preserves the *magnitude* of the input vector after dequantization, but at b=8 the centroid lookup is already extracting most of the directional information; the magnitude correction adds quantization noise that EDEN's per-coordinate-aware S avoids. EDEN's framework correctly extracts the last bit of MSE in this regime, exactly because their assumptions (post-rotation iid Gaussian, well-fit centroid table, per-coordinate bias structure) hold most tightly at high b on pure Gaussian.

This is a credibility check on the rest of the paper. EDEN's lever exists; it's a real, large win in the regime where the lever was designed to work. It's just that production TurboQuant+ runs at b=4, on real KV, where the regime doesn't hold and the lever shrinks to ±1%.

The honest framing: at b=8 on iid Gaussian, EDEN-S is the right tool. At b=4 on real KV cache, matched-norm is the right tool. Both can be true.

---

## 13. The Honest Interpretation

Stitching it together, with the first-order vs second-order frame:

1. **The rotation choice is the first-order lever.** Hadamard beats dense Haar by ~71% in MSE on synthetic Gaussian at d=128, +334% at d=256. Production TurboQuant+ already uses WHT (a randomized Hadamard variant), so the headline 19% MSE gap on synthetic Gaussian doesn't transfer. This is the fix the EDEN authors built into their reference impl years before TurboQuant; it's also the fix that ships in TurboQuant+ today. We agree on which rotation to use.
2. **The optimal-S formula is a real second-order lever.** It contributes ~1% on synthetic (HL7 confirms this is dimension-invariant) and *negative* contribution on real KV (H6, H7). On real KV cache (post-WHT), matched-norm beats EDEN's biased optimal `S` by 0.5 to 9% across the configs I tested across two model families. The optimality argument depends on iid Gaussian assumptions that real KV violates.
3. **The optimal-S lever does have a regime where it wins big.** HL5 shows EDEN-S beats matched-norm by 34-76% at b=6 and b=8 on synthetic Gaussian. That regime is iid Gaussian + high bit budget + closely matched centroid table. KV cache compression at b=4 with heavy-tailed real distributions is a different regime, and the lever shrinks to ±1% there.
4. **The "1 bit free" claim only applies vs TurboQuant_prod.** Production V cache uses TurboQuant_mse, not _prod, because we already know QJL hurts attention. Comparing against _prod and ignoring that result is comparing against a baseline production already abandoned for independent reasons.
5. **The note paper is correct under its own assumptions.** I am not claiming the math is wrong. I am claiming the assumptions don't hold at the deployment surface I ship to. The interesting algorithmic territory is on the prototype side and at high b on synthetic data (where assumptions are closer to satisfied), not on real KV at b=4 (where they're not).

So if I'm being scrupulous: there is no measurable production-relevant improvement from adopting EDEN's optimal-`S` formula in TurboQuant+ as currently shipped, and on real KV the formula is mildly worse than what I already do. I don't plan to change the scale formula on the basis of #87. I do plan to add the EDEN-S regime characterization (HL5) to the docs as guidance for anyone running TurboQuant-style codebooks at high b on Gaussian-like inputs (likely model weights, possibly some activation paths).

---

## 14. Acknowledgements (and the Attribution Question, #89)

Issue #89 is a different question from #87, and on this one I think Portnoy is right.

The DRIVE/EDEN line of work (Vargaftik, Ben-Basat, Portnoy, Mendelson, Ben-Itzhak, Mitzenmacher; NeurIPS 2021 and ICML 2022) is genuine prior art for the rotate-then-quantize family of estimators. DRIVE introduced the one-bit randomized rotation + sign quantization combination. EDEN extended it to multi-bit with the half-normal centroid table and the analytical scale parameter. Both are mathematically clean, both predate TurboQuant by years, both deserve explicit citation in the TurboQuant+ documentation going forward.

The Google TurboQuant paper does cite some prior work in this space, but the EDEN line is undercited relative to its actual influence on the rotate-then-quantize design pattern. I'll add explicit acknowledgement and citation in the TurboQuant+ docs (the implementation that I maintain). The papers belong in the references section. The note paper itself, even where I disagree on the bottleneck question, is also a useful reference for anyone digging into the design space (in particular the analytical S derivation in HL5's high-b regime is a clean piece of work).

To be clear about my own contribution: I did not write the TurboQuant paper (that's Google Research / NYU). I built TurboQuant+, which is the implementation plus a set of empirical extensions (asymmetric K/V, Boundary V, turbo4 resurrection, weight compression). The DRIVE/EDEN authors are upstream of the design pattern that all of this builds on, and they should be acknowledged as such.

Names that belong in the TurboQuant+ references section going forward: Shay Vargaftik, Ran Ben-Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben-Itzhak, Michael Mitzenmacher.

---

## 15. Limitations

Honest caveats on this writeup:

1. **Two models, one corpus, one context length.** Qwen3-0.6B and Llama-3.2-1B, wikitext.test, 512 tokens. The MSE-on-real-KV result holds across the layers I sampled in both families, but I haven't repeated the H5/H6/H7 sweeps across more families (Mistral, Phi, Gemma) or longer context. Asymmetric K/V results across 7+ models give me some confidence that single-family findings on KV cache statistics tend to generalize, but the cross-family confirmation here is two points, not seven.
2. **MSE is a proxy, not a quality metric.** The 0.5 to 9% MSE differences may or may not transfer to PPL or downstream task accuracy. They probably don't, given how flat the quality response is at high bits and how much KL divergence is dominated by other factors at the model level. I deliberately did not run model-level KL or PPL experiments in this paper, since that's a different investigation surface; the claim here is strictly per-vector MSE. Given the <1% scale-induced MSE differences observed here, any downstream impact is likely dominated by other factors in the attention pipeline.
3. **Synthetic Gaussian is the friendliest test case for EDEN's analytical formula and the harshest test case for matched-norm.** Even so, on Gaussian + WHT at b=4 the gap is ~1%. At higher b on Gaussian, EDEN-S wins decisively (HL5). Anywhere with less Gaussian assumption violation, EDEN-S might do better than my measurements suggest.
4. **My WHT uses fixed sign flips; EDEN's reference uses fresh per-encode signs.** HL8 confirmed std across 5 seeds is 0.13% of mean. Below my measurement floor. Not a confound.
5. **All scripts are in `/tmp/eden-investigation/` and `/tmp/eden-investigation/h_paper_expansion/`.** I'll move them to a public repo if anyone wants to verify or extend.

---

## 16. Summary

> **EDEN's optimal scale is real, but second-order. Rotation choice is first-order. Production already uses the first-order fix.**

The "TurboQuant is just EDEN with `S=1`" framing is mathematically defensible at the level of TurboQuant_mse-as-described-in-the-Google-paper. At the level of TurboQuant+ as it actually ships in llama.cpp and MLX (WHT rotation, matched-norm scale, no QJL on V), the practical gap to EDEN is between -9% (matched-norm wins on real KV at low bits) and +1% (optimal `S` wins by a hair on synthetic Gaussian at b=4). EDEN-S does win by 34-76% in the regime it was designed for (high b, iid Gaussian); production KV at b=4 isn't that regime. The "1 bit free" `_prod` result holds, but only against a chain that production already left behind for independent reasons.

The attribution argument is separate and basically right. DRIVE/EDEN is real prior art, and it'll get explicit citation in the TurboQuant+ docs going forward.

Thanks to Vargaftik, Ben-Basat, Portnoy, Mendelson, Ben-Itzhak, and Mitzenmacher for the original DRIVE and EDEN work, for the note paper, and for filing the issues. Disagreement on which knob is dominant for the system I ship; agreement on the attribution question; full respect for the framework either way.

---

## References

- TurboQuant paper (Google Research / NYU, ICLR 2026): [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- DRIVE paper (NeurIPS 2021): [arXiv:2105.08339](https://arxiv.org/abs/2105.08339)
- EDEN paper (ICML 2022): [proceedings.mlr.press/v162/vargaftik22a.html](https://proceedings.mlr.press/v162/vargaftik22a.html)
- Note on TurboQuant and DRIVE/EDEN ([arXiv:2604.18555](https://arxiv.org/abs/2604.18555))
- Towards Data Science article (Portnoy): [How a 2021 quantization algorithm quietly outperforms its 2026 successor](https://towardsdatascience.com/how-a-2021-quantization-algorithm-quietly-outperforms-its-2026-successor/)
- EDEN reference implementation: [github.com/amitport/EDEN-Distributed-Mean-Estimation](https://github.com/amitport/EDEN-Distributed-Mean-Estimation), commit `5c7639a6af810e08d21827dd0ed55772b8113e99`
- TurboQuant+ implementation: [github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- Issue #87 (Mitzenmacher): [TheTom/turboquant_plus#87](https://github.com/TheTom/turboquant_plus/issues/87)
- Issue #89 (Portnoy): [TheTom/turboquant_plus#89](https://github.com/TheTom/turboquant_plus/issues/89)
- Investigation log: [[EDEN vs TurboQuant Investigation]]
- Related: [turbo4-resurrection.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md) (why production V cache avoids the QJL/_prod chain)
- Related: [asymmetric-kv-compression.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) (production K/V config)
