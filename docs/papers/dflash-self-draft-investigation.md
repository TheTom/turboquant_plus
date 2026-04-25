# Speculative Decoding on GDN Hybrid Architectures: A 31-Experiment Investigation

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## Abstract

We investigate whether speculative decoding can be made to work dynamically on Qwen3.5/3.6 GatedDeltaNet (GDN) hybrid architectures — without pre-trained draft models, without external training infrastructure, and without user complexity. Over 31 experiments across 5 days, we systematically explored layer-skip self-drafting, linear aligners, MLP draft heads, Medusa-style parallel prediction, block diffusion, multi-layer target conditioning, and full z-lab recipe replication.

Key findings: (1) single-hidden-state conditioning caps at 36% acceptance regardless of model size or training duration; (2) multi-layer target features break this ceiling (13% → flat across 16 positions with KV injection); (3) GDN recurrent state creates a verification cost floor that limits theoretical speedup to ~1.33x on 75% GDN architectures; (4) tape-replay rollback (recording innovation deltas during verify, replaying only accepted steps) solves the Mamba state corruption problem we could not crack with snapshot/restore; (5) training DFlash drafts from scratch on consumer hardware (M5 Max) is not currently feasible — convergence requires 300K+ gradient steps achievable only on cluster-scale compute.

We document every experiment, including the 28 that failed, as a reference for others working on speculative decoding for hybrid architectures.

---

## 1. Motivation

DFlash (z-lab, 2026) claims 6x lossless decode acceleration via block diffusion drafting, 2.5x faster than EAGLE-3. The technique requires a pre-trained draft model (~500M-1B params) per target model, downloaded from HuggingFace.

Our goal was to achieve similar speedups **dynamically** — runtime-only, no extra downloads, no training, no user complexity. This aligns with TurboQuant+'s philosophy: post-training optimizations that work on any model out of the box.

Target platform: Apple Silicon (M-series), running Qwen3.6-35B-A3B-4bit (40 layers: 30 GDN + 10 full attention).

---

## 2. Architecture Background

### 2.1 Qwen3.5/3.6 Hybrid Architecture

Qwen3.5/3.6 alternates two layer types:
- **GatedDeltaNet (GDN):** Recurrent (Mamba-style). Conv1D state + gated delta state. Sequential dependency: each token's state depends on the previous token's state.
- **Full attention:** Standard multi-head attention with KV cache. Parallel: N tokens can be verified in one matmul.

`full_attention_interval=4` → attention at layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39. The remaining 30 layers are GDN.

### 2.2 Implications for Speculative Decoding

During multi-token verification:
- Attention layers (25%): parallel. Cost of verifying N tokens ≈ cost of 1.
- GDN layers (75%): sequential. Cost of verifying N tokens = N × cost of 1.

Total verify cost: `0.25 + 0.75 × N` single-token equivalents.

Maximum theoretical speedup as N→∞ with 100% acceptance:

$$\text{speedup}_\text{max} = \frac{1}{0.25 + 0.75/N} \to 1.33\times$$

For comparison, pure-attention models: verify cost = 1 regardless of N. Maximum speedup = N×.

This ceiling is architectural, not algorithmic. No draft quality improvement changes it.

**Update (2026-04-25):** pupposandro & davideciffa demonstrated 3.43-5.46x on the same hybrid architecture using tree-aware GDN kernels (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree_persist`) that walk the DDTree structure through recurrent state directly, avoiding the sequential bottleneck. Our ceiling analysis assumed sequential GDN verification; tree-aware kernels may invalidate this constraint.

---

## 3. Experiment Log

### 3.1 Phase 1: Layer-Skip Self-Draft (Experiments 1-3)

**Hypothesis:** Use the target model's first K layers as a fast drafter. Same weights, fewer layers, cheaper per-token cost.

**Experiment 1: Stateless self-draft.**
Each draft token sees ONLY its own embedding through K layers. No KV cache history.

| Draft Layers | tok/s | Avg Accept | vs Baseline |
|:---:|---:|:---:|:---:|
| 10/40 (25%) | 113 | 3.1/4 | 1.13x |
| 20/40 (50%) | 64 | 2.5/4 | 0.64x |
| 30/40 (75%) | 52 | 2.5/4 | 0.52x |

Output is gibberish ("ofabah ofsworth..."). Acceptance is pure luck — the LM head expects layer 40 output space, not layer 10.

**Experiment 2: Cache-aware self-draft.**
Give the drafter its own KV cache. Draft tokens see full context through K layers.

| Exit Layer | Layers Skipped | Acceptance | Cost Savings |
|:---:|:---:|:---:|:---:|
| 39 (skip 1) | 1 | 71.8% | 1.8% |
| 38 (skip 2) | 2 | 34.9% | ~4% |
| 36 (skip 4) | 4 | 18.1% | ~8% |
| 32 (skip 8) | 8 | 3.4% | ~18% |

Two complementary cliffs: acceptance collapses from 72% to 3% over 8 layers (the LM head can't read intermediate states), while skipping 1 layer saves only 1.8% of compute. The savings × acceptance product never exceeds 1.0. Best theoretical case (exit@39, N=3, α=0.72): 0.80x = 20% slowdown.

**Experiment 3: Linear aligner.**
Fit W = lstsq(h_K, h_40) from prefill hidden states. 4M params, fits in ~100ms.

Calibration results: 7.5x improvement at skip-8 (2.1% → 15.9%). Decode results: 0% at all depths.

Root cause: **prefill/decode distribution shift.** During prefill, all positions attend bidirectionally. During decode, each position sees only left-context via KV cache. The aligner learned a mapping that doesn't exist in decode space.

### 3.2 Phase 2: Trained Draft Heads (Experiments 4-6)

**Experiment 4: Prefill-trained MLP.**
2M param MLP trained on (h_t, next_token) pairs from prefill. Training accuracy: 100% at 5 epochs. Decode acceptance: 1%. Same distribution shift as Experiment 3.

**Experiment 5: Decode-path MLP (small scale).**
Train on actual decode hidden states (autoregressive generation with KV cache). 2K samples, 1M params, 10 epochs: 5-7% acceptance. Better than prefill-trained but too small.

**Experiment 6: Decode-path MLP (full scale).**
100K decode-path tokens, 59M params, 100 epochs (19,600 gradient steps). Training: 17 min data collection + 40 min training.

| Prompt Type | Acceptance |
|---|:---:|
| Philosophy | 42.0% |
| Code (B-tree C++) | 38.0% |
| Science (nuclear fusion) | 32.7% |
| Creative (French onion soup) | 33.3% |
| Analysis (democracy vs auth) | 35.3% |
| **Overall** | **36.3%** |

**Scaling trajectory:**

| Scale | Data | Params | Epochs | Steps | Acceptance |
|---|---:|---:|---:|---:|:---:|
| Tiny | 2K | 1M | 10 | ~40 | 5% |
| Small | 10K | 17M | 20 | ~400 | 13% |
| Medium | 50K | 17M | 20 | ~2K | 15% |
| Large | 20K | 59M | 15 | ~600 | 17% |
| **Full** | **100K** | **59M** | **100** | **19,600** | **36%** |

Both data AND training duration matter. This is the single-hidden-state ceiling: ~36% at scale with h_t conditioning only.

### 3.3 Phase 3: End-to-End Integration (Experiments 7-8)

**Experiment 7: Speculative decode E2E.**

| Version | Approach | Speed | Output |
|---|---|:---:|---|
| v1 | No cache fix | 1.68x | Garbage (corrupted Mamba) |
| v2 | Snapshot + replay accepted | 0.49x | N/A (replay too expensive) |
| v3 | Snapshot + restore + refresh | 0.54x | Diverges at position 2 |

The GDN corruption problem: on partial rejection, target Mamba state includes rejected tokens' contributions. Snapshot/restore loses accepted tokens' contributions. Replay costs ≥ 2 full forwards per round.

**Experiment 8: Separate draft model.**
21.5M param draft with its own GDN-like recurrent layers. Own cache, independent from target.

Result: still garbage. The issue is the TARGET's Mamba during verification, not the draft's cache. On partial rejection: target Mamba restored → accepted tokens' contributions lost → KV/Mamba inconsistency → degeneration.

The math that makes DFlash work despite this: at 85% acceptance, P(all 8 match) = 27%. Those rounds yield 9 tokens / 1 forward = huge win. At our 36%: P(all 4 match) = 1.7%. Too rare. **Block diffusion's 85% acceptance is required, not optional, for GDN spec decode economics.**

### 3.4 Phase 4: Parallel Prediction (Experiments 9-10)

**Experiment 9: Medusa parallel heads.**
8 independent prediction heads sharing one backbone, each predicting a different future position from h_t. 46M params, 20K samples, 30 epochs.

Per-position acceptance: 2.3-4.7% across all positions. P(all N match): 0.0% for N≥2.

**Experiment 10: Block diffusion (self-attention, no target conditioning).**
Same as Medusa but with bidirectional self-attention across positions. 234.9M params, 4 RefineBlock layers, 3 refinement steps. 20K tokens, 30 epochs.

Per-position acceptance: 0.8-2.8%. Self-attention didn't help.

Root cause: predictions from a single hidden state are independently ~2-5%. Inter-position self-attention can't help when initial predictions are all wrong — refining garbage against garbage produces garbage. Without conditioning on target intermediate states, parallel prediction is fundamentally limited.

### 3.5 Phase 5: Multi-Layer Conditioning (Experiments 11-16)

**Experiment 11: Multi-layer features.**
Extract hidden states from 7 target layers [0, 1, 10, 20, 30, 38, 39] (boundary-weighted per TurboQuant+ layer sensitivity findings). Cross-attention to target features + self-attention across positions.

| Position | v2 (single h_t) | v3 (multi-layer) | DFlash (reference) |
|:---:|:---:|:---:|:---:|
| t+1 | 2.8% | **13.0%** | ~88% |
| t+2 | 2.2% | **5.7%** | ~88% |
| t+3+ | 1-2% | 1-2% | ~88% |

Multi-layer features provide 5x improvement at t+1. Signal decays sharply with distance. Right direction, undertrained and undersized.

**Experiment 12: z-lab architecture match (v4A).**
5 target conditioning layers, GQA attention, single-pass. 406M params, 20K tokens. Result: 2-4% all positions. Undertrained.

**Experiment 13: 100K data scale (v4B).**
Same architecture, 100K tokens, cosine LR. Result: 3-4% flat. Data scale is NOT the bottleneck at this architecture size.

**Experiment 14: KL distillation loss (v4C).**
Top-64 logit distillation with T=2. Result: 1-2%. WORSE. Soft loss dilutes the gradient. The model needs sharp token-level feedback, not fuzzy probability matching.

**Experiment 15: Paper-faithful architecture (v5).**
Matched DFlash's actual architecture: KV injection into draft K/V projections, exponential position weighting (γ=4), anchor sampling (512 per sequence).

Result: **13-15% FLAT across all 16 positions.** The architecture works — flat acceptance at every position, not just t+1.

But 15% per-position ≠ 15% consecutive: P(all 4 match at 15%) = 0.05%.

**Experiment 16: v5 E2E speed test.**
Result: 0% consecutive acceptance, 0.30x effective speed. The draft cost is pure overhead at this acceptance level.

### 3.6 Phase 6: Pre-Trained Drafts (Experiments 17-18)

**Experiment 17: z-lab 3.5 draft on 3.6 target.**
67% acceptance, 0.69x = 31% SLOWER. Draft doesn't transfer across model versions.

**Experiment 18: z-lab 3.6 draft (WIP, 2000 training steps).**
75% acceptance, 0.92x = 8% slower. Even z-lab hadn't cracked 3.6 at time of testing.

### 3.7 Phase 7: Training Interventions (Experiments 19-27)

**Experiment 19: Scheduled sampling (v6).**
Teacher forcing ratio decays 1.0→0.3. Result: 2-4% flat, 0.03 consecutive. WORSE than v5. Model becomes overly conservative when forced to condition on its own bad predictions during training.

**Experiments 20-24: Ablation sweep.**

| Exp | Change | t+1 Accept | vs v5 Baseline |
|:---:|---|:---:|---|
| 20 | 8 draft layers (more capacity) | 2.3% | Worse (overfits) |
| 21 | 8 target features (more conditioning) | 2.7% | Worse (overfits) |
| 22 | 8L + 8F combined | 2.7% | Worse |
| 23 | 60 epochs (more training) | 2.5% | Saturated |
| 24 | Diverse test prompts | 1.6% | Worst (overfits to style) |

At 20K training tokens, adding capacity or features worsens performance (overfitting). More epochs saturate. The bottleneck is training data scale AND gradient steps, not architecture.

**Experiments 25-27: Diagnostics.**
Top-k oracle (is correct token in top-5/10?), overfit single batch (can architecture memorize 256 tokens?), scale plan for v5. Experiment 27 superseded by Experiment 28.

### 3.8 Phase 8: Full-Scale Replication (Experiments 28-31)

**Experiments 28-29: Full z-lab recipe.**
800K target-generated tokens (117 min collection at 114 tok/s). 474M param draft, 6 epochs.

Run 1: 0.4% acceptance. Masking bug — training fed all 16 ground truth tokens as noise input. Draft learned to copy input, not predict.

Run 2 (mask fix): 7.1% acceptance. Healthy loss curve (27.0 → 0.11), generalizing but undertrained.

**Experiment 30: MTP layer as zero-training drafter.**
Qwen3.6-35B-A3B ships with a built-in MTP (multi-token prediction) layer: 1 full MoE transformer layer, 475MB at 4-bit. spiritbuun reports 80%+ acceptance using this on Qwen3.5 with no training.

We extracted MTP weights manually (mlx_lm converter strips them). Built forward pass by deep-copying a real model layer. Result: 0%. Forward pass bugs: expert weight format mismatch, missing chat template, RoPE offset handling (MTP attention needs target's KV cache offset, not fresh cache).

Status: parked with forward pass bugs. MTP covers ~20-30% of popular models (Qwen, DeepSeek). For those models, 80%+ acceptance with zero training is the best deal in speculative decoding.

**Experiment 31: bstnxbt model class + full scale.**
Used bstnxbt's actual `DFlashDraftModel` class (gated Q attention). 474M params, 800K tokens, 12 epochs, 12,624 gradient steps. Loss: 29.3 → 0.008 (clean halving every epoch).

Acceptance: 0.6%. WORSE than v5 (13-15%).

Root cause: mode collapse to marginal distribution. Draft predicts `" a" " thinking" " process" ":"` for every prompt — the most common Qwen3.6 thinking prefix. At 134K samples, predicting common tokens gives lower loss than learning the feature-to-token mapping.

Architecture validation: overfit test on 20 samples → 98% accuracy. Fine-tune from z-lab weights → 100% in 50 steps. The architecture works. Training dynamics at scale cause mode collapse. z-lab solves this with full sequence context (Flex Attention, persistent draft cache accumulation). Our single-position features don't provide enough conditioning signal.

---

## 4. The Tape-Replay Solution

Analysis of bstnxbt/dflash-mlx revealed the solution to our Experiment 7-8 Mamba corruption problem: **tape-replay rollback.**

During speculative verify, record an "innovation tape" per GDN timestep:
- `tape[t]`: innovation delta δ — shape [B, T, Hv, Dv]
- `tape_k[t]`: keys — shape [B, T, Hk, Dk]
- `tape_g[t]`: decay gates — [B, T, Hv] or [B, T, Hv, Dk]
- `tape_qkv[t]`: raw embeddings for conv state

On partial acceptance (accept first n of N):
1. Restore snapshot to pre-verify state
2. Replay only accepted steps: `state = state * g[t] + k[t] * tape[t]` for t = 0..n-1
3. Conv state rebuilt from last (conv_kernel_size - 1) tokens of qkv tape

Cost: O(n_accepted), not O(context_length). One Metal kernel dispatch per Mamba layer.

We tried the extremes (ignore state → corruption; full replay → too expensive). Tape-replay is the middle ground: record the minimum information needed (δ, k, g), replay only accepted steps through a cheap recurrent kernel.

---

## 5. The z-lab Recipe (Reverse-Engineered)

From `z-lab/Qwen3.5-35B-A3B-DFlash` config.json and dflash.py:

| Parameter | Value |
|---|---|
| Draft params | ~500M (8 layers, 2048 hidden, GQA 32Q/4KV) |
| Target layers | 5: [1, 10, 19, 28, 37] (uniform) |
| Block size | 16 tokens per cycle |
| Training data | ~800K samples, **target-generated** |
| Optimizer | AdamW, lr=6e-4, cosine, 0.04 warmup |
| Epochs | 6 (~300K gradient steps) |
| Loss | CE with exponential decay: w_k = exp(-(k-1)/γ), γ=7 |
| Shared with target | Embedding + LM head (frozen) |
| Draft attention | Bidirectional within block (no causal mask) |
| KV injection | Target features projected into draft K/V, concatenated with draft's own K/V |

Key differentiators from our experiments:
- **500M params** vs our 438K-59M. Proper small transformer, not a head.
- **Target-generated training data.** Eliminates distribution shift (our Experiment 4 killer).
- **~300K gradient steps** vs our ~1K-20K. 15-300x more optimization.

---

## 6. Cross-References to TurboQuant+ Research

Findings from our KV cache and weight compression work that transfer to speculative decoding:

1. **K >> V for prediction quality** (asymmetric-kv-compression.md): K determines attention routing via softmax (exponential amplification). Draft models should prioritize K-space fidelity.

2. **Boundary layers carry disproportionate information** (layer-aware-v-compression.md): First/last 2 layers are 37-91% more sensitive. Multi-layer conditioning should oversample boundaries.

3. **Simpler > complex** (turbo4-resurrection.md): Removing QJL correction improved turbo4 quality. More refinement steps may not be better.

4. **90% attention sparsity at 32K** (sparse-v-dequant.md): Draft model's job gets easier at long context. Block diffusion should scale better than autoregressive drafting.

5. **Error amplification varies by family** (weight-compression-tq4.md): Llama propagates quantization error 6-8x more aggressively than Qwen. Draft accuracy will vary by model family.

---

## 7. Consolidated Results

| Exp | Architecture | Conditioning | Accept | Status |
|:---:|---|---|:---:|---|
| 1 | Stateless self-draft | None | 0% | Dead |
| 2 | Cache-aware self-draft | Own cache | 3-72% | Dead (cost cliff) |
| 3 | Linear aligner | lstsq(h_K, h_40) | 0% decode | Dead (distribution shift) |
| 4 | Prefill MLP | h_t (prefill) | 1% | Dead (distribution shift) |
| 5 | Decode MLP (2K) | h_t (decode) | 5-7% | Too small |
| 6 | Decode MLP (100K) | h_t (decode, scaled) | 36% | Single-h_t ceiling |
| 7 | Spec decode E2E | h_t + cache mgmt | N/A | Dead (Mamba corruption) |
| 8 | Separate GDN draft | Own recurrent model | N/A | Dead (target Mamba) |
| 9 | Medusa parallel | h_t, independent | 2-5% | Dead (no inter-position) |
| 10 | Block diffusion v2 | h_t, self-attention | 2-3% | Dead (insufficient signal) |
| 11 | Multi-layer v3 | 7 target layers | 13% t+1 | Right direction |
| 12 | z-lab arch v4A | 5 layers, GQA | 2-4% | Undertrained |
| 13 | 100K data v4B | 5 layers, 100K | 3-4% | Data not bottleneck |
| 14 | KL distill v4C | Top-64 logit, T=2 | 1-2% | Worse (diluted gradient) |
| **15** | **Paper-faithful v5** | **KV inject + exp weight** | **13-15% flat** | **Architecture works** |
| 16 | v5 E2E speed | Same draft | 0% consec | Not viable at 15% |
| 17 | z-lab 3.5→3.6 | Pre-trained draft | 67% | Doesn't transfer |
| 18 | z-lab 3.6 (WIP) | Pre-trained draft | 75% | 0.92x (still slow) |
| 19 | Scheduled sampling | v5 + teacher decay | 2-4% | Worse (conservative) |
| 20-24 | Ablation sweep | Various | 1.6-2.7% | All worse (overfit) |
| 25-27 | Diagnostics | — | — | Planning |
| 28 | z-lab recipe (broken) | 474M, 800K, all GT | 0.4% | Masking bug |
| 29 | z-lab recipe (fixed) | 474M, masked noise | 7.1% | Generalizing |
| 30 | MTP layer | Built-in, no training | 0% | Forward pass bugs |
| **31** | **bstnxbt class** | **474M, 800K, 12 ep** | **0.6%** | **Mode collapse** |

---

## 8. Viable Paths Forward

1. **MTP approach:** Zero training for Qwen/DeepSeek models that ship MTP layers. 80%+ acceptance reported. Requires implementing MTP forward pass with correct RoPE offset handling.

2. **Fine-tune z-lab drafts:** Start from pre-trained weights, adapt to Config-I quants or new model versions. Much less training than from scratch.

3. **Tree-aware GDN kernels:** pupposandro & davideciffa's `ggml_ssm_conv_tree` and `ggml_gated_delta_net_tree_persist` demonstrate 3.43-5.46x on GDN hybrids using DDTree verification. Porting these to Metal would break the sequential verification ceiling.

4. **Smaller drafts for pure-attention models:** 50M params converges 10x faster. Pure-attention models (Llama, GPT, Phi) don't have the GDN ceiling — even 36% acceptance gives 1.6x.

---

## 9. References

- DFlash: https://arxiv.org/abs/2602.06036
- dflash-mlx (working MLX implementation): https://github.com/bstnxbt/dflash-mlx
- z-lab pre-trained drafts: https://huggingface.co/z-lab
- DDTree: https://liranringel.github.io/ddtree/
- pupposandro 207 tok/s on RTX 3090: https://x.com/pupposandro/status/2046264488832213174
- TurboQuant+: https://github.com/TheTom/turboquant_plus
