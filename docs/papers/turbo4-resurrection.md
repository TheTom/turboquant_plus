# Resurrecting TurboQuant4: From PPL 679 to Beating q4_0

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## Abstract

TurboQuant4 (turbo4) was broken. Completely broken — PPL 679 on Metal, catastrophically slow prefill on CUDA, and a 1-bit error correction mechanism (QJL) that actively made quality worse. The community consensus was "do not bother using turbo4."

This document describes how we took turbo4 from unusable to the best quantized KV cache format we've tested: PPL +0.23% vs q8_0, better than q4_0, matching turbo3 decode speed, and near-parity prefill. The path involved finding 7 bugs, running a QJL ablation that proved the paper's own correction mechanism is harmful for attention, and redesigning the format around 4-bit PolarQuant with optimal centroids.

---

## 1. Where We Started

### The State of turbo4 on March 28, 2026

turbo4 was the 4-bit variant of TurboQuant: 3-bit PolarQuant quantization of WHT-rotated KV cache vectors, plus a 1-bit QJL (Quantized Johnson-Lindenstrauss) residual correction. 4.25 bits per value, 3.76x compression.

On paper, turbo4 should have been strictly better than turbo3 (3.5 bits). The extra bit was supposed to correct the PolarQuant residual error via QJL projection, giving near-lossless quality at only marginally less compression.

In practice:
- **Metal (our fork):** PPL 679.27. Completely unusable. The SET_ROWS kernel was writing turbo3-format data into turbo4 blocks.
- **CUDA (buun's fork):** PPL degraded from -0.28% at 2K to +3.69% at 64K. QJL noise accumulated with context length. Prefill was 51.9% of q8_0 — catastrophically slow.
- **Community verdict:** "@spiritbuun: turbo3 is better in every way. Do not bother using turbo4."

### The Numbers That Mattered

| Metric | turbo4 (broken) | turbo3 | q8_0 |
|--------|----------------|--------|------|
| PPL (Metal) | 679.27 | 6.1756 | 6.1109 |
| Decode (Metal) | 44.43 tok/s | 78.24 | 86.01 |
| PPL 64K (CUDA) | +3.69% | +0.49% | baseline |
| Prefill (CUDA) | 51.9% of q8_0 | 99.3% | baseline |

---

## 2. Finding the Bugs

A code analysis of the turbo4 Metal path revealed 7 bugs. Three were critical, explaining the PPL 679.

### Bug 1: SET_ROWS Used turbo3 Packing for turbo4 Blocks

The `kernel_set_rows_turbo` was a shared template for both turbo3 and turbo4. It used turbo3's 2+1 bit packing scheme (2 low bits in `qs[]`, 1 high bit in `signs[]`) for 32-element blocks. But turbo4 uses 128-element blocks with packed 3-bit indices spanning byte boundaries. The shared template wrote turbo3-format data into turbo4 blocks, corrupting every KV cache write.

**Fix:** Dedicated `kernel_set_rows_turbo4` with correct 128-element block handling.

**Impact:** PPL 679 → 6.19.

### Bug 2: QJL Matrix Multiplication Missing in Dequant

The Metal dequant read QJL signs as raw ±1 values and multiplied them by a scale factor. The C reference code applied a full matrix-vector multiply (`turbo_qjl_matrix_t × signs`). Without the matrix multiply, the QJL "correction" was random noise.

### Bug 3: SET_ROWS Missing QJL Step Entirely

The SET_ROWS kernel performed PolarQuant quantization but never computed the QJL residual, projection, or signs. The `signs[]` field in the KV cache was either zero-initialized or contained stale data. The dequant then "corrected" with this garbage.

### Bugs 4-7: Supporting Issues

4. **3-bit packing format mismatch** in SET_ROWS (turbo3 2+1 scheme applied to turbo4's packed 3-bit format)
5. **rnorm uninitialized** in C reference quantizer (QJL scale computed from garbage)
6. **Metal computed rnorm from wrong-basis residual** (residual in rotated space, not normalized space)
7. **WHT used instead of QJL random matrix** in Metal (may have been intentional approximation, but breaks C reference symmetry)

---

## 3. The QJL Ablation

After fixing Bug 1 (SET_ROWS), turbo4 PPL dropped from 679 to 6.19. Still slightly worse than turbo3 (6.18). The natural question: is QJL actually helping?

We disabled QJL in the dequant by removing the sign correction:

```metal
// Original: cache[i] = (recon[i] + signs_f[i] * qjl_scale) * norm;
cache[i] = recon[i] * norm;  // PolarQuant only
```

**Result:**

| Config | PPL |
|--------|-----|
| turbo4 with QJL | 6.1894 |
| turbo4 without QJL | 6.1756 |
| turbo3 (reference) | 6.1756 |

QJL made quality **worse**. Removing it made turbo4 identical to turbo3.

### Why QJL Fails for Attention

This was independently confirmed by three groups:
- **buun (CUDA):** turbo4 PPL degrades from -0.28% at 2K to +3.69% at 64K. QJL noise accumulates with context.
- **scos-lab (independent impl):** GPT-2 b=3 MSE gives +7.6% PPL, QJL (Prod) gives +300% PPL. "QJL variance is amplified by softmax."
- **Arclabs001 (YATQ):** 3-bit MSE-only top-1 71.0% vs +QJL 61.2% (-9.8%). "QJL eliminates bias but explodes variance. Softmax tolerates uniform bias, not variance."

The TurboQuant paper's own ablation (OpenReview) confirms: MSE-only (no QJL, no two-tier channel) still beats baselines. The QJL correction is not needed.

**Decision:** Drop QJL entirely.

---

## 4. The Speed Investigation

With QJL dropped, turbo4 matched turbo3 quality (6.1756). But decode was still 48 tok/s vs turbo3's 78. A component isolation analysis identified the bottlenecks:

| Component | tok/s lost | % of gap |
|-----------|-----------|---------|
| 128-block FA tiling overhead | 13.7 | 33% |
| 3-bit byte-spanning unpack | 24.3 | 58% |
| QJL sign unpack + scale | 3.6 | 9% |

### The 3-Bit Unpack Problem

turbo4's original 3-bit index extraction:
```metal
int bit_offset = j * 3;
int byte_idx = bit_offset / 8;
int bit_pos = bit_offset % 8;
uint16_t raw = (uint16_t)xb->qs[byte_idx];
if (byte_idx + 1 < QK_TURBO4 * 3 / 8) {
    raw |= (uint16_t)xb->qs[byte_idx + 1] << 8;
}
uint8_t idx = (uint8_t)((raw >> bit_pos) & 0x7);
```

Per element: division, modulo, conditional second byte read, shift, mask. 128 iterations. 58% of the total decode cost.

turbo3's byte-aligned 2+1 bit scheme:
```metal
idx = (qs[j/4] >> ((j%4)*2)) & 0x3;
if (signs[j/8] & (1<<(j%8))) idx |= 4;
```

Simple shifts and masks, no byte-spanning.

**Fix:** Switch turbo4 to 2+1 bit packing (same as turbo3). Since QJL was dropped, the `signs[]` field was free for the 3rd index bit.

**Impact:** 48.05 → 63.79 tok/s (+33%).

### The Full-Block Cache Problem

turbo4's dequant allocated `float cache[128]` on the stack, dequantized all 128 elements, then returned only 4. The FA vec kernel called this 32 times per block position — each call did the full 128-element unpack redundantly.

turbo3 avoided this by directly extracting the needed elements from the 32-element block.

**Fix:** Direct per-element extraction for turbo4, identical pattern to turbo3.

**Impact:** 63.79 → 78.30 tok/s (+23%). Now matching turbo3.

---

## 5. The 4-Bit Breakthrough

With QJL dropped and the dequant fixed, turbo4 was functionally "turbo3 in 128-element blocks." Same quality, same speed, different block layout. The original promise of turbo4 — better quality than turbo3 — was unfulfilled.

The fix: use the full 4 bits for PolarQuant instead of splitting 3+1 (PolarQuant+QJL). 16 optimal centroids instead of 8.

### Computing 4-Bit Centroids

The optimal scalar quantizer for N(0, 1/√d) with 16 levels has centroids given by the conditional expectation within each quantile bin. For d=128:

```
[-0.1739, -0.1172, -0.0895, -0.0688, -0.0513, -0.0356, -0.0210, -0.0069,
  0.0069,  0.0210,  0.0356,  0.0513,  0.0688,  0.0895,  0.1172,  0.1739]
```

Packing: simple nibbles. 2 indices per byte, no byte-spanning. 4 bits per element with `(qs[j/2] >> ((j%2)*4)) & 0xF`. Even simpler than the 2+1 scheme.

### Results

| Config | PPL | vs q8_0 | Decode tok/s | vs q8_0 |
|--------|-----|---------|-------------|---------|
| q8_0 | 6.1109 | — | 85.71 | 1.00x |
| **turbo4 4-bit** | **6.1250** | **+0.23%** | **79.87** | **0.93x** |
| q4_0 | 6.1424 | +0.52% | 83.36 | 0.97x |
| turbo3 | 6.1756 | +1.06% | 76.84 | 0.90x |

turbo4 4-bit has:
- Better quality than turbo3 (4.5x smaller PPL gap to q8_0)
- Better quality than q4_0
- Better compression than q4_0 (3.76x vs 3.56x)
- Matching turbo3 decode speed
- Near-parity q8_0 prefill (96%)

---

## 6. The Full Journey

| Step | PPL | Decode tok/s | Change |
|------|-----|-------------|--------|
| Broken (shared SET_ROWS) | 679.27 | 44.43 | — |
| + Dedicated SET_ROWS | 6.1894 | 44.43 | Fixed block corruption |
| + Drop QJL | 6.1756 | 48.05 | QJL was hurting quality |
| + 2+1 bit packing | 6.1756 | 63.79 | Eliminated byte-spanning reads |
| + Direct-extract dequant | 6.1756 | 78.30 | Eliminated redundant full-block cache |
| + 4-bit PolarQuant | **6.1250** | **79.87** | 16 optimal centroids |

Total improvement: PPL 679 → 6.13, decode 44 → 80 tok/s.

---

## 7. Validation

### Codex Review

OpenAI Codex (gpt-5.3-codex) reviewed the final diff and caught an out-of-bounds memory access: the `block_turbo4_0` struct had `qs[48]` (sized for 3-bit packing) but 4-bit nibble packing writes to `qs[0..63]`. The struct was updated to `qs[64]`, maintaining the 68-byte block size.

### Cross-Validation

- buun's CUDA turbo4 (old QJL format) confirmed QJL degradation at long context
- scos-lab's independent implementation confirmed MSE > QJL for attention
- Arclabs001's YATQ analysis quantified QJL variance explosion through softmax
- The TurboQuant paper's own ablation (OpenReview) confirms MSE-only beats baselines

### Long-Context Stability

| Context | turbo4 4-bit | turbo3 | turbo4 vs turbo3 |
|---------|-------------|--------|-----------------|
| 512 | 6.1250 | 6.1756 | turbo4 wins |
| 8K | 5.5546 | 5.5700 | turbo4 wins |
| 16K | 5.0813 | 5.0630 | ~tied (within noise) |

No degradation trend — unlike buun's QJL turbo4 which degraded from -0.28% at 2K to +3.69% at 64K.

---

## 8. What We Learned

1. **QJL is harmful for KV cache attention.** The 1-bit correction eliminates bias but explodes variance. Softmax amplifies variance. Three independent groups confirmed this.

2. **Byte-spanning bit extraction is catastrophically expensive on GPU.** The 3-bit packed format cost 58% of total decode time. Simple nibble packing or byte-aligned 2+1 schemes are dramatically faster.

3. **Full-block dequant caching doesn't work when the block is 128 elements.** The FA kernel calls dequant 32x per position — each call doing a full 128-element unpack and returning 4 elements. Direct extraction eliminates this 32x redundancy.

4. **More centroids > error correction.** 16 optimal PolarQuant centroids give better quality than 8 centroids + QJL correction, at the same bit rate, with simpler implementation and faster execution.

5. **The original paper's design was suboptimal for this application.** TurboQuant was designed for general vector quantization. In the specific context of KV cache attention (where softmax amplifies variance), the QJL component is counterproductive.

---

## 9. Current Status

turbo4 4-bit PolarQuant is merged to `feature/turboquant-kv-cache` (main). `TURBO4_USE_4BIT` ifdef enables 4-bit on Metal by default, legacy 3-bit+QJL on CUDA until ported.

### Prefill Context Scaling

| Context | turbo4 tok/s | turbo3 tok/s | q8_0 tok/s | turbo4/q8_0 |
|---------|-------------|-------------|-----------|------------|
| 2K | 2682 | 2708 | 2665 | 1.01x |
| 4K | 2370 | 2289 | 2255 | 1.05x |
| 8K | 2041 | 2054 | 2002 | 1.02x |
| 16K | 1621 | 1698 | 1605 | 1.01x |
| 32K | 1141 | 1204 | 1098 | 1.04x |

turbo4 prefill matches or exceeds q8\_0 at all context lengths. Compressed cache moves less data over the memory bus.

### Summary

**What works:**
- PPL: +0.23% vs q8_0 (best quantized KV cache quality we've tested)
- Decode: 79.87 tok/s, 93% of q8_0 (faster than turbo3's 76.84)
- Prefill: 101-105% of q8_0 across all context lengths
- NIAH: 31/33 (93.9%) — beats q8_0 (30/33, 90.9%)
- KLD: 0.0096 (40% lower than turbo3's 0.0161; note q4_0 is slightly lower at 0.0081 despite worse PPL — more centroids help top-token prediction more than overall distribution matching)
- Dense model: matches q8_0 decode (17.25 vs 17.17 tok/s)
- Real-world PDF (24K): 63.7 tok/s decode (20% faster than turbo3's 53.3)
- Sparse V: active and compatible
- Long context: stable (no QJL degradation)

### Centroid Investigation (Resolved)

buun tested our 4-bit centroids on CUDA and initially found they worked better on hd128 but slightly worse on hd256. His agent suspected our centroids were non-standard.

```
Ours (d=128, σ=0.0884):  outer centroid = 0.1739  (1.97σ)
buun's (N(0,1) scaled):  outer centroid = 0.2416  (2.73σ)
```

**Resolution:** Our centroids are standard Lloyd-Max for N(0, 1/√128), which is the correct post-FWHT distribution. buun's empirical validation showed they're within 0.2-0.3% of optimal computed from real KV data. The -0.19% "beats q8_0" result at 2K was a short-context artifact — at 8K it was +0.82%, same as Gaussian centroids.

### Why turbo4 Fixes the head_dim=128 Gap

The hd128 quality gap is **not** about centroids, FWHT mixing stages, or InnerQ equalization. It's fundamental to bit rate and dimensionality.

Attention logits are `q·k / √d`. Quantization error variance in each logit:

```
Var(Δlogit) = MSE_per_element / d
```

At fixed MSE: hd128 has **2x the logit variance** of hd256 (128 vs 256 dimensions to average over). More variance → more softmax ranking errors → worse PPL.

Going from 8→16 centroids (turbo3→turbo4) roughly halves MSE per element. So turbo4 on hd128 ≈ turbo3 on hd256 in terms of logit noise.

buun's CUDA data confirms:

| Config | hd128 vs q8_0 | Explanation |
|--------|-------------|-------------|
| turbo3 (8 centroids) | +3.33% | High noise, few centroids, few dims |
| turbo4 Gaussian centroids | +1.20% | 3x better — just from 16 centroids |
| turbo4 our centroids (2K) | -0.19% | Short-context artifact |
| turbo4 our centroids (8K) | +0.82% | Same as Gaussian — centroids are already optimal |

**Conclusion:** The centroids are a solved problem. The hd128 fix is more bits (turbo4), not better centroids, not InnerQ, not CAT alignment. Beautifully simple.

**What's next:**
- Head-dim-dependent centroid codebooks (d=128 and d=256)
- Asymmetric K/V (turbo3-K + turbo4-V — blocked by cross-type FA kernel instantiation)
- Cross-model validation (see [cross-model-validation.md](../cross-model-validation.md))
- Upstream integration
- CUDA port

---

## Acknowledgments

- **@spiritbuun** — CUDA turbo4 data that motivated the investigation, "do not bother" verdict that we proved could be reversed
- **scos-lab** — Independent MSE > QJL finding, K/V norm disparity data
- **Arclabs001/YATQ** — Quantified QJL variance explosion through softmax
- **OpenAI Codex** — Caught the out-of-bounds struct bug
- **Sean Rasch** — turbo4 Python test coverage (PR #41)

---

## Addendum: Independent Validation (2026-03-31)

The finding that QJL hurts quality for attention workloads has been independently confirmed:

- **@Aaryan-Kapoor** (2026-03-25): Independently implemented Algorithm 1 (MSE) only, without QJL. Convergence on the PolarQuant-only approach from a separate implementation.
- **@scos-lab** (2026-03-28): Measured PPL degradation quantitatively: "Paper (Prod keys) +300% vs MSE (both) +7.6% on GPT-2. QJL adds variance that softmax amplifies. Low variance (MSE) beats unbiasedness (Prod)."
- **@AmesianX** — Blackwell DGX Spark (2026-03-30): Initial QJL implementation hurt quality. Root cause identified: correlation between MSE WHT and QJL SRHT rotations. Fixed by using independent sign patterns. First working QJL via independent random seeds.
- **@Arclabs001/YATQ** (2026-03-28): Quantified QJL variance explosion through softmax. Found that QJL works when using separate random seeds for MSE vs QJL rotations ("WHT + QJL + MSE is the solution!"), confirming the root cause is rotation correlation, not QJL itself.
- **@elusznik** — upstream PR #21089 (2026-03-27): Implemented CPU TurboQuant without QJL for the first upstream PR, scoping QJL as follow-up work.
- **@varjoranta** — [vibhavagarwal5/vllm PR #1](https://github.com/vibhavagarwal5/vllm/pull/1) (2026-04-02): Fixed vLLM TurboQuant default from 4-bit to FP8 values. Cites @TheTom (turbo4-resurrection) as source for "value precision is the quality bottleneck." Tested on A100 with Qwen3-8B — FP8 values match FP16 baseline, 4-bit produces garbage. @MidasMining confirmed 100% on 14-check reasoning benchmark with FP8. Independent vLLM validation that naive low-bit V quantization without rotation fails exactly as predicted.
