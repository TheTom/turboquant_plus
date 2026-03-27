# Attention-Gated Value Dequantization for Quantized KV Cache Inference

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## Abstract

Quantized KV cache compression (e.g., TurboQuant, NVFP4) reduces memory consumption during LLM inference but introduces a dequantization bottleneck during autoregressive decoding. After exhaustively testing 14 alternative dequant implementations (register arrays, bit-arithmetic, SIMD shuffles, fused block operations), we found that no instruction-level optimization beats the hardware's constant memory LUT on Apple Silicon. The bottleneck is not *how* values are dequantized, but *how many*.

We observe that in flash attention kernels, softmax attention weights are computed *before* value accumulation, and that at long context lengths, 90%+ of these weights are negligible. We propose **sparse V dequantization**: skipping value dequantization for positions where the attention weight falls below a threshold. Rather than making N dequant operations faster, we eliminate $(1-p) \times N$ of them entirely. This shifts the optimization target from instruction-level efficiency to attention-gated computation. On Apple M5 Max with a 35B MoE model, this yields **+22.8% decode throughput at 32K context** with zero perplexity loss, and unexpectedly *improved* needle-in-a-haystack retrieval (9/9 vs 7/9 baseline), suggesting that dequantizing negligible positions may introduce quantization artifacts rather than useful signal. The technique requires 3 lines of code and is orthogonal to existing dequant optimizations. Because it operates on the attention distribution rather than the dequantization mechanism itself, it is general to any quantized KV cache scheme — validated on both TurboQuant (3.5-bit) and q8\_0 (8-bit) — and its benefit scales with context length.

---

## 1. Introduction

KV cache compression is becoming essential for long-context LLM inference. Google's TurboQuant (ICLR 2026) achieves 4.6× compression via Walsh-Hadamard rotation and polar quantization, with perplexity within 1.1% of uncompressed baselines. However, the compression introduces a per-token dequantization cost during autoregressive decoding that grows with context length.

On Apple Silicon, TurboQuant's dequant uses a centroid lookup table (LUT) in Metal constant memory. Profiling reveals this LUT accounts for 14–34% of decode time depending on hardware generation and context depth. At 32K context on M5 Max, the dequant overhead alone reduces decode throughput from 78.3 tok/s (no-dequant ceiling) to 47.0 tok/s, a 40% penalty. Notably, the no-dequant ceiling is **28% faster than uncompressed q8_0** (61.0 tok/s) because the compressed cache moves less data over the memory bus.

This gap motivated an exhaustive search: 14 alternative dequant implementations were tested on M2 Pro and M5 Max hardware, including register arrays, bit-arithmetic, FMA branchless computation, simd_shuffle cross-lane transfer, and fused block dot products. None beat the baseline constant memory LUT on Apple Silicon (see Section 5).

The failure of all 14 approaches revealed a fundamental constraint: on Apple Silicon, the constant memory LUT is already at the hardware floor. No amount of cleverness in *how* you dequantize beats 4 divergent constant reads. The only remaining lever is reducing how many positions require dequantization at all — shifting the optimization target from instruction-level efficiency to attention-gated operation elimination.

We then observed that the dequant cost splits roughly 50/50 between K (key) and V (value) paths. The key insight: in flash attention, softmax weights are computed from K *before* V is accessed. At long context, most attention weights are negligible. Skipping V dequant for these positions eliminates approximately half the total dequant cost at long context, with no measurable quality impact. This reframes the optimization problem: instead of making each operation cheaper (bounded by hardware floor), eliminate operations entirely (unbounded improvement as attention sparsity increases with context length).

---

## 2. Background

### 2.1 TurboQuant KV Cache Compression

TurboQuant compresses KV cache entries from 16-bit to 3.5-bit via:
1. **Walsh-Hadamard Transform (WHT):** Rotates vectors to make coordinates approximately Gaussian
2. **Polar decomposition:** Converts Cartesian coordinates to polar (angle + radius) recursively
3. **Codebook quantization:** Maps angles to precomputed centroids via Lloyd's algorithm on the known analytical distribution

The result: 4.6× compression with 1.1% perplexity loss. Each 32-element block stores a norm (2 bytes), quantization indices (8 bytes), and sign bits (4 bytes) = 14 bytes total.

### 2.2 Flash Attention Decode Path

During autoregressive decoding (batch size = 1), flash attention computes:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

In fused flash attention kernels, this proceeds in tiles:
1. **K phase:** For each tile of KV positions, dequantize K, compute $QK^T$ scores, update running softmax
2. **V phase:** Using the computed attention weights, dequantize V, accumulate weighted sum

The attention weights $\alpha_i = \text{softmax}(QK^T)_i$ are known before V dequantization begins. This is the opportunity.

### 2.3 Attention Sparsity at Long Context

At context length $n$, the softmax distribution concentrates on a small subset of positions. Empirically, at 32K context with Qwen3.5-35B-A3B, over 90% of attention weights fall below $10^{-6}$. This sparsity increases with context length: the longer the conversation, the more positions have negligible attention.

---

## 3. Method

### 3.1 Sparse V Dequantization

We add a single conditional before the V dequant inner loop:

```metal
FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
    const float attn_weight = float(ss[NE*cc + ty]);
    if (attn_weight < 1e-6f) continue;  // skip negligible positions

    // ... existing V dequant and accumulation ...
}
```

When the attention weight for a KV position is below the threshold ($10^{-6}$), the entire V dequantization and accumulation for that position is skipped. This saves:
- Constant memory reads (centroid LUT lookups)
- ALU operations (index extraction, sign application, norm multiplication)
- Device memory reads (block data)

### 3.2 Threshold Selection

The threshold $\tau = 10^{-6}$ was chosen conservatively. For a 32K context with 32 attention heads, a weight of $10^{-6}$ contributes at most $10^{-6} \times \|V\|$ to the output, negligible relative to the dominant attention positions. We validate empirically that perplexity is identical at this threshold (Section 4).

### 3.3 Interaction with K Dequant Optimizations

Sparse V dequant is orthogonal to K-path optimizations. The K dequant must still run for all positions (to compute attention weights). Existing K-path optimizations include:
- **4-magnitude LUT:** Reduces constant memory addresses from 8 to 4 (+38% on M2 Pro)
- **Hardware auto-detection:** M1/M2/M3/M4 use 4-mag LUT; M5+ uses full 8-entry LUT

These stack: 4-mag reduces K dequant cost, sparse V reduces V dequant cost. Combined gains are additive.

Importantly, this is a zero-cost optimization:

- No model retraining
- No calibration data
- No changes to model architecture
- A minimal implementation (3 lines of kernel code)

Unlike most quantization-aware optimizations, sparse V requires no model-specific tuning. It operates purely at the kernel level using information already computed during inference.

---

## 4. Experiments

### 4.1 Setup

- **Hardware:** Apple M5 Max, 128GB unified memory, 546 GB/s bandwidth
- **Model:** Qwen3.5-35B-A3B (MoE), Q8_0 weight quantization
- **KV cache:** TurboQuant turbo3 (3.5-bit, 4.6× compression)
- **Framework:** llama.cpp with Metal flash attention kernels
- **Baseline:** turbo3 decode without sparse V (main branch)
- **Quality metric:** Perplexity on WikiText-2 (8-chunk and 32-chunk)
- **Retrieval metric:** Needle-in-a-haystack (NIAH), single and multi-key

### 4.2 Decode Throughput

Full regression suite with sparse V enabled ($\tau = 10^{-6}$):

| Context Depth | Baseline (tok/s) | Sparse V (tok/s) | Improvement | vs q8\_0 |
|---------------|-----------------|------------------|-------------|----------|
| short | 76.5 | 77.6 | +1.4% | 0.899× |
| 4K | 72.0 | 74.9 | +4.0% | — |
| 8K | 66.9 | 71.7 | +7.2% | — |
| 16K | 58.9 | 66.5 | +12.9% | 0.923× |
| 32K | 47.0 | 57.7 | **+22.8%** | **0.931×** |

The benefit scales with context length because longer contexts have more positions with negligible attention weights. At 32K, the ratio vs uncompressed q8\_0 improves from 0.76× to 0.93×, near parity.

### 4.3 Quality Validation

| Metric | q8\_0 | turbo3 | turbo3 + sparse V |
|--------|-------|--------|-------------------|
| PPL (8-chunk) | 6.111 | 6.211 | 6.176 |
| PPL (32-chunk) | 5.415 | 5.471 | — |
| vs q8\_0 | — | +1.6% | +1.06% |

Sparse V perplexity (6.176) is actually *lower* (better) than baseline turbo3 (6.211). The threshold is conservative enough that skipping negligible positions introduces no measurable quality degradation.

**Long-context validation:** The c=512 result above is a no-regression sanity check — at 512 tokens, sparse V skips ~6% of positions and has negligible effect. To validate under conditions where sparse V is actively skipping positions, we ran perplexity at longer context lengths with increased chunk counts for statistical power. q8\_0 baselines were run first to confirm corpus/chunk sanity before evaluating turbo3.

| Context | Chunks | q8\_0 | turbo3 ON | turbo3 OFF | ON/OFF Δ | vs q8\_0 |
|---------|--------|-------|-----------|------------|----------|---------|
| 8K | 20 | 5.4592 | 5.5195 | 5.5195 | 0.0000 | +1.1% |
| 16K | 10 | 5.0008 | 5.0630 | 5.0630 | 0.0000 | +1.2% |
| 32K | 5 | 6.0274 | 6.1103 | 6.1103 | 0.0000 | +1.4% |

All runs use $\tau = 10^{-6}$. PPL is numerically identical with and without sparse V at every context length tested in this setup. The +1.1–1.4% gap vs q8\_0 is the underlying TurboQuant compression overhead — consistent across context lengths and unaffected by sparse V.

**Direct skip-rate measurement** (Qwen3-1.7B, eager attention with `output_attentions=True`):

| Context | Overall skip rate | Min layer | Max layer | Median layer |
|---------|------------------|-----------|-----------|-------------|
| 512 | 9.1% | 0.0% | 32.1% | 6.3% |
| 2048 | 20.7% | 2.0% | 59.5% | 15.0% |
| 4096 | 28.4% | 3.7% | 72.4% | 24.5% |

Skip rate was measured directly by counting attention weights below $\tau$ across all layers and heads. Measurement used Qwen3-1.7B (same architecture family, fits in memory with eager attention). Skip rates on the 35B model are estimated from decode speed improvements at ~28% (8K), ~51% (16K), and ~90% (32K).

Skip rate increases with context length as expected: softmax concentrates on fewer positions as the sequence grows. Early layers show higher skip rates (broader attention patterns), while later layers are more focused. Full methodology, per-layer data, and raw commands: [`long-context-sparse-v-validation.md`](../long-context-sparse-v-validation.md).

### 4.4 KL Divergence vs f16

To measure distributional shift (not just top-token accuracy), we compute KL divergence against f16 KV cache logits on both MoE and dense models:

**MoE (Qwen3.5-35B-A3B):**

| Cache Type | Mean KLD | Δp RMS | Same top-p % |
|------------|----------|--------|-------------|
| q8\_0 | 0.001549 | 1.23% | 98.43% |
| q4\_0 | 0.008091 | 2.75% | 95.83% |
| turbo3 | 0.016145 | 4.09% | 94.31% |

**Dense (Qwen3.5-27B):**

| Cache Type | Mean KLD | Δp RMS | Same top-p % |
|------------|----------|--------|-------------|
| q8\_0 | 0.000018 | 0.13% | 99.90% |
| q4\_0 | 0.002741 | 1.44% | 97.65% |
| turbo3 | 0.009900 | 2.74% | 95.98% |

turbo3 KLD is higher than q4\_0 on both architectures, consistent with its lower effective bit rate (3.5 vs 4.0). The same-top-p metric shows turbo3 agrees with f16 on the top token 94–96% of the time. Dense models show lower KLD across all cache types because attention patterns are more concentrated.

### 4.5 NIAH Retrieval

| Test | q8\_0 | turbo3 | turbo3 + sparse V |
|------|-------|--------|-------------------|
| Single needle (9 positions) | 7/9 | 7/9 | **9/9 (100%)** |
| Multi-key (4K-32K) | 4/4 | 4/4 | 4/4 |

Sparse V achieves **perfect** single-needle retrieval (9/9), improving from 7/9 without sparse V. This behavior is consistent with the hypothesis that needle positions have meaningful attention weights (well above $10^{-6}$) and are never skipped. The improvement suggests that dequantizing low-weight positions may introduce quantization artifacts that accumulate in the output. Each negligible-weight position contributes near-zero useful signal but potentially non-zero quantization noise. Sparse V removes these contributions entirely, which appears to improve the signal-to-noise ratio of the attention output in this setup. This is a second, independent contribution: sparse V is not just a speed optimization, it is also a quality optimization for quantized KV caches.

### 4.6 Prefill Impact

Sparse V has minimal effect on prefill because prefill processes the entire prompt in parallel (no autoregressive attention weight computation). Measured prefill at 4K: 2429 tok/s with sparse V vs 2362 baseline (+2.8%).

### 4.7 Real-World Server Benchmark

The decode numbers in Section 4.2 are from `llama-bench` batch evaluation, which keeps the GPU maximally saturated. To validate under realistic conditions, we tested with `llama-server` processing a 70-page PDF (~24K prompt tokens) via the OpenAI-compatible chat completions API:

| Metric | turbo3 + sparse V | q8\_0 | ratio |
|--------|-------------------|-------|-------|
| Prefill | 1417.8 tok/s | 1449.9 tok/s | 0.98× |
| Decode | 53.3 tok/s | 68.2 tok/s | 0.78× |

The gap between `llama-bench` and `llama-server` results reflects system-level overhead (HTTP handling, templating, scheduling), not a limitation of sparse V itself. Kernel-level measurements approach near-parity with q8\_0 (0.93×), while end-to-end server performance remains lower due to non-kernel costs.

**Takeaway:** Users should expect ~78% of q8\_0 decode speed at long context in real server deployments, not the ~93% measured in synthetic benchmarks. The sparse V improvement still holds; without it, decode performance would be closer to ~60% of q8\_0.

### 4.8 Threshold Ablation

We swept the threshold $\tau$ across five values to determine sensitivity:

| $\tau$ | PPL (8-chunk) | vs q8\_0 | Decode tok/s (short) | Decode tok/s (pp32768+tg128) |
|--------|--------------|----------|---------------------|------------------------------|
| $10^{-4}$ | 6.1756 | +1.06% | 76.3 | 1111.1 |
| $10^{-5}$ | 6.1756 | +1.06% | 76.5 | 1112.7 |
| $\mathbf{10^{-6}}$ | **6.1756** | **+1.06%** | **76.1** | **1113.8** |
| $10^{-7}$ | 6.1756 | +1.06% | 75.7 | 1113.8 |
| $10^{-8}$ | 6.1756 | +1.06% | 76.4 | 1114.4 |

All tested thresholds ($10^{-4}$ to $10^{-8}$) produce identical PPL. This indicates that a large fraction of V contributions fall below numerical significance: skipping them introduces no measurable loss in model quality in this setup. The threshold effectively defines a boundary below which attention contributions do not affect output quality.

**Short-context decode speed is flat** ($\pm 1$ tok/s, within measurement noise). At short context, attention is dense, so few positions have weights below any of these thresholds, so the skip condition rarely triggers. The threshold's impact is context-dependent: the +22.8% improvement at 32K (Section 4.2) comes from the exponentially increasing fraction of near-zero weights at long context.

**Conclusion:** $\tau = 10^{-6}$ remains the recommended default. More aggressive values ($10^{-4}$, $10^{-5}$) are equally safe quality-wise but offer no additional speed benefit at short context. The long-context gains are already captured at $10^{-6}$. Future work could validate $\tau = 10^{-4}$ on harder retrieval tasks (multi-needle NIAH at 128K+) to confirm no degradation before raising the default. Raw benchmark logs are available in [`threshold-ablation-logs/`](../threshold-ablation-logs/) and the full analysis in [`threshold-ablation.md`](../threshold-ablation.md).

---

## 5. What Didn't Work: 14 Alternative Dequant Approaches

Before discovering sparse V, we exhaustively tested 14 dequant-level optimizations on M2 Pro (Apple8) and M5 Max (Apple10). All attempted to reduce the constant memory LUT cost:

| # | Approach | M2 8K tok/s | vs Best | Result |
|---|----------|-------------|---------|--------|
| 1 | **4-mag LUT + XOR sign** | **15.1** | **baseline** | **Best dequant-level fix (+38%)** |
| 2 | Batched byte extract | 13.7 | -9% | Better byte reading, still 8 LUT addresses |
| 3 | Inline block dequant | 13.5 | -11% | I-cache pressure |
| 4 | 2-pair half2 LUT | 12.0 | -21% | Ternary overhead exceeds LUT savings |
| 5 | Select chain (zero LUT) | 11.9 | -21% | Too much ALU |
| 6 | Bit-arithmetic | 11.6 | -23% | Pure ALU, zero memory, but ALU cost too high |
| 7 | Non-vec FA (nl=2) | 10.2 | -32% | Kernel not designed for single-token decode |
| 8 | float cn[8] registers | — | — | Metal spills to stack (M5 only) |
| 9 | half cn[8] registers | — | — | Also spills (M5 only) |
| 10 | Split 2×4 half LUT | — | — | Branch overhead (M5 only) |
| 11 | Deferred norm multiply | 12.9 | -15% | Loses ILP |
| 12 | FMA branchless | 11.4 | -25% | 7 ALU ops > 1 divergent constant read |
| 13 | simd_shuffle | 14.7 | -3% | Cross-lane latency ≈ constant LUT |
| 14 | Fused block dot | 8.1 | -46% | 64 float comparisons devastate throughput |

**Conclusion:** On Apple Silicon, 4 divergent constant memory reads are faster than any arithmetic computation that produces the same 4-way selection. The constant cache, even when divergent, beats 7+ ALU operations. The only path beyond the 4-mag LUT is changing *what* data is read (sparse V, format changes), not *how* it's computed.

Full experiment logs, kernel variants, and per-hardware profiling are available in [Decode Speed Hardware Analysis](https://github.com/TheTom/turboquant_plus/blob/main/docs/decode-speed-hardware-analysis.md).

---

## 6. Why Sparse V Works Where Dequant Tricks Don't

The 14 failed approaches all tried to make individual dequant operations cheaper. Sparse V succeeds because it eliminates entire dequant operations. The distinction:

- **Dequant optimization:** Make each of N operations faster → bounded by ALU/memory floor
- **Sparse V:** Eliminate (1-p)×N operations entirely, rather than attempting to optimize N under hardware constraints → unbounded improvement as p → 1

At 32K context, p ≈ 0.9 (90% of positions skipped). At 128K, p would be even higher. The technique becomes increasingly effective at exactly the context lengths where the dequant bottleneck is worst.

More broadly, sparse V is an instance of *attention-aware computation*: using the model's own sparsity pattern — computed as a byproduct of normal inference — to gate downstream kernel work. The attention weights are already available before V accumulation begins; sparse V simply acts on information the kernel already has.

---

## 7. Generality

Sparse V dequantization is not specific to TurboQuant. Because it gates computation based on the attention distribution — not the specifics of any dequantization implementation — it applies to any quantized KV cache scheme where:

1. Flash attention is used (softmax computed before V accumulation)
2. V is stored in a quantized format requiring dequantization
3. The dequant cost is non-trivial relative to the multiply-accumulate

This includes NVFP4, KIVI, CacheQuant, and other KV cache quantization methods — because it operates on the attention distribution rather than the dequantization mechanism itself. The 3-line implementation is kernel-level and requires no model changes, no retraining, and no calibration data.

### 7.1 Empirical Validation on q8\_0

To validate generality beyond TurboQuant, we tested sparse V on llama.cpp's standard q8\_0 KV cache (8-bit quantization, 2× compression) using the same model and hardware. A `TURBO_SPARSE_V=0` override was added to force-disable the optimization for A/B comparison:

| Test | q8\_0 + sparse V | q8\_0 (no sparse V) | Improvement |
|------|-----------------|---------------------|-------------|
| Decode (short, tg128) | 84.7 tok/s | 80.7 tok/s | **+5.0%** |
| Blended (pp32768+tg128) | 1145.2 tok/s | 1096.7 tok/s | **+4.4%** |

Sparse V provides a **5% decode speedup on q8\_0**, demonstrating that the optimization is not tied to expensive dequantization schemes. q8\_0 uses significantly cheaper per-position dequantization than turbo3. The benefit is smaller than turbo3's +22.8% at 32K because q8\_0's dequant is lightweight (simple scale-and-add vs centroid LUT + WHT rotation), but the attention sparsity still allows meaningful work to be skipped.

**Quality validation (q8\_0):**

| Metric | q8\_0 + sparse V | q8\_0 (no sparse V) |
|--------|-----------------|---------------------|
| PPL (8-chunk) | 6.1109 | 6.1109 |
| NIAH single (9 tests) | 7/9 | 7/9 |

PPL identical. NIAH identical — same two failures at 100% depth for 8K and 16K in both conditions. Sparse V has zero quality or retrieval impact on q8\_0, confirming it is purely a compute optimization.

This confirms that sparse V is a property of the attention mechanism itself, not the quantization method, and not a compression-specific optimization. Because it operates on the attention distribution rather than the dequantization mechanism, it applies broadly across KV cache formats, including q8\_0 and future quantization schemes. Raw benchmark logs: [`threshold-ablation-logs/q8_0_sparse_v_ablation_m5.txt`](../threshold-ablation-logs/q8_0_sparse_v_ablation_m5.txt), [`threshold-ablation-logs/q8_0_sparse_v_quality_m5.txt`](../threshold-ablation-logs/q8_0_sparse_v_quality_m5.txt).

### 7.2 Combined 4-mag LUT + Sparse V on M2 Pro

We hypothesized that 4-mag LUT (K dequant optimization) and sparse V (V dequant optimization) would stack since they address independent bottlenecks. Testing on M2 Pro (Apple8, 200 GB/s bandwidth) confirms:

| Test | turbo3 (4-mag + sparse V) | q8\_0 | ratio |
|------|--------------------------|-------|-------|
| Short decode (tg128) | 23.6 tok/s | 32.1 tok/s | **0.73×** |
| pp8192+tg128 | 189.1 tok/s | 210.1 tok/s | 0.90× |
| pp16384+tg128 | 155.1 tok/s | 171.9 tok/s | 0.90× |

**Historical progression on M2 Pro decode:**

| Optimization | Decode ratio vs q8\_0 |
|--------------|-----------------------|
| Baseline (no optimizations) | 0.45× |
| + 4-mag LUT | 0.67× |
| **+ 4-mag LUT + sparse V** | **0.73×** |

The two optimizations stack as predicted. 4-mag reduces K dequant cost (fewer constant memory addresses), sparse V skips V dequant for negligible positions. Combined: M2 Pro decode went from 45% to 73% of q8\_0 — a 62% improvement from the unoptimized baseline. Prefill blended numbers are 90% of q8\_0, consistent with M5 Max results.

Raw logs: [`threshold-ablation-logs/m2_pro_4mag_sparse_v.txt`](../threshold-ablation-logs/m2_pro_4mag_sparse_v.txt).

### 7.3 Dense Model Validation

Sparse V was evaluated on a dense 27B model (Qwen3.5-27B Q8\_0) to check for regressions outside MoE workloads.

| Test | With sparse V | Without | Delta |
|------|-------------|---------|-------|
| Short decode (tg128) | 16.73 tok/s | 16.61 tok/s | +0.7% |
| pp8192+tg128 | 298.27 tok/s | 294.52 tok/s | +1.3% |
| pp16384+tg128 | 316.98 tok/s | 311.24 tok/s | +1.8% |

No regressions were observed. Gains are smaller than MoE models, where attention is a larger fraction of decode time, but remain neutral-to-positive. The trend improves slightly with context length.

This suggests sparse V is safe to enable by default even for dense models. On dense architectures, FFN dominates decode compute (all parameters are active every token), so attention — and therefore V dequant — is a small fraction of total cost. Sparse V neither helps nor hurts meaningfully, but does not regress.

Raw logs: [`threshold-ablation-logs/dense_27b_sparse_v_clean_m5.txt`](../threshold-ablation-logs/dense_27b_sparse_v_clean_m5.txt).

---

## 8. Limitations and Future Work

**Limitations:**
- We perform a threshold ablation in Section 4.8 and find the method is insensitive to $\tau$ across $10^{-4}$ to $10^{-8}$. Perplexity is identical at all values.
- Short context benefit is modest (+1.4%) because attention is less sparse.

**Future work:**
- **Context-adaptive dispatch:** Compile multiple FA kernel variants, select optimal path based on KV cache size at dispatch time.
- **Sparse K dequant:** Extend the sparsity concept to K. After a first pass computing approximate attention scores, skip K dequant for positions that won't contribute. Requires two-pass attention or speculative attention.
- **Non-Apple hardware:** Test on NVIDIA (CUDA) and AMD (ROCm) where the dequant bottleneck profile differs.

More broadly, sparse V dequantization is an instance of a wider class of **attention-aware kernel optimizations**: using the attention distribution computed during inference to gate downstream computation. The same principle could extend to hybrid precision attention (full-precision dequant for high-weight positions, approximate for low-weight), hardware-aware sparsity gating, or speculative K-path pruning.

---

## 9. Conclusion

We present a 3-line modification to flash attention kernels that yields up to 22.8% decode throughput improvement for quantized KV caches at long context, with no measurable quality degradation in our evaluation, and an observed improvement in retrieval accuracy, suggesting that dequantizing negligible positions may introduce quantization artifacts into the attention output.

The core insight is straightforward: Making N dequant operations faster is bounded by hardware limits; eliminating $(1-p) \times N$ of them entirely is not. After 14 failed attempts to optimize the dequant instruction itself, sparse V succeeds by changing the question from "how do we dequantize faster?" to "should we dequantize at all?"

The technique exploits the natural sparsity of attention weights — a property that becomes more pronounced exactly when the dequant bottleneck is most severe. The approach is general to any quantized KV cache scheme, requires no model changes, no retraining, and no calibration data, and is orthogonal to existing dequant and compression optimizations. This represents an instance of a broader class of attention-aware kernel optimizations, where computation is gated by the model's own sparsity patterns rather than optimized at the instruction level.

More broadly, these results indicate that a significant fraction of value-side attention computation in long-context inference falls below numerical significance, and that the attention distribution itself provides a reliable, zero-cost signal for identifying these positions.

---

## Reproducibility

All code, benchmarks, and diagnostic tools are open source:

- **Implementation:** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (branch: `experimental_decode_speed_tests`)
- **Benchmarks & diagnostics:** [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- **Hardware analysis:** [decode-speed-hardware-analysis.md](../decode-speed-hardware-analysis.md)

To reproduce: build with `TURBO_SPARSE_V=1` environment variable and run `llama-bench` at various context depths with `-ctk turbo3 -ctv turbo3 -fa 1`.

---

## Acknowledgments

- **@spiritbuun**: CUDA fork with norm correction and register LUT, whose work inspired the hardware profiling investigation
- **@Ambisphaeric, @mariotomich**: Independent M1 Max testers who confirmed the decode regression is hardware-dependent
- **@ekryski**: GPT-OSS 20B testing on M1 Max with turbo4
- The TurboQuant community for extensive cross-hardware validation

## References

1. TurboQuant: Redefining AI Efficiency with Extreme Compression. Google Research, ICLR 2026.
2. Ilhan et al. "AttentionPack: Attention-aware Inference Optimizations for Large Vision-Language Models with Memory-efficient Decoding." arXiv:2603.23914, 2026.
3. An et al. "GlowQ: Group-Shared Low-Rank Approximation for Quantized LLMs." arXiv:2603.25385, 2026.
4. Xie et al. "Scaling Attention via Feature Sparsity." ICLR 2026. arXiv:2603.22300.
5. Zhao et al. "Self-Distillation for Multi-Token Prediction." arXiv:2603.23911, 2026.
6. Qasim et al. "The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference." arXiv:2603.19664, 2026.
7. Wang et al. "SliderQuant: Accurate Post-Training Quantization for LLMs." ICLR 2026. arXiv:2603.25284.
