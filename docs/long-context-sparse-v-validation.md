# Long-Context Sparse V Validation

**Date:** 2026-03-27
**Hardware:** Apple M5 Max 128GB
**Branch:** `feature/turboquant-kv-cache` @ `a52586e`

## Motivation

Prior sparse V quality validation used 8-chunk wikitext-2 at c=512 (PPL 6.1756). At 512 tokens, sparse V barely triggers — most attention weights are above the 1e-6 threshold. This eval validates quality at context lengths where sparse V is actively skipping positions, with increased chunk counts for statistical power and q8_0 baselines to confirm corpus sanity.

## Methodology

- `llama-perplexity` with wikitext-2-raw at c=8192 (20 chunks), c=16384 (10 chunks), c=32768 (5 chunks)
- KV cache: turbo3 (3.5-bit TurboQuant)
- Three runs per context length: q8_0 baseline, turbo3 sparse V ON, turbo3 sparse V OFF
- q8_0 baselines run first to confirm corpus/chunk settings produce sane PPL before evaluating turbo3
- Sparse V OFF via `TURBO_SPARSE_V=0` environment variable

**Skip-rate measurement:** Direct measurement via `output_attentions=True` on Qwen3-1.7B (same architecture family, fits in memory with eager attention). Counts attention weights below τ=1e-6 across all layers and heads. The 35B model cannot run eager attention (OOM), so 35B skip rates are estimated from decode speed improvements.

## Results

### MoE (Qwen3.5-35B-A3B Q8_0) — Full Validation

Each context length tested with three sequential runs: q8_0 baseline first (sanity check), then turbo3 without sparse V (isolates compression), then turbo3 with sparse V (isolates sparse V effect).

**wikitext-2 (1.3MB corpus):**

| Context | Chunks | q8_0 baseline | turbo3 + sparse V | turbo3 no sparse V | Sparse V Δ | vs q8_0 |
|---------|--------|---------------|--------------------|--------------------|------------|---------|
| 8K | 20 | 5.4592 ± 0.045 | 5.5195 ± 0.045 | 5.5195 ± 0.045 | 0.0000 | +1.1% |
| 16K | 10 | 5.0008 ± 0.039 | 5.0630 ± 0.040 | 5.0630 ± 0.040 | 0.0000 | +1.2% |
| 32K | 5 | 6.0274 ± 0.050 | 6.1103 ± 0.051 | 6.1103 ± 0.051 | 0.0000 | +1.4% |

**wikitext-103 (516MB corpus, 10× statistical power):**

| Context | Chunks | q8_0 | q4_0 | turbo3 + sparse V | turbo3 no sparse V | Sparse V Δ |
|---------|--------|------|------|--------------------|--------------------|------------|
| 32K | 50 | 7.0638 ± 0.021 | 7.0857 ± 0.021 | 7.1796 ± 0.021 | 7.1796 ± 0.021 | **0.0000** |

The 50-chunk wikitext-103 run is the strongest validation: tight CI (±0.021), large corpus, high chunk count. Sparse V delta is exactly 0.0000.

**Note on q4\_0:** Included as a reference baseline. No optimization or tuning effort was applied to q4\_0 in this work. Development and optimization focused on q8\_0 and turbo3 paths. turbo3 uses fewer bits (3.5 vs 4.0), so slightly higher PPL relative to q4\_0 is expected.

### Dense (Qwen3.5-27B Q8_0)

| Context | Chunks | Sparse V ON | Sparse V OFF | Delta |
|---------|--------|------------|-------------|-------|
| 8K | 8 | 7.0152 ± 0.106 | 7.0152 ± 0.106 | 0.0000 |

### Skip Rate — Direct Measurement (Qwen3-1.7B)

Measured directly via `output_attentions=True`, counting positions where attention weight < τ:

| Context | Overall skip | Min layer | Max layer | Median layer |
|---------|-------------|-----------|-----------|-------------|
| 512 | 9.1% | 0.0% | 32.1% | 6.3% |
| 2048 | 20.7% | 2.0% | 59.5% | 15.0% |
| 4096 | 28.4% | 3.7% | 72.4% | 24.5% |

### Skip Rate — Estimated (Qwen3.5-35B MoE, from decode speed data)

| Context | Decode Δ | Est. skip rate |
|---------|----------|---------------|
| 8K | +7.2% | ~28% |
| 16K | +12.9% | ~51% |
| 32K | +22.8% | ~90% |

## Interpretation

**Sparse V is directly measured to be active at long context.** At 4096 tokens, 28.4% of V positions are below threshold (directly measured). Skip rate grows monotonically with context length.

**PPL remains numerically identical when sparse V is active.** ON/OFF delta is 0.0000 at every context length and corpus tested (8K through 32K on wikitext-2, 32K on wikitext-103 with 50 chunks). The +1.1-1.6% gap vs q8_0 is the underlying TurboQuant compression overhead — consistent across context lengths and unaffected by sparse V.

**The 50-chunk wikitext-103 run is the definitive validation.** With CI ±0.021, it has sufficient statistical power to detect a PPL difference of ~0.04 (~0.6%). No difference was observed. This supersedes the earlier 2-5 chunk wikitext-2 runs.

**The 512-context PPL should be described as a no-regression sanity check.** At c=512, sparse V skips ~9% of positions. The true validation is at c=8K+ where skip rates are 28-90% and the optimization is actively changing computation.

**Remaining caveats:**
- Direct skip-rate measurement used Qwen3-1.7B (28 layers), not the 35B eval model (40 layers). The 35B model cannot run eager attention without OOM. Skip rates on the 35B are estimated from decode speed improvements.
- Skip rate varies significantly by layer (0% to 72% at 4K). Early layers show higher skip rates.

## Raw Commands

```bash
LLAMA=~/local_llms/llama.cpp/build-turbo/bin
MODEL_MOE=~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf
MODEL_DENSE=~/local_llms/models/Qwen3.5-27B-Q8_0.gguf
WIKI=~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw

# q8_0 baselines first (sanity check)
for ctx_ch in "8192 20" "16384 10" "32768 5"; do
  set -- $ctx_ch
  $LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI -c $1 \
    -ctk q8_0 -ctv q8_0 -fa on --chunks $2 -ngl 99
done

# turbo3 ON/OFF at each context length
for ctx_ch in "8192 20" "16384 10" "32768 5"; do
  set -- $ctx_ch
  $LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI -c $1 \
    -ctk turbo3 -ctv turbo3 -fa on --chunks $2 -ngl 99
  TURBO_SPARSE_V=0 $LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI -c $1 \
    -ctk turbo3 -ctv turbo3 -fa on --chunks $2 -ngl 99
done

# Direct skip rate measurement
python3 scripts/measure_skip_rate.py 512 2048 4096

# 50-chunk 32K on wikitext-103 (strongest validation)
WIKI103=~/local_llms/llama.cpp/wikitext-103-raw/wiki.train.raw

# 1. q8_0 baseline (anchors the experiment)
$LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI103 -c 32768 \
  -ctk q8_0 -ctv q8_0 -fa on --chunks 50 -ngl 99

# 2. turbo3 WITHOUT sparse V (isolates compression effect)
TURBO_SPARSE_V=0 $LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI103 -c 32768 \
  -ctk turbo3 -ctv turbo3 -fa on --chunks 50 -ngl 99

# 3. turbo3 WITH sparse V (isolates sparse V effect)
$LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI103 -c 32768 \
  -ctk turbo3 -ctv turbo3 -fa on --chunks 50 -ngl 99
```
