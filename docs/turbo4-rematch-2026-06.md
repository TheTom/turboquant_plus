# turbo4 improvements & cross-fork rematch (June 2026)

A round of turbo4 KV-cache work in [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) ([PR #197](https://github.com/TheTom/llama-cpp-turboquant/pull/197)), benchmarked head-to-head against the strongest external turbo4 fork (`spiritbuun/master`) on one RTX 5090. Both forks built from source; symmetric turbo4; Qwen3.6-35B-A3B (Q4_K_XL); KL divergence vs an f16-KV reference over 32768 tokens/depth.

## Result (both forks 4.125 bpw)

| depth | Mean KLD (ours / ref) | prefill t/s (ours / ref) | decode t/s (ours / ref) |
|------:|:---------------------:|:------------------------:|:-----------------------:|
| 2048  | **0.00930** / 0.00953 | **8334** / 8110 | **223** / 205 |
| 8192  | **0.00741** / 0.00753 | **8051** / 7984 | **222** / 200 |
| 16384 | **0.00840** / 0.00857 | **7760** / 7715 | **206** / 194 |
| 32768 | **0.00793** / 0.00796 | 7265 / 7297 | **189** / 188 |

Lower KLD at every depth; higher decode at every depth; higher prefill at 2048/8192/16384, dead-heat at 32768.

## What changed

1. **Corrected the 4-bit Lloyd-Max centroid table.** The *Python reference here was always correct* (`codebook.py` → `_lloyds_gaussian(16, σ=1/√d)`), but the C/CUDA port had drifted to a hardcoded table fit to σ≈0.064 — ~0.61× too narrow for the post-FWHT N(0,1/128) envelope (outer 0.1739 vs correct 0.2402). That clipped ~4.9% of the rotated tails → ~2.1× excess quantization MSE, inflating both PPL and the q·k logit variance (KLD). Re-deriving from the reference cut Mean KLD ~33%. A new regression test (`tests/test_codebook.py::test_4bit_centroids_pinned`, and the 3-bit equivalent) now pins the correct envelope so a port can be checked against it.

2. **Dropped a dead `rnorm` field → 4.125 bpw.** The C 4-bit block carried an unused fp16 `rnorm` (only the legacy 3-bit+QJL path reads it). The reference never had it; removing it from the port makes the block 66 B = 4.125 bpw, bit-identical quality. (README updated: turbo4 is 4.125 bpw / 3.9×, not 4.25 / 3.8×.)

3. **Fixed a perplexity-tool int32 overflow** (`i*nv` / `n_token*nv` indexing) that SIGSEGV'd KL-divergence runs at n_ctx ≥ ~28K — this was the "SEGV @ 32K" some comparison tables show, not an inference crash.

4. **Backported Programmatic Dependent Launch** (upstream #22522) — closes the decode kernel-launch-latency gap on Blackwell (helps f16 and turbo decode, quality-neutral).

5. **Fused turbo4 MMA decode path** — routes turbo4 token-generation onto the GQA-packed tensor-core kernel (inline dequant in the WHT-rotated domain). With grouped-query attention this reads each KV element once per head-group instead of per query head, which flattens decode at depth. Quality-neutral (KLD identical; divergence is f16 reduction-order noise, the same that separates f16-MMA from f16-VEC).

## Asymmetric quality config

Consistent with the K-dominates-KLD finding: **q8_0-K + turbo4-V (~6.3 bpw) cuts Mean KLD ~26%** vs symmetric turbo4 (−28% vs the reference fork), growing with depth. f16-K buys nothing further (K saturates at q8_0; the residual KLD is all V-side), so q8_0-K is the efficient K. Caveat: "V is free" holds at turbo4 (4-bit) but **breaks at turbo2** — q8_0-K + turbo2-V regresses KLD ~2.4× (the 2-bit V error dominates).

## Cross-backend

The changes compile cleanly under HIP/ROCm on an RX 9070 XT (gfx1201, ROCm 7.1), including the new turbo-MMA instances; the MMA decode path is gated to CUDA tensor cores and falls back to VEC on AMD. (turbo3/turbo2 are regression-free; their decode-at-depth still uses VEC — extending the MMA gate to them is straightforward follow-up.)
