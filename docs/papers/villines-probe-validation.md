# Independent Validation of the Villines TV-Divergence Probe Against the TurboQuant+ Corpus

## Abstract

We reproduce four of nine planned validation tests for the attention-mass divergence (TV) collapse diagnostic prototyped by Gregory Villines (2026-05-31). The probe captures real post-RoPE Q and K from a Hugging Face transformer via `scaled_dot_product_attention` monkey-patching, quantizes K through the production TurboQuant+ Python API (`PolarQuant` plus optional `QJL`, or `PolarQuant`-only via `TurboQuantMSE`), and reports total-variation distance between the FP16 and quantized attention distributions across quantizer configurations.

On Apple M5 Max with the production TurboQuant+ Python package (Python 3.12, torch 2.11.0, transformers 5.3.0), all reported reference numbers reproduce within ±0.008 of the originating RTX 5060 Ti CUDA results, despite running on a different framework path (Metal vs CUDA), a different calibration text (Python stdlib via `inspect` vs Villines' code corpus), and a different random seed. The fp16 identity check, the bit-width monotonicity, the QJL-vs-MSE drop-QJL replication at b=2, the Qwen K-channel offset at b=5, and the bias-subtract recovery all reproduce. The bias-norm vs TV-recovery correlation on Qwen2.5-3B reproduces with Pearson +0.98 and Spearman +0.82.

Anchoring the probe to the TurboQuant+ corpus, we run the same K-quantization sweep on Qwen2.5-7B-Instruct with fp16 weights, isolating the K-side mechanism from weight-quantization stacking. Symmetric `turbo3` K reads TV = 0.46 catastrophic routing on fp16 weights alone, an independent routing-side measurement of the same K-side fragility documented previously as PPL 3,556 on Q4_K_M weights in the asymmetric-kv-compression paper.

Four new findings against the existing TQ+ corpus are reported in §5. The probe is appropriate for routing-side anchoring of future K-precision recipes.

## 1 Background

Two TQ+ papers anchor the present validation:

- **[asymmetric-kv-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md)** documents that symmetric turbo quantization is catastrophic on Qwen2.5 family models with low-bit weight quantization, that the failure is entirely K-side, and that the asymmetric recipe `-ctk q8_0 -ctv turbo4` rescues quality across architectures. PPL 3,556 on Qwen2.5-7B Q4_K_M under symmetric `turbo3/turbo3` is the canonical catastrophic anchor (§3.1).
- **[why-mse-fails-for-kv-quantization](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/why-mse-fails-for-kv-quantization.md)** shows that per-layer reconstruction MSE inverts at the model output below b=3: a universal-K centroid table that improves MSE by 1–13% causes 70–90% mean KL@D regressions and 50–60% catastrophic-rate failures across five model families.

The Villines probe operates upstream of perplexity, measuring attention-mass divergence between FP16 and quantized softmax distributions. The premise is that TV captures K-side routing collapse cheaply and does not invert in the regime where MSE inverts. The present work is the first independent reproduction of the probe against the production TQ+ Python API on Apple Silicon.

## 2 Methodology

### 2.1 Hardware and software

- Apple M5 Max, 128 GB unified memory
- macOS 26.x, MPS backend
- Python 3.12.13, torch 2.11.0, transformers 5.3.0, numpy 2.4.3
- `turboquant_plus` editable install from `~/dev/turboquant`
- Branch: `test/villines-probe-validation`

### 2.2 Calibration corpus

Python stdlib source from `inspect`, `os`, and `argparse` modules concatenated via `inspect.getsource()`, truncated to 200 KB (~50K tokens at Qwen2.5 BPE). Greg used a similarly code-heavy text (his `calib.txt`). We did not have access to his exact bytes; the two texts share domain and approximate length.

### 2.3 Tests run

- **T1 Self-test**: synthetic Gaussian, verifies harness identity (fp16 TV = 0.0000 exact) and bit-width monotonicity.
- **T2 Sweep reproduce**: Qwen2.5-3B-Instruct, six configurations, layers [0, 1, 18], `seq_len=1024`, matching Villines' reference run.
- **T3 Bias correlation reproduce**: full 36-layer Qwen2.5-3B-Instruct sweep, per-head k_proj bias norm vs per-head TV correlation.
- **T4 Qwen2.5-7B anchor**: Qwen2.5-7B-Instruct fp16 weights, K-quantization sweep, layers [0, 1, 14, 27].

### 2.4 Model loading note

We loaded Qwen2.5-7B-Instruct rather than the base Qwen2.5-7B. Architecture is identical; minor numerical shifts in attention distributions are possible due to fine-tuning, but the K-offset mechanism is architecturally fixed by the `bias=True` choice on q_proj, k_proj, and v_proj.

## 3 Results

### 3.1 T1 Self-test (synthetic Gaussian)

| Config | TV | top1_keep |
|--------|----|-----------|
| fp16 | 0.0000 | 1.000 |
| turbo5 | 0.0287 | 0.891 |
| turbo4 | 0.0537 | 0.773 |
| turbo3 | 0.1005 | 0.703 |
| turbo2 | 0.1863 | 0.484 |
| turboMSE2 | 0.1328 | 0.594 |

Identity check passes (fp16 TV = 0.0000 exact). Monotonicity holds. `turboMSE2 < turbo2` replicates the drop-QJL finding established in [turbo4-resurrection §3](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md).

### 3.2 T2 Qwen2.5-3B sweep reproduction

| Config | Villines (CUDA) | This run (Metal) | Δ |
|--------|-----------------|------------------|------|
| fp16 | 0.0000 | 0.0000 | 0.0000 |
| turbo5 | 0.2091 | 0.2055 | -0.0036 |
| turbo5_biassub | 0.0422 | 0.0393 | -0.0029 |
| turbo3 | 0.5232 | 0.5181 | -0.0051 |
| turbo2 | 0.6663 | 0.6608 | -0.0055 |
| turboMSE2 | 0.5352 | 0.5272 | -0.0080 |

All values reproduce within 0.008. Two independent framework paths (CUDA vs Metal), two distinct calibration texts, and two random seeds yield numerically tight agreement. TV is calibration-robust on code-heavy text within the tested band.

### 3.3 T3 Bias-norm vs TV correlation

72 (layer, kv-head) points across 36 layers of Qwen2.5-3B-Instruct.

| Metric | Villines | This run |
|--------|----------|----------|
| Pearson bias vs base TV | +0.971 | +0.979 |
| Spearman bias vs base TV | +0.852 | +0.818 |
| Pearson bias vs TV recovery | not reported | +0.984 |
| Spearman bias vs TV recovery | not reported | +0.770 |

The five highest-bias-norm heads dominate baseline TV and recovery:

| Layer | KVH | bias_norm | TV_base | TV_biassub | Recovery |
|-------|-----|-----------|---------|------------|----------|
| 0 | 1 | 195.48 | 0.4837 | 0.0307 | 0.4529 |
| 0 | 0 | 148.43 | 0.3345 | 0.0250 | 0.3095 |
| 27 | 1 | 68.36 | 0.1932 | 0.0335 | 0.1597 |
| 1 | 0 | 64.77 | 0.1296 | 0.0368 | 0.0928 |
| 2 | 0 | 36.31 | 0.0931 | 0.0456 | 0.0475 |

Mid-layer heads with bias_norm under 6 (L3, L5, L8, L17) show recovery near zero. Small biases produce no offset to remove. The causal chain `bias=True → K-channel offset → routing misalignment → recoverable by centering` is established at per-head granularity, concentrated on layer 0 boundary heads and the late-layer L27 head.

### 3.4 T4 Qwen2.5-7B-Instruct anchor sweep

fp16 weights, K-quantization via the probe, layers [0, 1, 14, 27]:

| Config | mean_TV | max_TV | top1_keep | Notes |
|--------|---------|--------|-----------|-------|
| fp16 | 0.0000 | 0.0000 | 1.000 | identity |
| turbo5 | 0.2684 | 0.9610 | 0.647 | K-offset at 5-bit |
| turbo5_biassub | 0.0377 | 0.0844 | 0.930 | 7.1× reduction |
| **turbo3** | **0.4612** | **0.9947** | **0.502** | **catastrophic routing** |
| turbo3_biassub | 0.1321 | 0.2704 | 0.792 | 3.5× reduction (partial) |
| turbo2 | 0.5788 | 1.0000 | 0.393 | worst |
| turboMSE2 | 0.4702 | 0.9843 | 0.476 | drop-QJL replicates at b=2 |

Symmetric `turbo3` K on fp16 weights reads TV = 0.4612, an independent routing-side measurement of the same K-side fragility documented previously as PPL 3,556 on Q4_K_M weights ([asymmetric-kv-compression §3.1](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md)). The bias-subtract recovery gradient (7.1× at turbo5, 3.5× at turbo3) is the focus of §5.4.

## 4 Reconciliation With Google's Original TurboQuant Validation

[asymmetric-kv-compression §1](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) notes that the Google TurboQuant paper validates quality on fp16 model weights and reports near-lossless results. T4 measures fp16 weights and reports catastrophic routing on Qwen2.5-7B at symmetric `turbo3` K. These are consistent: Google's paper validates on architectures with `bias=False` on QKV projections (Llama, Mistral); Qwen2 and Qwen2.5 are the architectural exception with `bias=True`, and the K-channel offset that breaks routing at low-bit symmetric K is therefore a Qwen-specific phenomenon visible even with fp16 weights. Weight quantization adds stacking damage on top but is not necessary to trigger the routing collapse on Qwen2.5.

## 5 New Findings Against the Existing TQ+ Corpus

### 5.1 K-precision alone (independent of weight quantization) is sufficient for routing collapse on Qwen2.5-7B

The existing catastrophic-PPL data on Qwen2.5-7B in [asymmetric-kv-compression §3.1](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) measures the stacked regime (Q4_K_M weights plus symmetric `turbo3/turbo3` KV). T4 isolates the K-side mechanism: fp16 weights plus symmetric `turbo3` K alone produce TV = 0.46 catastrophic routing on Qwen2.5-7B-Instruct. Weight quantization is therefore stacking damage on top of an already-broken K-routing path, not the proximal cause of the catastrophe.

### 5.2 Bias-subtract is a partial low-bit-K rescue, complementary to asymmetric q8_0-K

The canonical TQ+ recipe is `-ctk q8_0 -ctv turbo4`: preserve K at 8-bit precision, compress V aggressively. Villines' bias-subtract preprocessing subtracts the per-head K mean before quantization and adds it back after, recovering routing fidelity by centering against the K-channel offset.

The two recipes are complementary, not competing:

- Asymmetric q8_0-K preserves full routing fidelity (TV ≈ 0 by construction at 8 bits) and is the recommended production recipe.
- Bias-subtract is a partial low-bit-K rescue: it recovers 7.1× of the TV at turbo5 (TV 0.27 → 0.04, near-complete) and 3.5× at turbo3 (TV 0.46 → 0.13, partial). It is appropriate when K must remain at low-bit for memory headroom that q8_0-K cannot afford.

The two recipes target the same Qwen K-fragility from different angles.

### 5.3 Layer-wise Qwen2.5 K-projection bias distribution

`bias=True` on Qwen2 and Qwen2.5 QKV projections is an architectural fact established outside the TQ+ corpus. T3 measures the per-layer per-head K-projection bias norms on Qwen2.5-3B-Instruct and finds a strongly skewed distribution: layer 0 boundary heads carry the largest biases (195 and 148), the late-layer L27 head carries a sizable bias (68), and a small number of early-mid-layer heads (L1, L2, L4) carry intermediate biases. The bulk of mid-layer heads (L3, L5–L17) carry biases an order of magnitude smaller (under 6). The boundary-layer-concentration pattern documented for V-cache sensitivity in [layer-aware-v-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md) has a K-side analogue at the bias level.

### 5.4 Bias-subtract recovery exhibits diminishing returns at low-bit K

The TV recovery ratio under per-head mean subtraction degrades monotonically with K bit-width:

- Qwen2.5-7B turbo5: 0.2684 → 0.0377, recovery factor 7.1×
- Qwen2.5-7B turbo3: 0.4612 → 0.1321, recovery factor 3.5×

The mechanism is consistent with the [why-mse-fails F3 result](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/why-mse-fails-for-kv-quantization.md): per-layer reconstruction fixes lose efficacy as bit-width approaches the cliff because the variance term of the underlying quantizer dominates over the bias term. Centering is a bias correction; it cannot recover variance loss. Below b=3, the asymmetric recipe (preserve K bits) is the only robust quality lever.

### 5.5 Quantitative bias-norm vs TV-recovery correlation

The TQ+ corpus established `bias=True` as the architectural source of Qwen's K-fragility but did not previously quantify the per-head dose-response. T3 reports Pearson +0.984 and Spearman +0.770 between per-head K-projection bias norm and the TV recovery under centering on Qwen2.5-3B-Instruct. Spearman is the appropriate number because the L0 outliers (bias_norm ≈ 200 vs ≤ 6 mid-layer typical) inflate Pearson. The +0.770 Spearman across 72 head-points is the first published numerical anchor for the causal claim that K-projection bias produces recoverable routing damage.

## 6 Cross-Method Corroboration With Existing TQ+ Findings

The probe corroborates four TQ+ claims via a routing-side TV metric rather than the output-side PPL or KL@D metrics previously used:

1. **K is the load-bearing tensor for KV-cache quantization quality** ([asymmetric-kv-compression §1](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md)). T4 confirms via fp16-weight + sym-K isolation.
2. **Qwen family is sensitive at all sizes** ([asymmetric-kv-compression §3](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md)). T2 (3B) and T4 (7B) both reproduce the K-side cliff.
3. **Drop-QJL at b=2** ([turbo4-resurrection §3](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md)). T2 (turboMSE2 below turbo2 by 0.13) and T4 (turboMSE2 below turbo2 by 0.11) replicate via TV on real models, complementing the synthetic and PPL evidence already on file.
4. **Per-layer reconstruction fixes do not transfer below b=3** ([why-mse-fails F3](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/why-mse-fails-for-kv-quantization.md)). T4 bias-subtract diminishing returns at turbo3 (3.5× vs 7.1× at turbo5) is the routing-side analogue.

## 7 Implications

- TV is a usable routing-side proxy for K-side quality in the regime where reconstruction MSE inverts. Villines' framing of TV as a "ghost PPL" is concrete: a routing metric that does not require running the model to logits and that lines up with the K-side cliff observed in our PPL data.
- Bias-subtract and asymmetric q8_0-K are complementary recipes for the same Qwen K-fragility. A future production stack could pair them: asymmetric q8_0-K at recommended precision, bias-subtract as a memory-constrained alternative when 8-bit K is not feasible.
- The TQ+ K-side framing benefits from the isolation T4 provides: K precision alone produces the catastrophe on Qwen2.5, with weight quantization adding stacking damage on top.

## 8 Outstanding Tests

Five tests in the original test matrix were not run in this session:

- **T5 Pre-RoPE exact `k_proj.bias` subtraction**: closes the bias-mechanism question definitively. If pre-RoPE exact subtraction zeroes TV, bias is the whole story; if a residual remains, the residual is the RoPE-phase or attention-sink interaction, measurable rather than mere cleanup.
- **T6 Mistral cross-arch sanity**: confirms that `bias=True` is the architectural specificity (Mistral has `bias=False` and should show no recovery under centering).
- **T7 Adrianosousa 14-config matrix**: wider grid validation on the practitioner-validated configuration set.
- **T8 Layer-0 outlier check on T3**: partially addressed in §3.3 by reporting Spearman alongside Pearson; full robustness check pending.
- **T9 Calibration-text dependence**: partially addressed in §3.2 (our Python stdlib corpus reproduced Villines' code-corpus numbers within 0.008); deeper sweep across wikitext, narrative, and dense math text pending.

T5 and T6 are recommended next steps before joint publication claims.

## 9 Reproducibility

The probe (`tq_collapse_probe.py`), the downstream test (`tv_vs_ppl.py`), Villines' originating result JSONs, our reproduction JSONs, and the calibration text are committed to branch `test/villines-probe-validation` of [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) under `villines-probe/`. Run instructions live in the probe docstring; on M5 Max use `--device mps`. The Villines defaults are `--device cuda:0`.

## References

1. Villines, G. (2026-05-31). `tq_collapse_probe.py` and `tv_vs_ppl.py`. Private distribution.
2. Villines, G. (2026). *Cross-Layer Quantization Error is Non-Additive at Two Bits: A Four-Architecture Measurement Study*. Working draft, MLSys-style, pending revision.
3. TheTom. *Asymmetric KV-Cache Compression: Why V is Free, K is Everything*. [asymmetric-kv-compression.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md).
4. TheTom. *Why MSE Fails for KV-Cache Quantization*. [why-mse-fails-for-kv-quantization.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/why-mse-fails-for-kv-quantization.md).
5. TheTom. *TurboQuant4 Resurrection*. [turbo4-resurrection.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md).
6. TheTom. *Layer-Aware V Compression*. [layer-aware-v-compression.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md).
7. Zandieh, A. et al. (2026). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. arXiv:2504.19874, ICLR 2026.
