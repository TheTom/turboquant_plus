# When Quantized Beats fp16: A KV-Rotation Investigation, and Why PPL Lies on gemma-class Instruct Models

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## Abstract

Master llama.cpp PR [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) adds a Walsh–Hadamard rotation inside the attention kernel for any quantized KV cache. Our fork ships TurboQuant's own kernel-level WHT rotation, so the master rotation looked redundant, then looked harmful: an external user ([@erazortt](https://github.com/erazortt), TheTom/turboquant_plus#88) reported that turning master's rotation off rescued KLD on gemma-4 26B-A4B Q6_K_XL with symmetric turbo*. The original fork shipped with rotation defaulted OFF (binary `LLAMA_ATTN_ROT_DISABLE` env). We set out to find a *better* default than that.

We did not. Three iterations of "smart" defaults all failed differently: v2 enabled rotation broadly and broke symmetric turbo; v3 added a per-side gate that skipped turbo types and turned out to be the *worst* asymmetric config on gemma-4 (+6.8% PPL on q8/turbo4); a v4 candidate that allowed V-rotation regardless of V type would have crashed phi-4. Across 7 model families on `-ctk q8_0 -ctv turbo4`, the optimal rotation policy splits four ways. Inside the gemma-4 family alone, three sizes want three different optima. We landed on rotation OFF on both sides plus two new env knobs, `LLAMA_ATTN_ROT_K_OVERRIDE=1` and `LLAMA_ATTN_ROT_V_OVERRIDE=1`, that let users opt each side in independently. The effective default is identical to the original fork; the contribution is per-side control and the matrix that documents which models want which knob.

While running this matrix we hit a second, larger problem. On three gemma-4 instruct GGUFs, quantized KV scores 7–42% *below* the fp16-KV baseline on wikitext-2 (model weights are still Q8_0 throughout; only the KV cache is being varied). The artifact is also present at q8/q8 with no rotation, so it is not a TurboQuant- or rotation-specific phenomenon. The effect persists from ctx=512 to ctx=2048 on the 26B-A4B MoE; on the dense 31B it disappears at ctx=2048 (short-context magnified there). KL divergence vs the fp16-KV reference, measured on the same setup, points the other way: V-only rotation lowers PPL by 3.9% but raises KLD by 6.1% and same-top-p agreement drops. PPL ranks the configurations backward; KLD ranks them in the order a faithful-quantization metric should. We are not the first to observe this pattern (ggerganov in PR #21038 explicitly recommended tracking KLD over PPL on similar data; @vektorprime in [#21394](https://github.com/ggml-org/llama.cpp/issues/21394) hit the same ranking flip on gemma-4 31B); what this paper adds is a controlled per-side rotation matrix, a direct PPL-vs-KLD ranking-inversion measurement on the same eval, and the resulting engineering policy. The recommendation for anyone validating KV-quantization changes on gemma-class instruct models is to use KLD against fp16-KV logits, not corpus PPL.

---

## 1. Background

### 1.1 Two rotations, one cache

TurboQuant compresses KV cache entries by first rotating each head vector with a Walsh–Hadamard transform (WHT), then quantizing the rotated coordinates with a polar codebook. The WHT spreads norm uniformly across coordinates, which makes the quantizer's Gaussian-shaped centroid grid a good fit. This is a *quantization-side* rotation: the rotation lives inside the cache format itself, applied at write time and undone implicitly by the codebook at read time. Our fork has shipped this rotation since the original TurboQuant integration.

Master llama.cpp PR [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) adds a different rotation. It rotates Q and K (`k_rot`) and rotates V with an involutory undo at the output (`v_rot`) inside the attention kernel itself, applied to *any* quantized K/V type. The rotation matrix is a Hadamard of size `nrot` (default 64, configurable via `LLAMA_ATTN_ROT_K_NROT`). The stated goal is the same as TurboQuant's WHT: reshape the per-head distribution so quantization centroids are well-matched. This is an *attention-side* rotation: applied during compute, independent of the cache format.

When both rotations are present, master's attention rotation composes with TurboQuant's quantization rotation. The composition can be helpful (stronger spreading of outliers), neutral (already well-conditioned), or harmful (the second rotation breaks the assumptions of the first). This paper is about which of those three you actually get on real models.

### 1.2 Prior reports of the same pattern

We are not the first to hit this artifact. Three independent reports cover adjacent terrain:

- **llama.cpp issue [#21394](https://github.com/ggml-org/llama.cpp/issues/21394) — "Gemma4 attn_rot_k and v = 0".** Master's PR #21038 *automatically disables* attention rotation on gemma-4 because its per-layer head_dim is heterogeneous (different layers have different head dimensions; SWA layers stay fp16; non-SWA can be quantized). When @stduhpf force-enabled it, @vektorprime ran gemma-4 31B at ctx=512 wikitext with q8_0 KV: PPL(Q) = 2289 vs PPL(base) = 1206 — quantized scoring nearly 2× *worse* than the fp16 reference, while their q4_0 cell scored *lower* than q8_0 (1573 vs 2289). Same model size, same context length, same corpus, same ranking-flip pattern we hit. PR [#21513](https://github.com/ggml-org/llama.cpp/pull/21513) was opened to "support attention rotation for heterogeneous iSWA" but the underlying eval-instability is unresolved.
- **AesSedai's tables inside PR [#21038](https://github.com/ggml-org/llama.cpp/pull/21038)** show several KV-quant configs with PPL strictly below the fp16 baseline while KLD is strictly above. The PR author (@ggerganov) responded: *"It seems important to track the KLD rather than PPL (maybe more significant for Qwen3.5)."* That is the same recommendation we land on in §4.3, made by the master rotation author about his own PR's data, before this investigation started.
- **localbench's KV-cache benchmark** ([substack](https://localbench.substack.com/p/kv-cache-quantization-benchmark)) measures gemma-4 26B-A4B q8_0 KLD = 0.377 vs Qwen at 0.04 — gemma-4 is an order of magnitude noisier under KV quantization than peer families using the same protocol.
- **llama.cpp issue [#22407](https://github.com/ggml-org/llama.cpp/issues/22407)** documents non-monotonic gemma-4 E4B base PPL across *weight* quantizations (BF16=7.11, Q4_K_M=23.06, Q3_K_M=919.46) — eval instability on this family is broader than just KV cache, plausibly tied to the per-layer-embedding (PLE) mechanism specific to E2B/E4B.

What this paper adds on top of those: a controlled per-side rotation matrix across 7 model families on the *same* setup (not just gemma), the cross-format matrix on q8/q8 vs q8/turbo4 vs t4/t4 showing the artifact survives KV format changes, a direct PPL-vs-KLD ranking inversion on the same eval (turning ggerganov's "track KLD" suggestion into a reproduced demonstration), and the engineering decision (per-side env knobs + documented per-model recommendations) that follows.

### 1.3 The trigger: erazortt #88, and the original fork default

@erazortt reported on TheTom/turboquant_plus#88 that gemma-4 26B-A4B Q6_K_XL with `-ctk turbo4 -ctv turbo4` lost noticeable quality after we picked up master's PR #21038, and that disabling master's rotation via `LLAMA_ATTN_ROT_DISABLE=1` rescued it. The fork already shipped with rotation defaulted OFF for exactly this case. The relevant code, prior to this investigation, was:

```cpp
const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE
    ? atoi(LLAMA_ATTN_ROT_DISABLE) : true;  // default ON-disable, i.e. OFF
```

The accompanying comment cited two reasons: TurboQuant's own kernel WHT already rotates the cache, and master's rotation crashed phi-4 with a graph hash overflow. The original maintainer had paid the cost of discovering both of these and made the right call. What was missing — and what erazortt's report exposed — was a way for users with non-turbo asymmetric configs (where rotation might help) to opt in per side, plus documentation pointing them at the env var. The investigation that follows is the search for a *better* default than OFF. There is no better default; there are per-side knobs that let users tune the rotation per model.

---

## 2. Three Iterations of "Smart" Defaults

Before the matrix in §3, we tried three candidate defaults that each looked correct on a small sample and failed on a wider one. The history matters because it is the reason the final policy is "default OFF + per-side knobs" rather than any of the heuristics we tried first.

### 2.1 v2 — broad enable on quantized KV

Initial response: enable rotation on both sides whenever the cache type is quantized and head-dim aligned, matching master's intent. This rolled out and immediately regressed @erazortt's symmetric `turbo4 / turbo4` config (the original report). On a gemma-4 26B-A4B Q6_K_XL KLD eval, rotation ON gave Mean KLD 2.589 / Same-top-p 52.9% vs rotation OFF Mean KLD 2.251 / Same-top-p 55.9% — clear regression on the model that triggered the report. v2 reverted within hours.

### 2.2 v3 — per-side gate that skips turbo types

The next candidate added a per-side gate that enabled rotation only when the cache type was *not* turbo, on the theory that turbo's own kernel WHT and master's attention WHT compose into a double-rotation that breaks the codebook. In code: `attn_rot_k = !is_turbo(type_k) && quantized && head_dim%64==0`, same for V. On q8/q8 this enables both sides (master parity); on `q8 / turbo4` it enables K-only; on `turbo4 / q8` it enables V-only; on `turbo4 / turbo4` it disables both. Looked sensible. PPL on Qwen3.5-2B q8/turbo4 was within standard error of OFF, which we took as a green light.

### 2.3 The v3 matrix that killed v3

When we ran v3 on the wider matrix, the gate turned out to be backwards on gemma-4. On `-ctk q8_0 -ctv turbo4`, gemma-4 26B-A4B Q8 results:

| Config | rot k/v | PPL | Δ vs OFF |
|--------|---------|-----|----------|
| OFF | 0/0 | 6273 ± 510 | — |
| **v3 default (K-only)** | 1/0 | **6700 ± 547** | **+6.8% (worst)** |
| broad (K+V) | 1/1 | 6176 ± 511 | −1.6% |
| V-only | 0/1 | 6027 ± 496 | **−3.9% (best)** |

v3's per-side gate enabled rotation on the q8 K side and disabled it on the turbo4 V side, which is exactly the *worst* asymmetric configuration on this model. The empirically best configuration on the same row is V-only — rotating the turbo V despite the "double-rotation" theory predicting it would break, and skipping the q8 K despite the same theory predicting it would help. The theory was wrong about both sides simultaneously.

### 2.4 v4 candidate — drop the V-side turbo guard

A v4 candidate, derived from the row above, was: keep the K-side turbo guard (PPL +52.7% on `t4 / t4` K-only at the time looked catastrophic — though §4.6 later shows that PPL number is itself part of the artifact and the cell is actually the lowest-KLD on its row), drop the V-side turbo guard, leave everything else the same. This would have given `q8/turbo4 → broad (k=1, v=1)` instead of v3's `K-only`, scoring 6176 (−1.6%) — better than v3, worse than V-only. Before we shipped it we ran the same configuration on phi-4. Phi-4 V-side rotation crashes the inference graph with a hash-table overflow. The v4 candidate would have bricked phi-4 inference for any user with a quantized V cache. We did not ship it.

### 2.5 v4 actual — back to default OFF, plus per-side knobs

The final v4 is functionally identical to the original fork default — rotation OFF on both sides — with two additions: per-side env-var overrides (`LLAMA_ATTN_ROT_K_OVERRIDE`, `LLAMA_ATTN_ROT_V_OVERRIDE`) and the matrix in §3 documenting which models benefit from which override. The legacy `LLAMA_ATTN_ROT_DISABLE=1` is preserved as a hard lock-out that blocks the per-side overrides for users who want a single switch to guarantee no rotation. The PR was originally framed as a fix; the honest framing is "feat: per-side env-knob opt-in for upstream attention rotation, plus a matrix telling you when to set them." The original default was correct from the start.

---

## 3. The Per-Side Rotation Matrix

### 3.1 Setup

- **Hardware:** Apple M5 Max, 128 GB unified memory, Metal flash attention.
- **llama.cpp:** TheTom fork at branch `fix/enable-attn-rot-by-default`, post-fix commit `817e913ec` (the prior `db3595a755a9` had a bug where the per-side override env knobs were silently no-ops; see Reproducibility for details). Build target `build-test/bin/llama-perplexity` (Metal-only fast iteration build, `EMBED=OFF`).
- **Corpus:** `wikitext-2-raw/wiki.test.raw`, ctx=512, 32 chunks (matches existing TurboQuant+ paper conventions).
- **KV formats under test:** primary `-ctk q8_0 -ctv turbo4` (asymmetric, the TurboQuant+ recommended default per [`asymmetric-kv-compression.md`](asymmetric-kv-compression.md)), plus `q8_0 / q8_0` (symmetric, master's intended use case) and `turbo4 / turbo4` (symmetric turbo) on the headline gemma-4 26B-A4B Q8 model.
- **Rotation control:** `LLAMA_ATTN_ROT_K_OVERRIDE` and `LLAMA_ATTN_ROT_V_OVERRIDE`, four configurations per model:
  - `OFF` — both env vars unset, no master rotation.
  - `K-only` — `LLAMA_ATTN_ROT_K_OVERRIDE=1`.
  - `V-only` — `LLAMA_ATTN_ROT_V_OVERRIDE=1`.
  - `broad` — both set.

The cache type and head-dim alignment guards inside `llama_kv_cache_unified` still apply: the override only takes effect on quantized types with `head_dim % 64 == 0`. nrot is left at the master default (64).

### 3.2 Cross-format matrix on gemma-4 26B-A4B Q8

To check whether the per-side rotation policy depends on the cache *format* and not just the model, we ran the full 4-config × 3-format matrix on the headline model (gemma-4 26B-A4B Q8, MoE+SWA). PPL ± SE, ctx=512, 32 chunks:

| KV format | OFF (0/0) | K-only (1/0) | V-only (0/1) | broad (1/1) | Δ best vs OFF |
|-----------|-----------|--------------|--------------|-------------|---------------|
| q8 / q8 | 9979 ± 832 | 10153 ± 847 (+1.7%) | 10451 ± 873 (+4.7%) | 10118 ± 846 (+1.4%) | OFF best |
| q8 / turbo4 | 6273 ± 510 | 6700 ± 547 (+6.8%) | **6027 ± 496 (−3.9%)** | 6176 ± 511 (−1.6%) | V-only |
| turbo4 / turbo4 | **5785 ± 471** | 8831 ± 728 (**+52.7%** — PPL only; KLD says K-only is BEST, see §4.6) | — | 7031 ± 582 (+21.5%) | depends on metric |

Two PPL-level takeaways, each of which §4.6 revisits under KLD and partially overturns:

- **K-side rotation on top of turbo K registers as destructive *by PPL*.** `t4 / t4` K-only costs +52.7% PPL. The naïve interpretation is that the two WHTs (TurboQuant's kernel-level on the K cache, master's attention-side on K) compose into a transform that no longer aligns with the TurboQuant codebook, with the resulting K error amplified by softmax. **§4.6 partially overturns this:** under the KLD oracle the same K-only cell is *closer* to the fp16-KV reference than OFF (KLD 2.029 vs 2.133), so the +52.7% PPL is itself an artifact in the same family as the headline. The composition is not as destructive as PPL suggests; the real evidence against K-on-turbo-K rotation is the broad-cell KLD (+9.7%), not the K-only PPL.
- **V-side rotation on top of turbo V is fine on q8/turbo4, mixed on t4/t4.** `q8 / turbo4` V-only is the best PPL cell on that row, but its KLD is *worse* than OFF (§4.3). The `t4 / t4` V-only cell did not run cleanly under our build at the time of measurement and is omitted; the broad cell (which includes V-side turbo rotation) is +21.5% PPL and +9.7% KLD, so on `t4 / t4` V-side turbo rotation is genuinely worse by both metrics, but the K-only data above suggests that *most* of the broad-cell harm comes from the V composition, not from K composition as we originally inferred.

The PPL numbers for `q8 / q8` (~10000), `q8 / turbo4` (~6000), and `t4 / t4` (~5800) on the same model and same corpus are wildly inconsistent — quantization should produce a small, monotonic increase in PPL, not 2× swings. This is a preview of the artifact we document in §4. The KLD picture is much cleaner (q8/q8 KLD ~0.5, q8/turbo4 KLD ~1.7, t4/t4 KLD ~2.1 — monotonic with bit rate), reinforcing the §4.3 conclusion that KLD is the right metric on this model.

### 3.3 Cross-model matrix on q8/turbo4

The asymmetric `q8 / turbo4` configuration is the TurboQuant+ default and the row most users care about. PPL ± SE, ctx=512, 32 chunks, across 7 model families:

| Model | OFF | K-only | V-only | broad | Best |
|-------|-----|--------|--------|-------|------|
| gemma-4 31B Q8 (it) | 8685.45 ± 733.07 | 9009.24 ± 760.97 | **4924.15 ± 408.13** | 4818.38 ± 398.82 | V-only / broad (−43%) |
| gemma-4 26B-A4B Q8 (it) | 6273.74 ± 510.44 | 6700 ± 547 | 6027.71 ± 496.38 | 6176.40 ± 510.78 | V-only (−3.9%) |
| gemma-4 E2B Q4_K_L (it) | **114.77 ± 6.50** | 115.36 ± 6.54 | 122.41 ± 6.98 | 122.42 ± 6.98 | OFF (V-only +6.7%) |
| Qwen2.5-7B Q8 (it) | 6.140 ± 0.174 | 6.146 ± 0.174 | 6.135 ± 0.173 | 6.116 ± 0.172 | within ± SE |
| Qwen3.5-2B Q8 (base) | 10.794 ± 0.324 | 10.791 ± 0.324 | 10.692 ± 0.325 | 10.692 ± 0.325 | within ± SE |
| phi-4 Q8 (base) | 5.824 ± 0.152 | 5.818 ± 0.152 | crash | crash | OFF / K-only |
| Mistral-Small-24B Q4 (it) | 5.317 ± 0.131 | 5.318 ± 0.131 | 5.326 ± 0.131 | 5.326 ± 0.131 | within ± SE |

The phi-4 V-side crashes are graph-hash overflows reproducible across runs and warrant a separate fix; for the policy decision, "crashes" is a strong vote against any default that turns V-rotation on for phi-4. Note also that the gemma-4 26B-A4B and 31B "V-only wins" are partly artifact (§4); the gemma-4 E2B regression on the same row is real (PPL is in a believable range there).

### 3.4 Architecture-quirk hypotheses

We do not have a confirmed mechanism for why the rotation policy splits this way. The dominant hypothesis is now external to this paper: per llama.cpp issue [#21394](https://github.com/ggml-org/llama.cpp/issues/21394), gemma-4 has *heterogeneous per-layer head_dim* — different layers have different head dimensions, SWA layers stay fp16 while non-SWA layers are quantized, and a single Hadamard rotation matrix sized for the largest head_dim is applied across all of them. PR #21038 explicitly auto-disables rotation on gemma-4 in master for this reason. Our fork's pre-investigation default of OFF on all sides hid the same problem; turning rotation on per side via the env knobs re-exposes it.

Per-family guesses for the rest of the matrix:

- **gemma-4 (heterogeneous head_dim).** When the rotation matrix is sized for the largest head_dim and applied to a layer with a smaller head_dim, the involutory undo at the output is mathematically off — the WHT is not its own inverse on a non-matching dimension. The three sizes (E2B, 26B-A4B, 31B) differ in head_dim distribution, MoE-vs-dense, and weight-quant aggressiveness, which is consistent with the three sizes wanting three different rotation policies even though they share the head_dim heterogeneity. Note: softcap was *removed* in gemma-3 and is not present in gemma-4 (replaced by QK-norm per the [gemma-3 tech report](https://arxiv.org/html/2503.19786v1)), so we drop that as a candidate mechanism.
- **gemma-4 E2B/E4B (PLE).** E2B and E4B additionally use a per-layer embedding (PLE) mechanism that issue [#22407](https://github.com/ggml-org/llama.cpp/issues/22407) flags as the suspected cause of non-monotonic PPL under weight quantization. The same mechanism plausibly contributes to E2B's Q4_K_L row in §3.3 going the opposite direction from the larger gemma-4 sizes.
- **Qwen2.5 / Mistral-Small** are pure-global-attention dense models with uniform per-layer head_dim. Master's rotation has nothing to fix and produces no measurable shift either way.
- **Qwen3.5** is hybrid Mamba+attention; most layers don't maintain a traditional KV cache, so the cells where rotation could matter are a small fraction of the total compute. Effect is below the measurement floor.
- **phi-4** has been a long-standing trouble case for any attention-graph alteration on this fork due to a graph-node hash collision when extra rotation tensors are inserted. The bug is real and orthogonal to rotation quality.

The most actionable follow-up is varying `LLAMA_ATTN_ROT_K_NROT` away from 64 on gemma-4 31B: if the per-side gemma rows shift with nrot but the others don't, the heterogeneous head_dim mechanism is implicated. We run that ablation in §3.6.

### 3.5 What the matrix says

Two quality patterns and one separate-bug observation:

1. **gemma-4 splits three ways inside one architecture family.** 31B wants V-only (−43%, partly artifact per §4). 26B-A4B wants V-only (−3.9%, partly artifact). E2B wants OFF (V-only is +6.7%, in a believable PPL range). A per-architecture default would silently regress the variants we did not test. Note that *master* already auto-disables rotation on gemma-4 because of the heterogeneous head_dim ([#21394](https://github.com/ggml-org/llama.cpp/issues/21394)); our env knobs deliberately bypass that guard so users can opt in to the V-only configuration on the sizes where the matrix or their own KLD measurement supports it. We discuss this trade-off in §5.3.

2. **Qwen and Mistral don't care.** Every cell is within standard error of OFF. Whatever the rotation does on these models is below the measurement floor.

3. **phi-4 V-side crashes — separate llama.cpp bug, not policy evidence.** A graph-node hash collision overflows when an extra rotation tensor is inserted in one of phi-4's layer types. This is independent of rotation quality and would crash regardless of whether the rotation was helpful. We mention it because it constrains the *default* (you cannot ship a default that crashes some users) but it is not evidence about the rotation's quality on phi-4.

Master's PR #21038 default — rotate when the cache type is quantized and head-dim is power-of-2 aligned — is correct for several models in this matrix. By KLD it is in fact correct for gemma-4 26B-A4B q8/q8 (the master use case), where rotation lowers KLD by ~20% (§4.6). By PPL it is wrong for gemma-4 E2B, "catastrophic" on `t4 / t4` K-only (+52.7% PPL — but §4.6 shows this cell is the *best* on its row by KLD, so the catastrophic framing is itself a PPL artifact), within noise for Qwen/Mistral, and would trip the phi-4 graph bug. Auto-disabling on gemma-4 (as master does for the heterogeneity reason) handles three of the gemma rows but blocks the q8/q8 case where rotation actually helps and blocks the V-only configurations users may legitimately want. There is no single setting we could pick in the C++ that does not regress at least one tested model on at least one rotation knob *or* miss a real benefit on another.

### 3.6 nrot ablation on gemma-4 31B

Mechanism (3) in §4.4 (loss of position-specific outliers under rotation) and the heterogeneous head_dim mechanism in §3.4 both predict that varying the rotation tile size `LLAMA_ATTN_ROT_K_NROT` away from the master default of 64 should change the gemma-4 rotation rows. We swept K-only at four nrot values on gemma-4 31B Q8 q8/turbo4 (ctx=512, 32 chunks, vs OFF baseline 8685.45 ± 733.07):

| nrot | rotation tile size | PPL ± SE | Δ vs OFF |
|------|--------------------|----------|----------|
| 64 (master default) | 64 | 9009.24 ± 760.97 | +3.7% |
| 0 (largest pow-2 ≤ head_dim) | 256/512 | 9056.19 ± 764.91 | +4.3% |
| 128 | 128 | **8867.49 ± 748.73** | +2.1% |
| 256 | 256 | 9152.44 ± 774.93 | +5.4% |

K-only PPL on gemma-4 31B is weakly nrot-dependent — all four values land within 3% of each other, all worse than OFF, with no monotone trend. The "bigger tile = worse" prediction of the outlier-smearing mechanism (3) does not cleanly hold; nrot=128 is best, nrot=256 is worst, and 64 (the master default) is in between. This is *mild evidence against* outlier-smearing being the dominant K-side mechanism on this model. The heterogeneous-head_dim mechanism in §3.4 is harder to test from K-side alone because gemma-4 31B has head_dim 256/512 across layers; a tile size that perfectly matches one layer's head_dim still mismatches the others, so all four nrot values produce some mismatch on some layers. The clean test would be a homogeneous-head_dim model run at multiple nrot values, which we do not have for non-gemma in the same precision regime.

V-side nrot is hardcoded at 64 in this fork at the time of writing, so the V-only row is fixed regardless of `LLAMA_ATTN_ROT_K_NROT`. The broad row, which combines K-side at the swept `nrot` with V-side at fixed 64, *does* vary with `LLAMA_ATTN_ROT_K_NROT`:

| nrot | broad PPL ± SE | Δ vs OFF (8685) |
|------|----------------|-----------------|
| 64 (master default) | **4818.38 ± 398.82** | **−44.5%** |
| 0 (largest pow-2 ≤ head_dim, =512) | 5159.06 ± 427.58 | −40.6% |
| 128 | 5104.36 ± 424.21 | −41.2% |
| 256 | 4875.02 ± 404.18 | −43.9% |

Two patterns: (i) the master default nrot=64 is the best in this row (and tied with nrot=256 within standard error), consistent with smaller-tile rotation being more invertible across the heterogeneous head_dim layers; (ii) the broad row varies more with nrot than the K-only row (5.4% spread vs 3.2%), as expected since both K-side rotation and V-side rotation contribute to the result. Like the headline §4.3 finding, this entire row is mostly artifact — PPL drops below the fp16-KV baseline of 5320 at every nrot tested — so the nrot variation is variation *in artifact magnitude* rather than in real model quality. The cleaner test would be on a homogeneous-head_dim model where the artifact is absent; we do not have one in the same precision regime.

The weak nrot dependence on K-only and the moderate dependence on broad together do not strongly distinguish mechanism (2) (heterogeneous head_dim) from mechanism (3) (outlier smearing) in §4.4. Both predict *some* nrot dependence; neither predicts the specific shape we see. The cleanest mechanism test is still V-side nrot ablation on a model where V-side rotation is the dominant lever (gemma-4 31B q8/turbo4 V-only is the case we know lowers PPL by 43%), which requires a code change to the fork's hardcoded V-side nrot.

### 3.7 Decision: default OFF + per-side knobs (= original default + knobs)

The fork now ships with both sides defaulting to OFF — *which is the same default the fork shipped with before this investigation* — plus two new env knobs that allow each side to be turned on independently:

```bash
# Try V-only rotation (best on gemma-4 q8/turbo4, both 31B and 26B-A4B)
LLAMA_ATTN_ROT_V_OVERRIDE=1 ./llama-perplexity ...

# Both sides on (master's #21038 default, post-cache-and-head-dim guards)
LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 ./llama-perplexity ...
```

The legacy `LLAMA_ATTN_ROT_DISABLE` env var is preserved as a hard lock-out (`=1` forces rotation off and blocks the per-side overrides), so users who want a single switch to guarantee no rotation continue to have one.

The contribution of this PR is therefore not a default change. It is per-side control (the original env var was binary, both-on or both-off, which could not express @erazortt's asymmetric case) and the matrix above. The original maintainer who chose default OFF — citing TurboQuant's own kernel WHT and the phi-4 graph hash overflow — was correct. We took a long detour through v2 and v3 only to land back at that same default. The lesson is that whoever paid the cost of discovering "phi-4 crashes if you touch V rotation" had already paid for the right answer; the gap was that the env var was undocumented and binary, so users like @erazortt assumed no rotation control existed.

---

## 4. The PPL Artifact

### 4.1 What we observed

A note on terminology before the data: throughout this section, "fp16 KV" means *the KV cache is fp16; the model weights are still Q8_0 (or Q4_K_L for E2B)*. We did not test fp16 weights — those models are too large for our local hardware. The reference is "as faithful KV as the runtime allows on this weights config." Quantizing the KV cache below that should monotonically increase PPL.

While collecting fp16-KV baselines for the matrix above, three gemma-4 instruct GGUFs scored *lower* PPL with quantized KV than with fp16-KV on the same corpus.

fp16-KV baselines vs `q8_0 / turbo4` OFF, wikitext-2-raw, ctx=512, 32 chunks:

| Model | fp16-KV PPL | q8/turbo4 OFF PPL | Δ vs fp16-KV |
|-------|-------------|-------------------|--------------|
| gemma-4 26B-A4B Q8 (it) | 10813.67 ± 906.28 | 6273.74 ± 510.44 | **−42.0%** |
| gemma-4 31B Q8 (it) | 5319.99 ± 438.95 | 8685.45 ± 733.07 | +63.3% |
| gemma-4 E2B Q4_K_L (it) | 123.02 ± 7.09 | 114.77 ± 6.50 | **−6.7%** |
| Qwen2.5-7B Q8 (it) | 6.103 ± 0.173 | 6.140 ± 0.174 | +0.6% |
| Qwen3.5-2B Q8 (base) | 10.663 ± 0.324 | 10.794 ± 0.324 | +1.2% |
| phi-4 Q8 (base) | 5.822 ± 0.152 | 5.824 ± 0.152 | +0.03% |
| Mistral-Small-24B Q4 (it) | 5.312 ± 0.132 | 5.317 ± 0.131 | +0.1% |

The non-gemma rows are all in the "quantization adds a small amount of PPL" direction, as expected. Two of the three gemma-4 instruct models invert: quantization scores *better* than fp16-KV. The 31B inverts in the other direction (rotation V-only PPL = 4924, also below the fp16-KV baseline of 5320).

The artifact is **also present at q8/q8 KV**, not just at the asymmetric q8/turbo4. From the cross-format matrix in §3.2: gemma-4 26B-A4B q8/q8 OFF = 9979 vs fp16-KV baseline 10813 — quantizing K from fp16 to q8_0 alone (no turbo, no rotation) drops PPL by 7.7%. This rules out any explanation that depends on TurboQuant's WHT or master's rotation: the artifact is in the *act of quantizing the KV cache at all*, on this model class.

This is inconsistent with what a faithful KV-quantization should produce. Strictly speaking, quantization *can* lower PPL via noise injection (a known regularization-by-noise effect documented in classic dropout / weight-noise literature), so we are careful not to call this "mathematically impossible." But the *magnitude* here is the problem — a 7–42% PPL improvement from 8-bit KV is far beyond any plausible regularization regime; the KV cache stores activations that the model produced, not parameters being trained. The cleaner reading is that PPL on this corpus is not measuring what we want it to measure on these models. Independent corroboration: @vektorprime's gemma-4 31B q4_0 ran at PPL 1573 vs q8_0 at 2289 in [llama.cpp #21394](https://github.com/ggml-org/llama.cpp/issues/21394) — *more aggressive* KV quantization scoring better than less aggressive, on the same model size and context. AesSedai's tables in [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) show q5_1/q5_1 PPL strictly below fp16 baseline while KLD is strictly above. Same pattern, multiple observers, on the same model class.

### 4.2 Persistence at longer context: depends on the model, gemma-only at every ctx

A natural defense would be that ctx=512 is too short, that the artifact disappears at realistic contexts, or that *all* models behave strangely at longer ctx. None of those is true.

| Model | Context | fp16-KV PPL | q8/turbo4 OFF PPL | Δ vs fp16-KV |
|-------|---------|-------------|-------------------|--------------|
| **gemma-4 26B-A4B Q8 (it)** | 512 | 10813.67 ± 906.28 | 6273.74 ± 510.44 | **−42.0%** |
| **gemma-4 26B-A4B Q8 (it)** | 2048 | 17820.63 ± 1014.15 | 11510.51 ± 648.27 | **−35.4%** |
| gemma-4 31B Q8 (it) | 512 | 5319.99 ± 438.95 | 8685.45 ± 733.07 | +63.3% |
| gemma-4 31B Q8 (it) | 2048 | 7612.27 ± 610.20 | 15886 (prior session) | +109% |
| Qwen2.5-7B Q8 (it) | 2048 | 6.309 ± 0.119 | 6.309 ± 0.119 | +0.0% |
| phi-4 Q8 (base) | 2048 | 5.706 ± 0.101 | 5.721 ± 0.102 | +0.27% |
| Mistral-Small-24B Q4 (it) | 2048 | 5.002 ± 0.083 | 5.012 ± 0.083 | +0.20% |

On the 26B-A4B MoE, the artifact narrows from −42% to −35% but remains large. On the dense 31B, the inversion is absent at ctx=512 in the OFF row (it appears under V-only, where 4924 crosses below the 5320 baseline) and at ctx=2048 the OFF gap goes back to the expected positive direction. Three non-gemma models at ctx=2048 (Qwen2.5-7B, phi-4, Mistral-Small) all score in the expected `quantized ≥ fp16-KV` direction with Δ ≤ 0.3%. The artifact is gemma-class at *every* context we measured; it is not "all models break at long ctx."

For the headline 26B-A4B case, both PPL columns rise with context because wikitext-2 contains long-range structure that short windows hide; what matters is the sign of the gap, and it stays inverted on this model. The non-gemma rows at ctx=2048 also rule out "the eval itself is broken at ctx=2048 across the board."

### 4.3 KL divergence tells the right story

To get a metric that does not depend on the corpus distribution, we ran `llama-perplexity --kl-divergence` against an fp16-KV reference on three models: gemma-4 26B-A4B Q8 (the headline artifact case), gemma-4 E2B Q4_K_L (within-gemma cross-check), and Qwen2.5-7B Q8 (healthy non-gemma control). The reference logits were generated with the same model and corpus, fp16-KV, ctx=512, 32 chunks, and saved with `--kl-divergence-base`.

**Headline: gemma-4 26B-A4B Q8, q8/turbo4 KV.**

| Config | PPL(Q) | KL Divergence | RMS Δp | Same top-p |
|--------|--------|---------------|--------|------------|
| OFF (q8/turbo4) | 6273.74 ± 510.44 | **1.738 ± 0.036** | 16.68 ± 0.39 % | **60.48 ± 0.54 %** |
| V-only | 6027.71 ± 496.38 | 1.844 ± 0.036 | 17.66 ± 0.40 % | 60.12 ± 0.54 % |
| broad (K+V) | 6176.40 ± 510.78 | 1.857 ± 0.036 | 17.90 ± 0.40 % | 58.89 ± 0.54 % |

PPL ranks the three configurations as **V-only < broad < OFF** — V-only "wins" by 3.9%. KLD ranks them as **OFF < V-only < broad** — V-only is *worse* than OFF, broad is worst. Same-top-p agreement with the fp16 reference moves with KLD: OFF is highest, broad is lowest. RMS Δp moves with KLD as well. Three of the four metrics that compare against the fp16 distribution agree on the same ordering; the only metric that does not is corpus PPL.

Translation: with V-only rotation, the model emits a distribution that is *farther* from the fp16 reference than with rotation off, but happens to assign higher probability to the next token in this particular wikitext-2 slice. PPL is reading drift as improvement.

**Within-gemma cross-check: gemma-4 E2B Q4_K_L, q8/turbo4 KV** (the model where rotation *raises* PPL by +6.7%):

| Config | PPL(Q) | KL Divergence | RMS Δp | Same top-p |
|--------|--------|---------------|--------|------------|
| OFF | 114.77 ± 6.50 | **0.0672 ± 0.0024** | 7.22 ± 0.25 % | n/a |
| K-only | 115.36 ± 6.54 | 0.0686 ± 0.0024 | 7.30 ± 0.26 % | n/a |
| V-only | 122.41 ± 6.98 | 0.0931 ± 0.0033 | 8.26 ± 0.26 % | n/a |
| broad | 122.42 ± 6.98 | 0.0925 ± 0.0032 | 8.28 ± 0.27 % | n/a |

On E2B, **PPL and KLD agree**: OFF is best on both, V-only is worst on both (PPL +6.7%, KLD +38%). This is the regime where the model is operating at relatively low KLD (0.07 nats vs 1.7 on the 26B-A4B headline) and PPL is in a believable absolute range (~115 vs ~6000). When the regime is well-behaved, PPL is informative; when the regime is in a high-error regime as on 26B-A4B, PPL inverts. The same E2B row also confirms the architecture-quirk story for the family: V-side rotation hurts E2B by +38% KLD even though it "helps" the larger 26B-A4B by 3.9% PPL — the per-side optimum genuinely differs inside the gemma-4 family at the KLD level too, not just the PPL level.

**Healthy non-gemma control: Qwen2.5-7B Q8, q8/turbo4 KV.**

| Config | PPL(Q) | KL Divergence | RMS Δp | Same top-p |
|--------|--------|---------------|--------|------------|
| OFF | 6.140 ± 0.174 | 0.01255 ± 0.00048 | 3.32 ± 0.11 % | 95.38 ± 0.23 % |
| K-only | 6.146 ± 0.174 | **0.01158 ± 0.00036** | 3.13 ± 0.10 % | **95.59 ± 0.23 %** |
| V-only | 6.135 ± 0.173 | 0.01474 ± 0.00079 | 3.59 ± 0.11 % | 94.79 ± 0.25 % |
| broad | 6.116 ± 0.172 | 0.01305 ± 0.00053 | 3.52 ± 0.12 % | 95.11 ± 0.24 % |

Even on a *healthy* model with KLD two orders of magnitude smaller than gemma-4's (0.012 vs 1.74), PPL and KLD still rank the configurations differently. PPL ranks **broad < V-only < OFF < K-only**. KLD ranks **K-only < OFF < broad < V-only**. Same-top-p moves with KLD: K-only highest, V-only lowest. This is the same direction of disagreement we see on gemma-4 26B-A4B, just with much smaller magnitude. The PPL-vs-KLD disagreement is not a gemma-only phenomenon — it is a general property of using corpus PPL as a proxy for "closeness to fp16-KV" when the metric your real contract cares about is the latter. On Qwen2.5-7B the magnitude is small enough that the disagreement does not flip the practical decision (everything is within standard error of OFF anyway). On gemma-4 26B-A4B the magnitude is large enough that PPL points at a different "best" configuration than KLD does. The lesson: KLD is the metric that survives across both regimes; PPL is the one that fools you in the high-error regime.

This three-model comparison answers the natural reviewer concern about the headline trio: KLD is not "always different from PPL in a meaningless way." It is consistently the metric that ranks configurations by their distance from the fp16-KV reference across well-behaved (Qwen) and broken-regime (gemma) cases alike.

**Headline trio at 256 chunks** (8× more data, ~3× tighter CIs, vs fp16-KV reference PPL 20045.71 ± 604.41):

| Config | PPL(Q) | Δ vs OFF | KL Divergence | Δ KLD vs OFF |
|--------|--------|----------|---------------|--------------|
| OFF | 11673.35 ± 342.77 | — | **1.7067 ± 0.0125** | — |
| V-only | 11160.84 ± 332.03 | −4.4% | 1.8943 ± 0.0132 | +11.0% |
| broad (K+V) | 10785.84 ± 320.35 | −7.6% | 1.9193 ± 0.0133 | +12.5% |

The PPL-vs-KLD ranking inversion that we reported at 32 chunks (PPL says V-only/broad win; KLD says they lose) survives an 8× increase in sample size. The CIs on KLD shrink from ±0.036 (32 chunks) to ±0.013 (256 chunks); the +11.0% KLD penalty for V-only is now 14σ above OFF, not 3σ. The PPL "improvement" widens (−4.4% → −7.6% for broad) and the KLD penalty widens (+6.8% → +12.5% for broad). The two metrics diverge *more* with more data, not less. The artifact is not a small-sample fluctuation.

### 4.4 Mechanism: why PPL goes the wrong way here

Four candidate explanations and what the data says about each:

1. **Calibration mismatch between gemma-4-it and wikitext-2.** Wikitext-2 is unstructured Wikipedia text; gemma-4 instruct checkpoints were trained to produce dialogue-shaped continuations. The fp16 model is confidently wrong on the corpus's true continuations, and a noisier KV cache walks the distribution back toward something less peaked, which scores better on a corpus the model is mismatched to. KLD rises (drift from fp16) while PPL falls (less confidently wrong). The Hugging Face `transformers` documentation explicitly warns about this class of artifact: *"for instruction-tuned (or otherwise supervised fine-tuned) models the predictions get further from true probabilistic predictions"* ([HF perplexity doc](https://huggingface.co/docs/transformers/perplexity)). HF transformers issue [#40990](https://github.com/huggingface/transformers/issues/40990) reports the same shape on gpt-oss-20b: an instruct/MoE model gives "extremely high perplexity on WikiText-2 (raw)." The artifact in this paper is the negative-direction sibling of that report — the model is so mis-calibrated on web text that any noise that softens its distribution lowers PPL.

2. **Heterogeneous per-layer head_dim in gemma-4 (master rotation specific).** Per llama.cpp [#21394](https://github.com/ggml-org/llama.cpp/issues/21394), gemma-4 has different head dimensions per layer (and SWA layers stay fp16 while non-SWA can be quantized). Master's PR #21038 sizes the Hadamard rotation matrix once per cache and applies it across layers, so on layers whose head_dim does not match, the involutory undo at the output is mathematically off. This explains why rotation specifically (not just quantization) produces large per-side swings on gemma-4 — the rotation operates partially on a wrong basis. It does *not* explain the q8/q8 OFF row dropping below fp16-KV (no rotation is involved there); for that, mechanism (1) applies.

3. **Loss of position-specific outliers under rotation (when rotation is on).** Master's V-rotation applies a Hadamard transform of size `nrot=64` per head. If gemma-4-it relies on a small number of high-magnitude V coordinates that encode chat-style "reply now" signals, the WHT smears that magnitude across all 64 coordinates and the involutory undo only restores it cleanly when head_dim equals the rotation size. The smearing reduces the model's sharpness on its in-distribution outputs, which on out-of-distribution wikitext-2 is a net PPL improvement. This composes with mechanism (2): the heterogeneous head_dim makes the smearing imperfect *and* the undo imperfect. **The §3.6 K-side nrot ablation is a weak test of this mechanism** — outlier smearing is V-centric (V is what flows through to the output via the weighted sum; K only sets attention weights), and the K-side ablation is not the natural place to look for it. The fork hardcodes V-side nrot at 64, so a clean V-side ablation would require a code change. We flag this as the most actionable mechanism follow-up: vary V-side nrot, see whether the §4.3 V-only KLD penalty depends on it.

4. **(Dropped) Softcap.** Earlier drafts hypothesized that gemma's logit soft-capping interacted with rotated logits. The gemma-3 tech report ([arxiv 2503.19786](https://arxiv.org/html/2503.19786v1)) confirms that softcap was *removed* in gemma-3 and replaced by QK-norm, and gemma-4 inherits that. So softcap is not in play here. The vLLM gemma-4 31B FP8 issue ([#39407](https://github.com/vllm-project/vllm/issues/39407)) shows logit saturation can still happen in the family but via a different path.

The first mechanism (calibration mismatch) is sufficient on its own to explain the q8/q8 OFF row going sub-baseline (no rotation involved) and is consistent with the Hugging Face folklore-grade warning about instruction-tuned PPL. The second and third mechanisms (heterogeneous head_dim, outlier smearing) compose with the first whenever rotation is on; they explain why the per-side rotation matrix in §3 swings so wildly on gemma-4 and not on the others. The third predicts that varying `LLAMA_ATTN_ROT_K_NROT` should change the gemma rotation rows; we test that in §3.6.

The headline finding is empirical, not mechanistic: **on gemma-4 instruct GGUFs, corpus PPL on wikitext-2 reverses sign relative to KLD vs fp16-KV logits.** Whichever combination of mechanisms turns out to dominate, the practical implication is the same: do not trust PPL as the primary correctness oracle for KV-quantization changes on these models. ggerganov made the same recommendation in his own PR thread *("It seems important to track the KLD rather than PPL...")* before this investigation started; we confirm it with a controlled per-side ranking-inversion measurement on the same model AesSedai's tables flagged.

### 4.5 KLD reference noise floor

Metal flash attention reductions are not guaranteed bit-exact across runs in general. The fp16-KV `--kl-divergence-base` we score against is one realization of the "true" fp16-KV distribution; any KLD we measure could in principle include a nondeterminism floor. To bound it we built the fp16-KV reference twice on identical inputs (same model, same corpus, same flags, same chunk count) and scored run #2 against run #1.

| Quantity | Value |
|----------|-------|
| Reference run #1 PPL | 10813.6655 ± 906.28169 |
| Reference run #2 PPL | 10813.6655 ± 906.28169 (bit-exact identical) |
| KLD (run #2 vs run #1) | **0.000000 ± 0.000000** |
| RMS Δp (run #2 vs run #1) | 0.000 ± 0.000 % |

The fp16-KV `llama-perplexity` codepath on this model + corpus + Metal build is fully deterministic across runs. The KLD noise floor is exactly zero. Every KLD delta we report in §4.3 (smallest is 0.001 between Qwen2.5-7B configs, largest is 0.13 between gemma-4 26B-A4B configs) is therefore a real ranking signal, not floor noise. This closes the "your KLD reference is itself noisy" hammer.

This result is for the specific Metal build under test (commit `817e913ec`). Other backends or alternate kernel paths may not be bit-exact; users replicating on CUDA or HIP should re-measure their own floor before relying on small KLD deltas.

### 4.6 Cross-format KLD on gemma-4 26B-A4B

The cross-format matrix in §3.2 is PPL-only, which is awkward given that this section argues PPL is unreliable on this model. We re-ran the most important cells under the KLD oracle (vs fp16-KV reference PPL 10813.67):

**q8 / q8 KV** (PPL ranks OFF < broad < V-only; KLD ranks broad < V-only < OFF):

| Config | PPL ± SE | Δ PPL vs OFF | KLD ± SE | Δ KLD vs OFF |
|--------|----------|--------------|----------|--------------|
| OFF | 9979.55 ± 832.25 | — | 0.5783 ± 0.0182 | — |
| V-only | 10451.85 ± 873.03 | +4.7% | **0.4661 ± 0.0161** | **−19.4%** |
| broad | 10118.81 ± 846.06 | +1.4% | **0.4467 ± 0.0162** | **−22.7%** |

On q8/q8 the inversion *flips direction*: PPL says rotation hurts (V-only +4.7%, broad +1.4%); KLD says rotation **helps** (V-only −19.4%, broad −22.7%). This is the master PR #21038 use case (rotation on standard quantized KV) — and on this model, the upstream rotation is in fact lowering KLD by ~20%, exactly as designed. The PPL on this model is so confused that it reads the rotation's actual quality benefit as a regression. The original maintainer's default OFF was right for the *fork* (which uses TurboQuant's own WHT and was hitting double-rotation issues elsewhere), but on plain q8/q8 the master rotation does what it claims, just not by a metric you can read off corpus PPL.

**turbo4 / turbo4 KV** (PPL ranks OFF < broad < K-only "catastrophic"; KLD ranks K-only < OFF < broad):

| Config | PPL ± SE | Δ PPL vs OFF | KLD ± SE | Δ KLD vs OFF |
|--------|----------|--------------|----------|--------------|
| OFF | 5785.07 ± 471.30 | — | 2.1334 ± 0.0405 | — |
| K-only ("catastrophic" in §3.2) | 8831.29 ± 727.89 | **+52.7%** | **2.0287 ± 0.0383** | **−4.9%** |
| broad | 7031.24 ± 582.09 | +21.5% | 2.3405 ± 0.0411 | +9.7% |

This is the cleanest PPL/KLD ranking-flip in the whole paper. The `t4 / t4` K-only cell that PPL flagged as **catastrophic at +52.7%** is actually the **lowest-KLD configuration on the row** — closer to the fp16-KV reference than OFF, by a margin (−0.10 nats) that is well above the 0.000 noise floor (§4.5). PPL is reading a +52.7% regression where KLD reads a 4.9% improvement. The "broad" cell is the only one that's worse than OFF by both metrics on this row — composing K-side rotation (which by KLD helps) with V-side rotation (which on `t4 / t4` apparently does not, since broad KLD is worse than K-only's KLD by 0.31 nats).

The §3 framing of "+52.7% K-only is catastrophic" was wrong. The honest framing is: "+52.7% PPL on a model where PPL is independently shown to be unreliable; KLD on the same row says K-only is the *best* cell." This does not change the engineering policy (we still default both sides OFF, since we cannot universally enable K-only on `t4 / t4` without the KLD oracle confirming each model), but it changes the language. We update §3.2 and §3.5 accordingly.

### 4.7 Cross-corpus check on wikitext-103

The largest reviewer hammer on the headline is "wikitext-2-specific artifact." We re-ran fp16-KV, q8/turbo4 OFF, and q8/q8 OFF on gemma-4 26B-A4B against wikitext-103 train (a different corpus, ~516MB vs ~1MB), 32 chunks at ctx=512:

| Corpus | KV format | fp16-KV PPL | q8/* PPL | Δ vs fp16-KV |
|--------|-----------|-------------|----------|--------------|
| wikitext-2 test | q8/turbo4 | 10813.67 ± 906.28 | 6273.74 ± 510.44 | **−42.0%** |
| wikitext-2 test | q8/q8 | 10813.67 ± 906.28 | 9979.55 ± 832.25 | **−7.7%** |
| **wikitext-103 train** | q8/turbo4 | 33845.11 ± 2864.90 | 19902.90 ± 1650.92 | **−41.2%** |
| **wikitext-103 train** | q8/q8 | 33845.11 ± 2864.90 | 31818.69 ± 2684.31 | **−6.0%** |

The artifact reproduces on the different corpus with both KV formats, same sign, nearly identical magnitude (q8/turbo4: −42.0% → −41.2%; q8/q8: −7.7% → −6.0%). The "wikitext-2 specific" defense is closed. Both PPL columns are wildly higher on wikitext-103 (the model is even more mis-calibrated on the train-split text), but the *ratio* between fp16-KV and quantized is preserved. This is consistent with the calibration-mismatch mechanism in §4.4: the artifact magnitude scales with how mis-calibrated the model is on the corpus, while the *direction* is set by the model's interaction with KV quantization, not by the specific text. The "more aggressive quantization → bigger artifact" pattern (q8/turbo4 inverts ~7× harder than q8/q8) holds on both corpora, which is what mechanism (1) predicts (more KV noise → more distribution softening). Both corpora we tested are still English web text; we cannot rule out that an in-distribution chat-shaped corpus would produce a different result, and §4.9 lists this as the most important remaining corpus gap.

### 4.8 Downstream completion probe

PPL and KLD are both proxies. We ran 3 short chat-shaped prompts on gemma-4 26B-A4B in two configs — fp16-KV (the reference) and q8/turbo4 OFF (the "−42% PPL win" / 1.7-nat-KLD config from §4.3) — using `llama-cli --single-turn`. The result was less obvious than either "broken under q8/turbo4" or "fine under q8/turbo4" would predict.

| Prompt | fp16-KV completion (40 tok) | q8/turbo4 OFF completion (40 tok) |
|--------|------------------------------|------------------------------------|
| "The capital of France is" | "**Paris**." | "**Paris**." |
| "Two plus two equals" | "Four" | "**four**" |
| "List three colors:" | "Red, Blue, Green (Primary/RGB)" | "Red, Blue, Yellow." |

Decode throughput: 92.1 t/s (fp16-KV) vs 73.3 t/s (q8/turbo4 OFF) — the throughput penalty is real, the answers are not broken. Both configs give the correct factual answer on all three prompts. The third prompt produces *different* lists ("Red, Blue, Green" vs "Red, Blue, Yellow") — both are valid three-color answers, and the divergence at the third token is consistent with the same-top-p ~60% measurement in §4.3 (token-level distribution drifts but stays in a "valid answer" region for simple factual queries). N=3 is not a benchmark; it is enough to show that the "−42% PPL improvement" config is *not* an obviously-broken model in the way the PPL number alone might suggest.

This recasts the §4.3 conclusion. KLD = 1.7 nats and same-top-p ≈ 60% do indicate a substantial token-level distribution shift relative to fp16-KV, *but* short factual completions are robust to that shift because the top-1 candidates remain in the same "right-answer cluster." PPL = −42% is still wrong-direction nonsense — quantization cannot give the model new information — but the model does not become unusable in the practical sense; it produces correct answers on prompts where many continuations would be acceptable. The honest takeaway is that PPL, KLD, and downstream-task accuracy each measure something different on this model:

- **PPL on wikitext-2** is mis-calibration-dominated and inverts.
- **KLD vs fp16-KV** is a faithful "distance from reference" metric (large here).
- **Downstream accuracy on simple chat-shaped prompts** is preserved despite the distance, because the answer space is small.

For KV-quant *evaluation* (which config is closer to fp16), KLD is the right oracle. For KV-quant *deployment* (does the model still answer correctly), task accuracy is the right oracle. PPL on this model class is the right oracle for nothing on its own. A more rigorous downstream probe (multi-turn reasoning, code generation, longer-form writing, NIAH) would likely surface degradation on q8/turbo4 OFF that simple factual completions hide. We flag this as the cleanest follow-up.

### 4.9 Scope and what we did *not* show

We are explicit about what is in evidence and what is not.

- **In evidence:** the artifact reproduces on three gemma-4-it GGUFs (26B-A4B, 31B, E2B) at ctx=512 and on the 26B-A4B MoE at ctx=2048. KLD ranks the configs in the intuitive order on the 26B-A4B; PPL does not. Independent reports ([#21394](https://github.com/ggml-org/llama.cpp/issues/21394), AesSedai in [#21038](https://github.com/ggml-org/llama.cpp/pull/21038), [localbench](https://localbench.substack.com/p/kv-cache-quantization-benchmark)) report the same shape on the same family.
- **In evidence:** the artifact also occurs at q8/q8 KV (no turbo, no rotation) on gemma-4 26B-A4B, so it is not a TurboQuant- or rotation-specific phenomenon. It is a property of *quantizing the KV cache at all* on this model class.
- **Not in evidence:** that the artifact generalizes to *all* instruction-tuned models. Our non-gemma instruct GGUFs in the matrix (Qwen2.5-7B-Instruct, Mistral-Small-24B-Instruct) score in the expected direction with small positive Δ. HF transformers [#40990](https://github.com/huggingface/transformers/issues/40990) reports a positive-direction artifact on gpt-oss-20b instruct/MoE; both directions are consistent with the calibration-mismatch mechanism. The artifact, as we have measured it, is **gemma-class instruct specific**, not "all instruct models."
- **Not in evidence:** that the gemma-4 *base* models would or would not show the artifact. We do not have a base gemma-4 GGUF locally to test against. The cleanest follow-up would be a paired base-vs-instruct run on the same gemma-4 size and corpus.
- **Not in evidence:** that other corpora (wikitext-103 long contexts, c4, in-distribution chat data) would reproduce the inversion. wikitext-2 at short context is the only corpus we measured.

---

## 5. Practical Recommendations

### 5.1 For users of the fork

The default is `LLAMA_ATTN_ROT_K_OVERRIDE` unset and `LLAMA_ATTN_ROT_V_OVERRIDE` unset, which is rotation OFF on both sides. This is safe across every model in our matrix.

If you are running gemma-4 31B (or another model where the matrix or your own measurements show a clear V-side win), opt in:

```bash
LLAMA_ATTN_ROT_V_OVERRIDE=1 \
  ./llama-server -m gemma-4-31B-Q8_0.gguf \
  -ctk q8_0 -ctv turbo4 -fa on
```

If you want to disable rotation under any circumstance (including ignoring future env-var overrides), set `LLAMA_ATTN_ROT_DISABLE=1`. This remains the hard lock-out.

### 5.2 For anyone validating KV-quantization changes on gemma-class instruct models

Use KLD against fp16 logits, not corpus PPL. Concretely:

```bash
# 1. Generate the fp16 reference once
./llama-perplexity \
  -m gemma-4-26B-A4B-it-Q8_0.gguf \
  -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 32 \
  --kl-divergence-base /tmp/kld-base-gemma26 \
  -ngl 99 --no-warmup

# 2. Score every candidate config against that reference
./llama-perplexity \
  -m gemma-4-26B-A4B-it-Q8_0.gguf \
  -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 32 \
  --kl-divergence-base /tmp/kld-base-gemma26 \
  --kl-divergence \
  -ctk q8_0 -ctv turbo4 \
  -ngl 99 --no-warmup
```

Look at **Mean KLD** and **Same top p** as the primary signals. Both align with intuition (lossier KV = more drift). Treat the PPL column as an interesting side metric, not the primary oracle. On non-gemma instruct models, PPL and KLD agree, so PPL is fine; on gemma-class, they disagree, so KLD wins.

### 5.3 For the broader llama.cpp community

Master PR #21038's behavior — rotate by default when the KV type is quantized and head-dim is power-of-2 — is the right default for several models in our matrix. It is wrong on at least three counts (gemma-4 E2B Q4_K_L regresses, phi-4 V-side crashes, and `t4 / t4` K-only is catastrophic at +52.7% PPL though see §4 for the metric caveat). Master *already* handles the gemma-4 case by auto-disabling rotation on the family due to heterogeneous per-layer head_dim ([#21394](https://github.com/ggml-org/llama.cpp/issues/21394)). Our fork's env knobs deliberately bypass that guard. The trade-off:

- **In favor of bypassing:** the V-only configuration on gemma-4 26B-A4B q8/q8 lowers KLD by 19% (§4.6) — that is master's intended use case behaving as designed. Auto-disable removes the option entirely. Users who run gemma-4 with q8/q8 and want the master rotation's KLD benefit need a way to opt in.
- **Against bypassing:** the rotation operates on a wrong basis when applied across layers with different head_dim, so any "win" we measure is partly mathematical garbage that happens to score well on this corpus. The honest answer is "do not use this configuration in production until per-layer rotation lands (PR [#21513](https://github.com/ggml-org/llama.cpp/pull/21513))."

We document both sides and let users make the call. A two-knob opt-in surface (`_K_OVERRIDE`, `_V_OVERRIDE`) costs essentially nothing in the C++ and lets users tune for their actual config without forking. If we propose the same env knobs upstream, we will respect the gemma-4 auto-disable guard by default and require an explicit second flag (`LLAMA_ATTN_ROT_FORCE_HETEROGENEOUS=1`) to bypass it. On the fork, where the audience is power users actively testing TurboQuant variants, we leave the bypass in.

**Cost of "default OFF" in this fork.** The largest user-visible loss vs always-on rotation is gemma-4 31B q8/turbo4, where V-only saves ~43% PPL relative to OFF. KLD on the *same architecture family* (gemma-4 26B-A4B and E2B; we did not run KLD on 31B because its file size and our hardware budget capped the runs we could do) goes the *opposite* direction: V-only raises KLD on 26B-A4B by 11% (256-chunk), and on E2B by 38%. By the same family-level pattern, the gemma-4 31B PPL "win" is partly artifact and is unlikely to survive a KLD test. So the "cost" of leaving rotation OFF on the 31B is at most a partly-artifact PPL gain. On models where rotation effect is within standard error (Qwen, Mistral) there is no cost. On gemma-4 E2B and on `t4 / t4` K-only the "cost" is negative — turning the default on would *hurt* users (E2B at +38% KLD, `t4/t4` K-only at +52.7% PPL).

---

## 6. Limitations

1. **Single platform.** All measurements are on Apple M5 Max, Metal flash attention. The rotation matrix and the PPL artifact are both expected to be platform-agnostic (they are properties of the math, not the kernel), but neither has been re-run on CUDA or HIP for this paper. We treat this as the largest unresolved hammer; cross-backend repro is in scope for future work, not this paper.
2. **Cross-format matrix is single-model.** The full 4-config × 3-format matrix (§3.2) is gemma-4 26B-A4B Q8 only. The cross-model matrix (§3.3) covers the `q8 / turbo4` row only. Other asymmetric pairs (q8/turbo3, q8/turbo2) are not in either table; informal spot-checks suggest the same per-side splits but we have not run the full grid.
3. **`t4 / t4` V-only cell missing on 26B-A4B.** Reported as such in §3.2; not load-bearing for the policy decision (the broad cell at +21.5% PPL / +9.7% KLD already shows that K+V composed rotation hurts on this row, even though K-only alone helps by KLD) but a gap in the matrix.
4. **No paired base-vs-instruct gemma-4 PPL artifact run.** This is the most important follow-up. If a base gemma-4 of comparable size shows no artifact and the instruct variant does, that is direct evidence for the calibration-mismatch hypothesis in §4.4. We do not have a base gemma-4 GGUF locally.
5. **No fp16-weights baseline.** Throughout this paper the "fp16-KV reference" still uses Q8_0 (or Q4_K_L) model weights; only the KV cache is varied. We cannot isolate "weight-quantization × KV-quantization interaction" from "KV-quantization on this weights config" without an fp16-weights run, which is out of reach on the 26B and 31B models for our local hardware. On E2B Q4_K_L this could in principle be done; we did not.
6. **Mechanism for the PPL artifact unconfirmed.** §4.4 lists four candidates (calibration mismatch, heterogeneous head_dim, loss of position-specific outliers, dropped softcap). We have not isolated which dominates; the headline finding in §4.3 is empirical and survives regardless of which mechanism is correct.
7. **PPL/KLD measured in a high-error regime.** On gemma-4 26B-A4B q8/turbo4 the KLD trio is 1.7–1.9 nats and same-top-p agreement is ~60%; the model is not behaving fp16-faithfully under any of these KV configs. We use KLD as a *ranking* oracle (which config is closer to fp16) rather than as an absolute quality measure; our claim is "rotation X moves the distribution further from fp16 than rotation Y," not "rotation X gives a usable model." §4.9 addresses the "is it usable at all" question via downstream completion probe.
8. **KLD reference nondeterminism — measured to zero on Metal, untested elsewhere.** §4.5 shows the fp16-KV reference is bit-exact deterministic on this Metal build; KLD floor is exactly 0.000000. We have not measured this on CUDA or HIP. Users replicating on other backends should re-measure their floor before relying on small KLD deltas.
9. **Wikitext-2 is one corpus; wikitext-103 is a second but still web text.** Even after the §4.7 cross-corpus check, all corpora we test are unstructured English web text. We have not run a chat-shaped or in-distribution corpus that would let us *separate* "calibration mismatch on web text" from "PPL artifact on this model regardless of corpus."
10. **Independent reports we lean on are llama.cpp-thread-grade evidence, not papers.** The corroborating data from #21394, #21038 AesSedai tables, and localbench are forum-grade — community forks, single observers, no peer review. We rely on them for "this is not a one-author observation" but cannot rely on them for methodology rigor beyond what we can verify ourselves.

---

## 7. Conclusion

We started looking for a single defensible default for master's attention rotation on top of TurboQuant's quantization rotation. There isn't one. The optimal policy splits four ways across seven model families and three ways inside the gemma-4 family alone; on `t4 / t4` K-only one cell costs +52.7% PPL. The fork's default — both sides OFF — is the same default it shipped with before this investigation. The contribution is per-side env knobs (`LLAMA_ATTN_ROT_K_OVERRIDE`, `LLAMA_ATTN_ROT_V_OVERRIDE`) that let users opt each side in independently, and the matrix that documents which models want which knob. The original maintainer who chose default OFF was correct from the start; we paid the cost of three iterations of "smart" defaults to confirm it, and the lesson is that the gap users actually felt was discoverability and per-side control, not the default itself.

While running that matrix, we found that PPL on three gemma-4 instruct GGUFs goes the wrong way: quantized KV scores 7–42% *better* than fp16-KV on wikitext-2, the gap persists when context is increased from 512 to 2048 on the 26B-A4B MoE, on the wikitext-103 train corpus with the same magnitude (closes the "wikitext-2-specific" defense), at q8/q8 with no rotation as well as at q8/turbo4 (closes the "rotation-specific" defense), and KLD vs the fp16-KV reference points the *opposite* direction from PPL on the same eval. The cleanest single example: on `t4 / t4` K-only, PPL says +52.7% catastrophic regression, KLD says −4.9% improvement — same row, opposite direction. We are not the first to observe the pattern (vektorprime, AesSedai, localbench, ggerganov himself in PR #21038); what this paper adds is a controlled per-side rotation matrix, a direct PPL-vs-KLD ranking-inversion measurement on the same setup, KLD evidence on a healthy non-gemma control (Qwen2.5-7B) showing the disagreement is a continuum, and a measured KLD noise floor of exactly zero on Metal (closes the "your reference is itself noisy" hammer). A small downstream completion probe (§4.8) shows the q8/turbo4 OFF model still answers simple factual prompts correctly, so the recasting is: PPL is wrong-direction, KLD is correct-direction-for-distance-from-fp16, and downstream task accuracy is the third independent metric. Use KLD for KV-quant evaluation, not corpus PPL, on this model class.

The deeper question — why the gemma-4 instruct fine-tunes specifically produce a distribution where small KV noise improves cross-entropy on out-of-distribution text — is left open. The likely mechanisms are (i) calibration mismatch between an instruction-tuned, sharply peaked output distribution and an unstructured web-text corpus, with quantization noise acting as a softening prior, and (ii) the heterogeneous per-layer head_dim in gemma-4 that makes a single rotation tile inherently mismatched to some layers. The cleanest test is a paired base-vs-instruct gemma-4 PPL+KLD run on wikitext-2 plus an in-distribution chat corpus; we flag it as the next experiment.

---

## Reproducibility

### Build and weights

- **llama.cpp fork:** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), branch `fix/enable-attn-rot-by-default`, commit `817e913ec` (the prior `db3595a755a9` shipped with `attn_rot_disable` defaulting to `true` for legacy compatibility, which silently blocked the new per-side override env knobs via the `&& !attn_rot_disable` guard inside both override branches; `817e913ec` flips that default to `false`. Tests in this paper that use `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE` require the post-fix build). Per-side env knobs are in `src/llama-kv-cache.cpp` around the `attn_rot_k`/`attn_rot_v` initialization.
- **Build target:** `build-test/bin/llama-perplexity` (Metal-only fast iteration build, `EMBED=OFF`).
- **GGUF SHA-256s:**
  - `gemma-4-26B-A4B-it-Q8_0.gguf` — `1157ef475f418871da843a25ce2de867eb00d75732440015d9141362ecd0145b`
  - `gemma-4-31B-it-Q8_0.gguf` — `66ed05a73a36901fafe2e7f965917cc8df750dcdcc43f1b578cc37c63830b335`
  - `google_gemma-4-E2B-it-Q4_K_L.gguf` — `2c64c8ab879d9463abb5dc7ec4ed169c65350361ca8366b39600d183cfd5b270`
- **Corpus:** `wikitext-2-raw/wiki.test.raw`, the standard llama.cpp distribution copy.
- **Issue thread:** [TheTom/turboquant_plus#88](https://github.com/TheTom/turboquant_plus/issues/88) — original report from @erazortt that triggered the investigation.
- **Master PR being investigated:** [ggml-org/llama.cpp#21038](https://github.com/ggml-org/llama.cpp/pull/21038).
- **Independent prior reports:** [#21394](https://github.com/ggml-org/llama.cpp/issues/21394), [#22407](https://github.com/ggml-org/llama.cpp/issues/22407).

### KLD-base determinism

§4.5 shows the fp16-KV `--kl-divergence-base` reference is bit-exact deterministic on this Metal build for `llama-perplexity` (KLD floor = 0.000000). Other backends or alternate kernel paths may not be, so users replicating on CUDA or HIP should re-measure the floor on their build before relying on small KLD deltas.

To reproduce the per-side matrix:

```bash
# OFF  — both env vars unset
./llama-perplexity -m MODEL.gguf -f wikitext-2-raw/wiki.test.raw \
  -c 512 --chunks 32 -ctk q8_0 -ctv turbo4 -ngl 99 --no-warmup

# K-only
LLAMA_ATTN_ROT_K_OVERRIDE=1 ./llama-perplexity ...

# V-only
LLAMA_ATTN_ROT_V_OVERRIDE=1 ./llama-perplexity ...

# broad
LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 ./llama-perplexity ...
```

To reproduce the KLD evidence in §4.3:

```bash
# 1. Build the fp16 KV reference
./llama-perplexity -m gemma-4-26B-A4B-it-Q8_0.gguf \
  -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 32 \
  --kl-divergence-base /tmp/kld-base-gemma26 \
  -ngl 99 --no-warmup

# 2. Score each rotation config against it
[ENV] ./llama-perplexity -m gemma-4-26B-A4B-it-Q8_0.gguf \
  -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 32 \
  --kl-divergence-base /tmp/kld-base-gemma26 --kl-divergence \
  -ctk q8_0 -ctv turbo4 -ngl 99 --no-warmup
```

---

## Acknowledgments

- **@erazortt** — original gemma-4 26B-A4B Q6_K_XL regression report on TheTom/turboquant_plus#88. Without that report we would have shipped the v2 broad-enable default and broken his configuration silently.
- **@ggerganov** — author of master PR #21038 and of the "track KLD rather than PPL" recommendation that we re-derived independently and now confirm with controlled per-side measurements.
- **@vektorprime, @stduhpf, AesSedai** — independent reporters of the same PPL/KLD ranking-flip pattern in [#21394](https://github.com/ggml-org/llama.cpp/issues/21394) and the PR #21038 thread, on the same model class. Their data is what made this an "independently observed pattern" rather than a one-author observation.
- The **localbench** maintainers for the cross-family KLD methodology that confirmed gemma-4 is an order of magnitude noisier under KV quantization than peer families.

---

## References

### Master implementation and prior community reports

1. llama.cpp PR #21038 — attention-side WHT rotation for quantized KV (master). [ggml-org/llama.cpp#21038](https://github.com/ggml-org/llama.cpp/pull/21038). Includes the AesSedai PPL/KLD inversion tables and ggerganov's "track KLD rather than PPL" comment.
2. llama.cpp issue #21394 — "Gemma4 attn_rot_k and v = 0". [ggml-org/llama.cpp#21394](https://github.com/ggml-org/llama.cpp/issues/21394). vektorprime's gemma-4 31B PPL ranking-flip report.
3. llama.cpp PR #21513 — "support attention rotation for heterogeneous iSWA" (in flight). [ggml-org/llama.cpp#21513](https://github.com/ggml-org/llama.cpp/pull/21513). Master-side fix for the per-layer head_dim issue.
4. llama.cpp issue #22407 — "Extreme Perplexity Values with Gemma 4 E4B Base Quantizations". [ggml-org/llama.cpp#22407](https://github.com/ggml-org/llama.cpp/issues/22407). Independent gemma-4 PPL pathology under weight quantization.
5. vLLM issue #39407 — gemma-4-31B FP8 + softcap saturation. [vllm-project/vllm#39407](https://github.com/vllm-project/vllm/issues/39407).
6. HF transformers issue #40990 — gpt-oss-20b instruct/MoE wikitext-2 PPL pathology. [huggingface/transformers#40990](https://github.com/huggingface/transformers/issues/40990).
7. Hugging Face perplexity documentation (warning about instruction-tuned models). [huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity).
8. localbench KV-cache benchmark (cross-family KLD, including gemma-4 26B-A4B). [localbench.substack.com](https://localbench.substack.com/p/kv-cache-quantization-benchmark).

### Architecture references

9. Gemma 3 technical report (introduces QK-norm, removes softcap). [arXiv:2503.19786](https://arxiv.org/html/2503.19786v1).
10. Gemma 3 Hugging Face blog (5:1 SWA pattern, head_dim notes). [huggingface.co/blog/gemma3](https://huggingface.co/blog/gemma3).

### Methods being investigated

11. TurboQuant: Redefining AI Efficiency with Extreme Compression. Google Research, ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).

### Companion papers in this docs/papers/ directory

12. Asymmetric K/V Cache Compression. [asymmetric-kv-compression.md](asymmetric-kv-compression.md).
13. Sparse V Dequantization. [sparse-v-dequant.md](sparse-v-dequant.md).
14. TurboQuant4 Resurrection. [turbo4-resurrection.md](turbo4-resurrection.md).
