"""
tq_collapse_probe.py — Attention-routing collapse diagnostic for KV-cache
quantizers, wired to the REAL turboquant_plus API.

──────────────────────────────────────────────────────────────────────────
What this is
──────────────────────────────────────────────────────────────────────────

A method-agnostic diagnostic that answers one sharp question: when you
quantize the K cache with a given quantizer at a given bit-width, does
attention routing survive — and if it breaks, is the breakage concentrated
in a few heads (sparse) or spread across all of them (diffuse)?

The metric is top-k attention rank preservation, NOT reconstruction error.
Collapse in KV quantization is fundamentally attention misrouting: when the
quantized K scores reorder the top keys, attention mass flows to the wrong
tokens and the error compounds over layers. Perplexity and most NIAH setups
are nearly blind to this; reconstruction MSE is blind to it by construction
(it averages exactly the coordinate that determines the bucket-flip). This
probe measures the routing directly.

It captures REAL post-RoPE Q/K from a REAL model on REAL text (wrapping
scaled_dot_product_attention — model-agnostic, no per-architecture RoPE
glue), then for each (layer, head) compares the FP16 top-k attended keys
against the top-k recovered after quantizing K with the quantizer under
test, across a bit-width sweep.

──────────────────────────────────────────────────────────────────────────
The quantizer under test is the REAL turboquant_plus package
──────────────────────────────────────────────────────────────────────────

Quantizer configs are the single swappable axis. Each maps a name to a
callable K_dequantized = f(K_real) using the genuine installed classes:

    turbo{2,3,4,5}        TurboQuant (PolarQuant + QJL), bit_width=b
    turboMSE{2,3,4,5}     TurboQuantMSE (PolarQuant only, no QJL)
    turbo{b}_nonc         norm_correction=False
    turbo{b}_biassub      bias-subtract preprocessing then real TurboQuant

This is the clean comparison the field's "drop QJL" consensus turns on
(turbo vs turboMSE at matched bits) AND the Paper B reconciliation
(real TurboQuant vs the scalar reimplementation that produced 0/60).

NOTE on Paper B: the original 0/60 was produced by a scalar quantizer
(rotation + naive symmetric 4-bit, no codebook/QJL/norm-correction), NOT
by this package. Running 'turbo4' here on the same model+text is the
matched-config rerun that resolves the divergence.

──────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────

  pip install -e .            # in the turboquant_plus repo, gets the package
  pip install torch transformers
  python tq_collapse_probe.py \\
      --model Qwen/Qwen2.5-7B \\
      --text some_real_text.txt \\
      --configs turbo2 turboMSE2 turbo4 turboMSE4 \\
      --layers 0 1 14 27 \\
      --seq-len 2048 \\
      --topk 8

Run on a model where collapse is expected (Qwen2.5-7B at 2-bit) and a
bias-free control (Mistral-7B or Llama-2-7B) to separate the general
variance/rank collapse from the Qwen-specific bias artifact.
"""

import argparse
import json
import sys
import time

import numpy as np

# ── real quantizer package ───────────────────────────────────────────────
try:
    import turboquant as tq
except ImportError:
    sys.exit("turboquant package not importable — run `pip install -e .` in "
             "the turboquant_plus repo first. This probe measures the REAL "
             "package, not a reimplementation.")


# ─────────────────────────────────────────────────────────────────────────
# Quantizer registry — each config is a callable K -> K_dequantized,
# applied per-head on (n_vectors, head_dim) arrays using the real classes.
# bias_per_head (head_dim,) is subtracted pre-quant / added post-quant for
# the *_biassub configs.
# ─────────────────────────────────────────────────────────────────────────

def make_quantizer(config_name, d, seed=42):
    """Return a fn(K_np[n,d], bias[d] or None) -> K_dequant_np[n,d] using
    the genuine turboquant_plus classes. Raises on unknown config."""

    def with_bias(fn):
        def wrapped(K, bias=None):
            if bias is not None:
                K = K - bias[None, :]
            out = fn(K)
            if bias is not None:
                out = out + bias[None, :]
            return out
        return wrapped

    # parse bit width
    import re
    m = re.search(r'(\d+)', config_name)
    bw = int(m.group(1)) if m else 4

    nonc = config_name.endswith("_nonc")
    norm_corr = not nonc

    # fp16 identity — sanity check: must score ~1.0 recall against itself.
    # If it doesn't, the harness has a bug independent of any quantizer.
    if config_name.startswith("fp16"):
        return with_bias(lambda K: K.astype(np.float32))

    if config_name.startswith("turboMSE"):
        q = tq.TurboQuantMSE(d=d, bit_width=bw, seed=seed, norm_correction=norm_corr)
        def fn(K):
            idx, norms = q.quantize(K.astype(np.float32))
            return q.dequantize(idx, norms)
        return with_bias(fn)

    if config_name.startswith("turbo"):
        q = tq.TurboQuant(d=d, bit_width=bw, seed=seed, norm_correction=norm_corr)
        def fn(K):
            comp = q.quantize(K.astype(np.float32))
            return q.dequantize(comp)
        return with_bias(fn)

    raise ValueError(f"unknown config: {config_name}")


# ─────────────────────────────────────────────────────────────────────────
# Rank-preservation metric (pure numpy — verifiable without a model)
# ─────────────────────────────────────────────────────────────────────────

def routing_divergence(Q, K_fp16, K_quant, topk, causal=True, sample_q=128, seed=0):
    """For sampled query positions, measure how much the attention
    DISTRIBUTION moves when K is quantized — the quantity that actually
    tracks misrouting, as opposed to brittle top-k set membership.

    Returns dict with:
      tv          mean total-variation distance between FP16 and quantized
                  attention distributions = fraction of attention MASS
                  misrouted. 0 = identical, primary collapse metric.
                  (fp16-vs-fp16 gives exactly 0 — harness sanity.)
      top1_keep   fraction of queries where the single most-attended key
                  is preserved (sharp binary routing signal).
      topk_churn  1 - top-k set recall, kept as a SECONDARY sensitivity
                  indicator. High even at high bit-widths because it counts
                  benign reordering of near-tied low-mass keys; NOT a
                  collapse metric on its own.
      topk_mass   mean FP16 top-k softmax mass (attention concentration).

    Q, K_*: (n, d) post-RoPE for one head. Scores = Q @ K^T / sqrt(d).
    """
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    rng = np.random.default_rng(seed)

    valid = np.arange(topk + 1, n)
    if len(valid) == 0:
        return {"tv": np.nan, "top1_keep": np.nan,
                "topk_churn": np.nan, "topk_mass": np.nan}
    if len(valid) > sample_q:
        qpos = rng.choice(valid, size=sample_q, replace=False)
    else:
        qpos = valid

    tvs, top1s, churns, masses = [], [], [], []
    for i in qpos:
        ctx = i + 1 if causal else n
        q = Q[i]
        s_fp = (K_fp16[:ctx] @ q) * scale
        s_q = (K_quant[:ctx] @ q) * scale
        # softmax both
        p_fp = np.exp(s_fp - s_fp.max()); p_fp /= p_fp.sum()
        p_q = np.exp(s_q - s_q.max()); p_q /= p_q.sum()
        tvs.append(0.5 * np.abs(p_fp - p_q).sum())
        top1s.append(int(np.argmax(s_fp) == np.argmax(s_q)))
        k = min(topk, ctx)
        idx_fp = np.argpartition(-s_fp, k - 1)[:k]
        top_fp = set(idx_fp.tolist())
        top_q = set(np.argpartition(-s_q, k - 1)[:k].tolist())
        churns.append(1.0 - len(top_fp & top_q) / k)
        masses.append(float(p_fp[idx_fp].sum()))

    return {
        "tv": float(np.mean(tvs)),
        "top1_keep": float(np.mean(top1s)),
        "topk_churn": float(np.mean(churns)),
        "topk_mass": float(np.mean(masses)),
    }


def gini(x):
    """Gini coefficient of a non-negative array (concentration of damage)."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def frac_heads_holding(damage, frac=0.8):
    """Fraction of heads that together hold `frac` of total damage.
    Low value => sparse (few heads carry the collapse)."""
    d = np.sort(np.asarray(damage, dtype=float))[::-1]
    tot = d.sum()
    if tot == 0:
        return 1.0
    c = np.cumsum(d) / tot
    return float((np.searchsorted(c, frac) + 1) / len(d))


# ─────────────────────────────────────────────────────────────────────────
# Self-test (no torch, no model) — verifies metric + real-quantizer wiring
# ─────────────────────────────────────────────────────────────────────────

def self_test():
    print("Self-test: metric + real-quantizer wiring (synthetic vectors)")
    print("(synthetic Gaussian, so absolute numbers are not a result —")
    print(" this only confirms the probe is wired to the real package)\n")
    d = 128
    rng = np.random.default_rng(0)
    n = 256
    Q = rng.standard_normal((n, d)).astype(np.float32)
    K = rng.standard_normal((n, d)).astype(np.float32)

    for cfg in ["fp16", "turbo5", "turbo4", "turbo3", "turbo2", "turboMSE2"]:
        qfn = make_quantizer(cfg, d=d)
        Kq = qfn(K, None)
        m = routing_divergence(Q, K, Kq, topk=8)
        print(f"  {cfg:12s}  TV(mass misrouted)={m['tv']:.4f}  "
              f"top1_keep={m['top1_keep']:.3f}  [churn={m['topk_churn']:.2f}]")
    print("\nfp16 must read TV=0.0000 (identity). Synthetic Gaussian has no")
    print("real attention structure, so TV here is tiny for all configs —")
    print("the real model run is where collapse separation shows up.\n")


# ─────────────────────────────────────────────────────────────────────────
# Real capture path (torch + transformers) — runs on a GPU box, not here
# ─────────────────────────────────────────────────────────────────────────

def run_real(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    captured = {}  # layer_idx -> (Q, K) numpy, post-RoPE, one batch

    # Wrap SDPA to capture Q/K. Model-agnostic: grabs whatever Q/K the model
    # hands to attention after RoPE, no per-arch glue.
    orig_sdpa = F.scaled_dot_product_attention
    call_counter = {"i": 0}
    layer_of_call = {}

    def capturing_sdpa(query, key, value, *a, **kw):
        idx = call_counter["i"]
        call_counter["i"] += 1
        li = layer_of_call.get(idx, idx)
        if li in args.layers:
            # query/key: (batch, heads, seq, head_dim) — take batch 0
            captured[li] = (
                query[0].detach().float().cpu().numpy(),
                key[0].detach().float().cpu().numpy(),
            )
        return orig_sdpa(query, key, value, *a, **kw)

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    # map sequential SDPA calls to layer indices (one call per layer in a
    # single forward with no generation)
    for i in range(n_layers):
        layer_of_call[i] = i

    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids[:, :args.seq_len].to(args.device)
    print(f"  seq_len={ids.shape[1]}, capturing layers {args.layers}")

    F.scaled_dot_product_attention = capturing_sdpa
    try:
        with torch.no_grad():
            model(ids)
    finally:
        F.scaled_dot_product_attention = orig_sdpa

    if not captured:
        sys.exit("no layers captured — check --layers against model depth")

    head_dim = next(iter(captured.values()))[1].shape[-1]
    print(f"  head_dim={head_dim}\n")

    # Optional per-head K bias for *_biassub configs: estimate as the mean K
    # vector per head over the captured sequence (proxy for the projection
    # bias; for exact bias use model.model.layers[l].self_attn.k_proj.bias).
    results = {}
    t0 = time.time()
    for cfg_name in args.configs:
        ph_tv = {}     # (layer, qhead) -> attention mass misrouted (TV)
        ph_top1 = {}   # top-1 key preservation
        ph_churn = {}  # top-k set churn (secondary)
        ph_mass = {}   # attention concentration
        for li, (Qh, Kh) in captured.items():
            n_q = Qh.shape[0]
            n_kv = Kh.shape[0]
            group = max(1, n_q // n_kv)
            qfn = make_quantizer(cfg_name, d=head_dim)
            Kq_heads, K_heads = [], []
            for h in range(n_kv):
                K = Kh[h]
                bias = K.mean(0) if cfg_name.endswith("_biassub") else None
                Kq_heads.append(qfn(K, bias))
                K_heads.append(K)
            for qh in range(n_q):
                kv = min(qh // group, n_kv - 1)
                m = routing_divergence(Qh[qh], K_heads[kv], Kq_heads[kv],
                                       topk=args.topk)
                if np.isnan(m["tv"]):
                    continue
                ph_tv[(li, qh)] = m["tv"]
                ph_top1[(li, qh)] = m["top1_keep"]
                ph_churn[(li, qh)] = m["topk_churn"]
                ph_mass[(li, qh)] = m["topk_mass"]

        tv = np.array(list(ph_tv.values()))
        top1 = np.array(list(ph_top1.values()))
        churn = np.array(list(ph_churn.values()))
        mass = np.array(list(ph_mass.values()))
        results[cfg_name] = {
            "mean_tv": float(tv.mean()),         # primary: mass misrouted
            "max_tv": float(tv.max()),           # worst head
            "mean_top1_keep": float(top1.mean()),
            "mean_topk_churn": float(churn.mean()),  # secondary sensitivity
            "mean_topk_mass": float(mass.mean()),
            "gini_tv": gini(tv),
            "frac_heads_80pct_tv": frac_heads_holding(tv, 0.8),
            "per_head_tv": {f"{li}_{h}": v for (li, h), v in ph_tv.items()},
        }
        s = results[cfg_name]
        print(f"  {cfg_name:14s}  mass_misrouted(TV)={s['mean_tv']:.4f}  "
              f"worst={s['max_tv']:.4f}  top1_keep={s['mean_top1_keep']:.3f}  "
              f"gini={s['gini_tv']:.3f}  heads80%={s['frac_heads_80pct_tv']:.2f}  "
              f"[churn={s['mean_topk_churn']:.2f}]")
    elapsed = time.time() - t0

    print(f"\n  total {elapsed:.0f}s")
    print("\nReading the output:")
    print("  • CHECK FIRST: fp16 mass_misrouted(TV) must be 0.0000. Else bug.")
    print("  • mass_misrouted(TV) = fraction of attention mass that lands on")
    print("    different keys after quantization. PRIMARY collapse metric:")
    print("    ~0 = routing preserved, rising = collapse. Forgiving of benign")
    print("    near-tied reordering, unlike raw top-k churn.")
    print("  • top1_keep = fraction of queries whose single top key survives.")
    print("  • gini/heads80% on TV = is the misrouting concentrated in a few")
    print("    heads (sparse, targetable) or spread across all (diffuse)?")
    print("  • churn (bracketed) = top-k set reordering, SECONDARY only — high")
    print("    even at high bits from benign tail reordering; not collapse.")
    print("  • mean_recall near 1.0  => routing preserved, no collapse")
    print("  • low gini + high frac  => diffuse collapse (all heads hurt)")
    print("  • high gini + low frac  => SPARSE collapse (few heads carry it)")
    print("    sparse is the regime where targeted protection / bias-subtract")
    print("    has a target; diffuse means uniform mixed-precision territory.")
    print("  • compare turbo{b} vs turboMSE{b}: if MSE >= turbo at matched")
    print("    bits, the field's drop-QJL consensus replicates on this model.")

    payload = {
        "model": args.model, "text": args.text, "seq_len": int(ids.shape[1]),
        "layers": args.layers, "topk": args.topk, "configs": args.configs,
        "results": results, "elapsed_seconds": elapsed,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  saved to {args.output}")


def run_bias_correlation(args):
    """Causal diagnostic: do the heads with the largest k_proj bias norms
    line up with the heads where mean-centering recovers the most attention
    routing? Sweeps ALL layers (per-head granularity needs many points),
    extracts each KV head's k_proj.bias norm, and correlates it against the
    per-head TV recovery (baseline TV minus bias-subtracted TV).

    Draws the causal line: Qwen2.5 has large per-head K biases (bias_check)
    -> those biases create a post-RoPE offset -> the offset misroutes
    attention -> centering recovers it. If high-bias heads are exactly the
    rescued heads, the mechanism is established, not just correlational luck.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.bias_corr            # e.g. "turbo5"
    biassub = base + "_biassub"

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    n_layers = model.config.num_hidden_layers

    # capture Q/K for ALL layers
    captured = {}
    orig = F.scaled_dot_product_attention
    counter = {"i": 0}

    def cap(query, key, value, *a, **kw):
        idx = counter["i"]; counter["i"] += 1
        if idx < n_layers:
            captured[idx] = (query[0].detach().float().cpu().numpy(),
                             key[0].detach().float().cpu().numpy())
        return orig(query, key, value, *a, **kw)

    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids[:, :args.seq_len].to(args.device)

    F.scaled_dot_product_attention = cap
    try:
        with torch.no_grad():
            model(ids)
    finally:
        F.scaled_dot_product_attention = orig

    head_dim = next(iter(captured.values()))[1].shape[-1]

    # extract per-(layer, kv_head) k_proj bias norm
    def get_kproj_bias(li):
        attn = model.model.layers[li].self_attn
        b = getattr(attn.k_proj, "bias", None)
        if b is None:
            return None
        b = b.detach().float().cpu().numpy()
        n_kv = b.shape[0] // head_dim
        return b.reshape(n_kv, head_dim)

    # check the model even has k_proj bias
    sample_bias = get_kproj_bias(next(iter(captured.keys())))
    if sample_bias is None:
        print(f"\n{args.model} has no k_proj.bias — this architecture carries "
              "no K projection bias, so the bias-correlation diagnostic is N/A "
              "(and centering has nothing structural to remove). That is itself "
              "the control: the effect is specific to bias-carrying models.")
        return

    points = []  # (layer, kv_head, bias_norm, tv_base, tv_biassub, recovery)
    qfn_base = make_quantizer(base, d=head_dim)
    qfn_bs = make_quantizer(biassub, d=head_dim)
    for li, (Qh, Kh) in sorted(captured.items()):
        n_q, n_kv = Qh.shape[0], Kh.shape[0]
        group = max(1, n_q // n_kv)
        bias_heads = get_kproj_bias(li)
        for kv in range(n_kv):
            K = Kh[kv]
            mean_bias = K.mean(0)  # post-RoPE centering vector (the real fix)
            Kq_base = qfn_base(K, None)
            Kq_bs = qfn_bs(K, mean_bias)
            # average TV over the query heads in this kv group
            tvb, tvs = [], []
            for qh in range(kv * group, min((kv + 1) * group, n_q)):
                tvb.append(routing_divergence(Qh[qh], K, Kq_base, args.topk)["tv"])
                tvs.append(routing_divergence(Qh[qh], K, Kq_bs, args.topk)["tv"])
            tv_base = float(np.nanmean(tvb))
            tv_bs = float(np.nanmean(tvs))
            bnorm = float(np.linalg.norm(bias_heads[kv]))
            points.append((li, kv, bnorm, tv_base, tv_bs, tv_base - tv_bs))

    arr = np.array([(p[2], p[3], p[5]) for p in points])  # bnorm, tv_base, recovery
    bnorm, tv_base, recov = arr[:, 0], arr[:, 1], arr[:, 2]

    def pearson(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return pearson(ra.astype(float), rb.astype(float))

    print(f"\n  {len(points)} (layer, kv-head) points across {len(captured)} layers\n")
    print(f"  bias_norm vs baseline TV ({base}):   "
          f"pearson={pearson(bnorm, tv_base):+.3f}  spearman={spearman(bnorm, tv_base):+.3f}")
    print(f"  bias_norm vs TV recovery (centering): "
          f"pearson={pearson(bnorm, recov):+.3f}  spearman={spearman(bnorm, recov):+.3f}")
    print("\n  Prediction if the bias is causal: both positive — high-bias")
    print("  heads collapse more AND recover more under centering.\n")

    # show the extremes
    order = np.argsort(-bnorm)
    print(f"  {'layer':>5} {'kvh':>3} {'bias_norm':>10} {'TV_base':>9} "
          f"{'TV_bsub':>9} {'recovery':>9}")
    for j in list(order[:5]) + list(order[-5:]):
        li, kv, bn, tb, ts, rc = points[j]
        print(f"  {li:>5} {kv:>3} {bn:>10.2f} {tb:>9.4f} {ts:>9.4f} {rc:>9.4f}")

    payload = {
        "model": args.model, "base_config": base, "n_points": len(points),
        "pearson_bias_vs_baseTV": pearson(bnorm, tv_base),
        "spearman_bias_vs_baseTV": spearman(bnorm, tv_base),
        "pearson_bias_vs_recovery": pearson(bnorm, recov),
        "spearman_bias_vs_recovery": spearman(bnorm, recov),
        "points": [{"layer": p[0], "kv_head": p[1], "bias_norm": p[2],
                    "tv_base": p[3], "tv_biassub": p[4], "recovery": p[5]}
                   for p in points],
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  saved to {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--text", help="real text file for KV capture")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--configs", nargs="+",
                    default=["turbo2", "turboMSE2", "turbo4", "turboMSE4"])
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 14, 27])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--output", default="tq_collapse_probe.json")
    ap.add_argument("--self-test", action="store_true",
                    help="verify metric + real-quantizer wiring, no model")
    ap.add_argument("--bias-corr", default=None, metavar="CONFIG",
                    help="run the bias-vs-routing causal diagnostic with the "
                         "given baseline config (e.g. turbo5); sweeps ALL "
                         "layers and correlates k_proj bias norm against TV "
                         "recovery under centering")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.bias_corr:
        if not args.text:
            sys.exit("--text required for --bias-corr")
        run_bias_correlation(args)
        return
    if not args.text:
        sys.exit("--text required for the real run (or use --self-test)")
    run_real(args)


if __name__ == "__main__":
    main()
