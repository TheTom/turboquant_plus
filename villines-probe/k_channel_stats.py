"""
k_channel_stats.py — Per-channel K-cache statistics across architectures.

Companion to tq_collapse_probe.py. Captures post-RoPE K (same SDPA
monkey-patch, model-agnostic) and quantifies the candidate routing-collapse
mechanism that survives the falsified bias hypothesis: large fixed per-channel
K offsets, regardless of whether a k_proj bias term produces them.

Per (layer, kv_head) it reports:
    offset_norm      ||mean_t K[t,:]||_2 — the exact vector *_biassub removes.
                     Generalizes Qwen2.5's bias_norm to bias-free models.
    spread           E_t ||K[t,:] - mean||_2 — typical centered vector norm.
    offset_ratio     offset_norm / spread — anisotropy of the K cloud.
                     PolarQuant codebooks assume a roughly centered cloud;
                     offset_ratio >> 0 means the codebook spends its bits
                     re-encoding a constant.
    max_abs_chan_mu  largest |per-channel mean| (outlier-channel signature)
    max_chan_kurt    largest per-channel excess kurtosis (heavy-tail signature)

Usage:
    PYTHONPATH=. python3 villines-probe/k_channel_stats.py \
        --model Qwen/Qwen2.5-7B-Instruct --text villines-probe/calib.txt \
        --layers all --seq-len 2048 --device mps \
        --output villines-probe/kstats_qwen25_7b_instruct.json

TODO: extend with per-channel variance share (top-k channels' fraction of
total variance) if offset_ratio alone doesn't separate Qwen from Mistral.
"""

import argparse
import json
import sys
import time

import numpy as np


def head_stats(K):
    """K: (seq, head_dim) float32 -> dict of channel statistics."""
    mu = K.mean(0)                      # (d,) per-channel mean
    Kc = K - mu[None, :]
    offset_norm = float(np.linalg.norm(mu))
    spread = float(np.linalg.norm(Kc, axis=1).mean())
    var = Kc.var(0)
    std = np.sqrt(var + 1e-12)
    # per-channel excess kurtosis
    kurt = (Kc ** 4).mean(0) / (var + 1e-12) ** 2 - 3.0
    return {
        "offset_norm": offset_norm,
        "spread": spread,
        "offset_ratio": offset_norm / max(spread, 1e-12),
        "max_abs_chan_mu": float(np.abs(mu).max()),
        "mean_abs_chan_mu": float(np.abs(mu).mean()),
        "max_chan_std": float(std.max()),
        "median_chan_std": float(np.median(std)),
        "chan_std_max_over_median": float(std.max() / max(np.median(std), 1e-12)),
        "max_chan_kurt": float(kurt.max()),
        "median_chan_kurt": float(np.median(kurt)),
    }


def run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    n_layers = model.config.num_hidden_layers
    layers = (list(range(n_layers)) if args.layers == ["all"]
              else [int(x) for x in args.layers])

    # Online per-layer stats: keep only K (not Q), one layer at a time.
    stats = {}  # (layer, kv_head) -> dict
    orig_sdpa = F.scaled_dot_product_attention
    call_counter = {"i": 0}

    def capturing_sdpa(query, key, value, *a, **kw):
        li = call_counter["i"]
        call_counter["i"] += 1
        if li in layers:
            K = key[0].detach().float().cpu().numpy()  # (n_kv, seq, d)
            for h in range(K.shape[0]):
                stats[f"{li}_{h}"] = head_stats(K[h])
        return orig_sdpa(query, key, value, *a, **kw)

    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids[:, :args.seq_len].to(args.device)
    print(f"  seq_len={ids.shape[1]}, layers={len(layers)}")

    t0 = time.time()
    F.scaled_dot_product_attention = capturing_sdpa
    try:
        with torch.no_grad():
            model(ids)
    finally:
        F.scaled_dot_product_attention = orig_sdpa

    if not stats:
        sys.exit("no layers captured")

    ratios = np.array([s["offset_ratio"] for s in stats.values()])
    kurts = np.array([s["max_chan_kurt"] for s in stats.values()])
    summary = {
        "model": args.model,
        "n_head_points": len(stats),
        "offset_ratio_mean": float(ratios.mean()),
        "offset_ratio_p90": float(np.percentile(ratios, 90)),
        "offset_ratio_max": float(ratios.max()),
        "frac_heads_offset_ratio_gt_0p5": float((ratios > 0.5).mean()),
        "frac_heads_offset_ratio_gt_1": float((ratios > 1.0).mean()),
        "max_chan_kurt_median": float(np.median(kurts)),
        "max_chan_kurt_p90": float(np.percentile(kurts, 90)),
        "elapsed_seconds": time.time() - t0,
    }
    out = {"summary": summary, "per_head": stats,
           "seq_len": int(ids.shape[1]), "text": args.text}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--layers", nargs="+", default=["all"])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--output", default="k_channel_stats.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
