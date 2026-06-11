"""
torquant_certificate.py — Empirical evaluation of the TorQuant (Braun 2026)
gradient certificate and flatness ratio on real post-RoPE K activations.

TorQuant ("Searching the Hadamard Variety") proposes learning butterfly
rotor angles instead of the fixed WHT, and gives two pre-training
diagnostics that decide whether any headroom exists for a given model:

    rho          = max_i (H C H^T)_ii / (tr C / d)        [its eq 15]
                   first-order recoverable variance factor; implied bit
                   saving Delta_b = 0.5 * log2(rho)        [its eq 19]
    certificate  = || grad_Theta L(Theta_H) ||             [its eq 16]
                   stagewise closed form per butterfly pair: 2*gamma*(beta
                   - alpha), gamma = cross-pair covariance, beta - alpha =
                   variance imbalance, computed in the intermediate
                   coordinates entering each stage (its Prop 6.1 + App A).

The paper's C = E[yy^T] is the UNCENTERED second moment. Our offset
finding (villines-probe-validation §10) predicts that on Qwen the rank-1
mean term mu mu^T dominates C on the catastrophic heads (offset_ratio up
to 58), so the certificate should fire there — and mostly vanish once K
is centered. Whatever survives centering is TorQuant's real headroom.

So we compute, per (layer, kv_head), on the SAME captures the TV probe
uses: rho / Delta_b / certificate on raw K and on centered K, for a
Qwen2.5 vs Mistral contrast.

Simplifications vs the full TorQuant pipeline (noted, deliberate):
  - S = I (no per-pair scalings); the certificate question "does cross-
    pair covariance survive the Hadamard" is unchanged, S only reweights.
  - Flattening objective J (its Prop 6.1 form), not the query-weighted
    L_K; the paper derives the closed form for J and calls L_K its
    Sigma_q-weighted analogue.

Usage:
    PYTHONPATH=. python3 villines-probe/torquant_certificate.py \
        --model Qwen/Qwen2.5-7B-Instruct --text villines-probe/calib.txt \
        --layers all --seq-len 2048 --device mps \
        --output villines-probe/torquant_cert_qwen25.json
"""

import argparse
import json
import sys
import time

import numpy as np


def butterfly_stage_pairs(d, stage):
    """Cooley–Tukey wiring: stage l (1-based) couples (a, a+2^(l-1))
    within contiguous blocks of size 2^l. Yields (a, b) index pairs."""
    stride = 1 << (stage - 1)
    block = stride << 1
    for b0 in range(0, d, block):
        for a in range(b0, b0 + stride):
            yield a, a + stride

def apply_h2_stage(C, d, stage):
    """Apply the pi/4 butterfly (= H2 on each pair) of one stage to both
    sides of C: C <- B C B^T. B is symmetric orthogonal."""
    r = 1.0 / np.sqrt(2.0)
    for a, b in butterfly_stage_pairs(d, stage):
        ra, rb = C[a].copy(), C[b].copy()
        C[a], C[b] = r * (ra + rb), r * (ra - rb)
    for a, b in butterfly_stage_pairs(d, stage):
        ca, cb = C[:, a].copy(), C[:, b].copy()
        C[:, a], C[:, b] = r * (ca + cb), r * (ca - cb)
    return C

def torquant_diagnostics(K):
    """K: (seq, d) float64. Returns rho, delta_b, certificate for the
    uncentered second moment of K (the paper's C = E[yy^T])."""
    n, d = K.shape
    L = int(np.log2(d))
    assert (1 << L) == d, "head_dim must be a power of 2"
    C = (K.T @ K) / n

    # certificate: stream stages, gradient per pair at the WHT point
    grad_sq = 0.0
    Cw = C.copy()
    for stage in range(1, L + 1):
        # C' entering this stage = (B_{stage-1}...B_1) C (...)^T — Cw holds it
        for a, b in butterfly_stage_pairs(d, stage):
            alpha, beta, gamma = Cw[a, a], Cw[b, b], Cw[a, b]
            g = 2.0 * gamma * (beta - alpha)
            grad_sq += g * g
        Cw = apply_h2_stage(Cw, d, stage)

    # after all L stages Cw = H C H^T — reuse it for rho
    diag = np.diag(Cw)
    tau = np.trace(C) / d
    rho = float(diag.max() / max(tau, 1e-30))
    return {
        "rho": rho,
        "delta_b": float(0.5 * np.log2(max(rho, 1.0))),
        "certificate": float(np.sqrt(grad_sq)),
        # scale-free version so heads of different magnitude compare:
        "certificate_normalized": float(np.sqrt(grad_sq) / max(tau * tau, 1e-30)),
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

    results = {}  # "layer_kvhead" -> {raw: {...}, centered: {...}, offset_ratio}
    orig_sdpa = F.scaled_dot_product_attention
    counter = {"i": 0}

    def capturing_sdpa(query, key, value, *a, **kw):
        li = counter["i"]
        counter["i"] += 1
        if li in layers:
            K = key[0].detach().cpu().numpy().astype(np.float64)  # (n_kv, seq, d)
            for h in range(K.shape[0]):
                Kh = K[h]
                mu = Kh.mean(0)
                spread = np.linalg.norm(Kh - mu[None, :], axis=1).mean()
                results[f"{li}_{h}"] = {
                    "raw": torquant_diagnostics(Kh),
                    "centered": torquant_diagnostics(Kh - mu[None, :]),
                    "offset_ratio": float(np.linalg.norm(mu) / max(spread, 1e-12)),
                }
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
    if not results:
        sys.exit("no layers captured")

    def agg(which, key):
        v = np.array([r[which][key] for r in results.values()])
        return {"mean": float(v.mean()), "p90": float(np.percentile(v, 90)),
                "max": float(v.max())}

    summary = {
        "model": args.model,
        "n_head_points": len(results),
        "rho_raw": agg("raw", "rho"),
        "rho_centered": agg("centered", "rho"),
        "delta_b_raw": agg("raw", "delta_b"),
        "delta_b_centered": agg("centered", "delta_b"),
        "cert_norm_raw": agg("raw", "certificate_normalized"),
        "cert_norm_centered": agg("centered", "certificate_normalized"),
        "elapsed_seconds": time.time() - t0,
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_head": results,
                   "seq_len": int(ids.shape[1])}, f, indent=1)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--layers", nargs="+", default=["all"])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--output", default="torquant_cert.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
