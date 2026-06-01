"""
tv_vs_ppl.py — Does routing-side TV predict downstream quality?

Tests one falsifiable question: across KV-quant configs, does the
attention-mass divergence (TV, a routing-side metric measurable without
running the model to logits) predict the model's DOWNSTREAM ability to
retrieve a planted fact (a "needle NLL" = teacher-forced loss on the fact
after a full forward pass)?

This is the "ghost PPL" test. TV is cheap and upstream; needle-NLL is the
real downstream signal. If TV tracks needle-NLL across configs — especially
where reconstruction MSE provably does NOT (Turney, why-mse-fails) — then
TV is a usable collapse predictor.

Two kinds of forward pass, both single-shot (no generation loop):
  1. fp16 pass, no hooks: capture Q/K -> compute TV per config offline.
  2. per-config pass, quantizing hooks on k_proj/v_proj (REAL package):
     teacher-forced NLL on the planted-fact tokens = downstream signal.

Usage:
  python tv_vs_ppl.py --model Qwen/Qwen2.5-3B --text calib.txt \\
      --configs fp16 turbo5 turbo5_biassub turbo3 turbo2 \\
      --tv-layers 0 1 18 --seq-len 1024
"""

import argparse, json, sys, re
import numpy as np

try:
    import turboquant as tq
except ImportError:
    sys.exit("turboquant not importable — pip install -e . in turboquant_plus")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── real-package quantizer slot (same as the probe) ───────────────────────
def make_quantizer(config_name, d, seed=42):
    def with_bias(fn):
        def w(K, bias=None):
            if bias is not None: K = K - bias[None, :]
            out = fn(K)
            if bias is not None: out = out + bias[None, :]
            return out
        return w
    m = re.search(r'(\d+)', config_name)
    bw = int(m.group(1)) if m else 4
    norm_corr = not config_name.endswith("_nonc")
    if config_name.startswith("fp16"):
        return with_bias(lambda K: K.astype(np.float32))
    if config_name.startswith("turboMSE"):
        q = tq.TurboQuantMSE(d=d, bit_width=bw, seed=seed, norm_correction=norm_corr)
        return with_bias(lambda K: q.dequantize(*q.quantize(K.astype(np.float32))))
    if config_name.startswith("turbo"):
        q = tq.TurboQuant(d=d, bit_width=bw, seed=seed, norm_correction=norm_corr)
        return with_bias(lambda K: q.dequantize(q.quantize(K.astype(np.float32))))
    raise ValueError(config_name)


def routing_tv(Q, K_fp16, K_quant, topk=8, sample_q=128, seed=0):
    n, d = Q.shape
    scale = 1.0/np.sqrt(d); rng = np.random.default_rng(seed)
    valid = np.arange(topk+1, n)
    if len(valid) == 0: return np.nan
    qpos = rng.choice(valid, size=min(sample_q, len(valid)), replace=False)
    tvs = []
    for i in qpos:
        ctx = i+1; q = Q[i]
        s_fp = (K_fp16[:ctx] @ q)*scale; s_q = (K_quant[:ctx] @ q)*scale
        p = np.exp(s_fp-s_fp.max()); p/=p.sum()
        pq = np.exp(s_q-s_q.max()); pq/=pq.sum()
        tvs.append(0.5*np.abs(p-pq).sum())
    return float(np.mean(tvs))


# ── quantizing hooks for the downstream pass ──────────────────────────────
def install_quant_hooks(model, cfg_name, head_dim, n_kv):
    qfn = make_quantizer(cfg_name, d=head_dim)
    handles = []
    def mk(is_k):
        def hook(module, inp, out):
            bsz, seq, _ = out.shape
            x = out.view(bsz, seq, n_kv, head_dim)
            new = x.clone()
            for h in range(n_kv):
                K = x[0, :, h, :].detach().float().cpu().numpy()
                bias = K.mean(0) if cfg_name.endswith("_biassub") else None
                Kq = qfn(K, bias)
                new[0, :, h, :] = torch.from_numpy(Kq).to(out.dtype).to(out.device)
            return new.view(bsz, seq, n_kv*head_dim)
        return hook
    for layer in model.model.layers:
        handles.append(layer.self_attn.k_proj.register_forward_hook(mk(True)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(mk(False)))
    return handles


def clear_hooks(handles):
    for h in handles: h.remove()


NEEDLE = "Remember this: the secret access code is GLYPH-7392."
QUESTION = "\n\nThe secret access code is"
ANSWER = " GLYPH-7392"


def build_ids(tok, haystack_tokens, seq_len, device):
    needle = tok(NEEDLE, add_special_tokens=False, return_tensors="pt").input_ids[0]
    quest = tok(QUESTION, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ans = tok(ANSWER, add_special_tokens=False, return_tensors="pt").input_ids[0]
    budget = seq_len - len(needle) - len(quest) - len(ans) - 5
    h = haystack_tokens[:budget]
    insert = int(len(h)*0.5)
    ids = torch.cat([h[:insert], needle, h[insert:], quest, ans])
    n_ans = len(ans)
    return ids.unsqueeze(0).to(device), n_ans


def needle_nll(model, ids, n_ans):
    """Teacher-forced NLL on the answer tokens = downstream needle PPL."""
    labels = ids.clone()
    labels[:, :-n_ans] = -100  # only score the answer span
    with torch.no_grad():
        out = model(ids, labels=labels)
    return float(out.loss.item())


def answer_top1(model, ids, n_ans):
    """Is the first answer token the model's argmax given the context?"""
    prefix = ids[:, :-n_ans]
    first_ans = ids[0, -n_ans].item()
    with torch.no_grad():
        logits = model(prefix).logits[0, -1]
    return int(torch.argmax(logits).item() == first_ans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--text", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--configs", nargs="+",
                    default=["fp16","turbo5","turbo5_biassub","turbo3","turbo2"])
    ap.add_argument("--tv-layers", nargs="+", type=int, default=[0,1,18])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--output", default="tv_vs_ppl.json")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads

    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    haystack = tok(text, return_tensors="pt").input_ids[0]
    ids, n_ans = build_ids(tok, haystack, args.seq_len, args.device)
    print(f"  seq_len={ids.shape[1]}, answer_tokens={n_ans}, "
          f"n_kv={n_kv}, head_dim={head_dim}\n")

    # ── fp16 pass: capture Q/K at tv-layers for offline TV ──
    cap = {}
    orig = F.scaled_dot_product_attention
    ctr = {"i": 0}
    def capture(q,k,v,*a,**kw):
        i = ctr["i"]; ctr["i"] += 1
        if i in args.tv_layers:
            cap[i] = (q[0].detach().float().cpu().numpy(),
                      k[0].detach().float().cpu().numpy())
        return orig(q,k,v,*a,**kw)
    F.scaled_dot_product_attention = capture
    try:
        with torch.no_grad(): model(ids)
    finally:
        F.scaled_dot_product_attention = orig

    # ── per-config: TV (offline) + downstream needle NLL (hooked pass) ──
    rows = []
    for cfg in args.configs:
        # TV offline from captured fp16 Q/K
        qfn = make_quantizer(cfg, d=head_dim)
        tvs = []
        for li,(Qh,Kh) in cap.items():
            nq, nkv = Qh.shape[0], Kh.shape[0]
            grp = max(1, nq//nkv)
            for h in range(nkv):
                K = Kh[h]; bias = K.mean(0) if cfg.endswith("_biassub") else None
                Kq = qfn(K, bias)
                tvs.append(routing_tv(Qh[h*grp], K, Kq))
        mean_tv = float(np.nanmean(tvs))

        # downstream needle NLL with quantizing hooks
        if cfg == "fp16":
            nll = needle_nll(model, ids, n_ans)
            top1 = answer_top1(model, ids, n_ans)
        else:
            handles = install_quant_hooks(model, cfg, head_dim, n_kv)
            try:
                nll = needle_nll(model, ids, n_ans)
                top1 = answer_top1(model, ids, n_ans)
            finally:
                clear_hooks(handles)
        rows.append((cfg, mean_tv, nll, top1))
        print(f"  {cfg:16s}  TV={mean_tv:.4f}  needle_NLL={nll:7.3f}  "
              f"needle_PPL={np.exp(nll):9.2f}  answer_top1={'OK' if top1 else 'MISS'}")

    # ── does TV predict needle NLL? ──
    tv = np.array([r[1] for r in rows]); nll = np.array([r[2] for r in rows])
    def pear(a,b):
        return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else float("nan")
    def spear(a,b):
        return pear(np.argsort(np.argsort(a)).astype(float),
                    np.argsort(np.argsort(b)).astype(float))
    print(f"\n  TV vs needle_NLL across {len(rows)} configs: "
          f"pearson={pear(tv,nll):+.3f}  spearman={spear(tv,nll):+.3f}")
    print("\n  If positive and tight: TV predicts downstream collapse — a")
    print("  routing-side 'ghost PPL' that needs no logit eval. If flat or")
    print("  noisy: TV is diagnostic but not predictive; needle-NLL is")
    print("  measuring something TV misses (likely the value path or MLP).")

    with open(args.output,"w") as f:
        json.dump({"model":args.model,
                   "rows":[{"config":r[0],"tv":r[1],"needle_nll":r[2],
                            "needle_ppl":float(np.exp(r[2])),"answer_top1":r[3]}
                           for r in rows],
                   "pearson_tv_nll":pear(tv,nll),
                   "spearman_tv_nll":spear(tv,nll)}, f, indent=2)
    print(f"\n  saved to {args.output}")


if __name__ == "__main__":
    main()
