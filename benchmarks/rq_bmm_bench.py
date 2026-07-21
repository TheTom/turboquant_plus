#!/usr/bin/env python3
"""Benchmark: RotorQuant batched 3x3 matmul vs element-wise vs TurboQuant matmul on MPS."""
import torch, time, numpy as np, sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/turboquant_plus"))

device = "mps"
d_actual = 128
bits = 3
n_groups = (d_actual + 2) // 3  # 43
n_warmup = 20
n_iter = 200

from turboquant.codebook import optimal_centroids
from turboquant.clifford import make_random_rotor

d_eff = max(n_groups * 8, 64)
centroids = torch.tensor(optimal_centroids(bits - 1, d_eff), dtype=torch.float32, device=device)
n_levels = len(centroids)

# TQ rotation matrix (build on CPU)
G = torch.randn(d_actual, d_actual)
Pi, _ = torch.linalg.qr(G)
Pi = Pi.to(device)

# Build 3x3 rotation matrices from rotors
rng = np.random.default_rng(42)
rotors_list = []
for i in range(n_groups):
    r = make_random_rotor(rng)
    rotors_list.append([r[0], r[4], r[5], r[6]])
rotors_t = torch.tensor(rotors_list, dtype=torch.float32, device=device)

s = rotors_t[:, 0]
p = rotors_t[:, 1]
q = rotors_t[:, 2]
r = rotors_t[:, 3]
s2, p2, q2, r2 = s**2, p**2, q**2, r**2

M = torch.zeros(n_groups, 3, 3, device=device)
M[:, 0, 0] = s2 - p2 - q2 + r2
M[:, 0, 1] = 2*s*p - 2*q*r
M[:, 0, 2] = 2*s*q + 2*p*r
M[:, 1, 0] = -2*s*p - 2*q*r
M[:, 1, 1] = s2 - p2 + q2 - r2
M[:, 1, 2] = 2*s*r - 2*p*q
M[:, 2, 0] = -2*s*q + 2*p*r
M[:, 2, 1] = -2*s*r - 2*p*q
M[:, 2, 2] = s2 + p2 - q2 - r2

Mt = M.transpose(1, 2).contiguous()

print(f"Mac Mini M4 - MPS Benchmark (d={d_actual}, {bits}-bit)")
print(f"Rotation matrices: {n_groups} x 3x3 precomputed")
print()
hdr = f"  {'n':>6s}  {'TQ (d*d mm)':>14s}  {'RQ (elem-wise)':>16s}  {'RQ (3x3 bmm)':>14s}  {'bmm vs TQ':>10s}"
print(hdr)
print(f"  {'---':>6s}  {'---':>14s}  {'---':>16s}  {'---':>14s}  {'---':>10s}")


def fmt(us):
    if us < 1000:
        return f"{us:.0f} us"
    return f"{us/1000:.2f} ms"


for n in [1024, 4096, 16384, 65536]:
    x = torch.randn(n, d_actual, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    # TQ: d*d matmul + quantize
    torch.mps.synchronize()
    for _ in range(n_warmup):
        y = x @ Pi.T
        _ = (y.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = x @ Pi.T
        _ = (y.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
    torch.mps.synchronize()
    tq_us = (time.perf_counter() - t0) / n_iter * 1e6

    # RQ element-wise (old)
    sr = rotors_t[:, 0]; p12 = rotors_t[:, 1]; p13 = rotors_t[:, 2]; p23 = rotors_t[:, 3]

    def rq_ew(x_in):
        pad = (3 - d_actual % 3) % 3
        if pad > 0:
            x_in = torch.nn.functional.pad(x_in, (0, pad))
        ng = x_in.shape[-1] // 3
        xg = x_in.reshape(x_in.shape[0], ng, 3)
        mv = torch.zeros(x_in.shape[0], ng, 8, device=device)
        mv[:, :, 1] = xg[:, :, 0]
        mv[:, :, 2] = xg[:, :, 1]
        mv[:, :, 3] = xg[:, :, 2]
        t = torch.empty_like(mv)
        t[:,:,0] = sr[:ng]*mv[:,:,0] - p12[:ng]*mv[:,:,4] - p13[:ng]*mv[:,:,5] - p23[:ng]*mv[:,:,6]
        t[:,:,1] = sr[:ng]*mv[:,:,1] + p12[:ng]*mv[:,:,2] + p13[:ng]*mv[:,:,3] + p23[:ng]*mv[:,:,7]
        t[:,:,2] = sr[:ng]*mv[:,:,2] - p12[:ng]*mv[:,:,1] + p23[:ng]*mv[:,:,3] - p13[:ng]*mv[:,:,7]
        t[:,:,3] = sr[:ng]*mv[:,:,3] - p13[:ng]*mv[:,:,1] - p23[:ng]*mv[:,:,2] + p12[:ng]*mv[:,:,7]
        t[:,:,4] = sr[:ng]*mv[:,:,4] + p12[:ng]*mv[:,:,0]
        t[:,:,5] = sr[:ng]*mv[:,:,5] + p13[:ng]*mv[:,:,0]
        t[:,:,6] = sr[:ng]*mv[:,:,6] + p23[:ng]*mv[:,:,0]
        t[:,:,7] = sr[:ng]*mv[:,:,7] - p23[:ng]*mv[:,:,1] + p13[:ng]*mv[:,:,2] - p12[:ng]*mv[:,:,3]
        rr = torch.empty_like(t)
        rr[:,:,0] = sr[:ng]*t[:,:,0]+p12[:ng]*t[:,:,4]+p13[:ng]*t[:,:,5]+p23[:ng]*t[:,:,6]
        rr[:,:,1] = sr[:ng]*t[:,:,1]-p12[:ng]*t[:,:,2]-p13[:ng]*t[:,:,3]-p23[:ng]*t[:,:,7]
        rr[:,:,2] = sr[:ng]*t[:,:,2]+p12[:ng]*t[:,:,1]-p23[:ng]*t[:,:,3]+p13[:ng]*t[:,:,7]
        rr[:,:,3] = sr[:ng]*t[:,:,3]+p13[:ng]*t[:,:,1]+p23[:ng]*t[:,:,2]-p12[:ng]*t[:,:,7]
        rr[:,:,4] = sr[:ng]*t[:,:,4]-p12[:ng]*t[:,:,0]
        rr[:,:,5] = sr[:ng]*t[:,:,5]-p13[:ng]*t[:,:,0]
        rr[:,:,6] = sr[:ng]*t[:,:,6]-p23[:ng]*t[:,:,0]
        rr[:,:,7] = sr[:ng]*t[:,:,7]+p23[:ng]*t[:,:,1]-p13[:ng]*t[:,:,2]+p12[:ng]*t[:,:,3]
        flat = rr.reshape(rr.shape[0], -1)
        return (flat.unsqueeze(-1) - centroids).abs().argmin(dim=-1)

    torch.mps.synchronize()
    for _ in range(n_warmup):
        rq_ew(x)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rq_ew(x)
    torch.mps.synchronize()
    rq_ew_us = (time.perf_counter() - t0) / n_iter * 1e6

    # RQ batched 3x3 matmul (fast path)
    def rq_bmm(x_in):
        pad = (3 - d_actual % 3) % 3
        if pad > 0:
            x_in = torch.nn.functional.pad(x_in, (0, pad))
        ng = x_in.shape[-1] // 3
        batch = x_in.shape[0]
        xg = x_in.reshape(batch, ng, 3)
        # Use einsum for batched small matmul (avoids expand+reshape overhead)
        rotated = torch.einsum('bgi,gij->bgj', xg, M[:ng])
        # Quantize
        flat = rotated.reshape(batch, -1)
        idx = (flat.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        q_vals = centroids[idx].reshape(batch, ng, 3)
        # Inverse rotate
        deq = torch.einsum('bgi,gij->bgj', q_vals, Mt[:ng])
        return deq.reshape(batch, -1)[:, :d_actual]

    torch.mps.synchronize()
    for _ in range(n_warmup):
        rq_bmm(x)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rq_bmm(x)
    torch.mps.synchronize()
    rq_bmm_us = (time.perf_counter() - t0) / n_iter * 1e6

    ratio = rq_bmm_us / tq_us
    faster = "RQ" if ratio < 1 else "TQ"
    speedup = max(ratio, 1/ratio)
    print(f"  {n:>6d}  {fmt(tq_us):>14s}  {fmt(rq_ew_us):>16s}  {fmt(rq_bmm_us):>14s}  {faster} {speedup:.1f}x")

print()
print("bmm vs elem-wise speedup shows the benefit of the 3x3 matmul trick.")
