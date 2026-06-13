#!/usr/bin/env python3
"""Metal test - use file-based library loading."""
import sys, os, struct, ctypes, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Documents/turboquant_plus"))

from Metal import MTLCreateSystemDefaultDevice, MTLResourceStorageModeShared, MTLSizeMake
from Foundation import NSURL

dev = MTLCreateSystemDefaultDevice()
print(f"Device: {dev.name()}", flush=True)

# Load via file URL instead of NSData
lib_path = "/tmp/rotor_fused.metallib"
url = NSURL.fileURLWithPath_(lib_path)
library, err = dev.newLibraryWithURL_error_(url, None)
if not library:
    print(f"Library load failed: {err}", flush=True)
    sys.exit(1)
print(f"Library loaded: {library.functionNames()}", flush=True)

fn = library.newFunctionWithName_("rotor_full_fused")
assert fn, "Function not found"

pso, err = dev.newComputePipelineStateWithFunction_error_(fn, None)
assert pso, f"PSO error: {err}"
print(f"Pipeline ready, max threads: {pso.maxTotalThreadsPerThreadgroup()}", flush=True)

queue = dev.newCommandQueue()

d = 128
n_groups = (d + 2) // 3

# Identity rotors for correctness test
rotors = np.zeros((n_groups, 4), dtype=np.float32)
rotors[:, 0] = 1.0
cents = np.array([-0.15, -0.05, 0.05, 0.15], dtype=np.float32)
n_levels = len(cents)

batch = 8
x = np.random.randn(batch, d).astype(np.float32)
x = (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)

def mkbuf(arr):
    return dev.newBufferWithBytes_length_options_(arr.tobytes(), arr.nbytes, MTLResourceStorageModeShared)

buf_in = mkbuf(x)
buf_out = dev.newBufferWithLength_options_(batch * d * 4, MTLResourceStorageModeShared)
buf_r = mkbuf(rotors)
buf_c = mkbuf(cents)
params = struct.pack("IIII", batch, d, n_groups, n_levels)
buf_p = dev.newBufferWithBytes_length_options_(params, len(params), MTLResourceStorageModeShared)

print("Dispatching...", flush=True)
cmd = queue.commandBuffer()
enc = cmd.computeCommandEncoder()
enc.setComputePipelineState_(pso)
enc.setBuffer_offset_atIndex_(buf_in, 0, 0)
enc.setBuffer_offset_atIndex_(buf_r, 0, 1)
enc.setBuffer_offset_atIndex_(buf_c, 0, 2)
enc.setBuffer_offset_atIndex_(buf_out, 0, 3)
enc.setBuffer_offset_atIndex_(buf_p, 0, 4)

tg = MTLSizeMake(batch, n_groups, 1)
tg_size = MTLSizeMake(1, min(n_groups, pso.maxTotalThreadsPerThreadgroup()), 1)
enc.dispatchThreads_threadsPerThreadgroup_(tg, tg_size)
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

status = cmd.status()
print(f"Status: {status} (4=completed)", flush=True)

if cmd.error():
    print(f"Error: {cmd.error()}", flush=True)
else:
    # Read Metal buffer contents via memoryview
    import objc
    n_floats = batch * d
    buf_bytes = bytes(buf_out.contents().as_buffer(n_floats * 4))
    out = np.frombuffer(buf_bytes, dtype=np.float32).reshape(batch, d).copy()
    mse = np.mean(np.sum((x - out)**2, axis=-1))
    print(f"MSE (identity rotor, 4-level quant): {mse:.6f}", flush=True)
    print(f"Input[0,:4]:  {x[0,:4]}", flush=True)
    print(f"Output[0,:4]: {out[0,:4]}", flush=True)

# --- Benchmark ---
print("\n--- BENCHMARK ---", flush=True)
from turboquant.clifford import make_random_rotor
from turboquant.codebook import optimal_centroids

rng = np.random.default_rng(42)
real_rotors = np.zeros((n_groups, 4), dtype=np.float32)
for g in range(n_groups):
    r = make_random_rotor(rng)
    real_rotors[g] = [r[0], r[4], r[5], r[6]]

real_cents = optimal_centroids(2, max(n_groups*8, 64)).astype(np.float32)
n_levels = len(real_cents)

buf_r2 = mkbuf(real_rotors)
buf_c2 = mkbuf(real_cents)

print(f"d={d}, mse_bits=2, n_levels={n_levels}")

for batch in [1024, 4096, 16384, 65536]:
    x2 = rng.standard_normal((batch, d)).astype(np.float32)
    x2 = (x2 / np.linalg.norm(x2, axis=-1, keepdims=True)).astype(np.float32)

    buf_in2 = mkbuf(x2)
    buf_out2 = dev.newBufferWithLength_options_(batch * d * 4, MTLResourceStorageModeShared)
    params2 = struct.pack("IIII", batch, d, n_groups, n_levels)
    buf_p2 = dev.newBufferWithBytes_length_options_(params2, len(params2), MTLResourceStorageModeShared)

    tg = MTLSizeMake(batch, n_groups, 1)
    tg_sz = MTLSizeMake(1, min(n_groups, pso.maxTotalThreadsPerThreadgroup()), 1)

    for _ in range(20):
        c = queue.commandBuffer()
        e = c.computeCommandEncoder()
        e.setComputePipelineState_(pso)
        e.setBuffer_offset_atIndex_(buf_in2, 0, 0)
        e.setBuffer_offset_atIndex_(buf_r2, 0, 1)
        e.setBuffer_offset_atIndex_(buf_c2, 0, 2)
        e.setBuffer_offset_atIndex_(buf_out2, 0, 3)
        e.setBuffer_offset_atIndex_(buf_p2, 0, 4)
        e.dispatchThreads_threadsPerThreadgroup_(tg, tg_sz)
        e.endEncoding()
        c.commit()
        c.waitUntilCompleted()

    t0 = time.perf_counter()
    for _ in range(200):
        c = queue.commandBuffer()
        e = c.computeCommandEncoder()
        e.setComputePipelineState_(pso)
        e.setBuffer_offset_atIndex_(buf_in2, 0, 0)
        e.setBuffer_offset_atIndex_(buf_r2, 0, 1)
        e.setBuffer_offset_atIndex_(buf_c2, 0, 2)
        e.setBuffer_offset_atIndex_(buf_out2, 0, 3)
        e.setBuffer_offset_atIndex_(buf_p2, 0, 4)
        e.dispatchThreads_threadsPerThreadgroup_(tg, tg_sz)
        e.endEncoding()
        c.commit()
        c.waitUntilCompleted()
    us = (time.perf_counter() - t0) / 200 * 1e6
    fmt = f"{us:.0f} us" if us < 1000 else f"{us/1000:.2f} ms"
    print(f"  n={batch:>6d}: {fmt:>10s}  ({us/batch:.2f} us/vec)", flush=True)

print("\nDONE")
