"""Phase A: Validate TurboQuant with real KV cache tensors from Qwen3-1.7B.

Loads a small Qwen model, runs inference, captures real K/V tensors,
compresses them with TurboQuant, and measures quality degradation.

Usage:
    python3 benchmarks/validate_real_model.py

Requires: pip install transformers torch accelerate
"""

import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent dir for turboquant imports
sys.path.insert(0, ".")
from turboquant import TurboQuant, TurboQuantMSE, KVCacheCompressor
from turboquant.outlier import OutlierTurboQuant, _compute_channel_split
from turboquant.calibration import calibrate_outlier_channels


MODEL_NAME = "Qwen/Qwen3-1.7B"  # head_dim=128, same as 27B


def load_model():
    """Load model and tokenizer."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # fp32 for accuracy baseline
        device_map="cpu",  # CPU is fine for validation
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")
    return model, tokenizer


def extract_kv_cache(model, tokenizer, prompt: str) -> dict:
    """Run inference and extract KV cache tensors from all layers.

    Returns:
        Dict with 'k_cache' and 'v_cache', each shape (num_layers, num_kv_heads, seq_len, head_dim)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values

    # Handle DynamicCache (iterable of tuples) — K is [0], V is [1]
    k_tensors = []
    v_tensors = []

    for layer_kv in past_kv:
        layer_tuple = tuple(layer_kv)
        k_tensors.append(layer_tuple[0].squeeze(0).numpy())
        v_tensors.append(layer_tuple[1].squeeze(0).numpy())

    k_cache = np.stack(k_tensors)  # (num_layers, num_kv_heads, seq_len, head_dim)
    v_cache = np.stack(v_tensors)

    return {"k_cache": k_cache, "v_cache": v_cache}


def analyze_kv_distribution(kv: dict):
    """Analyze the distribution of real KV tensors vs our Gaussian assumption."""
    print("\n" + "=" * 70)
    print("KV CACHE TENSOR DISTRIBUTION ANALYSIS")
    print("=" * 70)

    for name, cache in [("K cache", kv["k_cache"]), ("V cache", kv["v_cache"])]:
        flat = cache.ravel()
        per_head = cache.reshape(-1, cache.shape[-1])  # (n_vectors, head_dim)
        norms = np.linalg.norm(per_head, axis=1)

        print(f"\n  {name}: shape {cache.shape}")
        print(f"    Value range:     [{flat.min():.4f}, {flat.max():.4f}]")
        print(f"    Mean:            {flat.mean():.6f}")
        print(f"    Std:             {flat.std():.6f}")
        print(f"    Vector norms:    mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"min={norms.min():.4f}, max={norms.max():.4f}")
        print(f"    Kurtosis:        {_kurtosis(flat):.2f} (Gaussian=3.0)")

    return kv


def compress_and_compare(kv: dict):
    """Compress real KV tensors and measure quality at various bit-widths."""
    print("\n" + "=" * 70)
    print("COMPRESSION QUALITY ON REAL KV TENSORS")
    print("=" * 70)

    k_cache = kv["k_cache"]
    v_cache = kv["v_cache"]
    num_layers, num_heads, seq_len, head_dim = k_cache.shape

    print(f"\n  Model KV shape: {k_cache.shape}")
    print(f"  Total vectors: {num_layers * num_heads * seq_len}")
    print(f"  Original size: {k_cache.nbytes + v_cache.nbytes:,} bytes "
          f"({(k_cache.nbytes + v_cache.nbytes) / 1024 / 1024:.1f} MB)")

    print(f"\n  {'Config':<22} {'K MSE':>12} {'V MSE':>12} {'K Cosine':>10} {'Ratio':>8}")
    print(f"  {'─' * 66}")

    configs = [
        ("Uniform 2-bit", 2, 2, "uniform"),
        ("Outlier 2.5-bit", 2.5, 2.5, "outlier"),
        ("Uniform 3-bit", 3, 3, "uniform"),
        ("Outlier 3.5-bit", 3.5, 3.5, "outlier"),
        ("Uniform 4-bit", 4, 4, "uniform"),
    ]

    results = {}
    for name, k_bits, v_bits, mode in configs:
        if mode == "uniform":
            compressor = KVCacheCompressor(head_dim=head_dim, k_bits=int(k_bits), v_bits=int(v_bits))
            compressed = compressor.compress(k_cache, v_cache)
            k_hat, v_hat = compressor.decompress(compressed)
            stats = compressor.memory_stats(seq_len, num_layers, num_heads)
            ratio = stats["compression_ratio"]
        else:
            # Outlier: compress each head individually
            k_hat, v_hat, ratio = _compress_outlier(k_cache, v_cache, k_bits, v_bits, head_dim)

        k_mse = np.mean((k_cache - k_hat) ** 2)
        v_mse = np.mean((v_cache - v_hat) ** 2)

        # Per-vector cosine similarity
        k_flat = k_cache.reshape(-1, head_dim)
        k_hat_flat = k_hat.reshape(-1, head_dim)
        cosines = _batch_cosine_sim(k_flat, k_hat_flat)

        print(f"  {name:<22} {k_mse:>12.8f} {v_mse:>12.8f} {np.mean(cosines):>10.6f} {ratio:>7.1f}×")
        results[name] = {"k_mse": k_mse, "v_mse": v_mse, "cosine": np.mean(cosines), "ratio": ratio}

    return results


def _compress_outlier(k_cache, v_cache, k_bits, v_bits, head_dim):
    """Compress with outlier strategy, per-head."""
    num_layers, num_heads, seq_len, _ = k_cache.shape
    k_hat = np.zeros_like(k_cache)
    v_hat = np.zeros_like(v_cache)

    k_n_outlier, _, _, _ = _compute_channel_split(head_dim, k_bits)
    v_n_outlier, _, _, _ = _compute_channel_split(head_dim, v_bits)

    for layer in range(num_layers):
        for head in range(num_heads):
            # K cache with outlier TurboQuant
            k_vecs = k_cache[layer, head]
            k_outlier_idx = calibrate_outlier_channels(k_vecs, k_n_outlier)
            k_oq = OutlierTurboQuant(head_dim, target_bits=k_bits, seed=42 + layer * 100 + head, outlier_idx=k_outlier_idx)
            for i in range(seq_len):
                c = k_oq.quantize(k_vecs[i])
                k_hat[layer, head, i] = k_oq.dequantize(c)

            # V cache with outlier PolarQuant (MSE-only, lower overhead)
            v_vecs = v_cache[layer, head]
            v_outlier_idx = calibrate_outlier_channels(v_vecs, v_n_outlier)
            v_oq = OutlierTurboQuant(head_dim, target_bits=v_bits, seed=42 + layer * 100 + head + 50, outlier_idx=v_outlier_idx)
            for i in range(seq_len):
                c = v_oq.quantize(v_vecs[i])
                v_hat[layer, head, i] = v_oq.dequantize(c)

    # Approximate ratio
    avg_bits = (k_bits + v_bits) / 2
    ratio = 32 / (avg_bits + 64 / head_dim)  # +64 bits for 2 norms per vector
    return k_hat, v_hat, ratio


def attention_quality_test(model, tokenizer, kv: dict):
    """Test attention computation quality with compressed KV cache."""
    print("\n" + "=" * 70)
    print("ATTENTION QUALITY TEST")
    print("=" * 70)

    k_cache = kv["k_cache"]
    v_cache = kv["v_cache"]
    num_layers, num_heads, seq_len, head_dim = k_cache.shape

    # Use last token's query against full KV cache for each head
    # This simulates what happens during autoregressive generation
    rng = np.random.default_rng(42)

    print(f"\n  Testing attention output quality per layer (using real K/V from layer)...")
    print(f"  {'Config':<20} {'Avg Attn Cosine':>16} {'Max Attn Error':>16}")
    print(f"  {'─' * 54}")

    for bits_label, k_bits, v_bits, mode in [
        ("3-bit uniform", 3, 3, "uniform"),
        ("3.5-bit outlier", 3.5, 3.5, "outlier"),
        ("4-bit uniform", 4, 4, "uniform"),
    ]:
        attn_cosines = []

        for layer in range(min(num_layers, 4)):  # test first 4 layers
            for head in range(num_heads):
                q = rng.standard_normal((1, head_dim)).astype(np.float32)
                k = k_cache[layer, head]
                v = v_cache[layer, head]

                # Original attention
                scores = q @ k.T / np.sqrt(head_dim)
                attn = _softmax(scores)
                out_orig = attn @ v

                # Compressed attention
                if mode == "uniform":
                    compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)
                    k_4d = k[np.newaxis, np.newaxis, :, :]
                    v_4d = v[np.newaxis, np.newaxis, :, :]
                    compressed = compressor.compress(k_4d, v_4d)
                    k_hat, v_hat = compressor.decompress(compressed)
                    k_c, v_c = k_hat[0, 0], v_hat[0, 0]
                else:
                    k_n_outlier, _, _, _ = _compute_channel_split(head_dim, k_bits)
                    v_n_outlier, _, _, _ = _compute_channel_split(head_dim, v_bits)
                    k_outlier_idx = calibrate_outlier_channels(k, k_n_outlier)
                    v_outlier_idx = calibrate_outlier_channels(v, v_n_outlier)
                    
                    k_oq = OutlierTurboQuant(head_dim, target_bits=k_bits, seed=42, outlier_idx=k_outlier_idx)
                    v_oq = OutlierTurboQuant(head_dim, target_bits=v_bits, seed=43, outlier_idx=v_outlier_idx)
                    k_c = np.array([k_oq.dequantize(k_oq.quantize(k[i])) for i in range(seq_len)])
                    v_c = np.array([v_oq.dequantize(v_oq.quantize(v[i])) for i in range(seq_len)])

                scores_c = q @ k_c.T / np.sqrt(head_dim)
                attn_c = _softmax(scores_c)
                out_comp = attn_c @ v_c

                cos = np.dot(out_orig.ravel(), out_comp.ravel()) / (
                    max(np.linalg.norm(out_orig) * np.linalg.norm(out_comp), 1e-10))
                attn_cosines.append(cos)

        print(f"  {bits_label:<20} {np.mean(attn_cosines):>16.6f} {1 - min(attn_cosines):>16.6f}")


def niah_test(model, tokenizer):
    """Simple Needle-in-a-Haystack test."""
    print("\n" + "=" * 70)
    print("NEEDLE-IN-A-HAYSTACK TEST")
    print("=" * 70)

    needle = "The secret code is TURBOQUANT42."
    haystack = "This is some filler text about various topics. " * 50
    prompt = f"{haystack}\n\n{needle}\n\n{haystack}\n\nWhat is the secret code?"

    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    print(f"\n  Prompt length: {seq_len} tokens")
    print(f"  Needle: '{needle}'")

    with torch.no_grad():
        # Generate with full precision KV cache
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    found = "TURBOQUANT42" in response
    print(f"  Response: {response[:100]}...")
    print(f"  Needle found: {'✅ YES' if found else '❌ NO'}")

    return found


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return np.mean(((x - m) / s) ** 4)


def _batch_cosine_sim(A, B):
    """Cosine similarity between corresponding rows."""
    dots = np.sum(A * B, axis=1)
    norms_a = np.linalg.norm(A, axis=1)
    norms_b = np.linalg.norm(B, axis=1)
    valid = (norms_a > 1e-10) & (norms_b > 1e-10)
    cos = np.zeros(len(A))
    cos[valid] = dots[valid] / (norms_a[valid] * norms_b[valid])
    return cos


def main():
    print("=" * 70)
    print("TURBOQUANT PHASE A: REAL MODEL VALIDATION")
    print(f"Model: {MODEL_NAME} (head_dim=128, same as Qwen 27B)")
    print("=" * 70)

    model, tokenizer = load_model()

    # Step 1: Extract real KV tensors
    prompt = ("Explain the concept of vector quantization in the context of "
              "large language model inference optimization, including KV cache "
              "compression techniques and their impact on memory usage and "
              "generation speed for long-context applications.")
    print(f"\n  Extracting KV cache for prompt ({len(prompt)} chars)...")
    t0 = time.perf_counter()
    kv = extract_kv_cache(model, tokenizer, prompt)
    t_extract = time.perf_counter() - t0
    print(f"  Extracted in {t_extract:.1f}s")
    print(f"  K shape: {kv['k_cache'].shape}, V shape: {kv['v_cache'].shape}")

    # Step 2: Analyze real KV distributions
    analyze_kv_distribution(kv)

    # Step 3: Compress and compare
    results = compress_and_compare(kv)

    # Step 4: Attention quality
    attention_quality_test(model, tokenizer, kv)

    # Step 5: NIAH
    niah_test(model, tokenizer)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Model: {MODEL_NAME}")
    print(f"  KV shape: {kv['k_cache'].shape}")
    for name, r in results.items():
        print(f"  {name}: ratio={r['ratio']:.1f}×, K cosine={r['cosine']:.4f}, K MSE={r['k_mse']:.8f}")

    print(f"\n  ✅ Phase A validation complete.")
    print(f"  Next: Phase B — port to llama.cpp for real inference testing.")


if __name__ == "__main__":
    main()
