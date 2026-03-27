#!/usr/bin/env python3
"""
Direct measurement of sparse V skip rate at various context lengths.

Uses a real transformer model (Qwen3-1.7B) with output_attentions=True
to capture actual attention weight distributions, then counts how many
positions fall below the sparse V threshold (1e-6).

This gives exact skip rates — not estimates from speed data.
"""

import torch
import json
import sys
import os
from pathlib import Path

def measure_skip_rates(
    model_name: str = "Qwen/Qwen3-1.7B",
    context_lengths: list[int] = [512, 2048, 4096, 8192],
    threshold: float = 1e-6,
    device: str = "mps",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # need full attention weights, not flash
    )
    model.eval()

    # Generate a long input (repeated text to fill context)
    base_text = "The quick brown fox jumps over the lazy dog. " * 500
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)

    results = []

    for ctx_len in context_lengths:
        print(f"\n=== Context length: {ctx_len} ===")

        # Truncate input to context length
        input_ids = torch.tensor([base_ids[:ctx_len]], device=device)
        actual_len = input_ids.shape[1]
        if actual_len < ctx_len:
            print(f"  Warning: only {actual_len} tokens available (requested {ctx_len})")

        print(f"  Running forward pass ({actual_len} tokens)...")
        with torch.no_grad():
            outputs = model(
                input_ids,
                output_attentions=True,
                use_cache=False,
            )

        # outputs.attentions is a tuple of (batch, heads, seq_len, seq_len) tensors
        # We care about the LAST token's attention (decode scenario)
        layer_stats = []
        total_positions = 0
        total_skipped = 0

        for layer_idx, attn_weights in enumerate(outputs.attentions):
            # attn_weights: (1, n_heads, seq_len, seq_len)
            # Last token's attention over all previous positions
            last_token_attn = attn_weights[0, :, -1, :]  # (n_heads, seq_len)

            n_heads, n_pos = last_token_attn.shape
            below_threshold = (last_token_attn < threshold).float()

            skip_rate = below_threshold.mean().item()
            skip_per_head = below_threshold.mean(dim=1)  # per-head average

            layer_total = n_heads * n_pos
            layer_skipped = int(below_threshold.sum().item())

            total_positions += layer_total
            total_skipped += layer_skipped

            layer_stats.append({
                "layer": layer_idx,
                "skip_rate": skip_rate,
                "min_head_skip": skip_per_head.min().item(),
                "max_head_skip": skip_per_head.max().item(),
                "median_head_skip": skip_per_head.median().item(),
            })

        overall_skip = total_skipped / total_positions if total_positions > 0 else 0

        result = {
            "context_length": actual_len,
            "threshold": threshold,
            "overall_skip_rate": overall_skip,
            "total_positions": total_positions,
            "total_skipped": total_skipped,
            "n_layers": len(outputs.attentions),
            "per_layer": layer_stats,
        }
        results.append(result)

        print(f"  Overall skip rate: {overall_skip:.4f} ({overall_skip*100:.1f}%)")
        print(f"  Positions: {total_skipped:,} / {total_positions:,} below τ={threshold}")

        # Per-layer summary
        skip_rates = [s["skip_rate"] for s in layer_stats]
        print(f"  Layer skip rates: min={min(skip_rates):.3f} max={max(skip_rates):.3f} median={sorted(skip_rates)[len(skip_rates)//2]:.3f}")

        # Free memory
        del outputs
        torch.mps.empty_cache() if device == "mps" else None

    return results


def print_summary(results):
    print("\n" + "=" * 70)
    print("SPARSE V SKIP RATE — DIRECT MEASUREMENT")
    print("=" * 70)

    print(f"\n| Context | Skip Rate | Skipped/Total | Min Layer | Max Layer | Median Layer |")
    print(f"|---------|-----------|---------------|-----------|-----------|--------------|")
    for r in results:
        n_layers = r["n_layers"]
        layer_skips = [s["skip_rate"] for s in r["per_layer"]]
        print(f"| {r['context_length']:>7} | {r['overall_skip_rate']*100:>7.1f}% | {r['total_skipped']:>10,}/{r['total_positions']:>10,} | {min(layer_skips)*100:>7.1f}% | {max(layer_skips)*100:>7.1f}% | {sorted(layer_skips)[n_layers//2]*100:>8.1f}% |")

    print("\nPer-layer detail (first and last 3 layers):")
    for r in results:
        ctx = r["context_length"]
        layers = r["per_layer"]
        print(f"\n  Context {ctx}:")
        for s in layers[:3]:
            print(f"    Layer {s['layer']:>2}: {s['skip_rate']*100:.1f}% (head range: {s['min_head_skip']*100:.1f}%-{s['max_head_skip']*100:.1f}%)")
        if len(layers) > 6:
            print(f"    ...")
        for s in layers[-3:]:
            print(f"    Layer {s['layer']:>2}: {s['skip_rate']*100:.1f}% (head range: {s['min_head_skip']*100:.1f}%-{s['max_head_skip']*100:.1f}%)")


if __name__ == "__main__":
    # Default context lengths to test
    contexts = [512, 2048, 4096, 8192]

    # Can override on command line: python3 measure_skip_rate.py 512 2048 4096
    if len(sys.argv) > 1:
        contexts = [int(x) for x in sys.argv[1:]]

    results = measure_skip_rates(context_lengths=contexts)
    print_summary(results)

    # Save raw results
    output_dir = Path(__file__).parent.parent / "docs" / "threshold-ablation-logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "skip_rate_direct_measurement.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {output_file}")
