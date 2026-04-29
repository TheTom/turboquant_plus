"""REFRACT v0.1 CLI.

Usage:
    python3 -m refract.cli score \\
        --model MODEL.gguf \\
        --reference "ctk=f16,ctv=f16" \\
        --candidate "ctk=q8_0,ctv=turbo4,attn_rot_v=0" \\
        --prompts refract/prompts/v0.1.jsonl \\
        --corpus ~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw \\
        --chunks 32 -c 512 -ngl 99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .axes.gtm import run_gtm
from .axes.kld import run_kld
from .report import json_report, text_report, to_json_string
from .runner import KVConfig
from .score import MIN_FLOOR, composite_score


def _add_score_parser(sub):
    p = sub.add_parser(
        "score",
        help="Run REFRACT v0.1 on a candidate KV config vs the reference.",
    )
    p.add_argument("--model", required=True, type=Path,
                   help="Path to the GGUF model.")
    p.add_argument("--reference", default="ctk=f16,ctv=f16",
                   help="Reference KV config (default: ctk=f16,ctv=f16).")
    p.add_argument("--candidate", required=True,
                   help="Candidate KV config to score.")
    p.add_argument("--prompts", required=True, type=Path,
                   help="Path to JSONL prompts file (e.g. prompts/v0.1.jsonl).")
    p.add_argument("--corpus", required=True, type=Path,
                   help="Path to plain-text corpus for KLD axis "
                        "(e.g. wiki.test.raw).")
    p.add_argument("--chunks", type=int, default=32,
                   help="--chunks for llama-perplexity (default: 32).")
    p.add_argument("-c", "--ctx", type=int, default=512,
                   help="Context size (default: 512).")
    p.add_argument("-ngl", "--n-gpu-layers", type=int, default=99,
                   help="-ngl flag (default: 99).")
    p.add_argument("--n-predict", type=int, default=128,
                   help="Tokens to greedy-decode per GTM prompt (default: 128).")
    p.add_argument("--seed", type=int, default=42,
                   help="Greedy seed (default: 42).")
    p.add_argument("--measure-floor", action="store_true",
                   help="Measure REFRACT(ref, ref) and abort if < %.1f."
                        % MIN_FLOOR)
    p.add_argument("--skip-gtm", action="store_true",
                   help="Skip Axis A. Composite uses KLD only (debug).")
    p.add_argument("--skip-kld", action="store_true",
                   help="Skip Axis B. Composite uses GTM only (debug).")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Path to write the JSON report to.")
    p.add_argument("--no-progress", action="store_true",
                   help="Suppress per-prompt progress output.")
    return p


def _run_score(args) -> int:
    ref_kv = KVConfig.parse(args.reference)
    cand_kv = KVConfig.parse(args.candidate)

    print(f"REFRACT v0.1")
    print(f"  model     : {args.model}")
    print(f"  reference : {ref_kv.label()}")
    print(f"  candidate : {cand_kv.label()}")
    print()

    # ---- Floor check ------------------------------------------------------
    floor_score = None
    if args.measure_floor:
        print("Measuring noise floor: REFRACT(ref, ref) ...")
        floor_gtm = run_gtm(
            model=args.model, reference_kv=ref_kv, candidate_kv=ref_kv,
            prompts_path=args.prompts, n_predict=args.n_predict,
            ctx=args.ctx, n_gpu_layers=args.n_gpu_layers, seed=args.seed,
            progress=not args.no_progress,
        )
        floor_kld = run_kld(
            model=args.model, corpus=args.corpus,
            reference_kv=ref_kv, candidate_kv=ref_kv,
            chunks=args.chunks, ctx=args.ctx,
            n_gpu_layers=args.n_gpu_layers,
            progress=not args.no_progress,
        )
        floor_composite = composite_score(floor_gtm.score, floor_kld.score)
        floor_score = floor_composite.composite
        print(f"  floor: {floor_score:.2f} (gtm={floor_gtm.score:.2f}, "
              f"kld={floor_kld.score:.2f}, kld nats={floor_kld.mean_kld:.6f})")
        if floor_score < MIN_FLOOR:
            print()
            print(f"ERROR: noise floor {floor_score:.2f} < {MIN_FLOOR}.")
            print("The reference itself is non-deterministic on this build.")
            print("KLD deltas vs this reference cannot be trusted. Aborting.")
            return 2
        print()

    # ---- GTM --------------------------------------------------------------
    if args.skip_gtm:
        gtm = _stub_gtm()
    else:
        print("Running Axis A (GTM)...")
        gtm = run_gtm(
            model=args.model, reference_kv=ref_kv, candidate_kv=cand_kv,
            prompts_path=args.prompts, n_predict=args.n_predict,
            ctx=args.ctx, n_gpu_layers=args.n_gpu_layers, seed=args.seed,
            progress=not args.no_progress,
        )
        print(f"  GTM score: {gtm.score:.2f}")

    # ---- KLD --------------------------------------------------------------
    if args.skip_kld:
        kld = _stub_kld(args.chunks, args.ctx)
    else:
        print("Running Axis B (KLD@D, corpus proxy)...")
        kld = run_kld(
            model=args.model, corpus=args.corpus,
            reference_kv=ref_kv, candidate_kv=cand_kv,
            chunks=args.chunks, ctx=args.ctx,
            n_gpu_layers=args.n_gpu_layers,
            progress=not args.no_progress,
        )
        print(f"  KLD score: {kld.score:.2f}  (mean KLD = {kld.mean_kld:.6f} nats)")

    # ---- Composite --------------------------------------------------------
    composite = composite_score(gtm.score, kld.score, floor_score=floor_score)

    print()
    print(text_report(
        model=str(args.model),
        reference_label=ref_kv.label(),
        candidate_label=cand_kv.label(),
        composite=composite,
        gtm=gtm, kld=kld,
    ))

    if args.json_out:
        rep = json_report(
            model=str(args.model),
            reference_label=ref_kv.label(),
            candidate_label=cand_kv.label(),
            composite=composite,
            gtm=gtm, kld=kld,
        )
        args.json_out.write_text(to_json_string(rep))
        print(f"\nJSON report written to {args.json_out}")

    return 0


# Stubs for --skip-gtm / --skip-kld dev modes (composite still computes).
def _stub_gtm():
    from .axes.gtm import GTMResult
    return GTMResult(
        score=100.0, full_match_rate=1.0,
        median_first_divergence=None, mean_prefix_agreement_length=0.0,
        n_prompts=0, n_tokens_each=0, per_prompt=[],
    )


def _stub_kld(chunks: int, ctx: int):
    from .axes.kld import KLDResult
    return KLDResult(
        score=100.0, mean_kld=0.0, ppl=None,
        rms_dp_pct=None, same_topp_pct=None,
        base_path="", chunks=chunks, ctx=ctx, is_self_reference=False,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="refract",
        description="REFRACT v0.1 — KV-cache quantization quality oracle.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_score_parser(sub)
    args = parser.parse_args(argv)
    if args.cmd == "score":
        return _run_score(args)
    parser.error(f"unknown subcommand: {args.cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
