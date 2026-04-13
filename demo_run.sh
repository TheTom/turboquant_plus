#!/bin/bash
# TurboQuant+ End-to-End Demo
# Hardware: Apple Silicon Mac with Metal
# Reproduces: Python prototype + llama.cpp inference + perplexity validation

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$HOME/llama-cpp-turboquant"
MODEL="$HOME/models/qwen2.5-1.5b-instruct-q8_0.gguf"
WIKI="$HOME/models/wikitext-2-raw.txt"
BENCH="$LLAMA_DIR/build/bin/llama-bench"
PPL="$LLAMA_DIR/build/bin/llama-perplexity"

echo "============================================================"
echo "TurboQuant+ Demo — $(date)"
echo "============================================================"
echo ""

# --- Step 1: Python prototype ---
echo ">>> Step 1: Python tests (551 tests)"
source "$REPO_DIR/.venv/bin/activate"
python3 -m pytest tests/ -q 2>&1 | tail -3
echo ""

echo ">>> Step 2: Compression demo"
python3 benchmarks/demo.py 2>&1 | grep -E "bit TurboQuant|MSE:|Cosine|Compression|demos complete"
echo ""

# --- Step 3: llama.cpp benchmarks ---
if [ ! -f "$BENCH" ]; then
    echo "ERROR: llama-bench not found at $BENCH"
    echo "Build llama-cpp-turboquant first (see README)."
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Download: hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf --local-dir ~/models"
    exit 1
fi

echo ">>> Step 3: llama-bench speed comparison (pp512 + tg128, 3 runs each)"
echo ""
echo "--- q8_0 baseline ---"
$BENCH -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 3 2>&1 | grep -E "^\|.*t/s"
echo ""
echo "--- turbo4 symmetric ---"
$BENCH -m "$MODEL" -ngl 99 -fa 1 -ctk turbo4 -ctv turbo4 -p 512 -n 128 -r 3 2>&1 | grep -E "^\|.*t/s"
echo ""
echo "--- q8_0-K + turbo4-V (asymmetric, recommended) ---"
$BENCH -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv turbo4 -p 512 -n 128 -r 3 2>&1 | grep -E "^\|.*t/s"
echo ""

# --- Step 4: Perplexity ---
if [ ! -f "$WIKI" ]; then
    echo "SKIP: wikitext-2-raw.txt not found, skipping perplexity"
else
    echo ">>> Step 4: Perplexity comparison (wikitext-2, 512 ctx, 10 chunks)"
    echo ""
    PPL_Q8=$($PPL -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c 512 --chunks 10 -f "$WIKI" 2>&1 | grep "Final estimate")
    PPL_T4=$($PPL -m "$MODEL" -ngl 99 -fa 1 -ctk turbo4 -ctv turbo4 -c 512 --chunks 10 -f "$WIKI" 2>&1 | grep "Final estimate")
    PPL_ASYM=$($PPL -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv turbo4 -c 512 --chunks 10 -f "$WIKI" 2>&1 | grep "Final estimate")

    echo "  q8_0 baseline:       $PPL_Q8"
    echo "  turbo4 symmetric:    $PPL_T4"
    echo "  q8_0-K + turbo4-V:   $PPL_ASYM"
fi

echo ""
echo "============================================================"
echo "Demo complete."
echo "============================================================"
