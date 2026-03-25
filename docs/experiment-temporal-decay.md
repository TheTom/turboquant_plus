# Experiment: Temporal Decay KV Cache Compression

Branch: `experiment/temporal-decay`

## Hypothesis
Recent tokens need high precision; older tokens can tolerate lower precision. Progressive requantization: turbo3 (3.5-bit) for recent → turbo2 (2.5-bit) for old. Reduces memory for long contexts.

## Research Findings

### Trigger Point
Best: `llama_kv_cache::update()` — runs between batches, exclusive cache access. Every ~256 tokens, scan for aged-out cells.

### Token Age Tracking
Already available: `llama_kv_cells` tracks `pos` per cell. Age = `seq_pos_max(seq_id) - cell.pos`.

### Implementation Plan

**Phase 1: 2-Tier MVP**
1. Add `uint8_t precision_tier` to cell tracking
2. Implement turbo2 (2.5-bit PolarQuant, 2-bit indices + no hi1 bit): simpler block, ~60% of turbo3 size
3. Re-quantization in `update()`: read turbo3 block → dequant → re-quantize as turbo2
4. Dequant path: check block type at runtime (turbo3 vs turbo2)
5. Threshold: tokens older than 70% of max_pos → turbo2

**Phase 2: 3-Tier**
- turbo3 (recent 30%) → turbo2 (mid 40%) → QJL-only (old 30%)
- Per-layer decay rates (late layers decay slower per layer-adaptive results)

### Blockers
- **turbo2 doesn't exist yet** — need new block type, quantize/dequant, Metal kernel
- **In-place re-quantization** — need to handle different block sizes in same cache tensor (turbo3 = 14 bytes/32 vals, turbo2 = ~10 bytes/32 vals)
- **Mixed-type within same cache layer** — flash attention needs to handle heterogeneous blocks

### Verdict
High impact but significant engineering. turbo2 needs to be built first. Estimate: 2-3 sessions.

## Status
DESIGN COMPLETE — needs turbo2 implementation before proceeding.
