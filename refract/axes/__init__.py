"""REFRACT axes: per-axis scoring modules.

v0.1 ships two of the four planned axes:

  - gtm: Greedy Trajectory Match
  - kld: KL Divergence vs fp16-KV reference (corpus proxy in v0.1)

v0.2+ will add R-NIAH (long-context retrieval) and PLAD (perturbation-locality
aware drift). See README.md for the v0.2 roadmap.
"""

from __future__ import annotations
