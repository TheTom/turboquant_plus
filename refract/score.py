"""REFRACT v0.1 composite scoring.

Combines per-axis scores into a single 0–100 number using the harmonic mean,
which penalises a single bad axis more aggressively than the arithmetic mean
(matches the "fail-loud" intent of the design).

Bands (tunable; align these with the paper's findings as we collect more data):

    [90, 100]  EXCELLENT — within reference noise / true equivalence
    [80,  90)  PASS      — minor drift, safe to deploy
    [60,  80)  DEGRADED  — visible drift, audit before use
    [ 0,  60)  FAIL      — flag and treat as broken

Floor verification:
    REFRACT(fp16-KV, fp16-KV) must be >= MIN_FLOOR (default 99.5).
    If it is not, the reference itself is non-deterministic on this build
    and KLD deltas cannot be trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Minimum REFRACT score the reference vs reference must hit. The paper §4.5
# shows bit-exact zero KLD on Metal, so 99.5 leaves headroom for non-Metal
# float-jitter without admitting a broken reference.
MIN_FLOOR = 99.5


def harmonic_mean(values: list[float]) -> float:
    """Harmonic mean clipped to [0, 100]. Returns 0 if any value is 0."""
    clean = [max(v, 0.0) for v in values]
    if not clean:
        return 0.0
    if any(v <= 0.0 for v in clean):
        return 0.0
    n = len(clean)
    h = n / sum(1.0 / v for v in clean)
    return min(max(h, 0.0), 100.0)


def band(score: float) -> str:
    if score >= 90.0:
        return "EXCELLENT"
    if score >= 80.0:
        return "PASS"
    if score >= 60.0:
        return "DEGRADED"
    return "FAIL"


@dataclass
class CompositeScore:
    """REFRACT v0.1 composite output."""

    composite: float                 # 0–100 (harmonic_mean of axis scores)
    band: str                        # EXCELLENT / PASS / DEGRADED / FAIL
    gtm_score: float                 # 0–100
    kld_score: float                 # 0–100
    floor_score: Optional[float] = None  # measured floor (ref vs ref)
    floor_ok: Optional[bool] = None
    floor_min: float = MIN_FLOOR
    notes: list[str] = field(default_factory=list)


def composite_score(
    gtm_score: float,
    kld_score: float,
    floor_score: Optional[float] = None,
) -> CompositeScore:
    """Combine GTM and KLD into the REFRACT v0.1 composite."""
    composite = harmonic_mean([gtm_score, kld_score])
    floor_ok: Optional[bool] = None
    notes: list[str] = []
    if floor_score is not None:
        floor_ok = floor_score >= MIN_FLOOR
        if not floor_ok:
            notes.append(
                f"Floor failed: REFRACT(ref, ref) = {floor_score:.2f} < {MIN_FLOOR}. "
                "Reference is non-deterministic on this build; KLD deltas are unreliable."
            )
    return CompositeScore(
        composite=composite,
        band=band(composite),
        gtm_score=gtm_score,
        kld_score=kld_score,
        floor_score=floor_score,
        floor_ok=floor_ok,
        notes=notes,
    )
