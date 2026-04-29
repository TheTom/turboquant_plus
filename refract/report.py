"""REFRACT v0.1 report card formatter.

Produces:
  - text() : ANSI-coloured human-readable report card with bar charts.
  - json() : machine-readable dict suitable for ML pipelines.

No external dependencies: plain ANSI escapes for colour. Set NO_COLOR=1 in
the env to suppress them.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import os
from dataclasses import asdict
from typing import Optional

from .axes.gtm import GTMResult
from .axes.kld import KLDResult
from .score import CompositeScore, MIN_FLOOR, band


def _use_color() -> bool:
    return not os.environ.get("NO_COLOR")


def _c(code: str, s: str) -> str:
    if not _use_color():
        return s
    return f"\033[{code}m{s}\033[0m"


def _band_color(b: str) -> str:
    return {
        "EXCELLENT": "32",  # green
        "PASS":      "32",
        "DEGRADED":  "33",  # yellow
        "FAIL":      "31",  # red
    }.get(b, "0")


def _bar(score: float, width: int = 40) -> str:
    """ANSI bar of length ``width`` representing 0–100."""
    fill = int(round(width * max(0.0, min(score, 100.0)) / 100.0))
    bar = "#" * fill + "-" * (width - fill)
    color = _band_color(band(score))
    return _c(color, f"[{bar}]")


def text_report(
    *,
    model: str,
    reference_label: str,
    candidate_label: str,
    composite: CompositeScore,
    gtm: GTMResult,
    kld: KLDResult,
    extras: Optional[dict] = None,
) -> str:
    """Render the report card as a human-readable string."""
    lines: list[str] = []
    bar_width = 40

    # Header
    lines.append("=" * 72)
    lines.append(_c("1", " REFRACT v0.1 — Reference-anchored Robust Acid-test"))
    lines.append("=" * 72)
    lines.append(f" model     : {model}")
    lines.append(f" reference : {reference_label}")
    lines.append(f" candidate : {candidate_label}")
    lines.append(f" timestamp : {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("-" * 72)

    # Floor
    if composite.floor_score is not None:
        floor_ok = composite.floor_ok
        tag = _c("32", "OK") if floor_ok else _c("31", "FAIL")
        lines.append(
            f" Noise floor (ref vs ref): "
            f"{composite.floor_score:6.2f}  (min {MIN_FLOOR})  [{tag}]"
        )
    else:
        lines.append(
            _c("33", " Noise floor: NOT MEASURED — pass --measure-floor to verify."),
        )
    lines.append("-" * 72)

    # Composite
    band_str = _c(_band_color(composite.band), composite.band)
    lines.append(
        f" REFRACT     : {composite.composite:6.2f}  "
        f"{_bar(composite.composite, bar_width)}  {band_str}"
    )
    lines.append("")
    lines.append(f" Axis A GTM  : {composite.gtm_score:6.2f}  "
                 f"{_bar(composite.gtm_score, bar_width)}")
    lines.append(f" Axis B KLD  : {composite.kld_score:6.2f}  "
                 f"{_bar(composite.kld_score, bar_width)}")
    lines.append("-" * 72)

    # GTM diagnostics
    lines.append(" GTM diagnostics")
    lines.append(f"   prompts                    : {gtm.n_prompts}")
    lines.append(f"   tokens decoded each        : {gtm.n_tokens_each}")
    lines.append(f"   full match rate            : {gtm.full_match_rate*100:5.1f} %")
    if gtm.median_first_divergence is not None:
        lines.append(
            f"   median first divergence    : token {gtm.median_first_divergence}"
        )
    else:
        lines.append("   median first divergence    : (all matched)")
    lines.append(
        f"   mean prefix agreement      : {gtm.mean_prefix_agreement_length:5.1f} tokens"
    )

    # KLD diagnostics
    lines.append("")
    lines.append(" KLD diagnostics")
    lines.append(f"   chunks x ctx               : {kld.chunks} x {kld.ctx}")
    lines.append(f"   mean KLD (nats)            : {kld.mean_kld:.6f}")
    if kld.ppl is not None:
        lines.append(f"   candidate PPL              : {kld.ppl:.4f}")
    if kld.rms_dp_pct is not None:
        lines.append(f"   RMS Δp (vs reference)      : {kld.rms_dp_pct:.2f} %")
    if kld.same_topp_pct is not None:
        lines.append(f"   same top-p (vs reference)  : {kld.same_topp_pct:.2f} %")

    if composite.notes:
        lines.append("-" * 72)
        for n in composite.notes:
            lines.append(_c("33", f" NOTE: {n}"))

    if extras:
        lines.append("-" * 72)
        for k, v in extras.items():
            lines.append(f" {k}: {v}")

    lines.append("=" * 72)
    return "\n".join(lines)


def json_report(
    *,
    model: str,
    reference_label: str,
    candidate_label: str,
    composite: CompositeScore,
    gtm: GTMResult,
    kld: KLDResult,
    include_per_prompt: bool = True,
    extras: Optional[dict] = None,
) -> dict:
    """Return a JSON-serialisable dict twin of the text report."""
    gtm_dict = asdict(gtm)
    if not include_per_prompt:
        gtm_dict.pop("per_prompt", None)
    return {
        "schema": "refract.report.v0.1",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "reference": reference_label,
        "candidate": candidate_label,
        "composite": asdict(composite),
        "axes": {
            "gtm": gtm_dict,
            "kld": asdict(kld),
        },
        "extras": extras or {},
    }


def to_json_string(report: dict) -> str:
    return _json.dumps(report, indent=2, default=str)
