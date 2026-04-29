"""Fast unit tests for REFRACT scoring math + parsers. No subprocess.

Run with:
    pytest refract/tests/test_unit.py
"""

from __future__ import annotations

import math

import pytest

from refract.axes.gtm import _diff, _tokenize_words
from refract.axes.kld import _kld_to_score
from refract.runner import KVConfig
from refract.score import (
    MIN_FLOOR,
    band,
    composite_score,
    harmonic_mean,
)


def test_kld_to_score_zero_is_perfect():
    assert _kld_to_score(0.0) == pytest.approx(100.0)


def test_kld_to_score_paper_headline():
    # Paper §4.3: gemma-4 26B-A4B q8/turbo4 OFF mean KLD = 1.738 nats
    # → 100 * exp(-1.738) ≈ 17.6
    s = _kld_to_score(1.738)
    assert 17.0 < s < 18.5


def test_harmonic_mean_basic():
    assert harmonic_mean([100, 100]) == pytest.approx(100.0)
    assert harmonic_mean([50, 50]) == pytest.approx(50.0)
    # Harmonic mean of (100, 0) is 0 — single bad axis dominates.
    assert harmonic_mean([100, 0]) == 0.0


def test_band_thresholds():
    assert band(95) == "EXCELLENT"
    assert band(85) == "PASS"
    assert band(70) == "DEGRADED"
    assert band(40) == "FAIL"


def test_composite_floor_pass():
    c = composite_score(95, 95, floor_score=99.9)
    assert c.band == "EXCELLENT"
    assert c.floor_ok is True
    assert not c.notes


def test_composite_floor_fail_emits_note():
    c = composite_score(95, 95, floor_score=80.0)
    assert c.floor_ok is False
    assert any("Floor failed" in n for n in c.notes)


def test_kvconfig_parse_basic():
    cfg = KVConfig.parse("ctk=q8_0,ctv=turbo4,attn_rot_v=0")
    assert cfg.ctk == "q8_0"
    assert cfg.ctv == "turbo4"
    assert cfg.attn_rot_v == 0
    assert cfg.attn_rot_k is None
    assert cfg.cli_args() == ["-ctk", "q8_0", "-ctv", "turbo4"]
    assert cfg.env() == {"LLAMA_ATTN_ROT_V_OVERRIDE": "0"}


def test_kvconfig_parse_label_roundtrip():
    spec = "ctk=q8_0,ctv=turbo4,attn_rot_v=1,attn_rot_k=0"
    cfg = KVConfig.parse(spec)
    label = cfg.label()
    # Label is order-stable; just check all fragments present.
    for frag in spec.split(","):
        assert frag in label


def test_kvconfig_extras_passed_through():
    cfg = KVConfig.parse("ctk=f16,ctv=f16,custom_flag=42")
    assert cfg.extras == {"custom_flag": "42"}
    assert "--custom_flag" in cfg.cli_args()
    assert "42" in cfg.cli_args()


def test_diff_identical_sequences():
    a = ["hello", "world"]
    first_div, prefix = _diff(a, a)
    assert first_div is None
    assert prefix == 2


def test_diff_divergence_at_token_2():
    a = ["a", "b", "c", "d"]
    b = ["a", "b", "X", "Y"]
    first_div, prefix = _diff(a, b)
    assert first_div == 2
    assert prefix == 2


def test_diff_one_is_prefix():
    a = ["a", "b", "c"]
    b = ["a", "b"]
    first_div, prefix = _diff(a, b)
    assert first_div == 2  # divergence at boundary
    assert prefix == 2


def test_tokenize_words_simple():
    assert _tokenize_words("hello world") == ["hello", "world"]
    assert _tokenize_words("  ") == []


def test_min_floor_constant():
    # Sanity: if anyone changes MIN_FLOOR, force them to read this test.
    assert MIN_FLOOR == 99.5
