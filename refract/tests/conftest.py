"""Pytest configuration for REFRACT tests.

Registers the `integration` marker locally so the validation test does not
emit a PytestUnknownMarkWarning, without modifying the project-wide
pyproject.toml.
"""

from __future__ import annotations


def pytest_configure(config):  # noqa: D401
    config.addinivalue_line(
        "markers",
        "integration: requires llama.cpp + a real GGUF; takes minutes",
    )
