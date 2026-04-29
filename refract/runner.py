"""llama.cpp subprocess wrappers used by REFRACT axes.

We deliberately avoid pulling in heavy bindings — the llama.cpp CLIs are
stable and easy to drive over subprocess.

Configuration:
    LLAMA_CPP_BIN_DIR   path to llama.cpp build bin/ dir.
                        Defaults to ~/local_llms/llama.cpp/build-test/bin
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_BIN_DIR = Path(
    os.path.expanduser(
        os.environ.get(
            "LLAMA_CPP_BIN_DIR",
            "~/local_llms/llama.cpp/build-test/bin",
        )
    )
)


def _bin(name: str) -> Path:
    """Resolve a llama.cpp binary path. Raises FileNotFoundError if missing."""
    p = DEFAULT_BIN_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"llama.cpp binary not found: {p}\n"
            f"Set LLAMA_CPP_BIN_DIR to the directory containing {name}."
        )
    return p


# ---------------------------------------------------------------------------
# KV config parsing
# ---------------------------------------------------------------------------


@dataclass
class KVConfig:
    """A KV-cache configuration to pass to llama.cpp.

    Parsed from a "key=value,key=value,..." string. Recognised keys:

        ctk         cache type for K  (e.g. f16, q8_0, q4_0, turbo4)
        ctv         cache type for V  (e.g. f16, q8_0, q4_0, turbo4)
        attn_rot_k  0/1, sets LLAMA_ATTN_ROT_K_OVERRIDE
        attn_rot_v  0/1, sets LLAMA_ATTN_ROT_V_OVERRIDE
        attn_rot_disable  1 sets LLAMA_ATTN_ROT_DISABLE=1 (hard lockout)

    Any unknown key is preserved as an llama-cli arg of the form ``--<key> <val>``.
    """

    ctk: str = "f16"
    ctv: str = "f16"
    attn_rot_k: Optional[int] = None
    attn_rot_v: Optional[int] = None
    attn_rot_disable: Optional[int] = None
    extras: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, spec: str) -> "KVConfig":
        cfg = cls()
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"bad KV spec fragment '{part}' (need key=value)")
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "ctk":
                cfg.ctk = v
            elif k == "ctv":
                cfg.ctv = v
            elif k == "attn_rot_k":
                cfg.attn_rot_k = int(v)
            elif k == "attn_rot_v":
                cfg.attn_rot_v = int(v)
            elif k == "attn_rot_disable":
                cfg.attn_rot_disable = int(v)
            else:
                cfg.extras[k] = v
        return cfg

    def env(self) -> dict:
        """Return the env-var overlay this config requires."""
        env: dict = {}
        if self.attn_rot_k is not None:
            env["LLAMA_ATTN_ROT_K_OVERRIDE"] = str(self.attn_rot_k)
        if self.attn_rot_v is not None:
            env["LLAMA_ATTN_ROT_V_OVERRIDE"] = str(self.attn_rot_v)
        if self.attn_rot_disable is not None:
            env["LLAMA_ATTN_ROT_DISABLE"] = str(self.attn_rot_disable)
        return env

    def cli_args(self) -> list[str]:
        """Return llama-cli/llama-perplexity flags for this KV config."""
        args = ["-ctk", self.ctk, "-ctv", self.ctv]
        for k, v in self.extras.items():
            args.extend([f"--{k}", v])
        return args

    def label(self) -> str:
        """Short human label, e.g. ``ctk=q8_0,ctv=turbo4,attn_rot_v=0``."""
        bits = [f"ctk={self.ctk}", f"ctv={self.ctv}"]
        if self.attn_rot_k is not None:
            bits.append(f"attn_rot_k={self.attn_rot_k}")
        if self.attn_rot_v is not None:
            bits.append(f"attn_rot_v={self.attn_rot_v}")
        if self.attn_rot_disable is not None:
            bits.append(f"attn_rot_disable={self.attn_rot_disable}")
        for k, v in self.extras.items():
            bits.append(f"{k}={v}")
        return ",".join(bits)


# ---------------------------------------------------------------------------
# llama-cli wrapper (used by GTM)
# ---------------------------------------------------------------------------


# Junk lines llama-cli prints around the actual completion. Strip them.
# Order matters — patterns are applied in sequence to the captured stdout.
_NOISE_PATTERNS = [
    re.compile(r"^\[End thinking\].*$", re.MULTILINE),
    re.compile(r"^\[ Prompt:.*\]$", re.MULTILINE),
    re.compile(r"^Exiting\.\.\..*$", re.MULTILINE),
    re.compile(r"^llama_perf_.*$", re.MULTILINE),
    re.compile(r"^Log end$", re.MULTILINE),
]


def _strip_noise(text: str) -> str:
    out = text
    for pat in _NOISE_PATTERNS:
        out = pat.sub("", out)
    return out


def run_completion(
    model: Path,
    prompt: str,
    kv: KVConfig,
    n_predict: int = 128,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    seed: int = 42,
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> tuple[str, dict]:
    """Greedy-decode ``n_predict`` tokens from ``prompt`` using llama-cli.

    Returns (completion_text, metadata). The completion text has the prompt
    echo and llama-cli noise stripped.

    NOTE: ``--single-turn`` is critical — without it llama-cli enters
    interactive mode and the subprocess hangs forever waiting on stdin.
    Discovered the hard way; see paper §4.8.

    TODO(v0.2): replace with a logits-capturing custom binary so we can
    compute true trajectory-KLD per generated step.
    """
    bin_path = _bin("llama-cli")

    cmd: list[str] = [
        str(bin_path),
        "-m", str(model),
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(ctx),
        "-ngl", str(n_gpu_layers),
        "--seed", str(seed),
        "--temp", str(temperature),
        "--single-turn",      # CRITICAL — see docstring
        "--no-conversation",  # plain completion mode (not chat template)
        "--no-display-prompt",
        "-fa", "on",
    ]
    cmd.extend(kv.cli_args())

    env = os.environ.copy()
    env.update(kv.env())

    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout,
        text=True,
    )

    completion = _strip_noise(proc.stdout).strip()
    meta = {
        "returncode": proc.returncode,
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-cli exited {proc.returncode}\n"
            f"cmd: {meta['cmd']}\n"
            f"stderr tail:\n{meta['stderr_tail']}"
        )
    return completion, meta


# ---------------------------------------------------------------------------
# llama-perplexity wrapper (used by KLD axis)
# ---------------------------------------------------------------------------


_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")
_KLD_MEAN_RE = re.compile(r"Mean\s+KLD:\s*([0-9.+\-eE]+)")
_RMS_DP_RE = re.compile(r"RMS Δp:\s*([0-9.]+)\s*%", re.UNICODE)
_TOPP_RE = re.compile(r"Same top-?p:\s*([0-9.]+)\s*%")


def run_perplexity_kld_base(
    model: Path,
    corpus: Path,
    kv: KVConfig,
    base_path: Path,
    chunks: int = 32,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    timeout: float = 7200.0,
) -> dict:
    """Build a KLD-base file with llama-perplexity --kl-divergence-base.

    Used to capture fp16-KV reference logits.
    """
    bin_path = _bin("llama-perplexity")
    cmd: list[str] = [
        str(bin_path),
        "-m", str(model),
        "-f", str(corpus),
        "-c", str(ctx),
        "--chunks", str(chunks),
        "-ngl", str(n_gpu_layers),
        "-fa", "on",
        "--kl-divergence-base", str(base_path),
    ]
    cmd.extend(kv.cli_args())

    env = os.environ.copy()
    env.update(kv.env())

    proc = subprocess.run(
        cmd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, timeout=timeout, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity --kl-divergence-base exited {proc.returncode}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    return {
        "base_path": str(base_path),
        "stdout_tail": proc.stdout[-1000:],
    }


def run_perplexity_kld(
    model: Path,
    corpus: Path,
    kv: KVConfig,
    base_path: Path,
    chunks: int = 32,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    timeout: float = 7200.0,
) -> dict:
    """Score a candidate KV config against the reference base file.

    Returns dict with mean_kld, ppl, rms_dp_pct, same_topp_pct.
    """
    bin_path = _bin("llama-perplexity")
    cmd: list[str] = [
        str(bin_path),
        "-m", str(model),
        "-f", str(corpus),
        "-c", str(ctx),
        "--chunks", str(chunks),
        "-ngl", str(n_gpu_layers),
        "-fa", "on",
        "--kl-divergence",
        "--kl-divergence-base", str(base_path),
    ]
    cmd.extend(kv.cli_args())

    env = os.environ.copy()
    env.update(kv.env())

    proc = subprocess.run(
        cmd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, timeout=timeout, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity --kl-divergence exited {proc.returncode}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )

    text = proc.stdout + "\n" + proc.stderr
    out = {
        "ppl": _first_float(_PPL_RE, text),
        "mean_kld": _first_float(_KLD_MEAN_RE, text),
        "rms_dp_pct": _first_float(_RMS_DP_RE, text),
        "same_topp_pct": _first_float(_TOPP_RE, text),
        "stdout_tail": proc.stdout[-1000:],
    }
    if out["mean_kld"] is None:
        raise RuntimeError(
            "Could not parse Mean KLD from llama-perplexity output. "
            f"Last 500 chars:\n{text[-500:]}"
        )
    return out


def _first_float(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.search(text)
    return float(m.group(1)) if m else None
