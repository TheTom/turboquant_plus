"""llama.cpp subprocess wrappers used by REFRACT axes.

We deliberately avoid pulling in heavy bindings — the llama.cpp CLIs are
stable and easy to drive over subprocess.

Configuration:
    LLAMA_CPP_BIN_DIR   path to llama.cpp build bin/ dir.
                        Defaults to ~/local_llms/llama.cpp/build-test/bin
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
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


# v0.3.1: backend dispatch. CLI sets this via set_active_backend() based on
# --backend flag or REFRACT_BACKEND env. When set, the legacy run_completion
# / run_completion_trajectory functions delegate to the backend so MLX or
# vLLM users get the right inference engine without touching axis code.
_ACTIVE_BACKEND = None  # type: ignore[var-annotated]


def set_active_backend(backend) -> None:
    """Set the active backend for subsequent run_completion* calls."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def get_active_backend():
    """Return the active backend, or None if dispatch is bypassed."""
    return _ACTIVE_BACKEND


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
#
# v0.1.1 NOTE: this fork's llama-cli emits the loading spinner AND the
# multi-line ASCII art banner to STDOUT (not stderr). v0.1's noise filter
# missed all of that, so GTM was comparing two captured banners against each
# other instead of two model generations. Detected when "ref" text in the
# JSON started with "Loading model... ▄▄ ▄▄ ██ ██..." across all prompts.
_NOISE_PATTERNS = [
    re.compile(r"^\[End thinking\].*$", re.MULTILINE),
    re.compile(r"^\[ Prompt:.*\]$", re.MULTILINE),
    re.compile(r"^Exiting\.\.\..*$", re.MULTILINE),
    re.compile(r"^llama_perf_.*$", re.MULTILINE),
    re.compile(r"^Log end$", re.MULTILINE),
    re.compile(r"^Loading model\.\.\..*$", re.MULTILINE),
    re.compile(r"^>\s.*$", re.MULTILINE),   # prompt echo
]

# After noise removal, the remaining stdout typically looks like:
#   <ASCII art banner using unicode block chars>
#   <blank line>
#   | The capital of France is Paris.
#   <blank line>
# We want only the generation body. Strategy: find the last line starting
# with "| " (the generation prefix), strip the leading "| ", and use
# everything from there to end-of-string. If no "| " line found, fall back
# to stripping unicode-block-only lines.
_BLOCK_CHARS_RE = re.compile(r"^[\s\u2580-\u259F]+$", re.MULTILINE)
_GEN_LINE_RE = re.compile(r"^\|\s.*", re.MULTILINE)


def _strip_noise(text: str) -> str:
    # llama-cli's spinner uses backspace control chars (\x08) inside the
    # loading line AND inside the "| " generation prefix. Strip all backspaces
    # before any pattern matching — otherwise "|\x08 \x08[Start thinking]"
    # never matches "^\|\s..." and the generation gets dropped.
    out = text.replace("\x08", "")
    for pat in _NOISE_PATTERNS:
        out = pat.sub("", out)
    # If a "| ..." generation line is present, keep only that and what
    # follows. This handles the canonical llama-cli output shape on this fork.
    matches = list(_GEN_LINE_RE.finditer(out))
    if matches:
        first_gen = matches[0].start()
        out = out[first_gen:]
        # Strip the leading "| " marker from each generation line
        out = re.sub(r"^\|\s?", "", out, flags=re.MULTILINE)
    # Drop ASCII-art banner lines (unicode block chars only)
    out = _BLOCK_CHARS_RE.sub("", out)
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
    apply_chat_template: bool = True,
    system: Optional[str] = None,
    reasoning: str = "off",
) -> tuple[str, dict]:
    """Greedy-decode ``n_predict`` tokens from ``prompt`` using llama-cli.

    Returns (completion_text, metadata). The completion text has the prompt
    echo and llama-cli noise stripped.

    v0.3 chat-template handling
    ---------------------------
    By default (``apply_chat_template=True``), llama-cli is invoked with
    ``--jinja`` so the model's own chat template (read from GGUF metadata)
    wraps the prompt as a user message before generation. This is required
    for instruct-tuned models to engage Q&A mode; raw completion gets the
    model continuing the prompt stylistically rather than answering it
    (see CHANGELOG v0.3 plan).

    ``system`` is the system message to prepend; useful when the prompt
    has a context+question split (R-NIAH puts the haystack here).
    ``reasoning`` controls llama-cli's ``-rea`` flag — ``"off"`` disables
    thinking traces deterministically so n_predict isn't burned on
    `<think>...</think>` before the answer lands.

    Set ``apply_chat_template=False`` for axes that need raw completion
    (none today — the corpus-driven KLD axis uses a different code path).

    NOTE: ``--single-turn`` is still critical — without it llama-cli
    enters interactive mode and the subprocess hangs forever waiting on
    stdin.
    """
    # v0.3.1: dispatch to active backend if non-llamacpp is set.
    if _ACTIVE_BACKEND is not None and getattr(_ACTIVE_BACKEND, "name", None) != "llamacpp":
        res = _ACTIVE_BACKEND.run_completion(
            model=model, prompt=prompt, kv_config_str=kv.label(),
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers,
            seed=seed, temperature=temperature, timeout=timeout,
            apply_chat_template=apply_chat_template,
            system=system, reasoning=reasoning,
        )
        return res.text, res.metadata

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
        # NOTE: do NOT pass --no-conversation. This fork's llama-cli rejects
        # it with "please use llama-completion instead" and prints help. The
        # bug surfaces silently because the help banner is captured as the
        # "completion" string. --single-turn alone gives plain non-interactive
        # completion behaviour without the chat template.
        "--no-display-prompt",
        "-fa", "on",
    ]
    if apply_chat_template:
        cmd.extend(["--jinja", "-rea", reasoning])
        if system:
            cmd.extend(["-sys", system])
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
        errors="replace",  # llama-cli/perplexity sometimes emits non-utf-8
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
_TOPP_RE = re.compile(r"Same\s+top[-\s]?p:\s*([0-9.]+)\s*%")


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
        errors="replace",  # llama-perplexity stderr can contain non-utf-8 bytes
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
        errors="replace",  # llama-perplexity stderr can contain non-utf-8 bytes
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


# ---------------------------------------------------------------------------
# Corpus identity (v0.1.3) — record what the KLD axis was scored against
# ---------------------------------------------------------------------------


# Sample size for corpus SHA. We hash the first MiB rather than the whole
# file because corpora can be hundreds of MB and this code runs every score.
# A 1MiB head-hash is enough to detect "wrong corpus passed" while staying
# cheap. If the user really swaps the tail of a giant corpus the check will
# miss it — that's a known limitation, not a security feature.
CORPUS_HASH_BYTES = 1024 * 1024


def corpus_identity(corpus: Path) -> dict:
    """Return ``{path, size_bytes, sha256_head}`` for ``corpus``.

    Used to (a) record corpus identity in JSON output and (b) reject KLD
    scoring when a candidate's base file was built from a different corpus.
    """
    p = Path(corpus)
    size = p.stat().st_size
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(CORPUS_HASH_BYTES))
    return {
        "path": str(p),
        "size_bytes": size,
        "sha256_head": h.hexdigest(),
        "sha256_head_bytes": min(size, CORPUS_HASH_BYTES),
    }


def write_corpus_sidecar(base_path: Path, corpus: Path) -> Path:
    """Write a ``<base>.corpus.json`` sidecar recording corpus identity.

    Called when a KLD base file is built so a later candidate run can verify
    it's scoring against a base built from the same corpus.
    """
    sidecar = Path(str(base_path) + ".corpus.json")
    sidecar.write_text(json.dumps(corpus_identity(corpus), indent=2))
    return sidecar


def read_corpus_sidecar(base_path: Path) -> Optional[dict]:
    """Read the corpus identity sidecar for a base file. Returns None if absent."""
    sidecar = Path(str(base_path) + ".corpus.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def assert_corpus_matches(base_path: Path, corpus: Path) -> None:
    """Verify ``corpus`` matches the identity sidecar for ``base_path``.

    No-op if no sidecar exists (base was built outside REFRACT, e.g.
    user-supplied via --kl-divergence-base). Raises RuntimeError on
    mismatch — rationale: silently scoring against a base from a different
    corpus produces meaningless KLD numbers and is the kind of foot-gun
    that fail-loud is meant to catch.
    """
    expected = read_corpus_sidecar(base_path)
    if expected is None:
        return  # no sidecar, can't verify; treat as user knows best
    actual = corpus_identity(corpus)
    mismatched = []
    if expected.get("sha256_head") != actual["sha256_head"]:
        mismatched.append("sha256_head")
    if expected.get("size_bytes") != actual["size_bytes"]:
        mismatched.append("size_bytes")
    if mismatched:
        raise RuntimeError(
            f"corpus identity mismatch on {mismatched}: KLD base file "
            f"{base_path} was built from\n  {expected.get('path')!r} "
            f"(size={expected.get('size_bytes')}, "
            f"sha256_head={expected.get('sha256_head')[:16]}…)\n"
            f"but you're now scoring against\n  {actual['path']!r} "
            f"(size={actual['size_bytes']}, "
            f"sha256_head={actual['sha256_head'][:16]}…).\n"
            f"Refusing to compute KLD against a different corpus — the "
            f"resulting nats would be meaningless. Rebuild the base or "
            f"point --corpus at the original file."
        )


# ---------------------------------------------------------------------------
# llama-tokenize wrapper (used by GTM v0.1.2+)
# ---------------------------------------------------------------------------


def run_completion_trajectory(
    model: Path,
    prompt: str,
    kv: KVConfig,
    n_predict: int = 128,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    seed: int = 42,
    temperature: float = 0.0,
    timeout: float = 300.0,
    apply_chat_template: bool = True,
    system: Optional[str] = None,
    reasoning: str = "off",
) -> tuple[list[int], dict]:
    """v0.1.4: greedy-decode and capture model-token IDs at decode time.

    Drives the patched ``llama-completion`` binary with
    ``REFRACT_TRAJECTORY=<tmpfile>`` set; the binary writes one JSONL record
    per sampled token (``{"step":N,"token_id":ID}``). We read the file back,
    return the ID sequence, and delete the file.

    Returns (token_ids, metadata). The token IDs are the model's own
    sampled tokens — no detokenize→retokenize round-trip, no whitespace-vs-
    model-token unit mismatch. This is the v0.1.4 fix for GTM's structural
    weakness (LIMITATIONS.md §1, §5).

    Requires the ``llama-completion`` binary to be built from the patched
    ``tools/completion/completion.cpp`` (REFRACT v0.1.4 patch). If the
    binary lacks the patch, the trajectory file will be empty and this
    function returns ``([], meta)``.
    """
    # v0.3.1: dispatch to active backend if non-llamacpp is set.
    if _ACTIVE_BACKEND is not None and getattr(_ACTIVE_BACKEND, "name", None) != "llamacpp":
        res = _ACTIVE_BACKEND.run_completion_trajectory(
            model=model, prompt=prompt, kv_config_str=kv.label(),
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers,
            seed=seed, temperature=temperature, timeout=timeout,
            apply_chat_template=apply_chat_template, system=system,
        )
        return res.token_ids, res.metadata

    bin_path = _bin("llama-completion")

    fd, traj_path = tempfile.mkstemp(prefix="refract-traj-", suffix=".jsonl")
    os.close(fd)
    os.unlink(traj_path)  # patched binary creates it itself

    cmd: list[str] = [
        str(bin_path),
        "-m", str(model),
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(ctx),
        "-ngl", str(n_gpu_layers),
        "--seed", str(seed),
        "--temp", str(temperature),
        "-no-cnv",
        "--no-display-prompt",
        "-fa", "on",
    ]
    # llama-completion supports --jinja and -sys but does NOT support
    # `-rea on|off`; reasoning-trace control there is done via
    # `--reasoning-format` which is a JSON-output knob, not a thinking
    # toggle. For trajectory we accept whatever the model emits (thinking
    # tokens are still real model tokens to compare).
    if apply_chat_template:
        cmd.append("--jinja")
        if system:
            cmd.extend(["-sys", system])
    cmd.extend(kv.cli_args())

    env = os.environ.copy()
    env.update(kv.env())
    env["REFRACT_TRAJECTORY"] = traj_path

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
            text=True,
            errors="replace",
        )
        meta = {
            "returncode": proc.returncode,
            "cmd": " ".join(shlex.quote(c) for c in cmd),
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
            "trajectory_path": traj_path,
        }
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-completion exited {proc.returncode}\n"
                f"cmd: {meta['cmd']}\n"
                f"stderr tail:\n{meta['stderr_tail']}"
            )

        token_ids: list[int] = []
        try:
            with open(traj_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    token_ids.append(int(rec["token_id"]))
        except FileNotFoundError:
            # Patched binary not present, OR sampling produced 0 tokens
            # (model emitted EOS immediately). Both are valid empty cases.
            pass
        meta["n_tokens"] = len(token_ids)
        return token_ids, meta
    finally:
        try:
            os.unlink(traj_path)
        except OSError:
            pass


def tokenize_to_ids(
    model: Path,
    text: str,
    timeout: float = 120.0,
) -> list[int]:
    """Tokenize ``text`` using the model's vocabulary via llama-tokenize.

    Returns a list of integer token IDs. Used by GTM v0.1.2+ to compare
    completions in true model-token units rather than whitespace tokens
    (which can over-count and produce unit-mismatch artifacts where the
    "matched prefix length" exceeds the actual --n-predict value).

    Empty string returns []. Non-utf-8 bytes in stderr are tolerated
    (errors='replace').
    """
    if not text:
        return []
    # Dispatch to active backend when it's not llama.cpp — MLX/vLLM/SGLang
    # provide their own tokenizer and avoid the llama-tokenize binary, which
    # may not be on PATH or may have a stale ABI when the host's llama.cpp
    # checkout has drifted.
    if _ACTIVE_BACKEND is not None and getattr(_ACTIVE_BACKEND, "name", None) != "llamacpp":
        return _ACTIVE_BACKEND.tokenize_to_ids(model=model, text=text, timeout=timeout)
    bin_path = _bin("llama-tokenize")
    cmd: list[str] = [
        str(bin_path),
        "-m", str(model),
        "--ids",
        "--no-bos",
        "--no-parse-special",
        "--log-disable",
        "--stdin",
    ]
    proc = subprocess.run(
        cmd,
        input=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize exited {proc.returncode}\n"
            f"stderr tail:\n{proc.stderr[-500:]}"
        )
    # Output looks like: "[1, 2, 3, 4, 5]\n"
    out = proc.stdout.strip()
    if not out or not out.startswith("["):
        return []
    inner = out.strip("[]\n ")
    if not inner:
        return []
    return [int(x.strip()) for x in inner.split(",") if x.strip()]
