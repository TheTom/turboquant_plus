# Copyright 2026 Tom Turney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware replay system for TurboQuant diagnostic profiles.

Parses diagnostic output from turbo-hardware-diag.sh, builds structured
hardware profiles, and enables comparison/replay across different hardware
configurations.

Usage:
    # Parse a diagnostic output file
    profile = HardwareProfile.from_diag_file("turbo-diag-20260326.txt")

    # Compare two profiles
    report = compare_profiles(baseline, target)

    # Predict performance for a hardware config
    predicted = predict_decode_speed(profile, context_depth=32768)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GPUInfo:
    """GPU hardware capabilities."""
    name: str = "unknown"
    family: str = "unknown"
    family_id: int = 0
    has_tensor: bool = False
    has_unified_memory: bool = False
    has_bfloat: bool = False
    recommended_max_working_set_mb: float = 0.0
    metal_version: str = "unknown"
    # CUDA fields
    cuda_compute_cap: str = ""
    cuda_vram_mb: float = 0.0


@dataclass
class SystemInfo:
    """System hardware specs (no PII)."""
    platform: str = "unknown"  # Darwin, Linux
    os_version: str = "unknown"
    arch: str = "unknown"
    cpu_brand: str = "unknown"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_gb: int = 0
    apple_silicon: bool = False
    chip_model: str = ""
    l1_dcache: int = 0
    l2_cache: int = 0
    gpu: GPUInfo = field(default_factory=GPUInfo)


@dataclass
class BenchResult:
    """Single benchmark measurement."""
    label: str
    cache_type_k: str
    cache_type_v: str
    context_depth: int = 0  # 0 = short/default
    mode: str = ""  # "prefill", "decode", "combined"
    tok_per_sec: float = 0.0
    stddev: float = 0.0
    wall_ms: int = 0
    env: str = ""  # e.g., "TURBO_LAYER_ADAPTIVE=2"


@dataclass
class LoadSnapshot:
    """System load at a point in time."""
    label: str
    timestamp: str = ""
    load_avg: str = ""
    free_ram_mb: float = 0.0
    swap_used: str = ""
    process_count: int = 0
    thermal: str = ""
    gpu_util: str = ""


@dataclass
class ModelInfo:
    """Model metadata."""
    filename: str = ""
    filesize_bytes: int = 0
    architecture: str = ""
    name: str = ""
    file_type: str = ""
    model_type: str = ""
    params: str = ""
    n_layer: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    n_expert: int = 0
    n_expert_used: int = 0
    n_ctx_train: int = 0
    n_embd: int = 0


@dataclass
class PPLResult:
    """Perplexity measurement."""
    cache_type: str
    chunks: int
    ppl: float
    stddev: float
    env: str = ""


@dataclass
class HardwareProfile:
    """Complete hardware profile from a diagnostic run."""
    diag_version: int = 0
    timestamp: str = ""
    system: SystemInfo = field(default_factory=SystemInfo)
    model: ModelInfo = field(default_factory=ModelInfo)
    benchmarks: list[BenchResult] = field(default_factory=list)
    ppl_results: list[PPLResult] = field(default_factory=list)
    load_snapshots: list[LoadSnapshot] = field(default_factory=list)
    build_commit: str = ""

    def to_json(self) -> str:
        """Serialize to JSON for storage/replay."""
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        """Save profile to JSON file."""
        Path(path).write_text(self.to_json())

    @classmethod
    def from_json(cls, path: str | Path) -> HardwareProfile:
        """Load profile from JSON file."""
        data = json.loads(Path(path).read_text())
        profile = cls()
        profile.diag_version = data.get("diag_version", 0)
        profile.timestamp = data.get("timestamp", "")
        profile.build_commit = data.get("build_commit", "")

        # System info
        si = data.get("system", {})
        profile.system = SystemInfo(
            platform=si.get("platform", "unknown"),
            os_version=si.get("os_version", "unknown"),
            arch=si.get("arch", "unknown"),
            cpu_brand=si.get("cpu_brand", "unknown"),
            cpu_cores_physical=si.get("cpu_cores_physical", 0),
            cpu_cores_logical=si.get("cpu_cores_logical", 0),
            ram_total_gb=si.get("ram_total_gb", 0),
            apple_silicon=si.get("apple_silicon", False),
            chip_model=si.get("chip_model", ""),
            l1_dcache=si.get("l1_dcache", 0),
            l2_cache=si.get("l2_cache", 0),
            gpu=GPUInfo(**si.get("gpu", {})),
        )

        # Model info
        mi = data.get("model", {})
        profile.model = ModelInfo(**mi)

        # Benchmarks
        for b in data.get("benchmarks", []):
            profile.benchmarks.append(BenchResult(**b))

        # PPL
        for p in data.get("ppl_results", []):
            profile.ppl_results.append(PPLResult(**p))

        # Load snapshots
        for ls in data.get("load_snapshots", []):
            profile.load_snapshots.append(LoadSnapshot(**ls))

        return profile

    @classmethod
    def from_diag_file(cls, path: str | Path) -> HardwareProfile:
        """Parse a turbo-hardware-diag.sh output file into a profile."""
        text = Path(path).read_text()
        return parse_diag_output(text)

    def get_decode_curve(self, cache_type: str = "turbo3", env: str = "") -> dict[int, float]:
        """Extract decode speed vs context depth curve."""
        curve = {}
        for b in self.benchmarks:
            if b.mode == "decode" and b.cache_type_k == cache_type and b.env == env:
                curve[b.context_depth] = b.tok_per_sec
        return dict(sorted(curve.items()))

    def get_prefill_curve(self, cache_type: str = "turbo3", env: str = "") -> dict[int, float]:
        """Extract prefill speed vs context depth curve."""
        curve = {}
        for b in self.benchmarks:
            if b.mode == "prefill" and b.cache_type_k == cache_type and b.env == env:
                curve[b.context_depth] = b.tok_per_sec
        return dict(sorted(curve.items()))

    def get_ratio_curve(self, cache_type: str = "turbo3",
                        baseline: str = "q8_0", mode: str = "decode",
                        env: str = "") -> dict[int, float]:
        """Compute turbo3/q8_0 ratio at each context depth."""
        target = {}
        base = {}
        for b in self.benchmarks:
            if b.mode != mode:
                continue
            if b.cache_type_k == cache_type and b.env == env:
                target[b.context_depth] = b.tok_per_sec
            elif b.cache_type_k == baseline and b.env == "":
                base[b.context_depth] = b.tok_per_sec

        ratios = {}
        for depth in sorted(set(target.keys()) & set(base.keys())):
            if base[depth] > 0:
                ratios[depth] = target[depth] / base[depth]
        return ratios


    def find_decode_inflection(self, cache_type: str = "turbo3") -> Optional[int]:
        """Find context depth where decode ratio drops most steeply.

        Returns the depth where the gradient (ratio change per depth doubling)
        is most negative, indicating constant cache thrashing onset.
        Returns None if insufficient data.
        """
        ratios = self.get_ratio_curve(cache_type, "q8_0", "decode")
        if len(ratios) < 3:
            return None

        depths = sorted(ratios.keys())
        worst_gradient = 0.0
        inflection_depth = None

        for i in range(1, len(depths)):
            d0, d1 = depths[i - 1], depths[i]
            r0, r1 = ratios[d0], ratios[d1]
            if d0 > 0 and d1 > d0:
                # Gradient: ratio change per context doubling
                gradient = (r1 - r0) / (d1 - d0) * d0  # normalized
                if gradient < worst_gradient:
                    worst_gradient = gradient
                    inflection_depth = d1

        return inflection_depth

    def flag_unreliable_measurements(self) -> list[str]:
        """Flag measurements at 1K context (Metal async dispatch artifact)."""
        warnings = []
        for b in self.benchmarks:
            if b.context_depth == 1024 and b.tok_per_sec > 10000:
                warnings.append(
                    f"Unreliable measurement: {b.label} at 1K context "
                    f"({b.tok_per_sec:.0f} tok/s — Metal async dispatch artifact)"
                )
        return warnings


def parse_diag_output(text: str) -> HardwareProfile:
    """Parse raw turbo-hardware-diag.sh output into a HardwareProfile."""
    profile = HardwareProfile()
    lines = text.split('\n')

    # Header
    for line in lines:
        if line.startswith("TURBO_DIAG_VERSION="):
            profile.diag_version = int(line.split("=", 1)[1])
        elif line.startswith("TURBO_DIAG_TIMESTAMP="):
            profile.timestamp = line.split("=", 1)[1]
        elif line.startswith("TURBO_DIAG_MODEL="):
            profile.model.filename = line.split("=", 1)[1]

    # Hardware tags
    for line in lines:
        if not line.startswith("[HW]"):
            continue
        kv = line[len("[HW] "):]
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key == "os":
            # "os=Darwin os_version=25.3.0 arch=arm64" — compound line
            parts = kv.split()
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k == "os":
                        profile.system.platform = v
                    elif k == "os_version":
                        profile.system.os_version = v
                    elif k == "arch":
                        profile.system.arch = v
        elif key == "cpu_brand":
            profile.system.cpu_brand = val
        elif key == "cpu_cores_physical":
            profile.system.cpu_cores_physical = _int(val)
        elif key == "cpu_cores_logical":
            profile.system.cpu_cores_logical = _int(val)
        elif key == "ram_total_gb":
            profile.system.ram_total_gb = _int(val)
        elif key == "apple_silicon":
            profile.system.apple_silicon = val.lower() == "true"
        elif key == "chip_model":
            profile.system.chip_model = val
        elif key == "l1_dcache":
            profile.system.l1_dcache = _int(val)
        elif key == "l2_cache":
            profile.system.l2_cache = _int(val)

    # GPU info
    for line in lines:
        if "[GPU]" in line or "[METAL]" in line:
            content = re.sub(r'^\[(GPU|METAL)\]\s*', '', line)
            if "GPU name:" in content:
                profile.system.gpu.name = content.split("GPU name:")[-1].strip()
            elif "GPU family:" in content:
                fam = content.split("GPU family:")[-1].strip()
                profile.system.gpu.family = fam
                # Extract family ID
                m = re.search(r'\((\d+)\)', fam)
                if m:
                    profile.system.gpu.family_id = int(m.group(1))
            elif "has tensor" in content:
                profile.system.gpu.has_tensor = "true" in content.lower()
            elif "has unified" in content:
                profile.system.gpu.has_unified_memory = "true" in content.lower()
            elif "has bfloat" in content:
                profile.system.gpu.has_bfloat = "true" in content.lower()
            elif "recommendedMax" in content:
                m = re.search(r'([\d.]+)\s*MB', content)
                if m:
                    profile.system.gpu.recommended_max_working_set_mb = float(m.group(1))
        if "[METAL_TENSOR]" in line and "has tensor" in line:
            profile.system.gpu.has_tensor = "true" in line.lower()

    # Model info
    for line in lines:
        if not line.startswith("[MODEL]"):
            continue
        content = line[len("[MODEL] "):]
        if "general.name" in content:
            profile.model.name = content.split("=")[-1].strip()
        elif "general.architecture" in content or "arch " in content:
            profile.model.architecture = content.split("=")[-1].strip()
        elif "file type" in content and "file format" not in content:
            profile.model.file_type = content.split("=")[-1].strip()
        elif "model type" in content:
            profile.model.model_type = content.split("=")[-1].strip()
        elif "model params" in content:
            profile.model.params = content.split("=")[-1].strip()
        elif "n_layer" in content:
            profile.model.n_layer = _int(content.split("=")[-1].strip())
        elif "n_head " in content and "n_head_kv" not in content:
            profile.model.n_head = _int(content.split("=")[-1].strip())
        elif "n_head_kv" in content:
            profile.model.n_head_kv = _int(content.split("=")[-1].strip())
        elif "n_expert " in content and "used" not in content:
            profile.model.n_expert = _int(content.split("=")[-1].strip())
        elif "n_expert_used" in content:
            profile.model.n_expert_used = _int(content.split("=")[-1].strip())
        elif "n_ctx_train" in content:
            profile.model.n_ctx_train = _int(content.split("=")[-1].strip())
        elif "n_embd" in content:
            profile.model.n_embd = _int(content.split("=")[-1].strip())
        elif "filename=" in content:
            profile.model.filename = content.split("=", 1)[1]
        elif "filesize_bytes=" in content:
            profile.model.filesize_bytes = _int(content.split("=", 1)[1])

    # Benchmarks — parse llama-bench table output
    _parse_bench_results(lines, profile)

    # PPL results
    _parse_ppl_results(lines, profile)

    # Load snapshots
    _parse_load_snapshots(lines, profile)

    # Build info
    for line in lines:
        if line.startswith("[BUILD]"):
            profile.build_commit = line[len("[BUILD] "):].strip()

    return profile


def _parse_bench_results(lines: list[str], profile: HardwareProfile) -> None:
    """Parse [BENCH_START] tags and llama-bench table rows."""
    current_label = ""
    current_ctk = ""
    current_env = ""

    for line in lines:
        # Track current benchmark context
        if "[BENCH_START]" in line:
            m = re.search(r'label="([^"]*)"', line)
            if m:
                current_label = m.group(1)
            m = re.search(r'ctk=(\S+)', line)
            if m:
                current_ctk = m.group(1)
            m = re.search(r'env="([^"]*)"', line)
            if m:
                current_env = m.group(1)

        # Parse llama-bench table rows
        if line.startswith("|") and ("pp" in line or "tg" in line):
            _parse_bench_table_row(line, current_label, current_ctk, current_env, profile)


def _parse_bench_table_row(line: str, label: str, ctk: str, env: str,
                           profile: HardwareProfile) -> None:
    """Parse a single llama-bench markdown table row."""
    # Format: | model | size | params | backend | threads | ctk | ctv | batch | test | t/s |
    cols = [c.strip() for c in line.split("|")]
    if len(cols) < 10:
        return

    # Find the test column (contains ppXXXX or tgXXXX)
    test_col = ""
    tps_col = ""
    for i, col in enumerate(cols):
        if re.match(r'(pp|tg)\d+', col):
            test_col = col
            # t/s is typically the next column
            if i + 1 < len(cols):
                tps_col = cols[i + 1]
            break

    if not test_col:
        return

    # Parse test type and context
    mode = ""
    context = 0
    if test_col.startswith("pp") and "+tg" in test_col:
        mode = "combined"
        m = re.match(r'pp(\d+)\+tg(\d+)', test_col)
        if m:
            context = int(m.group(1))
    elif test_col.startswith("pp"):
        mode = "prefill"
        m = re.match(r'pp(\d+)', test_col)
        if m:
            context = int(m.group(1))
    elif test_col.startswith("tg"):
        mode = "decode"
        m = re.search(r'd(\d+)', test_col)
        if m:
            context = int(m.group(1))

    # Parse tok/s
    tps = 0.0
    stddev = 0.0
    m = re.match(r'([\d.]+)\s*±\s*([\d.]+)', tps_col)
    if m:
        tps = float(m.group(1))
        stddev = float(m.group(2))
    elif re.match(r'[\d.]+', tps_col):
        tps = float(re.match(r'[\d.]+', tps_col).group())

    # Extract cache type from the row itself
    row_ctk = ctk
    for col in cols:
        col_stripped = col.strip()
        if col_stripped in ("q8_0", "turbo3", "turbo4", "f16", "q4_0"):
            row_ctk = col_stripped
            break

    result = BenchResult(
        label=label,
        cache_type_k=row_ctk,
        cache_type_v=row_ctk,
        context_depth=context,
        mode=mode,
        tok_per_sec=tps,
        stddev=stddev,
        env=env,
    )
    profile.benchmarks.append(result)


def _parse_ppl_results(lines: list[str], profile: HardwareProfile) -> None:
    """Parse PPL results from output."""
    current_ctk = ""
    current_chunks = 0
    current_env = ""

    for line in lines:
        if "[PPL_START]" in line:
            m = re.search(r'ctk=(\S+)', line)
            if m:
                current_ctk = m.group(1)
            m = re.search(r'chunks=(\d+)', line)
            if m:
                current_chunks = int(m.group(1))
            m = re.search(r'env="([^"]*)"', line)
            if m:
                current_env = m.group(1) if m.group(1) else ""

        if "Final estimate: PPL =" in line:
            m = re.search(r'PPL = ([\d.]+) \+/- ([\d.]+)', line)
            if m:
                profile.ppl_results.append(PPLResult(
                    cache_type=current_ctk,
                    chunks=current_chunks,
                    ppl=float(m.group(1)),
                    stddev=float(m.group(2)),
                    env=current_env,
                ))


def _parse_load_snapshots(lines: list[str], profile: HardwareProfile) -> None:
    """Parse [LOAD_SNAPSHOT] tags."""
    current_snap = None

    for line in lines:
        if "[LOAD_SNAPSHOT] label=" in line:
            if current_snap:
                profile.load_snapshots.append(current_snap)
            label = line.split("label=")[1].split()[0]
            current_snap = LoadSnapshot(label=label)
            m = re.search(r'timestamp=(\S+)', line)
            if m:
                current_snap.timestamp = m.group(1)
        elif current_snap and "[LOAD_SNAPSHOT]" in line:
            content = line.split("[LOAD_SNAPSHOT]")[1].strip()
            if "load_avg=" in content:
                current_snap.load_avg = content.split("load_avg=")[1].strip()
            elif "process_count=" in content:
                current_snap.process_count = _int(content.split("process_count=")[1].strip())
            elif "approx_free_ram=" in content or "mem_available_mb=" in content:
                m = re.search(r'(\d+)', content.split("=")[-1])
                if m:
                    current_snap.free_ram_mb = float(m.group(1))
            elif "swap_used=" in content:
                current_snap.swap_used = content.split("swap_used=")[1].strip()
            elif "thermal=" in content:
                current_snap.thermal = content.split("thermal=")[1].strip()
            elif "gpu_util=" in content or "gpu_ioreg=" in content:
                current_snap.gpu_util = content.split("=", 1)[1].strip()

    if current_snap:
        profile.load_snapshots.append(current_snap)


def _int(val: str) -> int:
    """Safe int parse."""
    try:
        return int(re.sub(r'[^\d-]', '', val.strip()))
    except (ValueError, TypeError):
        return 0


# ============================================================
# Comparison and Analysis
# ============================================================

@dataclass
class ComparisonReport:
    """Result of comparing two hardware profiles."""
    baseline_name: str
    target_name: str
    hardware_diff: dict = field(default_factory=dict)
    decode_ratio_curve: dict = field(default_factory=dict)  # depth → (baseline_ratio, target_ratio)
    prefill_ratio_curve: dict = field(default_factory=dict)
    ppl_comparison: dict = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render comparison as markdown table."""
        lines = [f"# TurboQuant Hardware Comparison: {self.baseline_name} vs {self.target_name}\n"]

        if self.hardware_diff:
            lines.append("## Hardware Differences\n")
            lines.append("| Property | Baseline | Target |")
            lines.append("|----------|----------|--------|")
            for key, (bval, tval) in self.hardware_diff.items():
                lines.append(f"| {key} | {bval} | {tval} |")
            lines.append("")

        if self.decode_ratio_curve:
            lines.append("## Decode Speed Ratio (turbo3/q8_0)\n")
            lines.append("| Context | Baseline | Target | Delta |")
            lines.append("|---------|----------|--------|-------|")
            for depth, (br, tr) in sorted(self.decode_ratio_curve.items()):
                delta = tr - br if tr and br else 0
                flag = " ⚠️" if delta < -0.1 else ""
                lines.append(f"| {depth:,} | {br:.3f}x | {tr:.3f}x | {delta:+.3f}{flag} |")
            lines.append("")

        if self.anomalies:
            lines.append("## Anomalies\n")
            for a in self.anomalies:
                lines.append(f"- {a}")
            lines.append("")

        return "\n".join(lines)


def compare_profiles(baseline: HardwareProfile, target: HardwareProfile) -> ComparisonReport:
    """Compare two hardware profiles and identify differences."""
    report = ComparisonReport(
        baseline_name=baseline.system.chip_model or baseline.system.cpu_brand,
        target_name=target.system.chip_model or target.system.cpu_brand,
    )

    # Hardware differences
    hw_fields = [
        ("CPU", baseline.system.cpu_brand, target.system.cpu_brand),
        ("RAM (GB)", str(baseline.system.ram_total_gb), str(target.system.ram_total_gb)),
        ("GPU Family", baseline.system.gpu.family, target.system.gpu.family),
        ("GPU Family ID", str(baseline.system.gpu.family_id), str(target.system.gpu.family_id)),
        ("Tensor API", str(baseline.system.gpu.has_tensor), str(target.system.gpu.has_tensor)),
        ("Max Working Set (MB)", f"{baseline.system.gpu.recommended_max_working_set_mb:.0f}",
         f"{target.system.gpu.recommended_max_working_set_mb:.0f}"),
        ("Apple Silicon", str(baseline.system.apple_silicon), str(target.system.apple_silicon)),
    ]
    for name, bval, tval in hw_fields:
        if bval != tval:
            report.hardware_diff[name] = (bval, tval)

    # Decode ratio curves
    base_ratios = baseline.get_ratio_curve("turbo3", "q8_0", "decode")
    target_ratios = target.get_ratio_curve("turbo3", "q8_0", "decode")
    all_depths = sorted(set(base_ratios.keys()) | set(target_ratios.keys()))
    for depth in all_depths:
        br = base_ratios.get(depth, 0)
        tr = target_ratios.get(depth, 0)
        report.decode_ratio_curve[depth] = (br, tr)

    # Prefill ratio curves
    base_pf = baseline.get_ratio_curve("turbo3", "q8_0", "prefill")
    target_pf = target.get_ratio_curve("turbo3", "q8_0", "prefill")
    for depth in sorted(set(base_pf.keys()) | set(target_pf.keys())):
        bp = base_pf.get(depth, 0)
        tp = target_pf.get(depth, 0)
        report.prefill_ratio_curve[depth] = (bp, tp)

    # PPL comparison
    for bp in baseline.ppl_results:
        for tp in target.ppl_results:
            if bp.cache_type == tp.cache_type and bp.env == tp.env:
                report.ppl_comparison[f"{bp.cache_type}_{bp.env or 'uniform'}"] = (bp.ppl, tp.ppl)

    # Detect anomalies
    for depth, (br, tr) in report.decode_ratio_curve.items():
        if br > 0 and tr > 0:
            if tr < br * 0.5:
                report.anomalies.append(
                    f"Decode ratio at {depth:,} is {tr:.3f}x on target vs {br:.3f}x on baseline "
                    f"({tr/br:.0%} of expected). Constant cache thrashing suspected."
                )
            elif tr < 0.5:
                report.anomalies.append(
                    f"Decode ratio at {depth:,} is {tr:.3f}x — below 0.5x threshold. "
                    f"Hardware may not support turbo3 decode at this context depth."
                )

    # Tensor API warning
    if baseline.system.gpu.has_tensor and not target.system.gpu.has_tensor:
        report.anomalies.append(
            "Target lacks Tensor API (M1/M2/M3/M4). "
            "Turbo3 constant cache performance will be significantly worse."
        )

    return report


def predict_decode_from_baseline(baseline: HardwareProfile,
                                  target_gpu_family_id: int,
                                  target_has_tensor: bool) -> dict[int, float]:
    """Predict target decode ratios from baseline profile using a simple model.

    The model: constant cache throughput scales with GPU generation.
    M1 (family 1007): ~3x worse constant cache than M5 (family 1010)
    for divergent access patterns. Missing Tensor API adds additional penalty.
    """
    base_ratios = baseline.get_ratio_curve("turbo3", "q8_0", "decode")
    if not base_ratios:
        return {}

    # Simple constant cache degradation model:
    # Higher GPU family = better constant cache handling
    base_family = baseline.system.gpu.family_id
    if base_family == 0 or target_gpu_family_id == 0:
        return base_ratios  # Can't predict without family info

    # Family ID gap (e.g., 1010 - 1007 = 3 generations)
    gen_gap = base_family - target_gpu_family_id

    # Tensor API penalty: lacking tensor API means older Metal shader paths,
    # smaller constant cache, worse divergent access handling
    tensor_penalty = 1.0
    if baseline.system.gpu.has_tensor and not target_has_tensor:
        tensor_penalty = 1.3  # 30% additional overhead for missing tensor API

    # Degradation factor per context depth:
    # At short context, constant cache isn't the bottleneck (minimal degradation)
    # At long context, each generation gap increases the penalty exponentially
    predicted = {}
    for depth, ratio in base_ratios.items():
        # How much of the speed loss is from constant cache vs inherent overhead
        inherent_overhead = 1.0 - ratio  # e.g., 0.08 at 0.92x
        # Constant cache portion grows with context (exponential, not linear)
        # Real data shows exponential degradation at long context
        cache_fraction = min(1.0, (depth / 16384) ** 1.5) if depth > 0 else 0.0
        cache_penalty = inherent_overhead * cache_fraction
        non_cache_penalty = inherent_overhead - cache_penalty

        # Scale cache penalty by generation gap + tensor API penalty
        # Each generation ~1.5-2x worse for divergent constant access
        scaled_cache_penalty = cache_penalty * (1.8 ** gen_gap) * tensor_penalty

        predicted_ratio = max(0.01, 1.0 - non_cache_penalty - scaled_cache_penalty)
        predicted[depth] = round(predicted_ratio, 3)

    return predicted
