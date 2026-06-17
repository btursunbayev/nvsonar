"""Compile and run CUDA benchmark kernels"""

import ctypes
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

KERNELS_DIR = Path(__file__).parent / "kernels"
CACHE_DIR = Path.home() / ".nvsonar" / "cache"


@dataclass
class MemoryResult:
    """Memory bandwidth benchmark result"""

    read_gbps: float
    write_gbps: float
    copy_gbps: float


@dataclass
class ComputeResult:
    """Compute throughput benchmark result"""

    tflops: float


@dataclass
class PCIeResult:
    """PCIe bandwidth benchmark result"""

    h2d_gbps: float
    d2h_gbps: float


@dataclass
class BenchmarkResults:
    """All benchmark results"""

    memory: MemoryResult | None
    compute: ComputeResult | None
    pcie: PCIeResult | None


def _check_nvcc() -> str:
    """Find nvcc compiler"""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError(
            "CUDA toolkit not found. Install it to run benchmarks: "
            "https://developer.nvidia.com/cuda-downloads"
        )
    return nvcc


def _compile(kernel_name: str) -> Path:
    """Compile a .cu file to shared library, cache result"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cu_path = KERNELS_DIR / f"{kernel_name}.cu"
    so_path = CACHE_DIR / f"{kernel_name}.so"

    if not cu_path.exists():
        raise RuntimeError(
            f"Kernel source {kernel_name}.cu not found. "
            f"Reinstall nvsonar: pip install --force-reinstall nvsonar"
        )

    # recompile if source is newer than cached .so
    if so_path.exists():
        if cu_path.stat().st_mtime <= so_path.stat().st_mtime:
            return so_path

    nvcc = _check_nvcc()

    cmd = [
        nvcc,
        "--shared",
        "--compiler-options",
        "-fPIC",
        "-o",
        str(so_path),
        str(cu_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to compile {kernel_name}.cu: {e.stderr}") from e

    return so_path


# ctypes struct matching the C BenchResult for memory
class _MemoryBenchResult(ctypes.Structure):
    _fields_ = [
        ("read_gbps", ctypes.c_double),
        ("write_gbps", ctypes.c_double),
        ("copy_gbps", ctypes.c_double),
        ("success", ctypes.c_int),
        ("error", ctypes.c_char * 256),
    ]


class _ComputeBenchResult(ctypes.Structure):
    _fields_ = [
        ("tflops", ctypes.c_double),
        ("success", ctypes.c_int),
        ("error", ctypes.c_char * 256),
    ]


class _PCIeBenchResult(ctypes.Structure):
    _fields_ = [
        ("h2d_gbps", ctypes.c_double),
        ("d2h_gbps", ctypes.c_double),
        ("success", ctypes.c_int),
        ("error", ctypes.c_char * 256),
    ]


def run_memory() -> MemoryResult:
    """Run memory bandwidth benchmark"""
    so_path = _compile("memory")
    lib = ctypes.CDLL(str(so_path))

    result = _MemoryBenchResult()
    lib.bench_memory(ctypes.byref(result))

    if not result.success:
        raise RuntimeError(f"Memory benchmark failed: {result.error.decode()}")

    return MemoryResult(
        read_gbps=result.read_gbps,
        write_gbps=result.write_gbps,
        copy_gbps=result.copy_gbps,
    )


def run_compute() -> ComputeResult:
    """Run compute throughput benchmark"""
    so_path = _compile("compute")
    lib = ctypes.CDLL(str(so_path))

    result = _ComputeBenchResult()
    lib.bench_compute(ctypes.byref(result))

    if not result.success:
        raise RuntimeError(f"Compute benchmark failed: {result.error.decode()}")

    return ComputeResult(tflops=result.tflops)


def run_pcie() -> PCIeResult:
    """Run PCIe bandwidth benchmark"""
    so_path = _compile("pcie")
    lib = ctypes.CDLL(str(so_path))

    result = _PCIeBenchResult()
    lib.bench_pcie(ctypes.byref(result))

    if not result.success:
        raise RuntimeError(f"PCIe benchmark failed: {result.error.decode()}")

    return PCIeResult(
        h2d_gbps=result.h2d_gbps,
        d2h_gbps=result.d2h_gbps,
    )


def run_benchmarks() -> BenchmarkResults:
    """Run all benchmarks"""
    memory = None
    compute = None
    pcie = None

    try:
        memory = run_memory()
    except RuntimeError:
        pass

    try:
        compute = run_compute()
    except RuntimeError:
        pass

    try:
        pcie = run_pcie()
    except RuntimeError:
        pass

    return BenchmarkResults(memory=memory, compute=compute, pcie=pcie)
