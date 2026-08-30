"""Shared fixtures and helpers for analysis layer tests."""

from nvsonar.monitor.hardware import ECCInfo, GPUInfo, GPUProcess, PCIeInfo
from nvsonar.monitor.metrics import Metrics
from nvsonar.monitor.throttle import ThrottleReason, ThrottleStatus

GPU_IDLE = ThrottleReason(
    bitmask=1,
    name="GPU Idle",
    severity="info",
    explanation="GPU has no active workload",
    action=None,
)
SW_POWER_CAP = ThrottleReason(
    bitmask=4,
    name="Software Power Cap",
    severity="warning",
    explanation="",
    action=None,
)
HW_THERMAL_SLOWDOWN = ThrottleReason(
    bitmask=64,
    name="Hardware Thermal Slowdown",
    severity="critical",
    explanation="",
    action=None,
)
SW_THERMAL_SLOWDOWN = ThrottleReason(
    bitmask=32,
    name="Software Thermal Slowdown",
    severity="warning",
    explanation="",
    action=None,
)


def make_throttle(*reasons: ThrottleReason) -> ThrottleStatus:
    bitmask = 0
    for r in reasons:
        bitmask |= r.bitmask
    return ThrottleStatus(raw_bitmask=bitmask, active_reasons=list(reasons))


def make_pcie(degraded: bool = False) -> PCIeInfo:
    return PCIeInfo(
        current_link_gen=2 if degraded else 4,
        max_link_gen=4,
        current_link_width=16,
        max_link_width=16,
        tx_throughput_kbps=None,
        rx_throughput_kbps=None,
    )


def make_ecc(correctable: int = 0, uncorrectable: int = 0) -> ECCInfo:
    return ECCInfo(correctable=correctable, uncorrectable=uncorrectable, ecc_enabled=True)


def make_metrics(
    *,
    gpu_util: int | None = 0,
    mem_util: int | None = 0,
    memory_used: int | None = 0,
    memory_total: int | None = 24 * 1024**3,
    gpu_clock: int | None = 1440,
    max_gpu_clock: int | None = 1440,
    temperature: float | None = 40.0,
    power_usage: float | None = 30.0,
    power_limit: float | None = 300.0,
    fan_speed: int | None = None,
    throttle: ThrottleStatus | None = None,
    pcie: PCIeInfo | None = None,
    ecc: ECCInfo | None = None,
    processes: list[GPUProcess] | None = None,
    errors: list[str] | None = None,
) -> Metrics:
    return Metrics(
        gpu_utilization=gpu_util,
        memory_utilization=mem_util,
        memory_used=memory_used,
        memory_total=memory_total,
        gpu_clock=gpu_clock,
        memory_clock=8000,
        max_gpu_clock=max_gpu_clock,
        temperature=temperature,
        power_usage=power_usage,
        power_limit=power_limit,
        fan_speed=fan_speed,
        throttle=throttle if throttle is not None else make_throttle(),
        pcie=pcie if pcie is not None else make_pcie(),
        ecc=ecc if ecc is not None else make_ecc(),
        processes=processes or [],
        errors=errors or [],
    )


def make_gpu_info(
    *,
    index: int = 0,
    name: str = "NVIDIA A30",
    memory_total: int = 24 * 1024**3,
) -> GPUInfo:
    return GPUInfo(
        index=index,
        name=name,
        uuid=f"GPU-0000000{index}-0000-0000-0000-000000000000",
        memory_total=memory_total,
        driver_version="550.54.14",
        cuda_version="12.4",
        pci_bus_id=f"00000000:0{index}:00.0",
    )


def make_process(pid: int = 1234, name: str = "python", used_memory_mb: int = 512) -> GPUProcess:
    return GPUProcess(pid=pid, name=name, used_memory=used_memory_mb * 1024**2)
