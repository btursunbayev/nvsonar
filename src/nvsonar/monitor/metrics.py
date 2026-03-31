"""GPU metrics collection via NVML"""

from dataclasses import dataclass, field

import pynvml as nvml

from .hardware import get_handle, get_pcie_info, get_ecc_info, get_gpu_processes, PCIeInfo, ECCInfo, GPUProcess
from .throttle import decode_throttle_reasons, ThrottleStatus


@dataclass
class Metrics:
    """GPU metrics snapshot"""

    # Utilization (None = unavailable)
    gpu_utilization: int | None
    memory_utilization: int | None

    # Memory (None = unavailable)
    memory_used: int | None
    memory_total: int | None

    # Clocks
    gpu_clock: int | None
    memory_clock: int | None
    max_gpu_clock: int | None

    # Thermal
    temperature: float | None

    # Power
    power_usage: float | None
    power_limit: float | None

    # Fan
    fan_speed: int | None

    # Throttle
    throttle: ThrottleStatus

    # PCIe
    pcie: PCIeInfo

    # ECC
    ecc: ECCInfo

    # Processes
    processes: list[GPUProcess] = field(default_factory=list)

    # Collection errors
    errors: list[str] = field(default_factory=list)

    @property
    def memory_used_pct(self) -> float | None:
        if self.memory_total is None or self.memory_used is None:
            return None
        if self.memory_total == 0:
            return None
        return (self.memory_used / self.memory_total) * 100

    @property
    def power_used_pct(self) -> float | None:
        if not self.power_usage or not self.power_limit:
            return None
        if self.power_limit == 0:
            return None
        return (self.power_usage / self.power_limit) * 100

    @property
    def clock_reduction_pct(self) -> float | None:
        if self.max_gpu_clock is None or self.gpu_clock is None:
            return None
        if self.max_gpu_clock == 0:
            return None
        reduction = 1 - (self.gpu_clock / self.max_gpu_clock)
        return max(0.0, reduction * 100)


class MetricsCollector:
    """Collects GPU metrics for a single device"""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._handle = get_handle(device_index)

    def collect(self) -> Metrics:
        """Collect current metrics snapshot"""
        h = self._handle
        errors = []

        try:
            utilization = nvml.nvmlDeviceGetUtilizationRates(h)
            gpu_util = utilization.gpu
            mem_util = utilization.memory
        except nvml.NVMLError:
            gpu_util = None
            mem_util = None
            errors.append("Utilization unavailable")

        try:
            memory_info = nvml.nvmlDeviceGetMemoryInfo(h)
            mem_used = memory_info.used
            mem_total = memory_info.total
        except nvml.NVMLError:
            mem_used = None
            mem_total = None
            errors.append("Memory info unavailable")

        try:
            gpu_clock = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_GRAPHICS)
        except nvml.NVMLError:
            gpu_clock = None

        try:
            mem_clock = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_MEM)
        except nvml.NVMLError:
            mem_clock = None

        try:
            max_gpu_clock = nvml.nvmlDeviceGetMaxClockInfo(h, nvml.NVML_CLOCK_GRAPHICS)
        except nvml.NVMLError:
            max_gpu_clock = None

        try:
            temperature = nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
        except nvml.NVMLError:
            temperature = None
            errors.append("Temperature unavailable")

        try:
            power_usage = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except nvml.NVMLError:
            power_usage = None

        try:
            power_limit = nvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except nvml.NVMLError:
            power_limit = None

        try:
            fan_speed = nvml.nvmlDeviceGetFanSpeed(h)
        except nvml.NVMLError:
            fan_speed = None

        throttle = decode_throttle_reasons(h)
        pcie = get_pcie_info(h)
        ecc = get_ecc_info(h)
        processes = get_gpu_processes(h)

        return Metrics(
            gpu_utilization=gpu_util,
            memory_utilization=mem_util,
            memory_used=mem_used,
            memory_total=mem_total,
            gpu_clock=gpu_clock,
            memory_clock=mem_clock,
            max_gpu_clock=max_gpu_clock,
            temperature=temperature,
            power_usage=power_usage,
            power_limit=power_limit,
            fan_speed=fan_speed,
            throttle=throttle,
            pcie=pcie,
            ecc=ecc,
            processes=processes,
            errors=errors,
        )
