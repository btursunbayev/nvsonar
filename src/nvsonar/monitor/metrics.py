"""GPU metrics collection via NVML"""

from dataclasses import dataclass, field

import pynvml as nvml

from .hardware import get_handle, get_pcie_info, get_ecc_info, get_gpu_processes, PCIeInfo, ECCInfo, GPUProcess
from .throttle import decode_throttle_reasons, ThrottleStatus


@dataclass
class Metrics:
    """GPU metrics snapshot"""

    # Utilization
    gpu_utilization: int
    memory_utilization: int

    # Memory
    memory_used: int
    memory_total: int

    # Clocks
    gpu_clock: int
    memory_clock: int
    max_gpu_clock: int

    # Thermal
    temperature: float

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
    def memory_used_pct(self) -> float:
        if self.memory_total == 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100

    @property
    def power_used_pct(self) -> float | None:
        if not self.power_usage or not self.power_limit:
            return None
        if self.power_limit == 0:
            return None
        return (self.power_usage / self.power_limit) * 100

    @property
    def clock_reduction_pct(self) -> float:
        if self.max_gpu_clock == 0:
            return 0.0
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
            gpu_util = 0
            mem_util = 0
            errors.append("Failed to read utilization, driver may not support this query")

        try:
            memory_info = nvml.nvmlDeviceGetMemoryInfo(h)
        except nvml.NVMLError:
            memory_info = type("MemInfo", (), {"used": 0, "total": 0})()
            errors.append("Failed to read memory info, driver may not support nvmlDeviceGetMemoryInfo_v2")

        try:
            gpu_clock = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_GRAPHICS)
        except nvml.NVMLError:
            gpu_clock = 0

        try:
            mem_clock = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_MEM)
        except nvml.NVMLError:
            mem_clock = 0

        try:
            max_gpu_clock = nvml.nvmlDeviceGetMaxClockInfo(h, nvml.NVML_CLOCK_GRAPHICS)
        except nvml.NVMLError:
            max_gpu_clock = 0

        try:
            temperature = nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
        except nvml.NVMLError:
            temperature = 0
            errors.append("Failed to read temperature")

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
            memory_used=memory_info.used,
            memory_total=memory_info.total,
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
