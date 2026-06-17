"""GPU hardware detection and PCIe/ECC diagnostics"""

import os
from dataclasses import dataclass

import pynvml as nvml


def _decode(value: str | bytes) -> str:
    """Decode bytes to string if needed"""
    return value.decode("utf-8") if isinstance(value, bytes) else value


class _NVMLContext:
    """NVML library context"""

    def __init__(self):
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            nvml.nvmlInit()
            self._initialized = True
            return True
        except nvml.NVMLError:
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized


_ctx = _NVMLContext()


def initialize() -> bool:
    return _ctx.initialize()


@dataclass
class GPUInfo:
    """Static GPU device information"""

    index: int
    name: str
    uuid: str
    memory_total: int
    driver_version: str
    cuda_version: str
    pci_bus_id: str


@dataclass
class PCIeInfo:
    """PCIe link state"""

    current_link_gen: int
    max_link_gen: int
    current_link_width: int
    max_link_width: int
    tx_throughput_kbps: int | None
    rx_throughput_kbps: int | None

    @property
    def is_degraded(self) -> bool:
        return (
            self.current_link_gen < self.max_link_gen
            or self.current_link_width < self.max_link_width
        )

    @property
    def degradation_reason(self) -> str | None:
        if not self.is_degraded:
            return None

        parts = []
        if self.current_link_gen < self.max_link_gen:
            parts.append(f"Gen{self.current_link_gen} (max Gen{self.max_link_gen})")
        if self.current_link_width < self.max_link_width:
            parts.append(f"x{self.current_link_width} (max x{self.max_link_width})")

        return "PCIe downgraded: " + ", ".join(parts)


@dataclass
class ECCInfo:
    """ECC error counts"""

    correctable: int
    uncorrectable: int
    ecc_enabled: bool

    @property
    def has_errors(self) -> bool:
        return self.correctable > 0 or self.uncorrectable > 0


@dataclass
class GPUProcess:
    """A compute process running on the GPU"""

    pid: int
    name: str
    used_memory: int  # bytes


def get_handle(device_index: int):
    """Get NVML device handle"""
    if not _ctx.is_initialized:
        if not _ctx.initialize():
            raise RuntimeError("Failed to initialize NVML")
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        return handle
    except nvml.NVMLError as e:
        raise RuntimeError(f"Failed to get GPU {device_index}: {e}") from e


def get_device_count() -> int:
    """Get number of available GPUs"""
    if not _ctx.is_initialized:
        return 0
    try:
        count = nvml.nvmlDeviceGetCount()
        return count
    except nvml.NVMLError:
        return 0


def get_gpu_info(device_index: int = 0) -> GPUInfo | None:
    """Get static GPU information"""
    if not _ctx.is_initialized:
        return None

    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)

        name = _decode(nvml.nvmlDeviceGetName(handle))
        uuid = _decode(nvml.nvmlDeviceGetUUID(handle))
        try:
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = memory.total
        except nvml.NVMLError:
            memory_total = 0
        driver = _decode(nvml.nvmlSystemGetDriverVersion())

        try:
            cuda_ver = nvml.nvmlSystemGetCudaDriverVersion()
            cuda_str = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}" if cuda_ver else "unknown"
        except nvml.NVMLError:
            cuda_str = "unknown"

        try:
            pci = nvml.nvmlDeviceGetPciInfo(handle)
            bus_id = _decode(pci.busId)
        except nvml.NVMLError:
            bus_id = "unknown"

        info = GPUInfo(
            index=device_index,
            name=name,
            uuid=uuid,
            memory_total=memory_total,
            driver_version=driver,
            cuda_version=cuda_str,
            pci_bus_id=bus_id,
        )
        return info
    except nvml.NVMLError:
        return None


def list_gpus() -> list[GPUInfo]:
    """List all available GPUs"""
    if not _ctx.initialize():
        return []

    devices = []
    for i in range(get_device_count()):
        info = get_gpu_info(i)
        if info:
            devices.append(info)
    return devices


def get_pcie_info(handle) -> PCIeInfo:
    """Get PCIe link state for a device"""
    try:
        current_gen = nvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
    except nvml.NVMLError:
        current_gen = 0

    try:
        max_gen = nvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
    except nvml.NVMLError:
        max_gen = 0

    try:
        current_width = nvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
    except nvml.NVMLError:
        current_width = 0

    try:
        max_width = nvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
    except nvml.NVMLError:
        max_width = 0

    try:
        tx = nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES)
    except nvml.NVMLError:
        tx = None

    try:
        rx = nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES)
    except nvml.NVMLError:
        rx = None

    pcie = PCIeInfo(
        current_link_gen=current_gen,
        max_link_gen=max_gen,
        current_link_width=current_width,
        max_link_width=max_width,
        tx_throughput_kbps=tx,
        rx_throughput_kbps=rx,
    )
    return pcie


def get_ecc_info(handle) -> ECCInfo:
    """Get ECC error counts"""
    try:
        mode = nvml.nvmlDeviceGetEccMode(handle)
        ecc_enabled = mode[0] == nvml.NVML_FEATURE_ENABLED
    except nvml.NVMLError:
        return ECCInfo(correctable=0, uncorrectable=0, ecc_enabled=False)

    correctable = 0
    uncorrectable = 0

    try:
        correctable = nvml.nvmlDeviceGetTotalEccErrors(
            handle, nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, nvml.NVML_VOLATILE_ECC
        )
    except nvml.NVMLError:
        pass

    try:
        uncorrectable = nvml.nvmlDeviceGetTotalEccErrors(
            handle, nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nvml.NVML_VOLATILE_ECC
        )
    except nvml.NVMLError:
        pass

    return ECCInfo(
        correctable=correctable,
        uncorrectable=uncorrectable,
        ecc_enabled=ecc_enabled,
    )


def get_gpu_processes(handle) -> list[GPUProcess]:
    """Get compute processes currently running on a GPU"""
    try:
        procs = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except nvml.NVMLError:
        return []

    result = []
    for p in procs:
        try:
            name = os.path.basename(_decode(nvml.nvmlSystemGetProcessName(p.pid)))
        except nvml.NVMLError:
            name = "unknown"
        used_mem = p.usedGpuMemory if p.usedGpuMemory is not None else 0
        result.append(GPUProcess(pid=p.pid, name=name, used_memory=used_mem))
    return result
