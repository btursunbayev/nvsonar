"""Theoretical GPU specs for benchmark comparison"""

from dataclasses import dataclass


@dataclass
class GPUSpecs:
    """Theoretical performance specs for a GPU model"""

    memory_bandwidth_gbps: float
    fp32_tflops: float
    pcie_gen: int
    pcie_bandwidth_gbps: float  # theoretical max for the PCIe gen


# theoretical specs per GPU model
# memory bandwidth from official NVIDIA specs
# fp32 from official boost clock TFLOPS
# pcie bandwidth = theoretical max for that gen (not GPU-specific)
SPECS = {
    # Turing
    "1650 ti": GPUSpecs(192, 3.0, 3, 15.75),
    "1660": GPUSpecs(192, 5.0, 3, 15.75),
    "1660 super": GPUSpecs(336, 5.0, 3, 15.75),
    "1660 ti": GPUSpecs(288, 5.5, 3, 15.75),
    "2060": GPUSpecs(336, 6.5, 3, 15.75),
    "2070": GPUSpecs(448, 7.5, 3, 15.75),
    "2070 super": GPUSpecs(448, 9.1, 3, 15.75),
    "2080": GPUSpecs(448, 10.1, 3, 15.75),
    "2080 super": GPUSpecs(496, 11.2, 3, 15.75),
    "2080 ti": GPUSpecs(616, 13.4, 3, 15.75),
    # Ampere consumer
    "3060": GPUSpecs(360, 12.7, 4, 31.5),
    "3060 ti": GPUSpecs(448, 16.2, 4, 31.5),
    "3070": GPUSpecs(448, 20.3, 4, 31.5),
    "3070 ti": GPUSpecs(608, 21.7, 4, 31.5),
    "3080": GPUSpecs(760, 29.8, 4, 31.5),
    "3080 ti": GPUSpecs(912, 34.1, 4, 31.5),
    "3090": GPUSpecs(936, 35.6, 4, 31.5),
    "3090 ti": GPUSpecs(1008, 40.0, 4, 31.5),
    # Ada Lovelace consumer
    "4060": GPUSpecs(272, 15.1, 4, 31.5),
    "4060 ti": GPUSpecs(288, 22.1, 4, 31.5),
    "4070": GPUSpecs(504, 29.1, 4, 31.5),
    "4070 super": GPUSpecs(504, 35.5, 4, 31.5),
    "4070 ti": GPUSpecs(504, 40.1, 4, 31.5),
    "4070 ti super": GPUSpecs(672, 44.1, 4, 31.5),
    "4080": GPUSpecs(717, 48.7, 4, 31.5),
    "4080 super": GPUSpecs(736, 52.0, 4, 31.5),
    "4090": GPUSpecs(1008, 82.6, 4, 31.5),
    # datacenter
    "a100": GPUSpecs(2039, 19.5, 4, 31.5),
    "a100 80gb": GPUSpecs(2039, 19.5, 4, 31.5),
    "h100": GPUSpecs(3350, 66.9, 5, 63.0),
    "l40": GPUSpecs(864, 90.5, 4, 31.5),
    "l40s": GPUSpecs(864, 91.6, 4, 31.5),
    "a10": GPUSpecs(600, 31.2, 4, 31.5),
    "a30": GPUSpecs(933, 10.3, 4, 31.5),
    "a40": GPUSpecs(696, 37.4, 4, 31.5),
    "v100": GPUSpecs(900, 15.7, 3, 15.75),
    "t4": GPUSpecs(320, 8.1, 3, 15.75),
}


def find_specs(gpu_name: str) -> GPUSpecs | None:
    """Find specs for a GPU by matching its name"""
    name = gpu_name.lower()

    # try exact matches first (longer names first to avoid partial matches)
    sorted_keys = sorted(SPECS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in name:
            return SPECS[key]

    return None
