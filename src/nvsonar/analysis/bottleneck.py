"""GPU bottleneck classification from single metrics snapshot"""

from dataclasses import dataclass
from enum import Enum

from nvsonar.monitor import Metrics


class BottleneckType(Enum):
    IDLE = "idle"
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BANDWIDTH_BOUND = "memory_bandwidth_bound"
    MEMORY_CAPACITY_BOUND = "memory_capacity_bound"
    POWER_LIMITED = "power_limited"
    THERMAL_THROTTLED = "thermal_throttled"
    DATA_STARVED = "data_starved"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


@dataclass
class BottleneckResult:
    """Bottleneck classification with confidence and warnings"""

    bottleneck: BottleneckType
    confidence: float  # 0.0 - 1.0
    detail: str
    warnings: list[str]


def classify(metrics: Metrics) -> BottleneckResult:
    """Classify GPU bottleneck from a single metrics snapshot"""
    warnings = _collect_warnings(metrics)

    gpu_util = metrics.gpu_utilization
    mem_util = metrics.memory_utilization
    mem_used_pct = metrics.memory_used_pct
    power_pct = metrics.power_used_pct
    clock_drop = metrics.clock_reduction_pct
    throttle = metrics.throttle

    # if critical metrics are unavailable, can't classify
    if gpu_util is None and mem_util is None:
        return BottleneckResult(
            BottleneckType.UNKNOWN, 0.10,
            "GPU metrics unavailable, cannot classify workload",
            warnings,
        )

    # use 0 as fallback for comparisons when one metric is available
    gu = gpu_util if gpu_util is not None else 0
    mu = mem_util if mem_util is not None else 0

    # --- idle ---
    if gu < 5 and mu < 5:
        return BottleneckResult(
            BottleneckType.IDLE, 0.95,
            "GPU has no active workload",
            warnings,
        )

    # --- hardware throttle (highest priority, masks real workload profile) ---

    hw_thermal = throttle.worst_severity == "critical" and any(
        r.name in ("Hardware Thermal Slowdown", "Hardware Slowdown")
        for r in throttle.active_reasons
    )
    sw_thermal = any(
        r.name == "Software Thermal Slowdown" for r in throttle.active_reasons
    )

    cd = f"{clock_drop:.0f}" if clock_drop is not None else "N/A"
    temp = metrics.temperature if metrics.temperature is not None else "N/A"

    if hw_thermal:
        return BottleneckResult(
            BottleneckType.THERMAL_THROTTLED, 0.95,
            f"Hardware thermal throttle active at {temp}C, clocks reduced {cd}%",
            warnings,
        )
    if sw_thermal:
        return BottleneckResult(
            BottleneckType.THERMAL_THROTTLED, 0.85,
            f"Driver thermal throttle at {temp}C, clocks reduced {cd}%",
            warnings,
        )

    # power limited
    sw_power_cap = any(r.name == "Software Power Cap" for r in throttle.active_reasons)
    if sw_power_cap and power_pct is not None and power_pct > 90:
        return BottleneckResult(
            BottleneckType.POWER_LIMITED, 0.90,
            f"Power draw at {power_pct:.0f}% of limit, clocks reduced {cd}%",
            warnings,
        )

    if power_pct is not None and power_pct > 95:
        return BottleneckResult(
            BottleneckType.POWER_LIMITED, 0.80,
            f"Power at {power_pct:.0f}% of limit",
            warnings,
        )

    # --- workload classification ---

    if mem_used_pct is not None and mem_used_pct > 95:
        return BottleneckResult(
            BottleneckType.MEMORY_CAPACITY_BOUND, 0.90,
            f"VRAM {mem_used_pct:.0f}% full, OOM risk",
            warnings,
        )

    if gu > 85 and mu < 50:
        conf = 0.85 if gu > 95 else 0.75
        return BottleneckResult(
            BottleneckType.COMPUTE_BOUND, conf,
            f"GPU {gu}% utilized, memory controller only {mu}%",
            warnings,
        )

    if mu > 80 and gu < 85:
        conf = 0.85 if mu > 90 else 0.75
        return BottleneckResult(
            BottleneckType.MEMORY_BANDWIDTH_BOUND, conf,
            f"Memory controller {mu}% busy, GPU at {gu}%",
            warnings,
        )

    if gu < 40 and mem_used_pct is not None and mem_used_pct > 50:
        return BottleneckResult(
            BottleneckType.DATA_STARVED, 0.70,
            f"GPU only {gu}% utilized despite {mem_used_pct:.0f}% VRAM used, "
            f"likely CPU or data loading bottleneck",
            warnings,
        )

    if gu > 70 and mu > 60:
        return BottleneckResult(
            BottleneckType.BALANCED, 0.65,
            f"Balanced workload, GPU {gu}%, memory controller {mu}%",
            warnings,
        )

    if gu >= 5 or mu >= 5:
        return BottleneckResult(
            BottleneckType.BALANCED, 0.40,
            f"Light workload, GPU {gu}%, memory controller {mu}%",
            warnings,
        )

    return BottleneckResult(
        BottleneckType.UNKNOWN, 0.20,
        "Unable to classify workload from available metrics",
        warnings,
    )


def _collect_warnings(metrics: Metrics) -> list[str]:
    """Check for secondary issues alongside the main diagnosis"""
    warnings = []

    # misleading utilization
    if (
        metrics.gpu_utilization is not None
        and metrics.gpu_utilization > 80
        and metrics.power_used_pct is not None
        and metrics.power_used_pct < 40
    ):
        warnings.append(
            f"GPU utilization is {metrics.gpu_utilization}% but power draw is only "
            f"{metrics.power_used_pct:.0f}%, GPU is not truly compute-saturated"
        )

    # PCIe degraded (skip if PCIe gen is 0 = not available)
    if metrics.pcie.is_degraded and metrics.pcie.max_link_gen > 0:
        warnings.append(metrics.pcie.degradation_reason)

    # ECC errors
    if metrics.ecc.uncorrectable > 0:
        warnings.append(
            f"{metrics.ecc.uncorrectable} uncorrectable ECC errors, "
            f"hardware may need replacement"
        )
    elif metrics.ecc.correctable > 0:
        warnings.append(
            f"{metrics.ecc.correctable} correctable ECC errors, monitor for increase"
        )

    # clock reduction without throttle reason
    # only warn if throttle status is not "no throttling"
    if (
        metrics.clock_reduction_pct is not None
        and metrics.clock_reduction_pct > 15
        and not metrics.throttle.is_throttled
        and metrics.throttle.worst_severity != "ok"
    ):
        warnings.append(
            f"Clocks {metrics.clock_reduction_pct:.0f}% below max "
            f"without active throttle reason"
        )

    # fan speed
    if metrics.fan_speed is not None and metrics.fan_speed > 90:
        warnings.append(f"Fan at {metrics.fan_speed}%, thermal stress")

    return warnings
