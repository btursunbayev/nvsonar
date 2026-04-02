"""Actionable recommendations from analysis results"""

from dataclasses import dataclass

from .bottleneck import BottleneckResult, BottleneckType
from .temporal import Pattern
from .outlier import Outlier


@dataclass
class Recommendation:
    """A prioritized recommendation with actions"""

    priority: int  # 1 = fix now, 2 = should fix, 3 = nice to know
    title: str
    explanation: str
    actions: list[str]


def recommend(
    bottleneck: BottleneckResult | None = None,
    patterns: list[Pattern] | None = None,
    outliers: list[Outlier] | None = None,
) -> list[Recommendation]:
    """Generate recommendations from analysis results"""
    recs = []

    if bottleneck:
        recs.extend(_from_bottleneck(bottleneck))

    for pattern in (patterns or []):
        rec = _from_pattern(pattern)
        if rec:
            recs.append(rec)

    for outlier in (outliers or []):
        rec = _from_outlier(outlier)
        if rec:
            recs.append(rec)

    # deduplicate by title
    seen = set()
    unique = []
    for r in recs:
        if r.title not in seen:
            seen.add(r.title)
            unique.append(r)

    unique.sort(key=lambda r: r.priority)
    return unique


def _from_bottleneck(result: BottleneckResult) -> list[Recommendation]:
    recs = []

    match result.bottleneck:
        case BottleneckType.THERMAL_THROTTLED:
            recs.append(Recommendation(
                priority=1,
                title="GPU is thermal throttling",
                explanation=result.detail,
                actions=[
                    "Check that all GPU fans are spinning",
                    "Improve case airflow or add fans",
                    "Clean dust from heatsink and filters",
                    "Lower ambient room temperature",
                    "If persistent, repaste GPU thermal compound",
                ],
            ))

        case BottleneckType.POWER_LIMITED:
            recs.append(Recommendation(
                priority=1,
                title="GPU is power limited",
                explanation=result.detail,
                actions=[
                    "Raise power limit: nvidia-smi -pl <watts>",
                    "Check that all GPU power cables are connected",
                    "Verify PSU has enough headroom for peak GPU draw",
                ],
            ))

        case BottleneckType.MEMORY_CAPACITY_BOUND:
            recs.append(Recommendation(
                priority=1,
                title="VRAM nearly full",
                explanation=result.detail,
                actions=[
                    "Reduce batch size",
                    "Enable gradient checkpointing",
                    "Use mixed precision (fp16/bf16) if not already",
                    "Try model parallelism to split across GPUs",
                    "Check for memory leaks in training loop",
                ],
            ))

        case BottleneckType.MEMORY_BANDWIDTH_BOUND:
            recs.append(Recommendation(
                priority=2,
                title="Memory bandwidth is the bottleneck",
                explanation=result.detail,
                actions=[
                    "This is normal for many ML workloads (especially inference)",
                    "Use mixed precision to reduce memory traffic",
                    "Increase batch size to improve compute-to-memory ratio",
                    "Consider a GPU with higher memory bandwidth (HBM)",
                ],
            ))

        case BottleneckType.COMPUTE_BOUND:
            recs.append(Recommendation(
                priority=3,
                title="GPU is compute bound",
                explanation=result.detail,
                actions=[
                    "This is good, GPU is fully utilized",
                    "Use mixed precision (fp16/bf16) for ~2x throughput on tensor cores",
                    "Enable torch.compile() or XLA compilation",
                    "Check if flash attention is enabled",
                ],
            ))

        case BottleneckType.DATA_STARVED:
            recs.append(Recommendation(
                priority=1,
                title="GPU is waiting for data",
                explanation=result.detail,
                actions=[
                    "Increase DataLoader num_workers",
                    "Enable pin_memory=True in DataLoader",
                    "Move training data to faster storage (SSD/NVMe)",
                    "Pre-process and cache data instead of on-the-fly transforms",
                    "Profile CPU usage to confirm the bottleneck",
                ],
            ))

        case BottleneckType.IDLE:
            recs.append(Recommendation(
                priority=3,
                title="GPU is idle",
                explanation="No workload detected",
                actions=["No action needed, GPU is not in use"],
            ))

        case BottleneckType.BALANCED:
            recs.append(Recommendation(
                priority=3,
                title="Balanced workload",
                explanation=result.detail,
                actions=[
                    "No single bottleneck detected",
                    "Profile with torch.profiler for deeper optimization",
                ],
            ))

        case _:
            pass

    # recommendations from warnings
    for warning in result.warnings:
        if "ECC" in warning and "uncorrectable" in warning:
            recs.append(Recommendation(
                priority=1,
                title="Hardware memory errors detected",
                explanation=warning,
                actions=[
                    "Check dmesg for Xid errors: sudo dmesg -T | grep Xid",
                    "Run NVIDIA field diagnostics",
                    "Contact GPU vendor for RMA if errors persist",
                ],
            ))
        elif "PCIe" in warning:
            recs.append(Recommendation(
                priority=2,
                title="PCIe link degraded",
                explanation=warning,
                actions=[
                    "Reseat the GPU in its PCIe slot",
                    "Check BIOS PCIe settings (Gen3/Gen4 forced?)",
                    "Try a different PCIe slot",
                    "Check for M.2 SSD lane sharing on consumer boards",
                ],
            ))
        elif "not truly compute-saturated" in warning:
            recs.append(Recommendation(
                priority=2,
                title="GPU utilization is misleading",
                explanation=warning,
                actions=[
                    "nvidia-smi GPU-Util only shows if ANY kernel is running",
                    "Low power + high util = small kernels or memory-latency-bound",
                    "Profile with Nsight Compute for real SM occupancy",
                ],
            ))

    return recs


def _from_pattern(pattern: Pattern) -> Recommendation | None:
    match pattern.name:
        case "clock_oscillation":
            return Recommendation(
                priority=1,
                title="GPU clocks are unstable",
                explanation=pattern.detail,
                actions=[
                    "Clocks bouncing between throttled and boost",
                    "Usually caused by thermal or power throttle cycling",
                    "Check cooling and power limit settings",
                    "Consider locking clocks: nvidia-smi -lgc <min>,<max>",
                ],
            )
        case "temperature_rising":
            return Recommendation(
                priority=1 if pattern.severity == "critical" else 2,
                title="Temperature is rising",
                explanation=pattern.detail,
                actions=[
                    "Cooling cannot keep up with current workload",
                    "Check fan speed and airflow",
                    "Will likely hit thermal throttle soon if trend continues",
                ],
            )
        case "utilization_dips":
            return Recommendation(
                priority=2,
                title="Periodic GPU idle gaps",
                explanation=pattern.detail,
                actions=[
                    "GPU periodically drops to near-zero utilization",
                    "Usually a data loading bottleneck between batches",
                    "Increase DataLoader num_workers",
                    "Enable prefetching or pin_memory",
                ],
            )
        case "memory_creep":
            return Recommendation(
                priority=2,
                title="VRAM usage is growing",
                explanation=pattern.detail,
                actions=[
                    "Memory is steadily increasing over time",
                    "Check for tensors not being freed in training loop",
                    "Look for growing lists or caches",
                    "Will eventually OOM if trend continues",
                ],
            )
        case "throttle_cycling":
            return Recommendation(
                priority=1 if pattern.severity == "critical" else 2,
                title="GPU throttle is cycling on and off",
                explanation=pattern.detail,
                actions=[
                    "GPU is repeatedly hitting and recovering from a limit",
                    "Check if power limit is set too close to workload demand",
                    "Improve cooling to prevent thermal cycling",
                    "Consider locking clocks: nvidia-smi -lgc <min>,<max>",
                ],
            )
    return None


def _from_outlier(outlier: Outlier) -> Recommendation | None:
    if outlier.severity != "critical":
        return None

    return Recommendation(
        priority=1,
        title=f"GPU {outlier.gpu_index} is an outlier ({outlier.metric})",
        explanation=outlier.detail,
        actions=[
            f"GPU {outlier.gpu_index} differs significantly from the group",
            "In distributed training, this GPU slows down all others",
            "Check this GPU's cooling, power cables, and PCIe seating",
            "Run nvidia-smi on this GPU specifically to compare",
        ],
    )
