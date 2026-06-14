"""Composite GPU health score (0-100)."""

from nvsonar.analysis.bottleneck import BottleneckResult, BottleneckType
from nvsonar.monitor import Metrics


def health_score(metrics: Metrics, bottleneck: BottleneckResult) -> int:
    """0-100 weighted average across thermal, clocks, ECC, power, memory, PCIe."""
    scores = {}

    is_idle = bottleneck.bottleneck == BottleneckType.IDLE

    # thermal: 100 if <70C, linear decay to 0 at 95C
    temp = metrics.temperature
    if temp is None:
        scores["thermal"] = 100
    elif temp < 70:
        scores["thermal"] = 100
    elif temp < 95:
        scores["thermal"] = int(100 * (95 - temp) / 25)
    else:
        scores["thermal"] = 0

    # power: 100 if <80% of limit, linear decay to 50 at 95%
    power_pct = metrics.power_used_pct
    if power_pct is None or power_pct < 80:
        scores["power"] = 100
    elif power_pct < 95:
        scores["power"] = int(100 - (power_pct - 80) * (50 / 15))
    else:
        scores["power"] = 50

    # clocks: 100 minus reduction percentage
    clock_drop = metrics.clock_reduction_pct
    if is_idle or clock_drop is None:
        scores["clocks"] = 100
    else:
        scores["clocks"] = max(0, int(100 - clock_drop))

    # memory: 100 if <80% used, linear decay to 0 at 100%
    mem_pct = metrics.memory_used_pct
    if mem_pct is None or mem_pct < 80:
        scores["memory"] = 100
    else:
        scores["memory"] = max(0, int(100 * (100 - mem_pct) / 20))

    # pcie: 100 if at max, 60 if degraded (skip if no PCIe or idle)
    if is_idle or metrics.pcie.max_link_gen == 0:
        scores["pcie"] = 100
    else:
        scores["pcie"] = 60 if metrics.pcie.is_degraded else 100

    # ecc: 100 if no errors, 30 if correctable, 0 if uncorrectable
    if metrics.ecc.uncorrectable > 0:
        scores["ecc"] = 0
    elif metrics.ecc.correctable > 0:
        scores["ecc"] = 30
    else:
        scores["ecc"] = 100

    weights = {
        "thermal": 0.20,
        "clocks": 0.20,
        "ecc": 0.20,
        "power": 0.15,
        "memory": 0.15,
        "pcie": 0.10,
    }

    total = sum(scores[k] * weights[k] for k in weights)

    if metrics.ecc.uncorrectable > 0:
        total *= 0.3
    if metrics.throttle.worst_severity == "critical":
        total *= 0.7
    if metrics.pcie.is_degraded and not is_idle and metrics.pcie.max_link_gen > 0:
        total *= 0.85

    return max(0, min(100, int(total)))
