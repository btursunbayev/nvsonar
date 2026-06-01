"""Multi-GPU outlier detection via Z-scores"""

import math
from dataclasses import dataclass

from nvsonar.monitor import Metrics


@dataclass
class Outlier:
    """A GPU metric that deviates from the group"""

    gpu_index: int
    metric: str
    value: float
    group_mean: float
    group_std: float
    z_score: float
    detail: str
    severity: str


def detect_outliers(
    gpu_metrics: dict[int, Metrics],
    z_threshold: float = 2.0,
) -> list[Outlier]:
    """Compare metrics across GPUs and flag outliers"""
    if len(gpu_metrics) < 2:
        return []

    outliers = []

    # metrics to compare: (extract fn, name, direction, detail template)
    checks = [
        (
            lambda m: float(m.temperature),
            "temperature",
            "high",
            "{value:.0f}C vs group avg {mean:.0f}C, possible cooling issue",
        ),
        (
            lambda m: float(m.gpu_utilization),
            "gpu_utilization",
            "low",
            "{value:.0f}% vs group avg {mean:.0f}%, straggler GPU",
        ),
        (
            lambda m: float(m.gpu_clock),
            "gpu_clock",
            "low",
            "{value:.0f} MHz vs group avg {mean:.0f} MHz, check throttling",
        ),
        (
            lambda m: float(m.power_usage) if m.power_usage else None,
            "power_draw",
            "both",
            "{value:.1f}W vs group avg {mean:.1f}W",
        ),
        (
            lambda m: m.memory_used_pct,
            "memory_used",
            "high",
            "{value:.1f}% vs group avg {mean:.1f}%, uneven distribution",
        ),
    ]

    for extract, metric_name, direction, template in checks:
        values = {}
        for idx, m in gpu_metrics.items():
            v = extract(m)
            if v is not None:
                values[idx] = v

        if len(values) < 2:
            continue

        mean = sum(values.values()) / len(values)
        variance = sum((v - mean) ** 2 for v in values.values()) / len(values)
        std = math.sqrt(variance)

        if std < 0.001:
            continue  # all GPUs identical

        for idx, value in values.items():
            z = (value - mean) / std

            flagged = False
            if direction == "high" and z > z_threshold:
                flagged = True
            elif direction == "low" and z < -z_threshold:
                flagged = True
                z = abs(z)
            elif direction == "both" and abs(z) > z_threshold:
                flagged = True
                z = abs(z)

            if not flagged:
                continue

            severity = "critical" if abs(z) > 3.0 else "warning"

            if metric_name == "temperature":
                ecc = gpu_metrics[idx].ecc
                if ecc.uncorrectable > 0:
                    severity = "critical"

            detail = template.format(value=value, mean=mean)
            outliers.append(
                Outlier(
                    gpu_index=idx,
                    metric=metric_name,
                    value=value,
                    group_mean=mean,
                    group_std=std,
                    z_score=z,
                    detail=detail,
                    severity=severity,
                )
            )

    # ECC outliers: any uncorrectable when peers have none
    for idx, m in gpu_metrics.items():
        if m.ecc.uncorrectable > 0:
            others_clean = all(
                gpu_metrics[j].ecc.uncorrectable == 0 for j in gpu_metrics if j != idx
            )
            if others_clean:
                outliers.append(
                    Outlier(
                        gpu_index=idx,
                        metric="ecc_errors",
                        value=float(m.ecc.uncorrectable),
                        group_mean=0.0,
                        group_std=0.0,
                        z_score=99.0,
                        detail=f"{m.ecc.uncorrectable} uncorrectable ECC errors "
                        f"while all other GPUs have 0",
                        severity="critical",
                    )
                )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    outliers.sort(key=lambda o: severity_order.get(o.severity, 99))
    return outliers
