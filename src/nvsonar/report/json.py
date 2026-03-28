"""Structured JSON report output"""

import json

from nvsonar.monitor import Metrics
from nvsonar.monitor.hardware import GPUInfo
from nvsonar.analysis.bottleneck import BottleneckResult
from nvsonar.analysis.temporal import Pattern
from nvsonar.analysis.recommendations import Recommendation


def build_report(
    gpu_info: GPUInfo,
    metrics: Metrics,
    bottleneck: BottleneckResult,
    patterns: list[Pattern] | None = None,
    recommendations: list[Recommendation] | None = None,
) -> dict:
    """Build a structured report dict for one GPU"""
    return {
        "gpu": {
            "index": gpu_info.index,
            "name": gpu_info.name,
            "uuid": gpu_info.uuid,
            "driver": gpu_info.driver_version,
            "cuda": gpu_info.cuda_version,
            "pci_bus_id": gpu_info.pci_bus_id,
        },
        "metrics": {
            "gpu_utilization": metrics.gpu_utilization,
            "memory_utilization": metrics.memory_utilization,
            "memory_used_mb": metrics.memory_used // (1024 ** 2),
            "memory_total_mb": metrics.memory_total // (1024 ** 2),
            "memory_used_gb": round(metrics.memory_used / (1024 ** 3), 1),
            "memory_total_gb": round(metrics.memory_total / (1024 ** 3), 1),
            "memory_used_pct": round(metrics.memory_used_pct, 1),
            "gpu_clock_mhz": metrics.gpu_clock,
            "max_gpu_clock_mhz": metrics.max_gpu_clock,
            "clock_reduction_pct": round(metrics.clock_reduction_pct, 1),
            "temperature_c": metrics.temperature,
            "power_usage_w": metrics.power_usage,
            "power_limit_w": metrics.power_limit,
            "power_used_pct": round(metrics.power_used_pct, 1) if metrics.power_used_pct else None,
            "fan_speed_pct": metrics.fan_speed,
        },
        "throttle": {
            "is_throttled": metrics.throttle.is_throttled,
            "worst_severity": metrics.throttle.worst_severity,
            "reasons": [
                {
                    "name": r.name,
                    "severity": r.severity,
                    "explanation": r.explanation,
                }
                for r in metrics.throttle.active_reasons
            ],
        },
        "pcie": {
            "current_gen": metrics.pcie.current_link_gen,
            "max_gen": metrics.pcie.max_link_gen,
            "current_width": metrics.pcie.current_link_width,
            "max_width": metrics.pcie.max_link_width,
            "is_degraded": metrics.pcie.is_degraded,
        },
        "ecc": {
            "enabled": metrics.ecc.ecc_enabled,
            "correctable": metrics.ecc.correctable,
            "uncorrectable": metrics.ecc.uncorrectable,
        },
        "analysis": {
            "bottleneck": bottleneck.bottleneck.value,
            "confidence": round(bottleneck.confidence, 2),
            "detail": bottleneck.detail,
            "warnings": bottleneck.warnings,
        },
        "patterns": [
            {
                "name": p.name,
                "severity": p.severity,
                "detail": p.detail,
            }
            for p in (patterns or [])
        ],
        "recommendations": [
            {
                "priority": r.priority,
                "title": r.title,
                "explanation": r.explanation,
                "actions": r.actions,
            }
            for r in (recommendations or [])
        ],
    }


def to_json(
    gpu_info: GPUInfo,
    metrics: Metrics,
    bottleneck: BottleneckResult,
    patterns: list[Pattern] | None = None,
    recommendations: list[Recommendation] | None = None,
    indent: int = 2,
) -> str:
    """Build report and serialize to JSON string"""
    report = build_report(gpu_info, metrics, bottleneck, patterns, recommendations)
    return json.dumps(report, indent=indent)
