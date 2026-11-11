"""Tests for the structured JSON report."""

import json

from nvsonar.analysis import classify
from nvsonar.analysis.recommendations import Recommendation
from nvsonar.analysis.temporal import Pattern
from nvsonar.report.json import build_report, to_json

from .conftest import (
    HW_THERMAL_SLOWDOWN,
    make_ecc,
    make_gpu_info,
    make_metrics,
    make_pcie,
    make_process,
    make_throttle,
)


def _report(metrics=None, **kwargs):
    metrics = metrics if metrics is not None else make_metrics()
    return build_report(make_gpu_info(), metrics, classify(metrics), **kwargs)


def test_report_carries_static_gpu_identity():
    gpu = _report()["gpu"]
    assert gpu["index"] == 0
    assert gpu["name"] == "NVIDIA A30"
    assert gpu["driver"] == "550.54.14"
    assert gpu["cuda"] == "12.4"


def test_memory_reported_in_both_mb_and_gb():
    metrics = make_metrics(memory_used=2 * 1024**3, memory_total=24 * 1024**3)
    m = _report(metrics)["metrics"]
    assert m["memory_used_mb"] == 2048
    assert m["memory_total_mb"] == 24576
    assert m["memory_used_gb"] == 2.0
    assert m["memory_used_pct"] == 8.3


def test_unavailable_metrics_serialize_as_null_not_zero():
    metrics = make_metrics(
        gpu_util=None,
        mem_util=None,
        memory_used=None,
        memory_total=None,
        temperature=None,
        power_usage=None,
        power_limit=None,
        gpu_clock=None,
        max_gpu_clock=None,
    )
    m = _report(metrics)["metrics"]
    for field in (
        "gpu_utilization",
        "memory_used_mb",
        "memory_used_pct",
        "temperature_c",
        "power_used_pct",
        "clock_reduction_pct",
    ):
        assert m[field] is None, field


def test_zero_power_draw_is_zero_not_null():
    """A real 0% reading must survive; only genuinely missing data becomes null."""
    metrics = make_metrics(power_usage=0.0, power_limit=300.0)
    assert _report(metrics)["metrics"]["power_used_pct"] == 0.0


def test_throttle_reasons_expanded_with_severity():
    metrics = make_metrics(throttle=make_throttle(HW_THERMAL_SLOWDOWN))
    throttle = _report(metrics)["throttle"]
    assert throttle["is_throttled"] is True
    assert throttle["worst_severity"] == "critical"
    assert throttle["reasons"][0]["name"] == "Hardware Thermal Slowdown"


def test_pcie_and_ecc_sections_reflect_hardware_state():
    metrics = make_metrics(pcie=make_pcie(degraded=True), ecc=make_ecc(correctable=3))
    report = _report(metrics)
    assert report["pcie"]["is_degraded"] is True
    assert report["pcie"]["current_gen"] == 2
    assert report["ecc"]["correctable"] == 3
    assert report["ecc"]["uncorrectable"] == 0


def test_analysis_section_matches_classifier_output():
    metrics = make_metrics(gpu_util=95, mem_util=20)
    bottleneck = classify(metrics)
    analysis = build_report(make_gpu_info(), metrics, bottleneck)["analysis"]
    assert analysis["bottleneck"] == bottleneck.bottleneck.value
    assert analysis["confidence"] == round(bottleneck.confidence, 2)


def test_patterns_and_recommendations_default_to_empty_lists():
    report = _report()
    assert report["patterns"] == []
    assert report["recommendations"] == []


def test_patterns_and_recommendations_are_serialized_when_present():
    report = _report(
        patterns=[Pattern("clock_oscillation", "warning", "clocks unstable")],
        recommendations=[Recommendation(1, "Improve cooling", "GPU is hot", ["clean fans"])],
    )
    assert report["patterns"][0]["severity"] == "warning"
    assert report["recommendations"][0]["priority"] == 1
    assert report["recommendations"][0]["actions"] == ["clean fans"]


def test_processes_converted_to_megabytes():
    metrics = make_metrics(processes=[make_process(pid=99, used_memory_mb=1024)])
    proc = _report(metrics)["processes"][0]
    assert proc["pid"] == 99
    assert proc["used_memory_mb"] == 1024


def test_to_json_emits_parseable_json():
    metrics = make_metrics()
    text = to_json(make_gpu_info(), metrics, classify(metrics))
    assert json.loads(text) == build_report(make_gpu_info(), metrics, classify(metrics))
