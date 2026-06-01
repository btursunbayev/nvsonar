"""Tests for bottleneck classification and warning collection."""

from nvsonar.analysis.bottleneck import BottleneckType, classify
from nvsonar.monitor.throttle import ThrottleReason

from .conftest import (
    GPU_IDLE,
    HW_THERMAL_SLOWDOWN,
    SW_POWER_CAP,
    SW_THERMAL_SLOWDOWN,
    make_ecc,
    make_metrics,
    make_pcie,
    make_throttle,
)


def test_idle_when_gpu_and_memory_quiet():
    result = classify(make_metrics(gpu_util=0, mem_util=0))
    assert result.bottleneck == BottleneckType.IDLE
    assert result.confidence >= 0.9


def test_compute_bound_when_gpu_saturated_memory_low():
    result = classify(make_metrics(gpu_util=95, mem_util=20))
    assert result.bottleneck == BottleneckType.COMPUTE_BOUND


def test_memory_bandwidth_bound_when_controller_saturated():
    result = classify(make_metrics(gpu_util=40, mem_util=90))
    assert result.bottleneck == BottleneckType.MEMORY_BANDWIDTH_BOUND


def test_memory_capacity_bound_when_vram_nearly_full():
    result = classify(
        make_metrics(
            gpu_util=80,
            mem_util=30,
            memory_used=int(0.97 * 24 * 1024**3),
            memory_total=24 * 1024**3,
        )
    )
    assert result.bottleneck == BottleneckType.MEMORY_CAPACITY_BOUND


def test_thermal_throttled_when_hardware_slowdown_active():
    result = classify(
        make_metrics(
            gpu_util=80,
            temperature=92.0,
            gpu_clock=600,
            max_gpu_clock=1500,
            throttle=make_throttle(HW_THERMAL_SLOWDOWN),
        )
    )
    assert result.bottleneck == BottleneckType.THERMAL_THROTTLED


def test_thermal_throttled_when_software_slowdown_active():
    result = classify(
        make_metrics(
            gpu_util=80,
            temperature=85.0,
            throttle=make_throttle(SW_THERMAL_SLOWDOWN),
        )
    )
    assert result.bottleneck == BottleneckType.THERMAL_THROTTLED


def test_power_limited_when_software_power_cap_active():
    result = classify(
        make_metrics(
            gpu_util=80,
            power_usage=295.0,
            power_limit=300.0,
            throttle=make_throttle(SW_POWER_CAP),
        )
    )
    assert result.bottleneck == BottleneckType.POWER_LIMITED


def test_data_starved_when_low_util_with_loaded_vram():
    result = classify(
        make_metrics(
            gpu_util=20,
            mem_util=5,
            memory_used=int(0.6 * 24 * 1024**3),
            memory_total=24 * 1024**3,
        )
    )
    assert result.bottleneck == BottleneckType.DATA_STARVED


def test_balanced_when_both_metrics_moderate():
    result = classify(make_metrics(gpu_util=75, mem_util=65))
    assert result.bottleneck == BottleneckType.BALANCED


def test_unknown_when_metrics_unavailable():
    result = classify(make_metrics(gpu_util=None, mem_util=None))
    assert result.bottleneck == BottleneckType.UNKNOWN


def test_no_false_positive_clock_warning_on_idle_gpu():
    """Regression: idle GPU with reduced clocks shouldn't fire the
    "without active throttle reason" warning when GPU Idle is the throttle reason.
    """
    metrics = make_metrics(
        gpu_util=0,
        mem_util=0,
        gpu_clock=210,
        max_gpu_clock=1440,
        throttle=make_throttle(GPU_IDLE),
    )
    result = classify(metrics)
    assert not any("without active throttle reason" in w for w in result.warnings)


def test_clock_warning_still_fires_when_clocks_reduced_with_no_explanation():
    info_reason = ThrottleReason(
        bitmask=2,
        name="Applications Clocks Setting",
        severity="info",
        explanation="",
        action=None,
    )
    metrics = make_metrics(
        gpu_util=50,
        mem_util=10,
        gpu_clock=300,
        max_gpu_clock=1500,
        throttle=make_throttle(info_reason),
    )
    result = classify(metrics)
    assert any("without active throttle reason" in w for w in result.warnings)


def test_uncorrectable_ecc_warning():
    metrics = make_metrics(ecc=make_ecc(uncorrectable=3))
    result = classify(metrics)
    assert any("uncorrectable ECC" in w for w in result.warnings)


def test_pcie_degradation_warning():
    metrics = make_metrics(pcie=make_pcie(degraded=True))
    result = classify(metrics)
    assert any("PCIe" in w for w in result.warnings)


def test_misleading_utilization_warning():
    metrics = make_metrics(
        gpu_util=90,
        mem_util=10,
        power_usage=60.0,
        power_limit=300.0,
    )
    result = classify(metrics)
    assert any("not truly compute-saturated" in w for w in result.warnings)
