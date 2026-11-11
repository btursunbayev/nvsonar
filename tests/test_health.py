"""Tests for the composite 0-100 health score."""

from nvsonar.analysis import classify
from nvsonar.analysis.bottleneck import BottleneckResult, BottleneckType
from nvsonar.analysis.health import health_score

from .conftest import HW_THERMAL_SLOWDOWN, make_ecc, make_metrics, make_pcie, make_throttle

IDLE = BottleneckResult(BottleneckType.IDLE, 0.95, "idle", [])
COMPUTE = BottleneckResult(BottleneckType.COMPUTE_BOUND, 0.85, "compute bound", [])


def test_healthy_idle_gpu_scores_near_perfect():
    assert health_score(make_metrics(), IDLE) >= 95


def test_missing_metrics_do_not_penalize_score():
    metrics = make_metrics(
        temperature=None,
        power_usage=None,
        power_limit=None,
        gpu_clock=None,
        max_gpu_clock=None,
        memory_used=None,
        memory_total=None,
    )
    assert health_score(metrics, IDLE) == 100


def test_temperature_decays_score_between_70_and_95():
    cool = health_score(make_metrics(temperature=60.0), COMPUTE)
    warm = health_score(make_metrics(temperature=85.0), COMPUTE)
    hot = health_score(make_metrics(temperature=99.0), COMPUTE)
    assert cool > warm > hot


def test_uncorrectable_ecc_collapses_score():
    metrics = make_metrics(ecc=make_ecc(uncorrectable=1))
    assert health_score(metrics, COMPUTE) < 30


def test_correctable_ecc_costs_less_than_uncorrectable():
    correctable = health_score(make_metrics(ecc=make_ecc(correctable=5)), COMPUTE)
    uncorrectable = health_score(make_metrics(ecc=make_ecc(uncorrectable=5)), COMPUTE)
    assert correctable > uncorrectable


def test_critical_throttle_applies_multiplier():
    clean = health_score(make_metrics(temperature=90.0), COMPUTE)
    throttled = health_score(
        make_metrics(temperature=90.0, throttle=make_throttle(HW_THERMAL_SLOWDOWN)), COMPUTE
    )
    assert throttled < clean


def test_degraded_pcie_penalized_under_load_but_not_when_idle():
    degraded = make_metrics(pcie=make_pcie(degraded=True))
    assert health_score(degraded, COMPUTE) < health_score(degraded, IDLE)


def test_reduced_clocks_only_penalized_under_load():
    metrics = make_metrics(gpu_clock=700, max_gpu_clock=1440)
    assert health_score(metrics, IDLE) > health_score(metrics, COMPUTE)


def test_full_vram_lowers_memory_component():
    full = make_metrics(memory_used=int(0.99 * 24 * 1024**3), memory_total=24 * 1024**3)
    empty = make_metrics(memory_used=0, memory_total=24 * 1024**3)
    assert health_score(full, COMPUTE) < health_score(empty, COMPUTE)


def test_score_stays_within_bounds_on_worst_case():
    metrics = make_metrics(
        gpu_util=100,
        mem_util=100,
        temperature=110.0,
        power_usage=299.0,
        power_limit=300.0,
        gpu_clock=200,
        max_gpu_clock=1440,
        memory_used=24 * 1024**3,
        memory_total=24 * 1024**3,
        ecc=make_ecc(correctable=100, uncorrectable=10),
        pcie=make_pcie(degraded=True),
        throttle=make_throttle(HW_THERMAL_SLOWDOWN),
    )
    score = health_score(metrics, classify(metrics))
    assert 0 <= score <= 100
