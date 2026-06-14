"""Tests for the Prometheus exporter metric format."""

import time

import pytest
from prometheus_client.parser import text_string_to_metric_families

from nvsonar.analysis import classify
from nvsonar.analysis.bottleneck import BottleneckType
from nvsonar.analysis.health import health_score
from nvsonar.exporter.cache import Snapshot
from nvsonar.exporter.collectors import NVSonarCollector

from .conftest import HW_THERMAL_SLOWDOWN, make_metrics, make_throttle


class FakeCache:
    """SnapshotCache stand-in that avoids touching NVML."""

    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self.scrape_errors = 0
        self.last_scrape_duration_s = 0.0

    def all(self):
        return self._snapshots


def _make_snapshot(gpu_index=0, name="A30", **metric_overrides):
    metrics = make_metrics(**metric_overrides)
    result = classify(metrics)
    return Snapshot(
        gpu_index=gpu_index,
        gpu_name=name,
        metrics=metrics,
        bottleneck=result,
        health_score=health_score(metrics, result),
        timestamp=time.time(),
    )


def _render(snapshots) -> str:
    """Run the collector and return the Prometheus text exposition."""
    from prometheus_client import generate_latest
    from prometheus_client.registry import CollectorRegistry

    registry = CollectorRegistry()
    registry.register(NVSonarCollector(FakeCache(snapshots)))
    return generate_latest(registry).decode("utf-8")


def _families(text):
    return {fam.name: fam for fam in text_string_to_metric_families(text)}


def test_exporter_emits_basic_metric_families():
    snap = _make_snapshot(gpu_util=80, temperature=65.0, power_usage=180.0, power_limit=300.0)
    families = _families(_render([snap]))

    assert "nvsonar_gpu_temp_celsius" in families
    assert "nvsonar_gpu_utilization_ratio" in families
    assert "nvsonar_gpu_power_watts" in families
    assert "nvsonar_gpu_health_score" in families


def test_utilization_is_ratio_not_percentage():
    snap = _make_snapshot(gpu_util=87)
    families = _families(_render([snap]))
    [sample] = families["nvsonar_gpu_utilization_ratio"].samples
    assert sample.value == pytest.approx(0.87)


def test_bottleneck_is_one_hot_across_types():
    snap = _make_snapshot(gpu_util=95, mem_util=20)
    families = _families(_render([snap]))
    samples = families["nvsonar_gpu_bottleneck"].samples

    active = [s for s in samples if s.value == 1.0]
    inactive = [s for s in samples if s.value == 0.0]

    assert len(active) == 1
    assert active[0].labels["type"] == BottleneckType.COMPUTE_BOUND.value
    assert len(inactive) == len(BottleneckType) - 1


def test_unavailable_metric_is_omitted():
    snap = _make_snapshot(gpu_util=80, temperature=None)
    families = _families(_render([snap]))
    assert families["nvsonar_gpu_temp_celsius"].samples == []


def test_throttle_emits_one_sample_per_active_reason():
    snap = _make_snapshot(
        gpu_util=80,
        throttle=make_throttle(HW_THERMAL_SLOWDOWN),
        temperature=92.0,
    )
    families = _families(_render([snap]))
    samples = families["nvsonar_gpu_throttle_active"].samples

    assert len(samples) == 1
    assert samples[0].labels["reason"] == "Hardware Thermal Slowdown"
    assert samples[0].labels["severity"] == "critical"


def test_ecc_counter_present_with_zero_baseline():
    snap = _make_snapshot(gpu_util=80)
    families = _families(_render([snap]))
    severities = {s.labels["severity"] for s in families["nvsonar_gpu_ecc_errors"].samples}
    assert severities == {"correctable", "uncorrectable"}


def test_labels_carry_gpu_index_and_name():
    snap_a = _make_snapshot(gpu_index=0, name="A30", gpu_util=70)
    snap_b = _make_snapshot(gpu_index=1, name="T4", gpu_util=40)
    families = _families(_render([snap_a, snap_b]))
    samples = families["nvsonar_gpu_utilization_ratio"].samples

    labels = {(s.labels["gpu"], s.labels["name"]) for s in samples}
    assert labels == {("0", "A30"), ("1", "T4")}


def test_self_monitoring_metrics_present():
    snap = _make_snapshot(gpu_util=50)
    families = _families(_render([snap]))
    assert "nvsonar_exporter_scrape_duration_seconds" in families
    assert "nvsonar_exporter_scrape_errors" in families


def test_empty_cache_produces_no_samples_but_no_crash():
    text = _render([])
    families = _families(text)
    for fam in families.values():
        gpu_samples = [s for s in fam.samples if "gpu" in s.labels]
        assert gpu_samples == []
