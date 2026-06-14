"""Prometheus collectors that translate NVSonar snapshots into metric families."""

from typing import Iterator

from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector

from nvsonar.analysis.bottleneck import BottleneckType
from nvsonar.exporter.cache import SnapshotCache

NAMESPACE = "nvsonar"


class NVSonarCollector(Collector):
    """Translates the SnapshotCache into Prometheus metrics on each scrape."""

    def __init__(self, cache: SnapshotCache):
        self._cache = cache

    def collect(self) -> Iterator:
        snapshots = self._cache.all()

        yield from self._gauge(
            snapshots,
            "gpu_temp_celsius",
            "GPU temperature in degrees Celsius",
            lambda s: s.metrics.temperature,
        )
        yield from self._gauge(
            snapshots,
            "gpu_utilization_ratio",
            "GPU compute utilization (0-1)",
            lambda s: _ratio(s.metrics.gpu_utilization),
        )
        yield from self._gauge(
            snapshots,
            "gpu_memory_controller_ratio",
            "GPU memory controller utilization (0-1)",
            lambda s: _ratio(s.metrics.memory_utilization),
        )
        yield from self._gauge(
            snapshots,
            "gpu_memory_used_bytes",
            "GPU memory used in bytes",
            lambda s: s.metrics.memory_used,
        )
        yield from self._gauge(
            snapshots,
            "gpu_memory_total_bytes",
            "GPU memory total in bytes",
            lambda s: s.metrics.memory_total,
        )
        yield from self._gauge(
            snapshots,
            "gpu_power_watts",
            "GPU power draw in watts",
            lambda s: s.metrics.power_usage,
        )
        yield from self._gauge(
            snapshots,
            "gpu_power_limit_watts",
            "GPU power limit in watts",
            lambda s: s.metrics.power_limit,
        )
        yield from self._gauge(
            snapshots,
            "gpu_sm_clock_mhz",
            "GPU SM clock in MHz",
            lambda s: s.metrics.gpu_clock,
        )
        yield from self._gauge(
            snapshots,
            "gpu_memory_clock_mhz",
            "GPU memory clock in MHz",
            lambda s: s.metrics.memory_clock,
        )
        yield from self._gauge(
            snapshots,
            "gpu_fan_speed_ratio",
            "GPU fan speed (0-1)",
            lambda s: _ratio(s.metrics.fan_speed),
        )
        yield from self._gauge(
            snapshots,
            "gpu_clock_reduction_ratio",
            "Reduction of current SM clock vs max (0-1)",
            lambda s: _ratio(s.metrics.clock_reduction_pct),
        )
        yield from self._gauge(
            snapshots,
            "gpu_health_score",
            "NVSonar GPU health score (0-100)",
            lambda s: s.health_score,
        )

        bottleneck = GaugeMetricFamily(
            _name("gpu_bottleneck"),
            "Active bottleneck classification (1 if active, 0 otherwise)",
            labels=["gpu", "name", "type"],
        )
        for snap in snapshots:
            for bt in BottleneckType:
                value = 1.0 if snap.bottleneck.bottleneck == bt else 0.0
                bottleneck.add_metric(
                    [str(snap.gpu_index), snap.gpu_name, bt.value],
                    value,
                )
        yield bottleneck

        throttle = GaugeMetricFamily(
            _name("gpu_throttle_active"),
            "Active throttle reason (1 if active)",
            labels=["gpu", "name", "reason", "severity"],
        )
        for snap in snapshots:
            for reason in snap.metrics.throttle.active_reasons:
                throttle.add_metric(
                    [str(snap.gpu_index), snap.gpu_name, reason.name, reason.severity],
                    1.0,
                )
        yield throttle

        ecc = CounterMetricFamily(
            _name("gpu_ecc_errors"),
            "Cumulative ECC error counts",
            labels=["gpu", "name", "severity"],
        )
        for snap in snapshots:
            ecc.add_metric(
                [str(snap.gpu_index), snap.gpu_name, "correctable"],
                snap.metrics.ecc.correctable,
            )
            ecc.add_metric(
                [str(snap.gpu_index), snap.gpu_name, "uncorrectable"],
                snap.metrics.ecc.uncorrectable,
            )
        yield ecc

        pcie_gen = GaugeMetricFamily(
            _name("gpu_pcie_link_gen"),
            "Current PCIe link generation",
            labels=["gpu", "name"],
        )
        pcie_width = GaugeMetricFamily(
            _name("gpu_pcie_link_width"),
            "Current PCIe link width (lanes)",
            labels=["gpu", "name"],
        )
        for snap in snapshots:
            if snap.metrics.pcie.max_link_gen > 0:
                pcie_gen.add_metric(
                    [str(snap.gpu_index), snap.gpu_name],
                    snap.metrics.pcie.current_link_gen,
                )
                pcie_width.add_metric(
                    [str(snap.gpu_index), snap.gpu_name],
                    snap.metrics.pcie.current_link_width,
                )
        yield pcie_gen
        yield pcie_width

        procs = GaugeMetricFamily(
            _name("gpu_process_count"),
            "Number of processes using this GPU",
            labels=["gpu", "name"],
        )
        for snap in snapshots:
            procs.add_metric(
                [str(snap.gpu_index), snap.gpu_name],
                len(snap.metrics.processes),
            )
        yield procs

        scrape_duration = GaugeMetricFamily(
            _name("exporter_scrape_duration_seconds"),
            "Duration of the last background collection cycle",
        )
        scrape_duration.add_metric([], self._cache.last_scrape_duration_s)
        yield scrape_duration

        scrape_errors = CounterMetricFamily(
            _name("exporter_scrape_errors"),
            "Total background collection errors since startup",
        )
        scrape_errors.add_metric([], self._cache.scrape_errors)
        yield scrape_errors

    @staticmethod
    def _gauge(snapshots, suffix, doc, extractor):
        family = GaugeMetricFamily(_name(suffix), doc, labels=["gpu", "name"])
        for snap in snapshots:
            value = extractor(snap)
            if value is not None:
                family.add_metric([str(snap.gpu_index), snap.gpu_name], float(value))
        yield family


def _name(suffix: str) -> str:
    return f"{NAMESPACE}_{suffix}"


def _ratio(value: float | int | None) -> float | None:
    """Convert a 0-100 percentage to a 0-1 ratio, preserving None."""
    if value is None:
        return None
    return value / 100.0
