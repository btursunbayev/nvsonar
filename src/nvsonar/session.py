"""GPU monitoring session"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from nvsonar.monitor import (
    initialize,
    get_device_count,
    get_gpu_info,
    MetricsCollector,
    Metrics,
)
from nvsonar.analysis import classify, TemporalAnalyzer, detect_outliers, recommend
from nvsonar.analysis.bottleneck import BottleneckResult


@dataclass
class GPUSnapshot:
    """Single timestamped metrics snapshot"""

    timestamp: float
    metrics: Metrics
    bottleneck: BottleneckResult


@dataclass
class GPUSummary:
    """Summary of a GPU's behavior during a session"""

    gpu_index: int
    gpu_name: str
    duration_seconds: float
    total_samples: int

    # time distribution
    idle_pct: float
    throttled_pct: float
    data_starved_pct: float

    # peaks
    peak_temperature: float
    peak_gpu_utilization: int
    peak_memory_used_pct: float
    peak_power_usage: float | None

    # averages
    avg_gpu_utilization: float
    avg_memory_utilization: float
    avg_temperature: float

    # dominant bottleneck
    dominant_bottleneck: str
    dominant_bottleneck_pct: float

    # patterns detected
    patterns: list[str]

    # recommendations
    recommendations: list[str]


@dataclass
class SessionResult:
    """Full session monitoring result"""

    duration_seconds: float
    gpu_summaries: list[GPUSummary]
    outlier_warnings: list[str]


class Session:
    """Monitors GPUs in a background thread during a workload"""

    def __init__(self, interval: float = 0.5, gpu_indices: list[int] | None = None):
        self._interval = interval
        self._gpu_indices = gpu_indices
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time: float = 0
        self._snapshots: dict[int, list[GPUSnapshot]] = {}
        self._temporals: dict[int, TemporalAnalyzer] = {}
        self._collectors: dict[int, MetricsCollector] = {}
        self._gpu_names: dict[int, str] = {}

    def start(self):
        """Start background monitoring"""
        if not initialize():
            raise RuntimeError("Failed to initialize NVML, no NVIDIA GPU found")

        device_count = get_device_count()
        if device_count == 0:
            raise RuntimeError("No GPUs detected")

        if self._gpu_indices is None:
            self._gpu_indices = list(range(device_count))

        for i in self._gpu_indices:
            self._collectors[i] = MetricsCollector(i)
            self._temporals[i] = TemporalAnalyzer(window_size=120)
            self._snapshots[i] = []

            info = get_gpu_info(i)
            self._gpu_names[i] = info.name if info else f"GPU {i}"

        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> SessionResult:
        """Stop monitoring and return analysis"""
        if self._gpu_indices is None:
            return SessionResult(duration_seconds=0, gpu_summaries=[], outlier_warnings=[])

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        duration = time.time() - self._start_time
        return self._analyze(duration)

    def _collect_loop(self):
        """Background collection loop"""
        while not self._stop_event.is_set():
            now = time.time()

            for i, collector in self._collectors.items():
                try:
                    metrics = collector.collect()
                    bottleneck = classify(metrics)
                    self._temporals[i].update(metrics)
                    self._snapshots[i].append(GPUSnapshot(
                        timestamp=now,
                        metrics=metrics,
                        bottleneck=bottleneck,
                    ))
                except Exception:
                    pass

            self._stop_event.wait(self._interval)

    def _analyze(self, duration: float) -> SessionResult:
        """Analyze collected data and build summary"""
        summaries = []

        # collect final metrics for outlier detection
        latest_metrics = {}
        for i, snaps in self._snapshots.items():
            if snaps:
                latest_metrics[i] = snaps[-1].metrics

        outliers = []
        if len(latest_metrics) > 1:
            outlier_results = detect_outliers(latest_metrics)
            for o in outlier_results:
                outliers.append(
                    f"GPU {o.gpu_index} {o.metric}: {o.detail}"
                )

        for i in self._gpu_indices:
            snaps = self._snapshots.get(i, [])
            if not snaps:
                continue

            total = len(snaps)

            # count bottleneck distribution
            bottleneck_counts: dict[str, int] = {}
            idle_count = 0
            throttled_count = 0
            starved_count = 0

            for s in snaps:
                btype = s.bottleneck.bottleneck.value
                bottleneck_counts[btype] = bottleneck_counts.get(btype, 0) + 1

                if btype == "idle":
                    idle_count += 1
                elif btype == "thermal_throttled":
                    throttled_count += 1
                elif btype == "data_starved":
                    starved_count += 1

            # also count power_limited as throttled
            throttled_count += bottleneck_counts.get("power_limited", 0)

            # dominant bottleneck (excluding idle)
            non_idle = {k: v for k, v in bottleneck_counts.items() if k != "idle"}
            if non_idle:
                dominant = max(non_idle, key=non_idle.get)
                dominant_pct = (non_idle[dominant] / total) * 100
            else:
                dominant = "idle"
                dominant_pct = 100.0

            # peaks and averages
            temps = [s.metrics.temperature for s in snaps]
            gpu_utils = [s.metrics.gpu_utilization for s in snaps]
            mem_utils = [s.metrics.memory_utilization for s in snaps]
            mem_used_pcts = [s.metrics.memory_used_pct for s in snaps]
            powers = [s.metrics.power_usage for s in snaps if s.metrics.power_usage is not None]

            # temporal patterns
            patterns = self._temporals[i].detect()
            pattern_strs = [f"[{p.severity}] {p.detail}" for p in patterns]

            # recommendations from dominant state
            last_bottleneck = snaps[-1].bottleneck
            recs = recommend(bottleneck=last_bottleneck, patterns=patterns)
            rec_strs = [f"[P{r.priority}] {r.title}" for r in recs if r.priority <= 2]

            summaries.append(GPUSummary(
                gpu_index=i,
                gpu_name=self._gpu_names.get(i, f"GPU {i}"),
                duration_seconds=duration,
                total_samples=total,
                idle_pct=(idle_count / total) * 100,
                throttled_pct=(throttled_count / total) * 100,
                data_starved_pct=(starved_count / total) * 100,
                peak_temperature=max(temps),
                peak_gpu_utilization=max(gpu_utils),
                peak_memory_used_pct=max(mem_used_pcts),
                peak_power_usage=max(powers) if powers else None,
                avg_gpu_utilization=sum(gpu_utils) / total,
                avg_memory_utilization=sum(mem_utils) / total,
                avg_temperature=sum(temps) / total,
                dominant_bottleneck=dominant,
                dominant_bottleneck_pct=dominant_pct,
                patterns=pattern_strs,
                recommendations=rec_strs,
            ))

        return SessionResult(
            duration_seconds=duration,
            gpu_summaries=summaries,
            outlier_warnings=outliers,
        )


# module-level session for simple API
_active_session: Session | None = None


def start(interval: float = 0.5, gpus: list[int] | None = None):
    """Start monitoring GPUs in the background"""
    global _active_session
    if _active_session is not None:
        raise RuntimeError("Monitoring already active, call stop() first")
    _active_session = Session(interval=interval, gpu_indices=gpus)
    _active_session.start()


def stop() -> SessionResult:
    """Stop monitoring and return session results"""
    global _active_session
    if _active_session is None:
        raise RuntimeError("No active monitoring session")
    result = _active_session.stop()
    _active_session = None
    return result


@contextmanager
def monitor(interval: float = 0.5, gpus: list[int] | None = None):
    """Context manager for GPU monitoring"""
    start(interval=interval, gpus=gpus)
    try:
        yield
    finally:
        result = stop()
        print_summary(result)


def print_summary(result: SessionResult):
    """Print session summary to terminal"""
    minutes = result.duration_seconds / 60

    for s in result.gpu_summaries:
        print(f"\n{'=' * 60}")
        print(f"GPU {s.gpu_index}: {s.gpu_name}")
        print(f"Session: {minutes:.1f} min, {s.total_samples} samples")
        print(f"{'=' * 60}")

        print(f"\n  Avg GPU utilization:    {s.avg_gpu_utilization:.0f}%")
        print(f"  Avg memory controller:  {s.avg_memory_utilization:.0f}%")
        print(f"  Avg temperature:        {s.avg_temperature:.0f}C")
        print(f"  Peak temperature:       {s.peak_temperature:.0f}C")
        print(f"  Peak VRAM:              {s.peak_memory_used_pct:.0f}%")
        if s.peak_power_usage is not None:
            print(f"  Peak power:             {s.peak_power_usage:.0f}W")

        print(f"\n  Time distribution:")
        print(f"    Idle:          {s.idle_pct:.1f}%")
        print(f"    Throttled:     {s.throttled_pct:.1f}%")
        print(f"    Data starved:  {s.data_starved_pct:.1f}%")

        if s.dominant_bottleneck != "idle":
            print(f"\n  Dominant bottleneck: {s.dominant_bottleneck} ({s.dominant_bottleneck_pct:.0f}% of session)")

        if s.patterns:
            print(f"\n  Patterns detected:")
            for p in s.patterns:
                print(f"    {p}")

        if s.recommendations:
            print(f"\n  Recommendations:")
            for r in s.recommendations:
                print(f"    {r}")

    if result.outlier_warnings:
        print(f"\n{'=' * 60}")
        print("Multi-GPU outliers:")
        for w in result.outlier_warnings:
            print(f"  {w}")

    print()
