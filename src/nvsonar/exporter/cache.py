"""Background snapshot cache for the Prometheus exporter"""

import threading
import time
from dataclasses import dataclass

from nvsonar.analysis import classify
from nvsonar.analysis.bottleneck import BottleneckResult
from nvsonar.analysis.health import health_score
from nvsonar.monitor import Metrics, MetricsCollector, get_gpu_info


@dataclass
class Snapshot:
    gpu_index: int
    gpu_name: str
    metrics: Metrics
    bottleneck: BottleneckResult
    health_score: int
    timestamp: float


class SnapshotCache:
    """Latest snapshot per GPU. A background thread refreshes it so a Prometheus scrape never blocks on NVML."""

    def __init__(self, gpu_indices: list[int], interval_s: float = 2.0):
        self._gpu_indices = list(gpu_indices)
        self._interval = interval_s
        self._collectors: dict[int, MetricsCollector] = {}
        self._names: dict[int, str] = {}
        self._snapshots: dict[int, Snapshot] = {}
        self._scrape_errors = 0
        self._last_scrape_duration = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        for i in self._gpu_indices:
            self._collectors[i] = MetricsCollector(i)
            info = get_gpu_info(i)
            self._names[i] = info.name if info else f"GPU{i}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="nvsonar-exporter")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._interval + 1)
            self._thread = None

    def all(self) -> list[Snapshot]:
        with self._lock:
            return list(self._snapshots.values())

    @property
    def scrape_errors(self) -> int:
        return self._scrape_errors

    @property
    def last_scrape_duration_s(self) -> float:
        return self._last_scrape_duration

    def _loop(self) -> None:
        while not self._stop.is_set():
            start = time.monotonic()
            for i in self._gpu_indices:
                try:
                    metrics = self._collectors[i].collect()
                    result = classify(metrics)
                    snap = Snapshot(
                        gpu_index=i,
                        gpu_name=self._names[i],
                        metrics=metrics,
                        bottleneck=result,
                        health_score=health_score(metrics, result),
                        timestamp=time.time(),
                    )
                    with self._lock:
                        self._snapshots[i] = snap
                except Exception:
                    self._scrape_errors += 1
            self._last_scrape_duration = time.monotonic() - start
            self._stop.wait(self._interval)
