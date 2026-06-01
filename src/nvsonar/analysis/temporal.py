"""Temporal pattern detection over GPU metrics history"""

import math
from collections import deque
from dataclasses import dataclass

from nvsonar.monitor import Metrics


@dataclass
class Pattern:
    """A detected temporal pattern"""

    name: str
    severity: str  # info, warning, critical
    detail: str


@dataclass
class _RollingStats:
    """Incremental mean/variance tracker (Welford's algorithm)"""

    n: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self._m2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def cv(self) -> float:
        """Coefficient of variation"""
        if self.mean == 0:
            return 0.0
        return self.std / abs(self.mean)


class TemporalAnalyzer:
    """Detects patterns across a sliding window of metrics snapshots"""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._clock_history: deque[int] = deque(maxlen=window_size)
        self._temp_history: deque[float] = deque(maxlen=window_size)
        self._gpu_util_history: deque[int] = deque(maxlen=window_size)
        self._mem_used_history: deque[float] = deque(maxlen=window_size)
        self._power_history: deque[float] = deque(maxlen=window_size)

        self._clock_stats = _RollingStats()
        self._sample_count = 0

        # throttle tracking
        self._throttle_history: deque[bool] = deque(maxlen=window_size)
        self._throttle_transitions = 0
        self._last_throttled: bool | None = None

    def update(self, metrics: Metrics):
        """Add a new metrics snapshot to the history"""
        if metrics.gpu_clock is not None:
            self._clock_history.append(metrics.gpu_clock)
            self._clock_stats.update(metrics.gpu_clock)
        if metrics.temperature is not None:
            self._temp_history.append(metrics.temperature)
        if metrics.gpu_utilization is not None:
            self._gpu_util_history.append(metrics.gpu_utilization)
        if metrics.memory_used_pct is not None:
            self._mem_used_history.append(metrics.memory_used_pct)
        if metrics.power_usage is not None:
            self._power_history.append(metrics.power_usage)

        # track throttle state transitions
        throttled = metrics.throttle.is_throttled
        self._throttle_history.append(throttled)
        if self._last_throttled is not None and throttled != self._last_throttled:
            self._throttle_transitions += 1
        self._last_throttled = throttled

        self._sample_count += 1

    @property
    def has_enough_data(self) -> bool:
        """Need at least 10 samples for meaningful pattern detection"""
        return self._sample_count >= 10

    def detect(self) -> list[Pattern]:
        """Run all pattern detectors and return findings"""
        if not self.has_enough_data:
            return []

        patterns = []

        osc = self._detect_clock_oscillation()
        if osc:
            patterns.append(osc)

        trend = self._detect_temperature_trend()
        if trend:
            patterns.append(trend)

        dips = self._detect_utilization_dips()
        if dips:
            patterns.append(dips)

        creep = self._detect_memory_creep()
        if creep:
            patterns.append(creep)

        throttle = self._detect_throttle_frequency()
        if throttle:
            patterns.append(throttle)

        return patterns

    def _detect_clock_oscillation(self) -> Pattern | None:
        """Detect GPU clock bouncing between throttled and boost"""
        if len(self._clock_history) < 10:
            return None

        values = list(self._clock_history)
        mean = sum(values) / len(values)
        if mean == 0:
            return None

        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        cv = std / mean

        if cv < 0.10:
            return None

        # confirm with zero-crossing rate
        midpoint = (min(values) + max(values)) / 2
        crossings = sum(
            1 for i in range(1, len(values)) if (values[i] > midpoint) != (values[i - 1] > midpoint)
        )
        crossing_rate = crossings / len(values)

        if crossing_rate < 0.2:
            return None

        spread = max(values) - min(values)
        severity = "warning" if cv < 0.20 else "critical"
        return Pattern(
            "clock_oscillation",
            severity,
            f"GPU clock oscillating between {min(values)}-{max(values)} MHz "
            f"(spread {spread} MHz, {crossings} transitions in last "
            f"{len(values)} samples)",
        )

    def _detect_temperature_trend(self) -> Pattern | None:
        """Detect sustained temperature increase"""
        if len(self._temp_history) < 20:
            return None

        values = list(self._temp_history)
        n = len(values)

        # least squares slope
        sum_x = n * (n - 1) / 2
        sum_xx = n * (n - 1) * (2 * n - 1) / 6
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))

        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # slope is C per sample, at 1Hz slope of 0.05 = 3C/min
        if slope < 0.05:
            return None

        rate_per_min = slope * 60
        current = values[-1]

        if slope > 0.15:
            severity = "critical"
        elif slope > 0.08:
            severity = "warning"
        else:
            severity = "info"

        return Pattern(
            "temperature_rising",
            severity,
            f"Temperature rising at ~{rate_per_min:.1f}C/min " f"(currently {current:.0f}C)",
        )

    def _detect_utilization_dips(self) -> Pattern | None:
        """Detect periodic utilization drops (data loading bottleneck)"""
        if len(self._gpu_util_history) < 15:
            return None

        values = list(self._gpu_util_history)
        mean = sum(values) / len(values)

        if mean < 40:
            return None  # overall low usage, not a dip pattern

        dip_threshold = max(20, mean * 0.3)
        dips = sum(1 for v in values if v < dip_threshold)
        dip_ratio = dips / len(values)

        if dip_ratio < 0.10:
            return None

        return Pattern(
            "utilization_dips",
            "warning",
            f"GPU utilization dropping to <{dip_threshold:.0f}% in "
            f"{dips}/{len(values)} samples (avg {mean:.0f}%), "
            f"likely data loading or CPU bottleneck",
        )

    def _detect_memory_creep(self) -> Pattern | None:
        """Detect monotonically increasing VRAM usage (possible memory leak)"""
        if len(self._mem_used_history) < 20:
            return None

        values = list(self._mem_used_history)

        # check if memory increased by more than 5% over the window
        start_avg = sum(values[:5]) / 5
        end_avg = sum(values[-5:]) / 5
        growth = end_avg - start_avg

        if growth < 5.0:
            return None

        # confirm it's a trend not a step
        chunk = max(1, len(values) // 5)
        chunks = [
            sum(values[i : i + chunk]) / chunk for i in range(0, len(values) - chunk + 1, chunk)
        ]
        increases = sum(1 for i in range(1, len(chunks)) if chunks[i] > chunks[i - 1])

        if increases < len(chunks) - 2:
            return None  # not consistently increasing

        return Pattern(
            "memory_creep",
            "warning",
            f"VRAM usage grew {growth:.1f}% over last {len(values)} samples "
            f"({start_avg:.1f}% to {end_avg:.1f}%), possible memory leak",
        )

    def _detect_throttle_frequency(self) -> Pattern | None:
        """Detect intermittent throttling (on/off cycling)"""
        if len(self._throttle_history) < 15:
            return None

        values = list(self._throttle_history)
        throttled_count = sum(1 for v in values if v)
        total = len(values)

        # skip if never throttled or always throttled
        if throttled_count == 0 or throttled_count == total:
            return None

        # count transitions within the window
        transitions = sum(1 for i in range(1, total) if values[i] != values[i - 1])

        if transitions < 3:
            return None

        throttled_pct = (throttled_count / total) * 100
        severity = "critical" if transitions > 10 else "warning"

        return Pattern(
            "throttle_cycling",
            severity,
            f"Throttle triggered {transitions} times in last {total} samples "
            f"(throttled {throttled_pct:.0f}% of the time)",
        )
