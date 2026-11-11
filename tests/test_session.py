"""Tests for the Python session-monitoring API."""

import time

import pytest

from nvsonar import session as session_module
from nvsonar.analysis import classify
from nvsonar.analysis.temporal import TemporalAnalyzer
from nvsonar.session import GPUSnapshot, Session, print_summary, start, stop

from .conftest import HW_THERMAL_SLOWDOWN, make_metrics, make_throttle


def snapshot(metrics=None, timestamp: float | None = None) -> GPUSnapshot:
    metrics = metrics if metrics is not None else make_metrics()
    return GPUSnapshot(
        timestamp=timestamp if timestamp is not None else time.time(),
        metrics=metrics,
        bottleneck=classify(metrics),
    )


def loaded_session(snapshots_by_gpu: dict[int, list[GPUSnapshot]], names=None) -> Session:
    """A Session primed with snapshots, bypassing NVML collection."""
    s = Session()
    s._gpu_indices = sorted(snapshots_by_gpu)
    s._snapshots = snapshots_by_gpu
    s._temporals = {i: TemporalAnalyzer() for i in snapshots_by_gpu}
    s._gpu_names = names or {i: "NVIDIA A30" for i in snapshots_by_gpu}
    return s


class FakeCollector:
    """Stands in for MetricsCollector, ending the loop after n samples."""

    def __init__(self, stop_event, samples: int, metrics=None, raises=False):
        self._stop_event = stop_event
        self._remaining = samples
        self._metrics = metrics if metrics is not None else make_metrics()
        self._raises = raises
        self.calls = 0

    def collect(self):
        self.calls += 1
        self._remaining -= 1
        if self._remaining <= 0:
            self._stop_event.set()
        if self._raises:
            raise RuntimeError("NVML read failed")
        return self._metrics


# --- analysis ---


def test_summary_reports_sample_count_and_duration():
    result = loaded_session({0: [snapshot() for _ in range(4)]})._analyze(12.0)
    assert result.duration_seconds == 12.0
    summary = result.gpu_summaries[0]
    assert summary.total_samples == 4
    assert summary.gpu_name == "NVIDIA A30"


def test_dominant_bottleneck_excludes_idle_time():
    busy = snapshot(make_metrics(gpu_util=95, mem_util=20))
    idle = snapshot(make_metrics(gpu_util=0, mem_util=0))
    summary = loaded_session({0: [idle] * 3 + [busy]})._analyze(10.0).gpu_summaries[0]
    assert summary.dominant_bottleneck == "compute_bound"
    assert summary.idle_pct == 75.0


def test_fully_idle_session_reports_idle_as_dominant():
    idle = snapshot(make_metrics(gpu_util=0, mem_util=0))
    summary = loaded_session({0: [idle] * 5})._analyze(10.0).gpu_summaries[0]
    assert summary.dominant_bottleneck == "idle"
    assert summary.dominant_bottleneck_pct == 100.0


def test_power_limited_time_counts_as_throttled():
    thermal = snapshot(
        make_metrics(
            gpu_util=90, mem_util=40, temperature=95.0, throttle=make_throttle(HW_THERMAL_SLOWDOWN)
        )
    )
    power = snapshot(make_metrics(gpu_util=90, mem_util=40, power_usage=298.0, power_limit=300.0))
    summary = loaded_session({0: [thermal, power]})._analyze(10.0).gpu_summaries[0]
    assert summary.throttled_pct == 100.0


def test_peaks_and_averages_computed_across_samples():
    snaps = [
        snapshot(make_metrics(gpu_util=20, mem_util=10, temperature=50.0, power_usage=100.0)),
        snapshot(make_metrics(gpu_util=100, mem_util=50, temperature=80.0, power_usage=250.0)),
    ]
    summary = loaded_session({0: snaps})._analyze(10.0).gpu_summaries[0]
    assert summary.peak_temperature == 80.0
    assert summary.peak_gpu_utilization == 100
    assert summary.peak_power_usage == 250.0
    assert summary.avg_gpu_utilization == 60.0
    assert summary.avg_temperature == 65.0


def test_unavailable_metrics_summarize_as_none_instead_of_crashing():
    """A GPU that never reports temperature or power must still produce a summary."""
    metrics = make_metrics(temperature=None, power_usage=None, power_limit=None)
    summary = loaded_session({0: [snapshot(metrics)] * 3})._analyze(10.0).gpu_summaries[0]
    assert summary.peak_temperature is None
    assert summary.avg_temperature is None
    assert summary.peak_power_usage is None
    assert summary.total_samples == 3


def test_averages_ignore_missing_samples_rather_than_diluting_them():
    snaps = [
        snapshot(make_metrics(temperature=None)),
        snapshot(make_metrics(temperature=80.0)),
    ]
    summary = loaded_session({0: snaps})._analyze(10.0).gpu_summaries[0]
    assert summary.avg_temperature == 80.0


def test_gpus_without_samples_are_omitted_from_summaries():
    result = loaded_session({0: [snapshot()], 1: []})._analyze(10.0)
    assert [s.gpu_index for s in result.gpu_summaries] == [0]


def test_outliers_need_more_than_one_gpu():
    hot = snapshot(make_metrics(gpu_util=90, temperature=95.0))
    assert loaded_session({0: [hot]})._analyze(10.0).outlier_warnings == []


def test_one_hot_gpu_flagged_against_a_cool_group():
    hot = snapshot(make_metrics(gpu_util=90, temperature=95.0))
    cool = snapshot(make_metrics(gpu_util=90, temperature=60.0))
    result = loaded_session({0: [hot], **{i: [cool] for i in range(1, 6)}})._analyze(10.0)
    assert any("GPU 0 temperature" in w for w in result.outlier_warnings)


def test_uniform_group_produces_no_outlier_warnings():
    cool = snapshot(make_metrics(gpu_util=90, temperature=60.0))
    result = loaded_session({i: [cool] for i in range(6)})._analyze(10.0)
    assert result.outlier_warnings == []


def test_recommendations_derived_from_the_final_state():
    metrics = make_metrics(
        gpu_util=90, mem_util=40, temperature=95.0, throttle=make_throttle(HW_THERMAL_SLOWDOWN)
    )
    summary = loaded_session({0: [snapshot(metrics)]})._analyze(10.0).gpu_summaries[0]
    assert summary.recommendations
    assert all(r.startswith("[P1]") or r.startswith("[P2]") for r in summary.recommendations)


# --- collection loop ---


def test_collect_loop_records_a_snapshot_per_sample():
    s = Session(interval=0)
    s._gpu_indices = [0]
    s._snapshots = {0: []}
    s._temporals = {0: TemporalAnalyzer()}
    s._collectors = {0: FakeCollector(s._stop_event, samples=3)}

    s._collect_loop()
    assert len(s._snapshots[0]) == 3


def test_collect_loop_survives_a_failing_read():
    s = Session(interval=0)
    s._gpu_indices = [0]
    s._snapshots = {0: []}
    s._temporals = {0: TemporalAnalyzer()}
    collector = FakeCollector(s._stop_event, samples=2, raises=True)
    s._collectors = {0: collector}

    s._collect_loop()
    assert collector.calls == 2
    assert s._snapshots[0] == []


def test_stop_before_start_returns_an_empty_result():
    result = Session().stop()
    assert result.gpu_summaries == []
    assert result.duration_seconds == 0


def test_stop_analyzes_collected_snapshots():
    s = loaded_session({0: [snapshot() for _ in range(3)]})
    s._start_time = time.time() - 5
    result = s.stop()
    assert result.gpu_summaries[0].total_samples == 3
    assert result.duration_seconds >= 5


# --- module-level API ---


def test_stop_without_an_active_session_raises():
    with pytest.raises(RuntimeError, match="No active monitoring session"):
        stop()


def test_starting_twice_raises(monkeypatch):
    monkeypatch.setattr(session_module, "_active_session", Session())
    with pytest.raises(RuntimeError, match="already active"):
        start()


def test_module_stop_returns_the_result_and_clears_the_session(monkeypatch):
    s = loaded_session({0: [snapshot()]})
    s._start_time = time.time()
    monkeypatch.setattr(session_module, "_active_session", s)

    result = stop()
    assert result.gpu_summaries[0].total_samples == 1
    assert session_module._active_session is None


# --- terminal summary ---


def test_print_summary_reports_stats_patterns_and_outliers(capsys):
    metrics = make_metrics(
        gpu_util=90, mem_util=40, temperature=95.0, throttle=make_throttle(HW_THERMAL_SLOWDOWN)
    )
    hot = snapshot(metrics)
    cool = snapshot(make_metrics(gpu_util=90, temperature=60.0))
    result = loaded_session({0: [hot] * 2, **{i: [cool] for i in range(1, 6)}})._analyze(120.0)

    print_summary(result)
    out = capsys.readouterr().out
    assert "GPU 0: NVIDIA A30" in out
    assert "2.0 min" in out
    assert "Peak temperature:" in out
    assert "Dominant bottleneck: thermal_throttled" in out
    assert "Recommendations:" in out
    assert "Multi-GPU outliers:" in out


def test_print_summary_shows_na_for_unavailable_metrics(capsys):
    metrics = make_metrics(temperature=None, power_usage=None, power_limit=None)
    result = loaded_session({0: [snapshot(metrics)]})._analyze(60.0)

    print_summary(result)
    out = capsys.readouterr().out
    assert "Peak temperature:" in out
    assert "N/A" in out
    assert "Peak power:" not in out
