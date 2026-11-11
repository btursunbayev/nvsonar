"""Tests for history persistence and trend detection."""

import json
import time

import pytest

from nvsonar import history
from nvsonar.analysis import classify
from nvsonar.history import (
    HistoryEntry,
    analyze_trends,
    load,
    print_history,
    save,
    save_from_metrics,
)

from .conftest import HW_THERMAL_SLOWDOWN, make_ecc, make_metrics, make_pcie, make_throttle

DAY = 86400


@pytest.fixture
def history_dir(tmp_path, monkeypatch):
    """Redirect history storage away from the user's home directory."""
    path = tmp_path / "history"
    monkeypatch.setattr(history, "HISTORY_DIR", path)
    return path


def make_entry(
    *,
    timestamp: float | None = None,
    gpu_index: int = 0,
    temperature: float = 60.0,
    gpu_clock: int = 1440,
    gpu_utilization: int = 80,
    ecc_correctable: int = 0,
    ecc_uncorrectable: int = 0,
    throttled: bool = False,
) -> HistoryEntry:
    return HistoryEntry(
        timestamp=timestamp if timestamp is not None else time.time(),
        gpu_index=gpu_index,
        gpu_name="NVIDIA A30",
        gpu_utilization=gpu_utilization,
        memory_utilization=40,
        memory_used_pct=50.0,
        temperature=temperature,
        power_usage=200.0,
        gpu_clock=gpu_clock,
        max_gpu_clock=1440,
        clock_reduction_pct=0.0,
        bottleneck="compute_bound",
        ecc_correctable=ecc_correctable,
        ecc_uncorrectable=ecc_uncorrectable,
        pcie_degraded=False,
        throttled=throttled,
    )


def series(count: int, *, minutes_ago: int | None = None, **kwargs) -> list[HistoryEntry]:
    """A run of entries one minute apart, oldest first, ending `minutes_ago` before now."""
    now = time.time()
    start = count if minutes_ago is None else minutes_ago + count
    return [make_entry(timestamp=now - (start - i) * 60, **kwargs) for i in range(count)]


# --- persistence ---


def test_save_then_load_round_trips_an_entry(history_dir):
    entry = make_entry(temperature=71.5)
    save(entry)
    loaded = load()
    assert len(loaded) == 1
    assert loaded[0] == entry


def test_load_returns_empty_when_no_history_exists(history_dir):
    assert load() == []


def test_entries_are_split_into_one_file_per_day_per_gpu(history_dir):
    now = time.time()
    save(make_entry(gpu_index=0, timestamp=now))
    save(make_entry(gpu_index=1, timestamp=now))
    save(make_entry(gpu_index=0, timestamp=now - DAY))
    assert len(list(history_dir.glob("gpu*.jsonl"))) == 3


def test_load_filters_by_gpu_index(history_dir):
    save(make_entry(gpu_index=0))
    save(make_entry(gpu_index=1))
    assert [e.gpu_index for e in load(gpu_index=1)] == [1]


def test_load_drops_entries_older_than_the_cutoff(history_dir):
    save(make_entry(timestamp=time.time() - 10 * DAY))
    save(make_entry(timestamp=time.time()))
    assert len(load(days=7)) == 1


def test_load_returns_entries_in_timestamp_order(history_dir):
    now = time.time()
    save(make_entry(timestamp=now))
    save(make_entry(timestamp=now - 100))
    timestamps = [e.timestamp for e in load()]
    assert timestamps == sorted(timestamps)


def test_corrupt_lines_are_skipped_not_fatal(history_dir):
    save(make_entry())
    path = next(history_dir.glob("gpu*.jsonl"))
    with open(path, "a") as f:
        f.write("not json\n")
        f.write("\n")
        f.write(json.dumps({"timestamp": time.time(), "unexpected": True}) + "\n")
    assert len(load()) == 1


def test_save_from_metrics_derives_entry_from_a_snapshot(history_dir):
    metrics = make_metrics(
        gpu_util=95,
        mem_util=20,
        temperature=72.0,
        ecc=make_ecc(correctable=2),
        pcie=make_pcie(degraded=True),
        throttle=make_throttle(HW_THERMAL_SLOWDOWN),
    )
    save_from_metrics(3, "NVIDIA A30", metrics, classify(metrics))

    entry = load()[0]
    assert entry.gpu_index == 3
    assert entry.temperature == 72.0
    assert entry.ecc_correctable == 2
    assert entry.pcie_degraded is True
    assert entry.throttled is True


# --- trend analysis ---


def test_no_trends_reported_below_ten_samples():
    assert analyze_trends(series(9)) == []


def test_stable_history_produces_no_trends():
    assert analyze_trends(series(20)) == []


def test_rising_temperature_detected():
    entries = series(10, temperature=60.0) + series(10, temperature=75.0)
    trends = {t.metric: t for t in analyze_trends(entries)}
    assert trends["temperature"].direction == "rising"
    assert "15.0C" in trends["temperature"].detail


def test_falling_temperature_detected():
    entries = series(10, temperature=80.0) + series(10, temperature=60.0)
    trends = {t.metric: t for t in analyze_trends(entries)}
    assert trends["temperature"].direction == "falling"


def test_small_temperature_drift_is_not_a_trend():
    entries = series(10, temperature=60.0) + series(10, temperature=62.0)
    assert "temperature" not in {t.metric for t in analyze_trends(entries)}


def test_growing_ecc_error_count_flagged_as_degradation():
    entries = series(10, ecc_correctable=0) + series(10, ecc_correctable=4)
    trends = {t.metric: t for t in analyze_trends(entries)}
    assert trends["ecc_errors"].direction == "rising"
    assert "degradation" in trends["ecc_errors"].detail


def test_more_frequent_throttling_flagged():
    entries = series(10, throttled=False) + series(10, throttled=True)
    assert "throttling" in {t.metric for t in analyze_trends(entries)}


def test_clock_degradation_measured_only_under_load():
    loaded = series(10, gpu_clock=1440, gpu_utilization=80) + series(
        10, gpu_clock=1200, gpu_utilization=80
    )
    assert "clock_speed" in {t.metric for t in analyze_trends(loaded)}

    idle = series(10, gpu_clock=1440, gpu_utilization=5) + series(
        10, gpu_clock=1200, gpu_utilization=5
    )
    assert "clock_speed" not in {t.metric for t in analyze_trends(idle)}


def test_rising_utilization_detected():
    entries = series(10, gpu_utilization=10) + series(10, gpu_utilization=90)
    trends = {t.metric: t for t in analyze_trends(entries)}
    assert trends["utilization"].direction == "rising"


def test_entries_missing_temperature_do_not_break_analysis():
    entries = series(20)
    for e in entries:
        e.temperature = None
    assert "temperature" not in {t.metric for t in analyze_trends(entries)}


# --- terminal output ---


def test_print_history_reports_when_nothing_recorded(history_dir, capsys):
    print_history()
    assert "No history data found" in capsys.readouterr().out


def test_print_history_summarizes_each_gpu_with_trends(history_dir, capsys):
    older = series(10, minutes_ago=10, temperature=60.0)
    newer = series(10, temperature=80.0, throttled=True)
    for entry in older + newer:
        save(entry)
    save(make_entry(gpu_index=1, ecc_correctable=3))

    print_history()
    out = capsys.readouterr().out
    assert "GPU 0: NVIDIA A30" in out
    assert "GPU 1: NVIDIA A30" in out
    assert "Samples" in out
    assert "Trends:" in out
    assert "ECC errors" in out
    assert "Throttled" in out
