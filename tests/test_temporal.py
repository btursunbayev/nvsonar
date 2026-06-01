"""Tests for temporal pattern detection."""

from nvsonar.analysis.temporal import TemporalAnalyzer

from .conftest import HW_THERMAL_SLOWDOWN, make_metrics, make_throttle


def _feed(analyzer, samples):
    for s in samples:
        analyzer.update(s)


def test_no_patterns_when_insufficient_data():
    analyzer = TemporalAnalyzer(window_size=60)
    _feed(analyzer, [make_metrics(gpu_clock=1500) for _ in range(5)])
    assert analyzer.detect() == []


def test_clock_oscillation_detected_when_clocks_bounce():
    analyzer = TemporalAnalyzer(window_size=60)
    # alternate between 600 and 1500 MHz
    samples = [make_metrics(gpu_clock=600 if i % 2 == 0 else 1500) for i in range(30)]
    _feed(analyzer, samples)
    patterns = analyzer.detect()
    assert any(p.name == "clock_oscillation" for p in patterns)


def test_no_clock_oscillation_when_clocks_stable():
    analyzer = TemporalAnalyzer(window_size=60)
    samples = [make_metrics(gpu_clock=1500) for _ in range(30)]
    _feed(analyzer, samples)
    patterns = analyzer.detect()
    assert not any(p.name == "clock_oscillation" for p in patterns)


def test_temperature_rising_detected_with_steady_increase():
    analyzer = TemporalAnalyzer(window_size=60)
    # 0.2C per sample => 12C/min at 1Hz => critical
    samples = [make_metrics(temperature=40.0 + 0.2 * i) for i in range(30)]
    _feed(analyzer, samples)
    patterns = analyzer.detect()
    assert any(p.name == "temperature_rising" for p in patterns)


def test_no_temperature_rising_when_flat():
    analyzer = TemporalAnalyzer(window_size=60)
    samples = [make_metrics(temperature=40.0) for _ in range(30)]
    _feed(analyzer, samples)
    patterns = analyzer.detect()
    assert not any(p.name == "temperature_rising" for p in patterns)


def test_utilization_dips_detected_when_periodic_drops():
    analyzer = TemporalAnalyzer(window_size=60)
    # high util most of the time but periodic near-zero dips
    samples = [make_metrics(gpu_util=5 if i % 5 == 0 else 90) for i in range(30)]
    _feed(analyzer, samples)
    patterns = analyzer.detect()
    assert any(p.name == "utilization_dips" for p in patterns)


def test_memory_creep_detected_when_vram_grows():
    analyzer = TemporalAnalyzer(window_size=60)
    total = 24 * 1024**3
    # grow from ~10% to ~30% over 30 samples
    samples = [
        make_metrics(
            memory_used=int((0.10 + 0.007 * i) * total),
            memory_total=total,
        )
        for i in range(30)
    ]
    _feed(analyzer, samples)
    patterns = analyzer.detect()
    assert any(p.name == "memory_creep" for p in patterns)


def test_throttle_cycling_detected_when_on_off_alternating():
    analyzer = TemporalAnalyzer(window_size=60)
    samples = []
    for i in range(30):
        throttle = make_throttle(HW_THERMAL_SLOWDOWN) if i % 3 == 0 else make_throttle()
        samples.append(make_metrics(throttle=throttle))
    _feed(analyzer, samples)
    patterns = analyzer.detect()
    assert any(p.name == "throttle_cycling" for p in patterns)
