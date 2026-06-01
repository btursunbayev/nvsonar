"""Tests for recommendation generation from analysis results."""

from nvsonar.analysis.bottleneck import BottleneckResult, BottleneckType
from nvsonar.analysis.outlier import Outlier
from nvsonar.analysis.recommendations import recommend
from nvsonar.analysis.temporal import Pattern


def _bottleneck(kind: BottleneckType, *, warnings=None) -> BottleneckResult:
    return BottleneckResult(
        bottleneck=kind,
        confidence=0.9,
        detail="test detail",
        warnings=warnings or [],
    )


def test_no_recommendations_when_no_inputs():
    assert recommend() == []


def test_thermal_throttle_yields_priority_one_recommendation():
    recs = recommend(bottleneck=_bottleneck(BottleneckType.THERMAL_THROTTLED))
    assert len(recs) == 1
    assert recs[0].priority == 1
    assert "thermal" in recs[0].title.lower()


def test_power_limited_yields_priority_one_recommendation():
    recs = recommend(bottleneck=_bottleneck(BottleneckType.POWER_LIMITED))
    assert any(r.priority == 1 and "power" in r.title.lower() for r in recs)


def test_data_starved_yields_dataloader_advice():
    recs = recommend(bottleneck=_bottleneck(BottleneckType.DATA_STARVED))
    actions = " ".join(a for r in recs for a in r.actions)
    assert "num_workers" in actions or "DataLoader" in actions


def test_compute_bound_yields_low_priority_recommendation():
    recs = recommend(bottleneck=_bottleneck(BottleneckType.COMPUTE_BOUND))
    assert recs and recs[0].priority == 3


def test_idle_yields_no_action_needed():
    recs = recommend(bottleneck=_bottleneck(BottleneckType.IDLE))
    assert recs and "no action" in recs[0].actions[0].lower()


def test_uncorrectable_ecc_warning_adds_hardware_recommendation():
    bottleneck = _bottleneck(
        BottleneckType.BALANCED,
        warnings=["5 uncorrectable ECC errors, hardware may need replacement"],
    )
    recs = recommend(bottleneck=bottleneck)
    assert any(r.priority == 1 and "memory errors" in r.title.lower() for r in recs)


def test_pattern_clock_oscillation_yields_recommendation():
    pattern = Pattern(name="clock_oscillation", severity="warning", detail="bouncing")
    recs = recommend(patterns=[pattern])
    assert any("unstable" in r.title.lower() for r in recs)


def test_pattern_memory_creep_yields_leak_advice():
    pattern = Pattern(name="memory_creep", severity="warning", detail="growing")
    recs = recommend(patterns=[pattern])
    assert any("vram" in r.title.lower() or "memory" in r.title.lower() for r in recs)


def test_critical_outlier_yields_recommendation():
    outlier = Outlier(
        gpu_index=2,
        metric="temperature",
        value=92.0,
        group_mean=60.0,
        group_std=2.0,
        z_score=16.0,
        detail="92C vs group avg 60C",
        severity="critical",
    )
    recs = recommend(outliers=[outlier])
    assert any("GPU 2" in r.title for r in recs)


def test_non_critical_outlier_skipped():
    outlier = Outlier(
        gpu_index=2,
        metric="temperature",
        value=70.0,
        group_mean=60.0,
        group_std=2.0,
        z_score=5.0,
        detail="70C vs group avg 60C",
        severity="warning",
    )
    recs = recommend(outliers=[outlier])
    assert recs == []


def test_recommendations_deduplicated_by_title():
    pattern_a = Pattern(name="memory_creep", severity="warning", detail="a")
    pattern_b = Pattern(name="memory_creep", severity="warning", detail="b")
    recs = recommend(patterns=[pattern_a, pattern_b])
    titles = [r.title for r in recs]
    assert len(titles) == len(set(titles))


def test_recommendations_sorted_by_priority():
    bottleneck = _bottleneck(BottleneckType.COMPUTE_BOUND)  # priority 3
    pattern = Pattern(name="clock_oscillation", severity="warning", detail="bouncing")  # priority 1
    recs = recommend(bottleneck=bottleneck, patterns=[pattern])
    priorities = [r.priority for r in recs]
    assert priorities == sorted(priorities)
