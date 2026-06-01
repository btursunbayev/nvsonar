"""Tests for multi-GPU outlier detection."""

from nvsonar.analysis.outlier import detect_outliers

from .conftest import make_ecc, make_metrics


def test_no_outliers_with_single_gpu():
    assert detect_outliers({0: make_metrics()}) == []


def test_no_outliers_when_all_gpus_identical():
    gpus = {i: make_metrics(temperature=60.0) for i in range(4)}
    assert detect_outliers(gpus) == []


def test_temperature_outlier_flagged_when_one_gpu_hotter():
    gpus = {i: make_metrics(temperature=60.0) for i in range(5)}
    gpus[5] = make_metrics(temperature=95.0)
    outliers = detect_outliers(gpus)
    assert any(o.gpu_index == 5 and o.metric == "temperature" for o in outliers)


def test_gpu_utilization_outlier_flagged_when_one_straggler():
    gpus = {i: make_metrics(gpu_util=92) for i in range(5)}
    gpus[5] = make_metrics(gpu_util=30)
    outliers = detect_outliers(gpus)
    assert any(o.gpu_index == 5 and o.metric == "gpu_utilization" for o in outliers)


def test_clock_outlier_flagged_when_one_throttled():
    gpus = {i: make_metrics(gpu_clock=1500) for i in range(5)}
    gpus[5] = make_metrics(gpu_clock=600)
    outliers = detect_outliers(gpus)
    assert any(o.gpu_index == 5 and o.metric == "gpu_clock" for o in outliers)


def test_ecc_outlier_flagged_when_one_gpu_has_uncorrectable():
    gpus = {
        0: make_metrics(ecc=make_ecc()),
        1: make_metrics(ecc=make_ecc()),
        2: make_metrics(ecc=make_ecc(uncorrectable=5)),
    }
    outliers = detect_outliers(gpus)
    ecc_outliers = [o for o in outliers if o.metric == "ecc_errors"]
    assert len(ecc_outliers) == 1
    assert ecc_outliers[0].gpu_index == 2
    assert ecc_outliers[0].severity == "critical"


def test_outliers_sorted_by_severity():
    gpus = {
        0: make_metrics(gpu_util=92, temperature=60.0),
        1: make_metrics(gpu_util=90, temperature=61.0),
        2: make_metrics(gpu_util=20, temperature=92.0, ecc=make_ecc(uncorrectable=2)),
    }
    outliers = detect_outliers(gpus)
    severities = [o.severity for o in outliers]
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    assert severities == sorted(severities, key=lambda s: severity_order.get(s, 99))
