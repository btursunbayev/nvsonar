"""Tests for flat CSV report rows."""

import csv
import io

from nvsonar.analysis import classify
from nvsonar.report.csv_report import CSV_FIELDS, report_to_csv_row, to_csv

from .conftest import (
    HW_THERMAL_SLOWDOWN,
    make_ecc,
    make_gpu_info,
    make_metrics,
    make_pcie,
    make_process,
    make_throttle,
)


def _row(metrics=None, gpu_index=0):
    metrics = metrics if metrics is not None else make_metrics()
    return report_to_csv_row(make_gpu_info(index=gpu_index), metrics, classify(metrics))


def test_row_covers_every_declared_field():
    assert set(_row()) == set(CSV_FIELDS)


def test_row_flattens_identity_and_metrics():
    row = _row(make_metrics(gpu_util=95, mem_util=20, temperature=72.0))
    assert row["gpu_index"] == 0
    assert row["gpu_name"] == "NVIDIA A30"
    assert row["gpu_utilization"] == 95
    assert row["temperature_c"] == 72.0
    assert row["bottleneck"] == "compute_bound"


def test_pcie_collapsed_into_readable_strings():
    row = _row(make_metrics(pcie=make_pcie(degraded=True)))
    assert row["pcie_gen"] == "2/4"
    assert row["pcie_width"] == "x16/x16"
    assert row["pcie_degraded"] is True


def test_throttle_flattened_to_flag_and_severity():
    row = _row(make_metrics(throttle=make_throttle(HW_THERMAL_SLOWDOWN)))
    assert row["is_throttled"] is True
    assert row["throttle_severity"] == "critical"


def test_ecc_counts_carried_through():
    row = _row(make_metrics(ecc=make_ecc(correctable=2, uncorrectable=1)))
    assert row["ecc_correctable"] == 2
    assert row["ecc_uncorrectable"] == 1


def test_processes_collapsed_to_count_and_pid_list():
    metrics = make_metrics(processes=[make_process(pid=11), make_process(pid=22)])
    row = _row(metrics)
    assert row["process_count"] == 2
    assert row["process_pids"] == "11,22"


def test_no_processes_yields_empty_pid_string():
    row = _row()
    assert row["process_count"] == 0
    assert row["process_pids"] == ""


def test_to_csv_writes_header_then_one_line_per_gpu():
    rows = [_row(gpu_index=0), _row(gpu_index=1)]
    parsed = list(csv.DictReader(io.StringIO(to_csv(rows))))
    assert len(parsed) == 2
    assert parsed[0]["gpu_index"] == "0"
    assert parsed[1]["gpu_index"] == "1"
    assert to_csv(rows).splitlines()[0] == ",".join(CSV_FIELDS)


def test_to_csv_ignores_extra_keys_instead_of_raising():
    row = _row()
    row["unexpected_column"] = "value"
    assert "unexpected_column" not in to_csv([row])


def test_to_csv_with_no_rows_still_writes_header():
    assert to_csv([]).strip() == ",".join(CSV_FIELDS)
