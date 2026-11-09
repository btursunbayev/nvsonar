"""Tests for terminal report card rendering."""

import pytest
from rich.console import Console

from nvsonar.analysis import classify
from nvsonar.analysis.recommendations import Recommendation
from nvsonar.analysis.temporal import Pattern
from nvsonar.report.card import _grade, print_report, print_report_plain

from .conftest import (
    HW_THERMAL_SLOWDOWN,
    make_ecc,
    make_gpu_info,
    make_metrics,
    make_pcie,
    make_process,
    make_throttle,
)


def render_rich(metrics, **kwargs) -> str:
    """Render print_report into a string instead of a terminal."""
    console = Console(width=120, force_terminal=False, no_color=True)
    with console.capture() as capture:
        print_report(make_gpu_info(), metrics, classify(metrics), console=console, **kwargs)
    return capture.get()


def render_plain(metrics, capsys, **kwargs) -> str:
    print_report_plain(make_gpu_info(), metrics, classify(metrics), **kwargs)
    return capsys.readouterr().out


@pytest.mark.parametrize(
    "score,expected",
    [
        (100, "A"),
        (90, "A"),
        (89, "B"),
        (75, "B"),
        (74, "C"),
        (50, "C"),
        (49, "D"),
        (25, "D"),
        (24, "F"),
        (0, "F"),
    ],
)
def test_grade_boundaries(score, expected):
    assert _grade(score)[0] == expected


def test_plain_report_shows_identity_grade_and_bottleneck(capsys):
    out = render_plain(make_metrics(gpu_util=95, mem_util=20), capsys)
    assert "GPU 0: NVIDIA A30" in out
    assert "Health:" in out
    assert "compute_bound" in out
    assert "confidence" in out


def test_plain_report_marks_unavailable_metrics_as_na(capsys):
    metrics = make_metrics(
        gpu_util=None,
        mem_util=None,
        memory_used=None,
        memory_total=None,
        temperature=None,
        gpu_clock=None,
        max_gpu_clock=None,
    )
    out = render_plain(metrics, capsys)
    assert out.count("N/A") >= 4


def test_plain_report_renders_processes_or_says_none(capsys):
    empty = render_plain(make_metrics(), capsys)
    assert "(none)" in empty

    busy = render_plain(make_metrics(processes=[make_process(pid=4242, name="train.py")]), capsys)
    assert "4242" in busy
    assert "train.py" in busy


def test_plain_report_includes_patterns_warnings_and_recommendations(capsys):
    out = render_plain(
        make_metrics(throttle=make_throttle(HW_THERMAL_SLOWDOWN), temperature=95.0),
        capsys,
        patterns=[Pattern("clock_oscillation", "warning", "clocks unstable")],
        recommendations=[Recommendation(1, "Improve cooling", "GPU is hot", ["clean the fans"])],
    )
    assert "clocks unstable" in out
    assert "Improve cooling" in out
    assert "clean the fans" in out


def test_plain_report_surfaces_collection_errors(capsys):
    out = render_plain(make_metrics(errors=["Temperature unavailable"]), capsys)
    assert "Errors:" in out
    assert "Temperature unavailable" in out


def test_plain_report_flags_degraded_pcie_link(capsys):
    out = render_plain(make_metrics(pcie=make_pcie(degraded=True)), capsys)
    assert "Gen2 x16" in out
    assert "max Gen4" in out


def test_rich_report_renders_without_raising_and_shows_metrics():
    out = render_rich(make_metrics(gpu_util=95, mem_util=20, temperature=72.0))
    assert "NVIDIA A30" in out
    assert "GPU utilization" in out
    assert "compute_bound" in out


def test_rich_report_handles_fully_unavailable_metrics():
    metrics = make_metrics(
        gpu_util=None,
        mem_util=None,
        memory_used=None,
        memory_total=None,
        temperature=None,
        power_usage=None,
        power_limit=None,
        gpu_clock=None,
        max_gpu_clock=None,
    )
    out = render_rich(metrics)
    assert "N/A" in out
    assert "unknown" in out


def test_rich_report_includes_optional_sections():
    out = render_rich(
        make_metrics(ecc=make_ecc(correctable=1), errors=["Fan speed unavailable"]),
        patterns=[Pattern("temp_trend", "critical", "temperature climbing")],
        recommendations=[Recommendation(2, "Check airflow", "warm", ["reseat the card"])],
    )
    assert "temperature climbing" in out
    assert "Check airflow" in out
    assert "reseat the card" in out
    assert "Fan speed unavailable" in out
