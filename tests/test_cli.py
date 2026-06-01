"""Tests for CLI helpers."""

import pytest
import typer

from nvsonar.cli import _parse_gpu_selection


def test_empty_string_returns_all_indices():
    assert _parse_gpu_selection("", 4) == [0, 1, 2, 3]


def test_all_keyword_returns_all_indices():
    assert _parse_gpu_selection("all", 4) == [0, 1, 2, 3]


def test_minus_one_returns_all_indices_for_backward_compat():
    assert _parse_gpu_selection("-1", 4) == [0, 1, 2, 3]


def test_single_index_returns_one_element_list():
    assert _parse_gpu_selection("2", 4) == [2]


def test_comma_separated_returns_listed_indices():
    assert _parse_gpu_selection("0,2,3", 4) == [0, 2, 3]


def test_whitespace_around_indices_is_tolerated():
    assert _parse_gpu_selection(" 0 , 1 , 2 ", 4) == [0, 1, 2]


def test_duplicate_indices_are_deduplicated_preserving_order():
    assert _parse_gpu_selection("1,0,1,2,0", 4) == [1, 0, 2]


def test_out_of_range_index_raises():
    with pytest.raises(typer.BadParameter):
        _parse_gpu_selection("5", 4)


def test_non_integer_raises():
    with pytest.raises(typer.BadParameter):
        _parse_gpu_selection("abc", 4)


def test_only_commas_raises():
    with pytest.raises(typer.BadParameter):
        _parse_gpu_selection(",,,", 4)
