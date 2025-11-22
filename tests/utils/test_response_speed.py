"""Tests for response speed helper."""

import omicverse.utils as utils

from omicverse.utils import calculate_response_speed


def test_calculates_character_throughput():
    result = calculate_response_speed(0.0, 2.0, "abcd")

    assert result.duration_seconds == 2.0
    assert result.characters == 4
    assert result.chars_per_second == 2.0
    assert result.tokens is None
    assert result.tokens_per_second is None


def test_calculates_token_throughput():
    result = calculate_response_speed(1.0, 3.0, "hello world", tokens=20)

    assert result.duration_seconds == 2.0
    assert result.chars_per_second == 11 / 2.0
    assert result.tokens_per_second == 10.0


def test_handles_zero_duration():
    result = calculate_response_speed(5.0, 5.0, "speed", tokens=5)

    assert result.duration_seconds == 0.0
    assert result.chars_per_second == 0.0
    assert result.tokens_per_second == 0.0


def test_rejects_negative_duration():
    try:
        calculate_response_speed(2.0, 1.0, "bad timing")
    except ValueError as exc:
        assert "end_time" in str(exc)
    else:
        raise AssertionError("Expected ValueError when end_time < start_time")


def test_utils_namespace_exports_module_and_callable():
    assert hasattr(utils, "response_speed")
    assert utils.calculate_response_speed is calculate_response_speed
