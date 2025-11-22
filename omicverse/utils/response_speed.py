"""Compute simple speed metrics for Agent responses.

This module provides a lightweight helper to measure how quickly an Agent
returns output. It is intentionally minimal and does not require any external
dependencies beyond the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResponseSpeed:
    """Container for Agent response speed metrics."""

    duration_seconds: float
    characters: int
    chars_per_second: float
    tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None


def calculate_response_speed(
    start_time: float, end_time: float, response_text: str, tokens: Optional[int] = None
) -> ResponseSpeed:
    """Calculate character and token throughput for an Agent response.

    Parameters
    ----------
    start_time:
        Timestamp (seconds) when the response began.
    end_time:
        Timestamp (seconds) when the response finished.
    response_text:
        The complete text returned by the Agent.
    tokens:
        Optional token count for the response. When provided, the function also
        reports tokens-per-second.

    Returns
    -------
    ResponseSpeed
        A dataclass with duration, character throughput, and optional token
        throughput. If ``end_time`` is equal to ``start_time``, throughput
        values default to ``0.0`` to avoid division by zero. A ``ValueError`` is
        raised when ``end_time`` precedes ``start_time``.
    """

    if end_time < start_time:
        raise ValueError("end_time cannot be earlier than start_time")

    duration = end_time - start_time
    characters = len(response_text or "")

    if duration == 0:
        chars_per_second = 0.0
        tokens_per_second = 0.0 if tokens is not None else None
    else:
        chars_per_second = characters / duration
        tokens_per_second = (tokens / duration) if tokens is not None else None

    return ResponseSpeed(
        duration_seconds=duration,
        characters=characters,
        chars_per_second=chars_per_second,
        tokens=tokens,
        tokens_per_second=tokens_per_second,
    )
