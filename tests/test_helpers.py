"""Tests for shared helper utility functions."""

import math

from utils.helpers import snap_to_tick


class TestSnapToTick:
    """Verify tick-rounding in all three directions."""

    def test_snap_down(self) -> None:
        """Floor to nearest tick below."""
        result = snap_to_tick(105.3, 10.0, direction="down")
        assert result == 100.0

    def test_snap_up(self) -> None:
        """Ceil to nearest tick above."""
        result = snap_to_tick(100.1, 10.0, direction="up")
        assert result == 110.0

    def test_snap_nearest(self) -> None:
        """Round to nearest tick (default)."""
        result = snap_to_tick(104.9, 10.0)
        assert result == 100.0

    def test_exact_tick_unchanged(self) -> None:
        """Price already on a tick boundary stays unchanged."""
        result = snap_to_tick(100.0, 10.0, direction="down")
        assert result == 100.0

    def test_small_tick_precision(self) -> None:
        """Works correctly with small tick sizes (e.g., 0.5)."""
        result = snap_to_tick(10.3, 0.5, direction="down")
        assert math.isclose(result, 10.0)
