"""Tests for theo engine pure functions (payoff calculations)."""

from theo.engine import lon_fly_payoff, strangle_value


class TestLonFlyPayoff:
    """Verify the piecewise LON_FLY payoff at key strike boundaries.

    Payoff structure:
        LON_FLY(S) = 2*max(0, 6200-S) + max(0, S-6200) - 2*max(0, S-6600) + 3*max(0, S-7000)
    """

    def test_below_all_strikes(self) -> None:
        """S well below 6200 activates only the first put term."""
        result = lon_fly_payoff(5000.0)
        expected = 2.0 * (6200.0 - 5000.0)
        assert result == expected == 2400.0

    def test_between_6200_and_6600(self) -> None:
        """S between first and second kink: only the call at 6200 is active."""
        result = lon_fly_payoff(6400.0)
        expected = 0.0 + 1.0 * (6400.0 - 6200.0) - 0.0 + 0.0
        assert result == expected == 200.0

    def test_above_7000(self) -> None:
        """S above all strikes: all terms active."""
        result = lon_fly_payoff(7500.0)
        expected = 0.0 + 1.0 * 1300.0 - 2.0 * 900.0 + 3.0 * 500.0
        assert result == expected == 1000.0

    def test_at_6600_boundary(self) -> None:
        """S exactly at 6600 kink point."""
        result = lon_fly_payoff(6600.0)
        expected = 0.0 + 1.0 * 400.0 - 0.0 + 0.0
        assert result == expected == 400.0


class TestStrangleValue:
    """Verify the tidal strangle payoff for 15-min height diffs.

    Payoff: max(0, 20 - diff) + max(0, diff - 25)
    """

    def test_inside_corridor_no_payoff(self) -> None:
        """Diff within [20, 25] corridor produces zero payoff."""
        assert strangle_value(22.0) == 0.0

    def test_below_20_put_active(self) -> None:
        """Diff below 20 activates the downside put."""
        assert strangle_value(15.0) == 5.0

    def test_above_25_call_active(self) -> None:
        """Diff above 25 activates the upside call."""
        assert strangle_value(30.0) == 5.0

    def test_at_boundary_20(self) -> None:
        """Diff exactly at 20 is zero payoff."""
        assert strangle_value(20.0) == 0.0

    def test_at_boundary_25(self) -> None:
        """Diff exactly at 25 is zero payoff."""
        assert strangle_value(25.0) == 0.0
