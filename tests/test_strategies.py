"""Tests for strategy computation pure functions."""

from execution.strategies import compute_order_size, lon_fly_payoff


class TestComputeOrderSize:
    """Verify position-aware order sizing logic."""

    def test_neutral_signal_flat_position(self) -> None:
        """Zero signal and flat position should produce symmetric sizes."""
        config = {"order_size": 5, "max_position": 50, "spread_ticks": 2}
        bid_size, ask_size = compute_order_size(0.0, 0, config)
        assert bid_size == ask_size
        assert bid_size > 0

    def test_positive_signal_biases_toward_buying(self) -> None:
        """Positive mispricing signal (theo > market) should increase bid size."""
        config = {"order_size": 5, "max_position": 50, "spread_ticks": 2}
        bid_size, ask_size = compute_order_size(0.5, 0, config)
        assert bid_size > ask_size

    def test_at_max_position_stops_building(self) -> None:
        """When at strategy max position, should stop building (bid_size=0)."""
        config = {"order_size": 5, "max_position": 50, "spread_ticks": 2}
        bid_size, _ask_size = compute_order_size(0.5, 50, config)
        assert bid_size == 0


class TestLonFlyPayoffConsistency:
    """Cross-check that strategies.lon_fly_payoff matches theo.engine version."""

    def test_matches_theo_engine(self) -> None:
        from theo.engine import lon_fly_payoff as theo_payoff

        for s in [5000.0, 6200.0, 6400.0, 6600.0, 7000.0, 7500.0, 8000.0]:
            assert lon_fly_payoff(s) == theo_payoff(s), f"Mismatch at S={s}"
