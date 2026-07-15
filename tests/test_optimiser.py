"""Tests for portfolio optimisation module."""

import numpy as np
import pandas as pd

from src.optimiser import (
    calculate_mean_variance,
    optimize_portfolio_mean_variance,
)


def _make_data_dict(n_assets: int = 2, n_days: int = 100) -> dict[str, pd.DataFrame]:
    """Helper: build a synthetic data_dict with Returns columns."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    data = {}
    for i in range(n_assets):
        data[f"ASSET{i+1}"] = pd.DataFrame(
            {
                "Price": np.random.randn(n_days) * 10 + 100 * (i + 1),
                "Returns": np.random.randn(n_days) * 0.01 * (i + 1),
            },
            index=[d.date() for d in dates],
        )
    return data


def _make_predicted_returns(data_dict: dict[str, pd.DataFrame], scale: float = 0.01) -> dict[str, float]:
    """Helper: build predicted_returns matching data_dict tickers."""
    return {ticker: float(np.random.randn() * scale) for ticker in data_dict.keys()}


class TestPortfolioOptimisation:
    """Test portfolio optimisation functions."""

    def test_calculate_mean_variance(self) -> None:
        """Test calculating mean and covariance from Returns columns."""
        data_dict = _make_data_dict()
        mean_returns, cov_matrix = calculate_mean_variance(data_dict)

        assert isinstance(mean_returns, pd.Series)
        assert isinstance(cov_matrix, pd.DataFrame)
        assert len(mean_returns) == 2
        assert cov_matrix.shape == (2, 2)
        assert "ASSET1" in mean_returns.index
        assert "ASSET2" in mean_returns.index

    def test_optimize_portfolio_mean_variance_basic(self) -> None:
        """Test basic portfolio optimisation uses predicted returns (not historical)."""
        data_dict = _make_data_dict()
        # Use VERY DIFFERENT predicted returns vs historical mean
        # If the optimizer uses historical mean instead, this test catches it
        predicted_returns = {
            "ASSET1": 0.10,   # very high
            "ASSET2": -0.05,  # very low (negative)
        }
        optimal_weights = optimize_portfolio_mean_variance(data_dict, predicted_returns)

        assert isinstance(optimal_weights, dict)
        assert len(optimal_weights) == 2
        assert np.isclose(sum(optimal_weights.values()), 1.0, rtol=1e-5)
        assert all(w >= 0 and w <= 1 for w in optimal_weights.values())
        assert "ASSET1" in optimal_weights
        assert "ASSET2" in optimal_weights

    def test_optimize_portfolio_uses_predicted_not_historical(self) -> None:
        """REGRESSION TEST (fixed 2026-07-15).

        Before the fix, optimize_portfolio_mean_variance used historical mean
        returns from calculate_mean_variance() as the expected returns (mu).
        This was wrong because historical mean != predicted future return.

        This test catches that bug: it gives predicted_returns that are wildly
        different from what historical mean would be, and verifies the optimizer
        behaves according to predicted_returns (not historical).

        With predicted ASSET1 = +0.50 (huge) and ASSET2 = -0.50 (huge negative),
        if the optimizer is using PREDICTED, it should heavily favor ASSET1.
        If it's using HISTORICAL, the result would be more balanced.
        """
        data_dict = _make_data_dict(n_assets=2, n_days=200)
        # Historical returns are random ~0.01 with std 0.01
        # So historical mean is ~0 ± a few percent
        predicted_returns = {
            "ASSET1": 0.50,   # 50% return (extremely bullish)
            "ASSET2": -0.50,  # -50% return (extremely bearish)
        }
        weights = optimize_portfolio_mean_variance(
            data_dict,
            predicted_returns,
            risk_aversion=1.0,  # low risk aversion so return dominates
        )

        # ASSET1 should get the MAJORITY of weight (it's predicted to return 50%)
        # If the optimizer were using historical mean (random ~0), the weights would
        # be much more balanced.
        assert weights["ASSET1"] > 0.7, (
            f"ASSET1 weight {weights['ASSET1']:.3f} is too low. "
            "Optimizer may still be using historical returns instead of predicted."
        )
        assert weights["ASSET2"] < 0.3, (
            f"ASSET2 weight {weights['ASSET2']:.3f} is too high. "
            "Optimizer may still be using historical returns instead of predicted."
        )

    def test_optimize_portfolio_missing_predicted_returns_raises(self) -> None:
        """Test that missing predicted_returns raises an error."""
        data_dict = _make_data_dict()

        # Empty predicted_returns should raise
        try:
            optimize_portfolio_mean_variance(data_dict, {})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Missing tickers should raise
        try:
            optimize_portfolio_mean_variance(data_dict, {"ASSET1": 0.05})  # missing ASSET2
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_optimize_portfolio_with_minimum_allocation(self) -> None:
        """Test portfolio optimisation with custom minimum allocation."""
        data_dict = _make_data_dict()
        predicted_returns = _make_predicted_returns(data_dict)
        min_allocation = 0.1  # 10% minimum

        optimal_weights = optimize_portfolio_mean_variance(
            data_dict, predicted_returns, minimum_allocation=min_allocation
        )

        assert isinstance(optimal_weights, dict)
        assert len(optimal_weights) == 2
        assert np.isclose(sum(optimal_weights.values()), 1.0, rtol=1e-5)
        assert all(w >= min_allocation for w in optimal_weights.values())
        assert all(w <= 1.0 for w in optimal_weights.values())