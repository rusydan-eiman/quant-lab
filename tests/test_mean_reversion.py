"""Tests for src/strategies/mean_reversion.py — Bollinger Bands, ADF, Hurst."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.mean_reversion import (
    compute_bollinger_bands,
    compute_hurst_exponent,
    generate_mean_reversion_signals,
)
from src.strategies.mean_reversion import test_adf_stationarity as adf_test


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_gbm_series(n: int = 1000, seed: int = 0, drift: float = 0.0003, vol: float = 0.01) -> pd.Series:
    """Random-walk series (Geometric Brownian Motion in disguise)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def _make_mean_reverting_series(n: int = 1000, seed: int = 1, mean: float = 100.0, theta: float = 0.05, vol: float = 1.0) -> pd.Series:
    """OU-like series: x_{t+1} = x_t + theta * (mean - x_t) + noise."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mean
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mean - x[i - 1]) + rng.normal(0, vol)
    return pd.Series(x, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def _make_trending_series(n: int = 1000, seed: int = 2, drift: float = 0.01) -> pd.Series:
    """Linear uptrend with noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(n) * drift + rng.normal(0, 1, n).cumsum() * 0.3
    return pd.Series(100 + x, index=pd.date_range("2020-01-01", periods=n, freq="D"))


# ---------------------------------------------------------------------------
# compute_bollinger_bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_returns_four_columns(self) -> None:
        s = _make_gbm_series(300)
        bands = compute_bollinger_bands(s, window=20)
        assert set(bands.columns) == {"middle", "upper", "lower", "band_width"}
        assert len(bands) == len(s)

    def test_window_minus_one_nans_then_real(self) -> None:
        s = _make_gbm_series(300)
        bands = compute_bollinger_bands(s, window=20)
        assert bands["middle"].iloc[:19].isna().all()
        assert bands["middle"].iloc[19:].notna().all()

    def test_band_ordering(self) -> None:
        s = _make_gbm_series(300)
        bands = compute_bollinger_bands(s, window=20)
        valid = bands.dropna()
        assert (valid["upper"] >= valid["middle"]).all()
        assert (valid["middle"] >= valid["lower"]).all()

    def test_band_width_positive(self) -> None:
        s = _make_gbm_series(300)
        bands = compute_bollinger_bands(s, window=20)
        assert (bands["band_width"].dropna() > 0).all()

    def test_invalid_window(self) -> None:
        s = _make_gbm_series(100)
        with pytest.raises(ValueError):
            compute_bollinger_bands(s, window=1)

    def test_invalid_num_std(self) -> None:
        s = _make_gbm_series(100)
        with pytest.raises(ValueError):
            compute_bollinger_bands(s, num_std=0.0)
        with pytest.raises(ValueError):
            compute_bollinger_bands(s, num_std=-1.0)

    def test_non_series_input(self) -> None:
        with pytest.raises(TypeError):
            compute_bollinger_bands([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# generate_mean_reversion_signals
# ---------------------------------------------------------------------------


class TestGenerateMeanReversionSignals:
    def test_returns_dataclass(self) -> None:
        s = _make_gbm_series(500)
        result = generate_mean_reversion_signals(s, window=20)
        assert hasattr(result, "positions")
        assert hasattr(result, "bands")

    def test_positions_only_minus_one_zero_one(self) -> None:
        s = _make_gbm_series(500)
        result = generate_mean_reversion_signals(s, window=20)
        assert set(result.positions.unique()).issubset({-1, 0, 1})

    def test_first_window_rows_flat(self) -> None:
        s = _make_gbm_series(500)
        result = generate_mean_reversion_signals(s, window=20)
        # First 20 rows correspond to NaN-band period, signal must be flat.
        assert (result.positions.iloc[:20] == 0).all()

    def test_no_signal_on_perfectly_smooth_trend(self) -> None:
        """A smoothly trending line should mostly stay flat because each day's
        price is inside the band."""
        s = pd.Series(
            np.linspace(100, 200, 200),
            index=pd.date_range("2020-01-01", periods=200, freq="D"),
        )
        result = generate_mean_reversion_signals(s, window=20, num_std=2.0)
        active_pct = float((result.positions != 0).mean())
        # Mostly flat on a trend.
        assert active_pct < 0.5

    def test_exit_on_middle_false(self) -> None:
        """When exit_on_middle=False, signals can last longer."""
        s = _make_gbm_series(500)
        with_mid = generate_mean_reversion_signals(s, window=20, exit_on_middle=True)
        without_mid = generate_mean_reversion_signals(s, window=20, exit_on_middle=False)
        # Both should return valid results without errors.
        assert isinstance(without_mid.positions, pd.Series)
        assert isinstance(with_mid.positions, pd.Series)


# ---------------------------------------------------------------------------
# test_adf_stationarity
# ---------------------------------------------------------------------------


class TestADFStationarity:
    def test_returns_dict_with_expected_keys(self) -> None:
        s = _make_gbm_series(500)
        result = adf_test(s)
        for key in [
            "t_statistic",
            "p_value",
            "critical_values",
            "lags_used",
            "n_obs",
            "is_mean_reverting",
            "method",
        ]:
            assert key in result

    def test_ou_style_series_indicates_mean_reverting(self) -> None:
        s = _make_mean_reverting_series(n=1000, theta=0.10)
        result = adf_test(s, max_lag=1)
        # Strong mean reversion => reject null => is_mean_reverting True.
        assert result["is_mean_reverting"]

    def test_trending_series_not_marked_mean_reverting(self) -> None:
        """A trending series should not be flagged as mean-reverting."""
        s = _make_trending_series(n=1000)
        result = adf_test(s, max_lag=1)
        assert not result["is_mean_reverting"]

    def test_insufficient_sample(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0])  # way too short
        result = adf_test(s)
        assert result["insufficient_sample"] is True
        assert result["is_mean_reverting"] is False

    def test_invalid_max_lag(self) -> None:
        s = _make_gbm_series(100)
        with pytest.raises(ValueError):
            adf_test(s, max_lag=0)

    def test_non_series_input(self) -> None:
        with pytest.raises(TypeError):
            adf_test([1, 2, 3])  # type: ignore[arg-type]

    def test_critical_values_present(self) -> None:
        s = _make_gbm_series(500)
        result = adf_test(s)
        cv = result["critical_values"]
        assert "1%" in cv and "5%" in cv and "10%" in cv
        # All negative (it's a test for unit root, rejections at negative values).
        assert cv["1%"] < 0
        assert cv["5%"] < 0
        assert cv["10%"] < 0
        # More negative at lower significance.
        assert cv["1%"] < cv["5%"] < cv["10%"]


# ---------------------------------------------------------------------------
# compute_hurst_exponent
# ---------------------------------------------------------------------------


class TestHurstExponent:
    def test_gbm_around_zero_point_five(self) -> None:
        """A Geometric Brownian Motion should have Hurst close to 0.5."""
        s = _make_gbm_series(n=5000, seed=42)
        h = compute_hurst_exponent(s, min_lag=2, max_lag=100)
        # Accept anything in [0.4, 0.6] for a GBM.
        assert 0.40 <= h <= 0.60

    def test_trending_series_above_zero_point_five(self) -> None:
        s = _make_trending_series(n=5000)
        h = compute_hurst_exponent(s, min_lag=2, max_lag=100)
        assert h > 0.5

    def test_returns_float(self) -> None:
        s = _make_gbm_series(500)
        h = compute_hurst_exponent(s)
        assert isinstance(h, float)

    def test_too_short_returns_nan(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        h = compute_hurst_exponent(s)
        assert np.isnan(h)

    def test_invalid_min_lag(self) -> None:
        s = _make_gbm_series(500)
        with pytest.raises(ValueError):
            compute_hurst_exponent(s, min_lag=1)

    def test_invalid_max_lag(self) -> None:
        s = _make_gbm_series(500)
        with pytest.raises(ValueError):
            compute_hurst_exponent(s, min_lag=2, max_lag=2)

    def test_non_series_input(self) -> None:
        with pytest.raises(TypeError):
            compute_hurst_exponent([1, 2, 3])  # type: ignore[arg-type]
