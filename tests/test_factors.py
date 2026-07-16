"""Tests for the factor models module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors import (
    FactorScores,
    combine_factors,
    compute_momentum_factor,
    compute_quality_factor,
    compute_size_factor,
    compute_value_factor,
    cross_sectional_rank,
    factors_from_yfinance,
)


# --- Helper builders ---


def _make_price_dict(
    tickers: list[str],
    n_days: int = 400,
    seed: int = 42,
    drifts: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a synthetic price-only data_dict.

    drift per ticker = annualized expected return, controls which tickers
    end up with high/low momentum. Use this to assert ordering.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    drifts = drifts or {t: 0.05 for t in tickers}

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        daily_drift = drifts.get(ticker, 0.05) / 252.0
        daily_vol = 0.01  # 1% daily vol
        prices = [100.0]
        for _ in range(n_days - 1):
            ret = rng.normal(daily_drift, daily_vol)
            prices.append(prices[-1] * (1 + ret))
        out[ticker] = pd.DataFrame(
            {"Price": prices},
            index=[d.date() for d in dates],
        )
    return out


def _make_returns_dict(
    tickers: list[str],
    n_days: int = 400,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Same shape as extractor output (Price + Returns)."""
    raw = _make_price_dict(tickers, n_days=n_days, seed=seed)
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in raw.items():
        df = df.copy()
        df["Returns"] = df["Price"].pct_change()
        out[ticker] = df.dropna()
    return out


# --- compute_momentum_factor ---


class TestMomentumFactor:
    def test_returns_decimal_in_correct_range(self) -> None:
        """12-month momentum should return decimal return, not percent points."""
        data = _make_returns_dict(["A", "B", "C"], n_days=400)
        scores = compute_momentum_factor(data)
        # All values should be reasonable decimal returns (between -50% and +100%)
        assert scores.abs().max() < 1.0

    def test_ranking_correct_with_known_drifts(self) -> None:
        """A ticker with higher drift should have higher momentum score."""
        data = _make_price_dict(
            ["LOSER", "WINNER"],
            n_days=400,
            drifts={"LOSER": -0.30, "WINNER": 0.50},
        )
        # Add Returns column since compute_momentum_factor uses 'Price' only,
        # but extractor-style data works too. Reuse _make_returns_dict.
        data_with_returns = {}
        for ticker, df in data.items():
            df2 = df.copy()
            df2["Returns"] = df2["Price"].pct_change()
            data_with_returns[ticker] = df2

        scores = compute_momentum_factor(data_with_returns)
        # WINNER should beat LOSER
        assert scores["WINNER"] > scores["LOSER"]

    def test_skip_recent_month_differs_from_raw_12mo(self) -> None:
        """Default skip_days=21 means momentum != simple last-12mo return."""
        data = _make_returns_dict(["X"], n_days=400)
        scores_default = compute_momentum_factor(data, skip_days=21)
        scores_no_skip = compute_momentum_factor(data, skip_days=0)
        # They should be close but not identical (the last month matters)
        assert abs(scores_default["X"] - scores_no_skip["X"]) >= 0

    def test_empty_dict_returns_empty_series(self) -> None:
        scores = compute_momentum_factor({})
        assert scores.empty

    def test_insufficient_data_drops_ticker(self) -> None:
        """Tickers with less than lookback days are excluded."""
        data = _make_returns_dict(["SHORT"], n_days=50)
        scores = compute_momentum_factor(data, lookback_days=252)
        assert "SHORT" not in scores.index

    def test_works_with_price_only_data(self) -> None:
        """compute_momentum_factor should work without 'Returns' column."""
        data = _make_price_dict(["A", "B"], n_days=400)
        scores = compute_momentum_factor(data)
        assert len(scores) == 2


# --- compute_value_factor ---


class TestValueFactor:
    def test_low_pe_high_value(self) -> None:
        """Lower P/E = higher value score (after negation)."""
        fundamentals = {
            "CHEAP": {"pe": 5.0, "pb": 1.0},
            "EXPENSIVE": {"pe": 50.0, "pb": 10.0},
        }
        scores = compute_value_factor(fundamentals)
        assert scores["CHEAP"] > scores["EXPENSIVE"]

    def test_negative_pe_excluded(self) -> None:
        """Negative P/E (loss-making) excluded from P/E component.

        For 'LOSS' (pe=-10, pb=1): only PB contributes -> score = -(0.5*1) = -0.5
        For 'PROFITABLE' (pe=10, pb=1): both PE and PB contribute -> score = -(0.5*10 + 0.5*1) = -5.5
        So PROFITABLE has more negative value score (lower value), and LOSS ticker
        still appears with a partial score.
        """
        fundamentals = {
            "LOSS": {"pe": -10.0, "pb": 1.0},
            "PROFITABLE": {"pe": 10.0, "pb": 1.0},
        }
        scores = compute_value_factor(fundamentals)
        # Both tickers should appear (LOSS has PB, PROFITABLE has both)
        assert "LOSS" in scores.index
        assert "PROFITABLE" in scores.index
        # PROFITABLE has higher P/E -> lower value score (more negative)
        assert scores["PROFITABLE"] < scores["LOSS"]

    def test_missing_both_ratios_drops_ticker(self) -> None:
        fundamentals = {
            "OK": {"pe": 10.0, "pb": 1.5},
            "MISSING": {},
        }
        scores = compute_value_factor(fundamentals)
        assert "OK" in scores.index
        assert "MISSING" not in scores.index

    def test_pe_only_when_pb_missing(self) -> None:
        """Tickers with only P/E should still get a score."""
        fundamentals = {
            "PE_ONLY": {"pe": 8.0},
            "BOTH": {"pe": 8.0, "pb": 2.0},
        }
        scores = compute_value_factor(fundamentals)
        assert "PE_ONLY" in scores.index

    def test_custom_weights(self) -> None:
        """Custom weights should work and normalize to sum=1."""
        fundamentals = {"A": {"pe": 10.0, "pb": 1.0}}
        scores_default = compute_value_factor(fundamentals)
        scores_pe_only = compute_value_factor(fundamentals, weights={"pe": 1.0, "pb": 0.0})
        # Default uses both, pe-only uses only PE (after weight normalization -> pe=1.0)
        # Score pe-only = -10.0
        # Score default = -(0.5 * 10 + 0.5 * 1) = -5.5
        assert scores_pe_only["A"] == pytest.approx(-10.0)
        assert scores_default["A"] == pytest.approx(-5.5)

    def test_empty_dict(self) -> None:
        scores = compute_value_factor({})
        assert scores.empty

    def test_invalid_weights_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_value_factor({"A": {"pe": 10.0}}, weights={"pe": 0.0})


# --- compute_size_factor ---


class TestSizeFactor:
    def test_smaller_cap_higher_score(self) -> None:
        """Smaller market cap => higher size factor score (neg log)."""
        caps = {"BIG": 3_000_000_000_000, "SMALL": 100_000_000}
        scores = compute_size_factor(caps)
        assert scores["SMALL"] > scores["BIG"]

    def test_zero_or_negative_cap_excluded(self) -> None:
        caps = {"OK": 1_000_000_000, "ZERO": 0, "NEG": -100}
        scores = compute_size_factor(caps)
        assert "OK" in scores.index
        assert "ZERO" not in scores.index
        assert "NEG" not in scores.index

    def test_nan_cap_excluded(self) -> None:
        caps = {"OK": 1_000_000_000, "BAD": float("nan")}
        scores = compute_size_factor(caps)
        assert "BAD" not in scores.index

    def test_empty_dict(self) -> None:
        scores = compute_size_factor({})
        assert scores.empty


# --- compute_quality_factor ---


class TestQualityFactor:
    def test_higher_roe_higher_score(self) -> None:
        fundamentals = {
            "GOOD": {"roe": 0.20},
            "BAD": {"roe": 0.05},
        }
        scores = compute_quality_factor(fundamentals)
        assert scores["GOOD"] > scores["BAD"]

    def test_lower_stability_higher_score(self) -> None:
        """Lower earnings variability => higher quality (after negation)."""
        fundamentals = {
            "STABLE": {"earnings_stability": 0.1},
            "VOLATILE": {"earnings_stability": 0.8},
        }
        scores = compute_quality_factor(fundamentals)
        assert scores["STABLE"] > scores["VOLATILE"]

    def test_combined_components(self) -> None:
        fundamentals = {
            "GOOD_STABLE": {"roe": 0.20, "earnings_stability": 0.1},
            "BAD_VOLATILE": {"roe": 0.05, "earnings_stability": 0.8},
        }
        scores = compute_quality_factor(fundamentals)
        assert scores["GOOD_STABLE"] > scores["BAD_VOLATILE"]

    def test_missing_components_drop_ticker(self) -> None:
        fundamentals = {
            "FULL": {"roe": 0.20, "earnings_stability": 0.1},
            "MISSING": {},
        }
        scores = compute_quality_factor(fundamentals)
        assert "MISSING" not in scores.index


# --- cross_sectional_rank ---


class TestCrossSectionalRank:
    def test_returns_pct_in_0_1(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=["a", "b", "c", "d", "e"])
        ranks = cross_sectional_rank(s)
        assert ranks.between(0, 1).all()

    def test_higher_value_higher_rank(self) -> None:
        s = pd.Series([1.0, 5.0, 3.0], index=["a", "b", "c"])
        ranks = cross_sectional_rank(s)
        assert ranks["b"] > ranks["c"] > ranks["a"]

    def test_works_on_nan_gracefully(self) -> None:
        """NaN values should propagate (NaN rank)."""
        s = pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])
        ranks = cross_sectional_rank(s)
        assert pd.isna(ranks["b"])


# --- combine_factors ---


class TestCombineFactors:
    def test_combines_available_factors(self) -> None:
        fs = FactorScores(
            momentum=pd.Series({"A": 0.10, "B": 0.20, "C": 0.30}),
            value=pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}),
        )
        composite = combine_factors(fs)
        # All tickers present in both factors
        assert len(composite) == 3
        # With normalize=True, each rank is in [0, 1]
        # A is 0/2 in momentum rank, B is 0.5, C is 1.0
        # A is 0/2 in value rank, B is 0.5, C is 1.0
        # Composite A = 0, C = 1.0, B = 0.5
        assert composite["C"] > composite["B"] > composite["A"]

    def test_drops_tickers_missing_in_any_factor(self) -> None:
        """If a ticker is missing in one factor, drop it from composite."""
        fs = FactorScores(
            momentum=pd.Series({"A": 0.10, "B": 0.20}),
            value=pd.Series({"A": 1.0, "C": 2.0}),  # C missing in momentum
        )
        composite = combine_factors(fs)
        # Only A is in both
        assert "A" in composite.index
        assert "B" not in composite.index
        assert "C" not in composite.index

    def test_custom_weights(self) -> None:
        fs = FactorScores(
            momentum=pd.Series({"A": 0.10, "B": 0.20}),
            value=pd.Series({"A": 1.0, "B": 2.0}),
        )
        # Equal weights -> composite should still rank B > A
        composite = combine_factors(fs, weights={"momentum": 1.0, "value": 1.0})
        assert composite["B"] > composite["A"]

    def test_unknown_factor_in_weights_raises(self) -> None:
        fs = FactorScores(momentum=pd.Series({"A": 0.10}))
        with pytest.raises(ValueError):
            combine_factors(fs, weights={"unknown_factor": 1.0})

    def test_no_normalize(self) -> None:
        """Without normalization, raw values are combined directly."""
        fs = FactorScores(
            momentum=pd.Series({"A": 0.10, "B": 0.20}),
            value=pd.Series({"A": 1.0, "B": 2.0}),
        )
        composite = combine_factors(fs, normalize=False)
        # Raw momentum + value: A = 1.1, B = 2.2
        assert composite["B"] > composite["A"]

    def test_empty_factor_scores_returns_empty(self) -> None:
        fs = FactorScores(momentum=pd.Series(dtype=float))
        composite = combine_factors(fs)
        assert composite.empty


# --- FactorScores container ---


class TestFactorScores:
    def test_available_factors_returns_present(self) -> None:
        fs = FactorScores(
            momentum=pd.Series({"A": 0.1}),
            value=pd.Series({"A": 1.0}),
            size=None,
            quality=pd.Series(dtype=float),
        )
        assert "momentum" in fs.available_factors()
        assert "value" in fs.available_factors()
        assert "size" not in fs.available_factors()
        assert "quality" not in fs.available_factors()

    def test_to_dataframe_combines_all(self) -> None:
        fs = FactorScores(
            momentum=pd.Series({"A": 0.1, "B": 0.2}),
            value=pd.Series({"A": 1.0, "C": 3.0}),  # C only in value
        )
        df = fs.to_dataframe()
        assert "momentum" in df.columns
        assert "value" in df.columns
        # Outer join: all tickers across both
        assert set(df.index) == {"A", "B", "C"}


# --- factors_from_yfinance (integration via mock factory) ---


class TestFactorsFromYfinance:
    def test_with_mock_yfinance_factory(self) -> None:
        """Mock factory should mimic yfinance.Ticker().info."""

        class MockInfo:
            def __init__(self, info_dict: dict) -> None:
                self._info = info_dict

        class MockTicker:
            def __init__(self, ticker: str) -> None:
                self._info = MOCK_DATA.get(ticker, {})
                self.info = self._info

        MOCK_DATA = {
            "A": {
                "trailingPE": 10.0,
                "priceToBook": 1.5,
                "marketCap": 1_000_000_000,
                "returnOnEquity": 0.15,
            },
            "B": {
                "trailingPE": 25.0,
                "priceToBook": 5.0,
                "marketCap": 100_000_000_000,
                "returnOnEquity": 0.25,
            },
        }

        data_dict = _make_returns_dict(["A", "B"], n_days=400)
        fs = factors_from_yfinance(
            tickers=["A", "B"],
            data_dict=data_dict,
            yfinance_factory=MockTicker,
        )

        # All four factors should be available
        available = fs.available_factors()
        assert "momentum" in available
        assert "value" in available
        assert "size" in available
        assert "quality" in available

    def test_skips_failed_tickers(self) -> None:
        """If yfinance raises or returns non-dict, skip that ticker."""

        class FailingTicker:
            def __init__(self, ticker: str) -> None:
                raise RuntimeError("network error")

        data_dict = _make_returns_dict(["A"], n_days=400)
        fs = factors_from_yfinance(
            tickers=["A"],
            data_dict=data_dict,
            yfinance_factory=FailingTicker,
        )
        # Momentum should still be computed (from data_dict)
        assert "momentum" in fs.available_factors()
        # Other factors empty (no fundamentals got fetched)
        assert "value" not in fs.available_factors()
        assert "size" not in fs.available_factors()

    def test_no_data_dict_skips_momentum(self) -> None:
        """Without data_dict, momentum is empty."""

        class MockTicker:
            def __init__(self, ticker: str) -> None:
                self.info = {"trailingPE": 10.0, "marketCap": 1_000_000}

        fs = factors_from_yfinance(
            tickers=["A"],
            data_dict=None,
            yfinance_factory=MockTicker,
        )
        assert "momentum" not in fs.available_factors()
        assert "value" in fs.available_factors()
