"""Factor models for cross-sectional portfolio allocation.

Provides four classical equity factors:
- Value (P/E, P/B)
- Momentum (12-month return, excluding the most recent month)
- Size (market cap, log)
- Quality (return on equity)

Each factor is computed per ticker as a cross-sectional rank (or z-score),
then optionally combined into a composite score.

References:
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on
  stocks and bonds. Journal of Financial Economics, 33(1), 3-56.
- Carhart, M. M. (1997). On persistence in mutual fund performance. The Journal
  of Finance, 52(1), 57-82.
- Asness, C. S., Frazzini, A., & Israel, R. (2019). Measuring the real estate
  premium. The Journal of Portfolio Management.

This module is designed to be:
1. Pure (no side effects, all inputs explicit)
2. Testable (no live API calls in the core; optional via factors_from_yfinance)
3. Composable (each factor is a separate function returning a pd.Series)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# Default lookback periods
DEFAULT_MOMENTUM_LOOKBACK_DAYS = 252  # ~12 months of trading days
DEFAULT_MOMENTUM_SKIP_DAYS = 21  # skip the most recent month (Jegadeesh & Titman 1993)
DEFAULT_VALUE_LOOKBACK_DAYS = 0  # P/E and P/B are point-in-time, no lookback needed


@dataclass
class FactorScores:
    """Container for per-ticker factor values.

    All attributes are pd.Series indexed by ticker symbol. Each value is the
    raw factor value (not yet ranked or z-scored).
    """

    momentum: pd.Series
    value: pd.Series | None = None
    size: pd.Series | None = None
    quality: pd.Series | None = None

    def available_factors(self) -> list[str]:
        """Return names of factors that have data (non-None and non-empty)."""
        out = []
        for name in ("momentum", "value", "size", "quality"):
            series = getattr(self, name)
            if series is not None and not series.empty:
                out.append(name)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Combine all factors into one DataFrame indexed by ticker.

        Columns: factor name, values are raw factor values. Tickers that lack
        data for a factor will have NaN.
        """
        frames = {}
        for name in self.available_factors():
            frames[name] = getattr(self, name)
        return pd.DataFrame(frames)


def compute_momentum_factor(
    data_dict: dict[str, pd.DataFrame],
    lookback_days: int = DEFAULT_MOMENTUM_LOOKBACK_DAYS,
    skip_days: int = DEFAULT_MOMENTUM_SKIP_DAYS,
) -> pd.Series:
    """Compute the 12-month momentum factor for each ticker.

    Definition (Jegadeesh & Titman 1993): cumulative return over the past
    `lookback_days` trading days, EXCLUDING the most recent `skip_days` days.
    The skip-month avoids the short-term reversal effect that contaminates
    raw 12-month returns.

    Args:
        data_dict: Dictionary mapping ticker to DataFrame with a 'Price' column
            and a date index.
        lookback_days: How many trading days back to measure momentum
            (default 252 = ~12 months).
        skip_days: Skip the most recent N days (default 21 = ~1 month).

    Returns:
        pd.Series indexed by ticker, values are decimal returns (e.g., 0.10 = +10%).
        Tickers with insufficient data are omitted from the result.

    Example:
        >>> scores = compute_momentum_factor(data_dict)
        >>> scores.head()
        AAPL    0.2341
        MSFT    0.1892
        ...
    """
    if not data_dict:
        return pd.Series(dtype=float)

    results: dict[str, float] = {}
    for ticker, df in data_dict.items():
        if "Price" not in df.columns:
            continue
        prices = df["Price"].dropna()
        # We need lookback + skip days of data minimum
        needed = lookback_days + skip_days
        if len(prices) < needed:
            # Try without the skip if we don't have enough
            if len(prices) < lookback_days:
                continue
            end_idx = len(prices)
            start_idx = end_idx - lookback_days
            momentum = float(prices.iloc[end_idx - 1] / prices.iloc[start_idx] - 1)
        else:
            # Skip the most recent skip_days days
            end_idx = len(prices) - skip_days
            start_idx = end_idx - lookback_days
            momentum = float(prices.iloc[end_idx - 1] / prices.iloc[start_idx] - 1)
        results[ticker] = momentum

    return pd.Series(results, name="momentum").sort_index()


def compute_value_factor(
    fundamentals: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Compute a value factor composite from P/E and P/B ratios.

    Args:
        fundamentals: Dictionary mapping ticker to a dict with keys:
            - 'pe': trailing P/E ratio (None or NaN if not available)
            - 'pb': price-to-book ratio (None or NaN if not available)
            Either or both may be present.
        weights: Optional dictionary mapping metric name ('pe', 'pb') to its
            weight in the composite. Defaults to equal weighting. Higher P/E
            and P/B indicate lower value (growth), so we negate.

    Returns:
        pd.Series indexed by ticker, values are NEGATIVE of weighted composite
        (so higher value = higher score, consistent with factor-investing intuition).

    Notes:
        - Tickers missing both ratios are omitted.
        - Negative P/E (loss-making firms) is set to NaN (not a useful value signal).
        - P/B <= 0 or NaN also excludes the ticker from the P/B component.
    """
    if not fundamentals:
        return pd.Series(dtype=float)

    if weights is None:
        weights = {"pe": 0.5, "pb": 0.5}

    # Normalize weights to sum to 1
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("weights must contain at least one positive value")
    weights = {k: v / total_w for k, v in weights.items()}

    results: dict[str, float] = {}
    for ticker, finfo in fundamentals.items():
        components: list[tuple[float, float]] = []  # (value, weight)
        if "pe" in weights and "pe" in finfo:
            pe = finfo["pe"]
            if pe is not None and not pd.isna(pe) and pe > 0:
                components.append((pe, weights["pe"]))
        if "pb" in weights and "pb" in finfo:
            pb = finfo["pb"]
            if pb is not None and not pd.isna(pb) and pb > 0:
                components.append((pb, weights["pb"]))
        if not components:
            continue
        # Higher ratio = lower value, so negate to get value score
        weighted_pe_pb = sum(val * w for val, w in components)
        results[ticker] = -weighted_pe_pb

    return pd.Series(results, name="value").sort_index()


def compute_size_factor(market_caps: dict[str, float]) -> pd.Series:
    """Compute a size factor from market capitalization.

    Args:
        market_caps: Dictionary mapping ticker to market cap in USD (or any
            consistent currency).

    Returns:
        pd.Series indexed by ticker, values are NEGATIVE log(market_cap).
        Negated so larger size = higher score (small-cap premium: smaller
        companies historically earn higher returns, so size factor rewards
        smaller cap; inverting log gives us "smaller = higher").

    Notes:
        Tickers with non-positive or NaN market cap are omitted.
    """
    if not market_caps:
        return pd.Series(dtype=float)

    results: dict[str, float] = {}
    for ticker, mc in market_caps.items():
        if mc is None or pd.isna(mc) or mc <= 0:
            continue
        # Smaller market cap => higher size score
        results[ticker] = -float(np.log(mc))

    return pd.Series(results, name="size").sort_index()


def compute_quality_factor(
    fundamentals: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Compute a quality factor composite from ROE and earnings stability.

    Args:
        fundamentals: Dictionary mapping ticker to a dict with keys:
            - 'roe': return on equity as a decimal (e.g., 0.18 = 18%)
            - 'earnings_stability': rolling coefficient of variation of earnings,
              LOWER is better. May be NaN.
        weights: Optional dictionary mapping metric name ('roe', 'earnings_stability')
            to its weight. Defaults to equal weighting.

    Returns:
        pd.Series indexed by ticker, values are composite quality score.
        Higher = better quality. Tickers missing data are omitted.

    Notes:
        For 'earnings_stability', we NEGATE so that lower variability becomes
        a higher score (we want stability to add to quality).
    """
    if not fundamentals:
        return pd.Series(dtype=float)

    if weights is None:
        weights = {"roe": 0.5, "earnings_stability": 0.5}

    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("weights must contain at least one positive value")
    weights = {k: v / total_w for k, v in weights.items()}

    results: dict[str, float] = {}
    for ticker, finfo in fundamentals.items():
        components: list[tuple[float, float]] = []
        if "roe" in weights and "roe" in finfo:
            roe = finfo["roe"]
            if roe is not None and not pd.isna(roe):
                components.append((roe, weights["roe"]))
        if "earnings_stability" in weights and "earnings_stability" in finfo:
            es = finfo["earnings_stability"]
            if es is not None and not pd.isna(es) and es > 0:
                # Lower is better, so negate
                components.append((-es, weights["earnings_stability"]))
        if not components:
            continue
        results[ticker] = sum(val * w for val, w in components)

    return pd.Series(results, name="quality").sort_index()


def cross_sectional_rank(scores: pd.Series, method: str = "average") -> pd.Series:
    """Rank tickers cross-sectionally (0..1 percentile rank by default).

    Args:
        scores: pd.Series of raw factor values indexed by ticker.
        method: Ranking method passed to pandas .rank(). Default 'average'.

    Returns:
        pd.Series of ranks in [0, 1], same index. Higher rank = higher factor score.
    """
    return scores.rank(method=method, pct=True)


def combine_factors(
    factor_scores: FactorScores,
    weights: dict[str, float] | None = None,
    normalize: bool = True,
) -> pd.Series:
    """Combine multiple factors into a single composite score.

    Args:
        factor_scores: FactorScores dataclass with available factor series.
        weights: Optional dict mapping factor name to its weight in the composite.
            If None, equal weights are used across available factors. Must contain
            only factor names present in `factor_scores.available_factors()`.
        normalize: If True (default), each factor is cross-sectionally ranked to
            [0, 1] before combining (so different scales don't dominate). If False,
            raw values are used (only safe if all factors are on the same scale).

    Returns:
        pd.Series indexed by ticker, values are composite scores. Higher = more
        attractive. Tickers missing any required factor are dropped.

    Raises:
        ValueError: If `weights` includes factors not in `factor_scores`.
    """
    available = factor_scores.available_factors()
    if not available:
        return pd.Series(dtype=float)

    if weights is None:
        weights = {f: 1.0 for f in available}
    else:
        unknown = set(weights) - set(available)
        if unknown:
            raise ValueError(f"Unknown factor names in weights: {unknown}")
        weights = {f: weights[f] for f in available if f in weights}

    # Normalize weights to sum to 1
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("weights must contain at least one positive value")
    weights = {k: v / total_w for k, v in weights.items()}

    # Normalize each factor to percentile rank
    if normalize:
        normalized = {
            name: cross_sectional_rank(getattr(factor_scores, name)) for name in weights
        }
    else:
        normalized = {name: getattr(factor_scores, name) for name in weights}

    # Inner-join across factors: only keep tickers present in all selected factors
    combined_index: pd.Index | None = None
    for name in weights:
        s = normalized[name]
        if combined_index is None:
            combined_index = s.index
        else:
            combined_index = combined_index.intersection(s.index)

    if combined_index is None or len(combined_index) == 0:
        return pd.Series(dtype=float)

    composite = pd.Series(0.0, index=combined_index)
    for name, w in weights.items():
        composite = composite.add(normalized[name].reindex(combined_index) * w, fill_value=0.0)

    return composite.sort_index()


def factors_from_yfinance(
    tickers: list[str],
    data_dict: dict[str, pd.DataFrame] | None = None,
    momentum_lookback_days: int = DEFAULT_MOMENTUM_LOOKBACK_DAYS,
    momentum_skip_days: int = DEFAULT_MOMENTUM_SKIP_DAYS,
    yfinance_factory: Callable[[str], object] | None = None,
) -> FactorScores:
    """Convenience: compute all four factors in one call using yfinance.

    This is the optional "production" path that talks to Yahoo Finance. The
    core factor functions (compute_*) are pure and don't touch the network.

    Args:
        tickers: List of ticker symbols.
        data_dict: Pre-loaded price data (from extractor.extract_data). If None,
            momentum factor will not be computed (since we can't compute it
            without price history). It's recommended to pass this.
        momentum_lookback_days: Lookback for momentum (passed to compute_momentum_factor).
        momentum_skip_days: Skip for momentum (passed to compute_momentum_factor).
        yfinance_factory: Optional factory that returns an object with .info
            (mimicking yfinance.Ticker). Useful for testing. Defaults to
            `lambda ticker: yfinance.Ticker(ticker)`.

    Returns:
        FactorScores with whatever factors could be computed. Tickers that
        lack data for a factor will be absent from that factor's series.
    """
    # Lazy import so that the module can be imported without yfinance installed
    if yfinance_factory is None:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for factors_from_yfinance. "
                "Install via `pip install yfinance` or pass yfinance_factory."
            ) from e
        yfinance_factory = lambda t: yf.Ticker(t)

    fundamentals: dict[str, dict[str, float]] = {}
    market_caps: dict[str, float] = {}

    for ticker in tickers:
        try:
            info = yfinance_factory(ticker).info
        except Exception:
            # Skip ticker on any failure; do not abort the whole loop
            continue

        if not isinstance(info, dict):
            continue

        # Value: P/E and P/B
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        if pe is not None or pb is not None:
            fundamentals[ticker] = {}
            if pe is not None:
                fundamentals[ticker]["pe"] = float(pe)
            if pb is not None:
                fundamentals[ticker]["pb"] = float(pb)

        # Size: market cap
        mc = info.get("marketCap")
        if mc is not None:
            market_caps[ticker] = float(mc)

        # Quality: ROE (earnings_stability would require a long history;
        # we leave it as NaN for now since yfinance doesn't give us
        # rolling earnings data easily)
        if ticker in fundamentals:
            roe = info.get("returnOnEquity")
            if roe is not None:
                fundamentals[ticker]["roe"] = float(roe)
            # earnings_stability omitted - requires separate computation

    momentum = pd.Series(dtype=float)
    if data_dict is not None:
        momentum = compute_momentum_factor(
            data_dict,
            lookback_days=momentum_lookback_days,
            skip_days=momentum_skip_days,
        )

    return FactorScores(
        momentum=momentum,
        value=compute_value_factor(fundamentals) if fundamentals else None,
        size=compute_size_factor(market_caps) if market_caps else None,
        quality=compute_quality_factor(fundamentals) if fundamentals else None,
    )
