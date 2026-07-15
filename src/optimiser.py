from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.settings import MAXIMUM_ALLOCATION, MINIMUM_ALLOCATION, RISK_AVERSION


def calculate_mean_variance(
    data_dict: dict[str, pd.DataFrame],
    lookback_days: int = 252,  # ~1 year of trading days
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calculate covariance matrix from Returns columns.

    Uses only the last N trading days (default: 252 days / ~1 year) of data.

    Args:
        data_dict: Dictionary where each key is a ticker symbol and each value
            is a DataFrame containing at least a "Returns" column representing
            periodic returns for that asset.
        lookback_days: Number of trading days to look back (default: 252)

    Returns:
        Tuple containing:
        - mean_returns: pd.Series of historical mean returns (kept for backward
            compatibility, but NOT used by the optimiser — see optimize_portfolio)
        - cov_matrix: pd.DataFrame covariance matrix of returns across all tickers

    Note:
        The mean_returns returned here are HISTORICAL actual returns. For
        portfolio optimization, use the predicted returns from Prophet instead
        (passed as predicted_returns to optimize_portfolio_mean_variance).
    """
    # For each ticker, take the last N days
    filtered_data = {}
    for ticker, df in data_dict.items():
        # Take last N rows (most recent data)
        filtered_df = df.tail(lookback_days)
        if len(filtered_df) > 0:
            filtered_data[ticker] = filtered_df

    if not filtered_data:
        # Fallback: use all data if filtering leaves nothing
        filtered_data = data_dict

    # Build returns DataFrame from filtered data
    returns_df = pd.DataFrame({ticker: df["Returns"] for ticker, df in filtered_data.items()})

    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    return mean_returns, cov_matrix


def optimize_portfolio_mean_variance(
    data_dict: dict[str, pd.DataFrame],
    predicted_returns: dict[str, float],
    minimum_allocation: float = MINIMUM_ALLOCATION,
    maximum_allocation: float = MAXIMUM_ALLOCATION,
    risk_aversion: float = RISK_AVERSION,
) -> dict[str, float]:
    """
    Optimise portfolio using mean-variance.

    Uses Prophet's predicted returns as expected returns (mu) and historical
    covariance for risk. This is the FIX for the previous bug where historical
    mean returns were incorrectly used as expected returns.

    Args:
        data_dict: Dictionary of DataFrames with 'Returns' column
            (used only for covariance matrix calculation)
        predicted_returns: Dictionary mapping ticker -> Prophet-predicted next-step
            return. THIS is the expected return (mu) used in optimisation.
        minimum_allocation: Minimum allocation for each asset (default: MINIMUM_ALLOCATION)
        maximum_allocation: Maximum allocation for each asset (default: MAXIMUM_ALLOCATION)
        risk_aversion: Risk-aversion coefficient (lambda) (default: RISK_AVERSION)

    Returns:
        Dictionary mapping ticker to optimal weight, where weights sum to 1.0

    Raises:
        ValueError: If optimisation fails, or if predicted_returns is missing
            tickers that exist in data_dict

    Note:
        Previous bug (fixed 2026-07-15):
        - Old code used calculate_mean_variance() which returned historical mean
          returns and called them "expected returns"
        - This was wrong: historical mean != predicted future return
        - Fix: predicted_returns parameter is now REQUIRED, used as mu
        - Covariance still comes from historical data (good estimate of risk)
    """
    if not predicted_returns:
        raise ValueError(
            "predicted_returns is required. Pass the output of model.predict_for_tickers()."
        )

    # Validate ticker alignment
    tickers = list(data_dict.keys())
    missing = [t for t in tickers if t not in predicted_returns]
    if missing:
        raise ValueError(f"predicted_returns missing tickers: {missing}")

    # Calculate covariance from historical data (risk estimate)
    _, cov = calculate_mean_variance(data_dict)

    # Build mu vector from PREDICTED returns (not historical)
    # This is the fix: use Prophet's predictions, not historical mean
    mu = pd.Series(
        {ticker: float(predicted_returns[ticker]) for ticker in tickers},
        index=tickers,
    )

    num_assets = len(tickers)

    # Objective: maximise expected return - (lambda/2) * variance
    # minimise negative of it
    def objective(weights: np.ndarray) -> float:
        port_return = float(np.dot(weights, mu))
        port_var = float(np.dot(weights.T, np.dot(cov, weights)))
        return -(port_return - 0.5 * risk_aversion * port_var)

    # Constraint: sum(weights) == 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Bounds: enforce minimum allocation per asset
    bounds = tuple((minimum_allocation, maximum_allocation) for _ in range(num_assets))

    # Initial guess: equal weights
    initial_weights = np.array([1 / num_assets] * num_assets)

    # Run optimizer
    result = minimize(
        objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if not result.success:
        raise ValueError(f"Optimisation failed: {result.message}")

    # Build a typed dictionary of weights to satisfy static type checking
    weights: dict[str, float] = {
        ticker: float(weight) for ticker, weight in zip(tickers, result.x, strict=True)
    }
    return weights