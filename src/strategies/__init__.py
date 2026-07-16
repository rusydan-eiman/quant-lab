"""Strategies package for quant-lab.

Each strategy module exposes a small API that produces a `pd.Series` of
position signals indexed by ticker symbol:

    positions: +1 = long, 0 = flat, -1 = short  (or fractional in [-1, 1])

Strategies are intentionally pure: take price data in, return signals out.
Portfolio construction (turning signals into weights) lives in
src/optimiser.py and src/pipeline.py.
"""

from .mean_reversion import (
    MeanReversionSignal,
    compute_bollinger_bands,
    compute_hurst_exponent,
    generate_mean_reversion_signals,
    test_adf_stationarity,
)

__all__ = [
    "MeanReversionSignal",
    "compute_bollinger_bands",
    "compute_hurst_exponent",
    "generate_mean_reversion_signals",
    "test_adf_stationarity",
]
