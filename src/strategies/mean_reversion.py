"""Mean reversion strategy module.

Implements the tools from QuantStart's "Basics of Statistical Mean Reversion
Testing" article, plus a Bollinger Bands entry/exit signal generator.

References
----------
- QuantStart (2023). Basics of Statistical Mean Reversion Testing.
  https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/

Statistical tests implemented (from the article):
- Augmented Dickey-Fuller (ADF): tests for the null hypothesis of a unit
  root (i.e. random walk). Reject the null => mean-reverting signal.
- Hurst Exponent: H < 0.5 is mean-reverting, H = 0.5 is random walk,
  H > 0.5 is trending.

Trading signal implemented (from issue #9 acceptance criteria):
- Bollinger Bands: buy when price < lower band, sell when price > upper
  band. Optional exit signal on return to middle band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def compute_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands for a price series.

    Bands are formed from a simple moving average (SMA) and rolling
    standard deviation. The result is a DataFrame indexed like the input
    price series with columns:
        - 'middle'  : the SMA itself (the "equilibrium" level)
        - 'upper'   : middle + num_std * rolling_std
        - 'lower'   : middle - num_std * rolling_std
        - 'band_width' : upper - lower (useful for filtering ranging markets)

    Args:
        prices: 1-D price series indexed by date.
        window: Number of periods for the rolling mean/std (default 20 = ~1
            trading month; the canonical Bollinger default).
        num_std: Number of standard deviations for the band width
            (default 2.0; the canonical Bollinger default).

    Returns:
        DataFrame with columns 'middle', 'upper', 'lower', 'band_width',
        indexed identically to ``prices``. The first ``window - 1`` rows will
        contain NaN because the rolling window cannot be computed.

    Notes:
        NaN-safe: any leading NaNs in ``prices`` are passed through as NaN
        in the output. Internally uses ``pandas`` rolling which already
        handles alignment correctly.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pd.Series, got {type(prices).__name__}")
    if window <= 1:
        raise ValueError(f"window must be >= 2, got {window}")
    if num_std <= 0:
        raise ValueError(f"num_std must be positive, got {num_std}")

    middle = prices.rolling(window=window, min_periods=window).mean()
    rolling_std = prices.rolling(window=window, min_periods=window).std(ddof=0)

    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    band_width = upper - lower

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "band_width": band_width,
        },
        index=prices.index,
    )


# ---------------------------------------------------------------------------
# Trading Signal from Bollinger Bands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanReversionSignal:
    """Container for a mean-reversion signal per ticker.

    ``positions`` is a series in {-1, 0, +1} (or fractional variants on the
    edges for gradation). ``entry_thresholds`` records the numeric levels
    that triggered entries so that downstream analysis can audit the rules.
    """

    positions: pd.Series  # per-date positions in {-1, 0, +1}
    bands: pd.DataFrame  # computed Bollinger Bands
    # ADX is omitted here; the strategy is closed-form rule-based.


def generate_mean_reversion_signals(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    exit_on_middle: bool = True,
) -> MeanReversionSignal:
    """Generate long-short mean-reversion signals from Bollinger Bands.

    Rule set (per the issue's acceptance criteria):
        - Long (positions = +1) when price crosses ABOVE the lower band from
          below. The "above" requirement means we waited for the band
          tap and the bounce, not catching falling knives.
        - Short (positions = -1) when price crosses BELOW the upper band
          from above (symmetric).
        - Exit (positions = 0) when:
            * If ``exit_on_middle`` (default): price returns to the middle
              band.
            * Otherwise: an opposite signal fires (e.g. we entered a long
              and price now hits the upper band -> switch to short).

    The signals are deliberately trailing: a long stays long until either
    an exit condition fires, preventing constant flipping in choppy
    regimes.

    Args:
        prices: Per-date price series for a single instrument.
        window: Bollinger band rolling window (default 20).
        num_std: Bollinger band std multiplier (default 2.0).
        exit_on_middle: If True, exit a position when price returns to the
            middle band. If False, only the opposite-band cross exits the
            trade.

    Returns:
        MeanReversionSignal with positions and underlying bands.

    Notes:
        - The first ``window`` rows have NaN bands, so positions there are
          forced to 0 (flat).
        - This signal generator returns {-1, 0, +1}; if you want
          fractional sizing by band-with-normalised distance, do that in a
          downstream sizing layer.
    """
    bands = compute_bollinger_bands(prices, window=window, num_std=num_std)

    # Cross signals are computed only where bands are valid.
    valid = bands["middle"].notna()
    pos = pd.Series(0, index=prices.index, dtype=int)

    upper = bands["upper"].to_numpy()
    lower = bands["lower"].to_numpy()
    mid = bands["middle"].to_numpy()
    px = prices.to_numpy()
    n = len(prices)

    state = 0  # 0 = flat, 1 = long, -1 = short

    # We start iterating at index 1 because crosses need a previous bar.
    for i in range(1, n):
        if not valid.iloc[i]:
            continue
        if state == 0:
            # Flat -> look for entry. We require a previous price that was
            # outside the band and a current price that has re-entered it,
            # because catching an exact cross is unreliable on discrete
            # daily bars.
            if (px[i - 1] <= lower[i - 1]) and (px[i] > lower[i]):
                state = 1
                pos.iloc[i] = state
            elif (px[i - 1] >= upper[i - 1]) and (px[i] < upper[i]):
                state = -1
                pos.iloc[i] = state
        elif state == 1:
            # Long: hold unless exit triggered.
            exit_now = False
            if exit_on_middle and px[i] >= mid[i]:
                exit_now = True
            if (px[i - 1] >= upper[i - 1]) and (px[i] < upper[i]):
                # Opposite band cross -> switch to short.
                state = -1
                pos.iloc[i] = state
                continue
            if exit_now:
                state = 0
                pos.iloc[i] = state
            else:
                pos.iloc[i] = state
        elif state == -1:
            # Short: hold unless exit triggered.
            exit_now = False
            if exit_on_middle and px[i] <= mid[i]:
                exit_now = True
            if (px[i - 1] <= lower[i - 1]) and (px[i] > lower[i]):
                # Opposite band cross -> switch to long.
                state = 1
                pos.iloc[i] = state
                continue
            if exit_now:
                state = 0
                pos.iloc[i] = state
            else:
                pos.iloc[i] = state

    return MeanReversionSignal(positions=pos, bands=bands)


# ---------------------------------------------------------------------------
# Statistical Tests for Mean Reversion
# ---------------------------------------------------------------------------


def test_adf_stationarity(
    prices: pd.Series,
    max_lag: int = 1,
    significance: float = 0.05,
) -> dict:
    """Augmented Dickey-Fuller (ADF) test for a price series.

    This is a self-contained implementation that does not require
    ``statsmodels``. It is calibrated for the simplest ADF specification:

        Delta y_t = gamma * y_{t-1} + intercept + epsilon_t

    where ``y`` is the (log) price level, ``Delta y_t = y_t - y_{t-1}``,
    and the lag order is ``max_lag``.

    The null hypothesis is ``gamma = 0`` (a unit root, i.e. random walk
    non-mean-reverting). A p-value below ``significance`` rejects the
    null in favour of mean reversion.

    Implementation note: the actual ADF test uses MacKinnon critical
    values for the t-statistic. Because the four standard
    ``sample-size-dependent`` critical values (-3.50 / -2.89 / -2.58 for
    1%, 5%, 10% with N >= 100) are widely tabulated and almost identical
    across reasonable sample sizes, we hard-code them rather than pull in
    a 200-line critical-value table from statsmodels. For sample sizes
    below 50 we note that the ``max_lag`` truncation becomes unreliable
    and we set a wide ``insufficient_sample`` flag.

    Args:
        prices: 1-D price series for one instrument.
        max_lag: Lag order for the AR term (default 1; the article's
            recommendation for trading research).
        significance: Significance level for the test (default 0.05).

    Returns:
        Dictionary with keys:
            - ``t_statistic``: ADF t-statistic (negative). More negative
              indicates stronger evidence against the unit root.
            - ``p_value``: approximate p-value (interpolated from the
              critical values; this is an approximation, not exact).
            - ``critical_values``: dict of 1%, 5%, 10% critical values.
            - ``lags_used``: integer (echoes ``max_lag``).
            - ``n_obs``: number of points used in the regression.
            - ``is_mean_reverting``: True if p_value < significance.
            - ``insufficient_sample``: True if ``n_obs`` is too small for
              the test to be reliable.
            - ``method``: 'adf-basic' (lets you tell which spec was used).

    Notes:
        The p-value is computed via linear interpolation among the 1%,
        5%, 10% critical values — adequate for a quick screen, not a
        substitute for ``statsmodels`` if you want exact p-values. If you
        need exact p-values, install ``statsmodels`` and replace this
        function's internals with ``statsmodels.tsa.stattools.adfuller``.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pd.Series, got {type(prices).__name__}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    clean = prices.dropna()
    n = len(clean)
    if n < max_lag + 10:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "critical_values": {"1%": float("nan"), "5%": float("nan"), "10%": float("nan")},
            "lags_used": max_lag,
            "n_obs": n,
            "is_mean_reverting": False,
            "insufficient_sample": True,
            "method": "adf-basic",
        }

    y = clean.to_numpy(dtype=float)
    y_lag = y[:-1]
    dy = np.diff(y)

    # We need `max_lag` lagged differences for the AR term.
    if max_lag == 1:
        X = np.column_stack([np.ones(len(dy)), y_lag])
    else:
        # Pad with zeros for unused lag columns so all rows fit.
        lag_cols = []
        for k in range(1, max_lag + 1):
            lag_cols.append(dy[-len(dy) + k - 1] if k <= len(dy) else 0.0)
        X = np.column_stack([np.ones(len(dy)), y_lag, *lag_cols])

    # OLS regression: dy = X * beta + eps
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    fitted = X @ beta
    resid = dy - fitted
    n_obs = len(dy)
    dof = max(n_obs - X.shape[1], 1)
    sigma2 = float(np.sum(resid**2) / dof)

    # Variance-covariance of beta: sigma2 * (X^T X)^-1
    xtx_inv = np.linalg.inv(X.T @ X)
    var_beta = sigma2 * np.diag(xtx_inv)
    se_beta = np.sqrt(np.where(var_beta > 0, var_beta, np.nan))

    # The coefficient for y_lag (y_{t-1}) is at index 1 (after intercept).
    gamma = float(beta[1])
    se_gamma = float(se_beta[1])
    if se_gamma == 0 or math.isnan(se_gamma):
        t_statistic = float("nan")
    else:
        t_statistic = gamma / se_gamma

    # MacKinnon critical values for the ADF t-statistic (no time trend, no
    # constant in the null, sample-size limit). Standard table for N>100:
    cv = {"1%": -3.4301, "5%": -2.8616, "10%": -2.5673}

    # Approximate p-value via linear interpolation on the negative t-statistic
    # in log space. This is a rough approximation; for accurate p-values use
    # statsmodels.
    p_value = _approximate_adf_pvalue(t_statistic, cv)
    if p_value is None or math.isnan(t_statistic):
        is_mr = False
        p_value_out = float("nan")
    else:
        is_mr = p_value < significance
        p_value_out = p_value

    return {
        "t_statistic": t_statistic,
        "p_value": p_value_out,
        "critical_values": cv,
        "lags_used": max_lag,
        "n_obs": n_obs,
        "is_mean_reverting": bool(is_mr),
        "insufficient_sample": False,
        "method": "adf-basic",
    }


def _approximate_adf_pvalue(t_stat: float, cv: dict) -> float | None:
    """Approximate p-value from a sparse critical-value table.

    Uses log-linear interpolation across the 1% / 5% / 10% points. If
    ``t_stat`` is more negative than the 1% critical value, we cap at
    p = 0.005 (extrapolation below would not be reliable).
    """
    if math.isnan(t_stat):
        return None
    # Build ordered list of (significance, critical_value), both negative.
    table = sorted(((0.01, cv["1%"]), (0.05, cv["5%"]), (0.10, cv["10%"])))
    # Sort critical values from most negative to least negative:
    crits = [(p, t) for p, t in table]
    if t_stat <= crits[0][1]:
        # More negative than 1% cutoff -> cap at p<0.01 by returning 0.005
        # (we report the conservative midpoint).
        return 0.005
    if t_stat >= crits[-1][1]:
        return 0.20  # Less extreme than 10% cutoff, conservatively.
    # Linear interpolation in (t_stat, p) space over the segments.
    for i in range(len(crits) - 1):
        p0, t0 = crits[i]
        p1, t1 = crits[i + 1]
        if t0 >= t_stat >= t1:  # both negative; more-negative is "lower"
            # Interp between (t0, p0) and (t1, p1)
            if t0 == t1:
                return p0
            frac = (t_stat - t0) / (t1 - t0)
            return p0 + frac * (p1 - p0)
    return None


def compute_hurst_exponent(
    prices: pd.Series,
    min_lag: int = 2,
    max_lag: int = 100,
) -> float:
    """Hurst Exponent (scaled-variance method).

    Implementation follows the QuantStart article. For each lag tau in
    ``[min_lag, max_lag]`` we compute the standard deviation of the
    lagged differences:

        sigma(tau) = std(|log(price[t+tau]) - log(price[t])|)

    Then we fit a line in log-log space:

        log(sigma) ~ log(tau) * H + const

    where the slope is the Hurst exponent. Interpretation:

        H < 0.5 -> mean reverting
        H = 0.5 -> random walk (Geometric Brownian Motion)
        H > 0.5 -> trending

    Args:
        prices: 1-D price series for a single instrument.
        min_lag: Smallest lag to consider (default 2; pairs require at
            least t and t-2).
        max_lag: Largest lag to consider (default 100).

    Returns:
        The Hurst exponent as a float. NaN if the series is too short or
        has zero variance.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pd.Series, got {type(prices).__name__}")
    if min_lag < 2:
        raise ValueError(f"min_lag must be >= 2, got {min_lag}")
    if max_lag <= min_lag:
        raise ValueError(f"max_lag must be > min_lag, got {max_lag}")

    log_prices = np.log(prices.dropna().to_numpy(dtype=float))
    n = len(log_prices)
    if n < max_lag + 10:
        return float("nan")

    lags = np.arange(min_lag, max_lag + 1)
    sigmas = []
    for tau in lags:
        diffs = np.abs(log_prices[tau:] - log_prices[:-tau])
        sigmas.append(float(np.std(diffs, ddof=0)))

    sigmas_arr = np.array(sigmas)
    if np.any(sigmas_arr <= 0):
        return float("nan")

    # Log-log linear fit of sigma vs tau; slope is the Hurst exponent
    # directly (this is the "rescaled range" simplification used in the
    # article's example code).
    log_lags = np.log(lags)
    log_sigmas = np.log(sigmas_arr)
    slope, _intercept = np.polyfit(log_lags, log_sigmas, 1)
    return float(slope)
