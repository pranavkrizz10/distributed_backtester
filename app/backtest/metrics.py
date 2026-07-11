"""Performance metrics + bootstrap significance test.

All metrics are computed from a strategy *daily return* series. Sharpe etc. are
intentionally implemented directly rather than pulled from a library so the
assumptions (annualization factor, ddof) are explicit and auditable.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return 0.0
    return float((r.mean() / sd) * math.sqrt(periods_per_year))


def equity_curve(returns: pd.Series, initial: float = 10_000.0) -> pd.Series:
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def bootstrap_sharpe_pvalue(
    returns: pd.Series,
    n_boot: int = 2000,
    periods_per_year: int = TRADING_DAYS,
    seed: int = 42,
) -> float:
    """One-sided p-value for H0: true Sharpe <= 0.

    Simplified i.i.d. bootstrap: resample daily returns with replacement and
    recompute the Sharpe each time. p = fraction of bootstrap Sharpes <= 0.
    (A block/stationary bootstrap would respect autocorrelation; noted as a
    known simplification.)
    """
    r = returns.dropna().to_numpy()
    if len(r) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    n = len(r)
    factor = math.sqrt(periods_per_year)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(r, size=n, replace=True)
        sd = sample.std(ddof=1)
        boots[i] = (sample.mean() / sd) * factor if sd > 0 else 0.0
    return float(np.mean(boots <= 0.0))
