"""Trading strategies.

Each strategy maps a price DataFrame + params to a *target position* series in
{-1, 0, +1}. The engine is responsible for the lookahead-safe shift and costs;
strategies must only ever look at data up to and including the current bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Small grids kept deliberately tiny so walk-forward optimization is fast.
SMA_GRID = [{"fast": f, "slow": s} for f in (10, 20, 50) for s in (50, 100, 200) if f < s]
MR_GRID = [{"window": w, "z_entry": z} for w in (10, 20, 30) for z in (1.0, 1.5, 2.0)]


def param_grid(strategy: str) -> list[dict]:
    return {"sma_crossover": SMA_GRID, "mean_reversion": MR_GRID}[strategy]


def max_lookback(strategy: str, params: dict) -> int:
    """Bars of warmup a strategy needs before it can emit a valid signal."""
    if strategy == "sma_crossover":
        return int(params["slow"])
    if strategy == "mean_reversion":
        return int(params["window"])
    return 0


def _sma_crossover(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["close"]
    fast = close.rolling(int(params["fast"]), min_periods=int(params["fast"])).mean()
    slow = close.rolling(int(params["slow"]), min_periods=int(params["slow"])).mean()
    # Long when fast above slow, flat otherwise.
    sig = (fast > slow).astype(float)
    sig[fast.isna() | slow.isna()] = 0.0
    return sig


def _mean_reversion(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["close"]
    window = int(params["window"])
    z_entry = float(params["z_entry"])
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    z = (close - ma) / sd

    # Stateful: enter long when z < -z_entry, exit when z >= 0;
    # enter short when z > z_entry, exit when z <= 0. Hold otherwise.
    z_arr = z.to_numpy()
    pos = np.zeros(len(z_arr))
    state = 0.0
    for i, zi in enumerate(z_arr):
        if np.isnan(zi):
            state = 0.0
        elif state == 0.0:
            if zi < -z_entry:
                state = 1.0
            elif zi > z_entry:
                state = -1.0
        elif state == 1.0 and zi >= 0.0:
            state = 0.0
        elif state == -1.0 and zi <= 0.0:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=close.index)


def compute_signal(df: pd.DataFrame, strategy: str, params: dict) -> pd.Series:
    if strategy == "sma_crossover":
        return _sma_crossover(df, params)
    if strategy == "mean_reversion":
        return _mean_reversion(df, params)
    raise ValueError(f"unknown strategy: {strategy}")
