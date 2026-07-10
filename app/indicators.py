"""Technical indicators. Pure functions over a close-price Series."""
from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing via EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0)  # if avg_loss == 0, RSI -> 100


def compute(close: pd.Series, indicator: str, period: int) -> pd.Series:
    if indicator == "sma":
        return sma(close, period)
    if indicator == "rsi":
        return rsi(close, period)
    raise ValueError(f"unknown indicator: {indicator}")
