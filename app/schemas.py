"""Pydantic request/response models."""
from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---- prices / indicators ----
class Bar(BaseModel):
    ts: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorResponse(BaseModel):
    ticker: str
    indicator: str
    period: int
    points: list[dict]  # [{"ts": ..., "value": ...}]
    cached: bool = False


# ---- backtest ----
Strategy = Literal["sma_crossover", "mean_reversion"]


class BacktestRequest(BaseModel):
    ticker: str
    start: dt.date
    end: dt.date
    strategy: Strategy = "sma_crossover"
    initial_cash: float = 10_000.0
    commission_bps: float = Field(1.0, description="per-trade commission in basis points of turnover")
    slippage_bps: float = Field(2.0, description="per-trade slippage in basis points of turnover")
    n_splits: int = Field(5, ge=2, le=20, description="walk-forward folds")

    def canonical(self) -> dict:
        """Stable dict used for cache hashing (order-independent)."""
        return {
            "ticker": self.ticker.upper(),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "strategy": self.strategy,
            "initial_cash": self.initial_cash,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "n_splits": self.n_splits,
        }


class BacktestAccepted(BaseModel):
    job_id: str
    status: str
    cache_hit: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    cache_hit: bool
    error: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    started_at: Optional[dt.datetime] = None
    finished_at: Optional[dt.datetime] = None


class BacktestResult(BaseModel):
    job_id: str
    status: str
    metrics: dict
    folds: list[dict]
    equity_curve: list[dict]
