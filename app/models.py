"""ORM models."""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Bar(Base):
    __tablename__ = "bars"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        UniqueConstraint("ticker", "ts", name="uq_bar_ticker_ts"),
        Index("ix_bars_ticker_ts", "ticker", "ts"),
    )


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class BacktestJob(Base):
    """A backtest run and its lifecycle state."""

    __tablename__ = "backtest_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, name="job_status"), nullable=False, default=JobStatus.queued
    )
    params_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=False)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(default=False)

    # CHECKPOINTING: completed walk-forward folds (each carrying its own OOS
    # returns) saved after every fold. On a retried/resumed run, the worker
    # loads this, skips folds already here, and continues from the next one.
    # Cleared back to None once the job reaches a terminal state (done/failed)
    # since the full result/error supersedes it at that point.
    checkpoint: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
