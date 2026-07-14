"""Async data ingestion engine.

Fetches historical OHLCV bars for many tickers concurrently and upserts them
into PostgreSQL.

Note on async: yfinance is a *synchronous* library under the hood, so each
provider call is offloaded to a thread via ``asyncio.to_thread``. Combined with
a semaphore (concurrency cap) and a crude inter-call delay (rate limit), this
gives genuine concurrent ingestion without blocking the event loop. Swapping in
Alpaca would replace the ``_fetch_one`` body with native async ``httpx`` calls.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .config import settings
from .db import AsyncSessionLocal, init_models
from .models import Bar

log = logging.getLogger("ingestion")

_rate_lock = asyncio.Lock()
_last_call = 0.0


async def _rate_limit() -> None:
    global _last_call
    async with _rate_lock:
        now = asyncio.get_event_loop().time()
        wait = settings.ingest_min_interval_s - (now - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = asyncio.get_event_loop().time()


def _fetch_sync(ticker: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(
        start=start.isoformat(), end=end.isoformat(), interval=interval, auto_adjust=False
    )
    if hist.empty:
        return hist
    hist = hist.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["open", "high", "low", "close", "volume"]]
    # Normalize index to tz-aware UTC.
    idx = hist.index
    hist.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    return hist


async def _fetch_one(
    ticker: str, start: dt.date, end: dt.date, interval: str
) -> tuple[str, pd.DataFrame]:
    last_exc: Exception | None = None
    for attempt in range(1, settings.ingest_max_retries + 1):
        try:
            await _rate_limit()
            df = await asyncio.to_thread(_fetch_sync, ticker, start, end, interval)
            return ticker, df
        except Exception as exc:  # network/provider hiccup -> backoff + retry
            last_exc = exc
            backoff = 0.5 * 2 ** (attempt - 1)
            log.warning("fetch %s failed (attempt %d): %s; retrying in %.1fs", ticker, attempt, exc, backoff)
            await asyncio.sleep(backoff)
    log.error("giving up on %s: %s", ticker, last_exc)
    return ticker, pd.DataFrame()


def _detect_gaps(df: pd.DataFrame) -> int:
    """Count missing business days within the returned range (rough heuristic)."""
    if df.empty:
        return 0
    expected = pd.bdate_range(df.index.min(), df.index.max(), tz="UTC")
    return int(len(expected) - len(df.index.unique()))


async def _store(ticker: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = [
        {
            "ticker": ticker.upper(),
            "ts": ts.to_pydatetime(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for ts, row in df.iterrows()
    ]
    async with AsyncSessionLocal() as session:
        stmt = pg_insert(Bar).values(records)
        # Idempotent: re-ingesting the same range is a no-op.
        stmt = stmt.on_conflict_do_nothing(constraint="uq_bar_ticker_ts")
        await session.execute(stmt)
        await session.commit()
    return len(records)


async def ingest_tickers(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
    interval: str = "1d",
) -> dict:
    await init_models()
    sem = asyncio.Semaphore(settings.ingest_max_concurrency)

    async def worker(t: str):
        async with sem:
            tk, df = await _fetch_one(t, start, end, interval)
            gaps = _detect_gaps(df)
            stored = await _store(tk, df)
            if gaps:
                log.info("%s: %d business-day gaps in returned range", tk, gaps)
            return {"ticker": tk, "bars": stored, "gaps": gaps}

    results = await asyncio.gather(*(worker(t) for t in tickers))
    summary = {"total_bars": sum(r["bars"] for r in results), "per_ticker": results}
    log.info("ingest complete: %s", summary["total_bars"])
    return summary
