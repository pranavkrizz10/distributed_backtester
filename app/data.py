"""Bar loading helpers shared by the API (async) and worker (sync)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .models import Bar

_COLS = ["open", "high", "low", "close", "volume"]


def _to_df(rows: list[Bar]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_COLS, index=pd.DatetimeIndex([], name="ts"))
    df = pd.DataFrame(
        [
            {
                "ts": r.ts,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )
    df = df.set_index("ts").sort_index()
    return df


def _as_dt(d, end: bool = False) -> dt.datetime | None:
    if d is None:
        return None
    # Accept ISO strings (e.g. "2018-01-01" or "2018-01-01T09:30:00").
    if isinstance(d, str):
        d = d.strip()
        if "T" in d or " " in d:
            d = dt.datetime.fromisoformat(d)
        else:
            d = dt.date.fromisoformat(d)
    if isinstance(d, dt.datetime):
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    # plain date -> start or end of day, UTC
    t = dt.time.max if end else dt.time.min
    return dt.datetime.combine(d, t, tzinfo=dt.timezone.utc)


def _build_stmt(ticker: str, start, end):
    stmt = select(Bar).where(Bar.ticker == ticker.upper())
    s, e = _as_dt(start), _as_dt(end, end=True)
    if s is not None:
        stmt = stmt.where(Bar.ts >= s)
    if e is not None:
        stmt = stmt.where(Bar.ts <= e)
    return stmt.order_by(Bar.ts)


async def load_bars_async(session: AsyncSession, ticker: str, start=None, end=None) -> pd.DataFrame:
    res = await session.execute(_build_stmt(ticker, start, end))
    return _to_df(list(res.scalars().all()))


def load_bars_sync(session: Session, ticker: str, start=None, end=None) -> pd.DataFrame:
    res = session.execute(_build_stmt(ticker, start, end))
    return _to_df(list(res.scalars().all()))
