"""FastAPI application: REST endpoints + live-progress WebSocket."""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from . import indicators as ind
from .cache import cache_get_json, cache_set_json, params_hash
from .config import settings
from .data import load_bars_async
from .db import get_session, init_models
from .jobs import run_backtest_job
from .models import BacktestJob, JobStatus
from .redis_client import (
    backtest_queue,
    get_async_redis,
    progress_channel,
    result_cache_key,
)
from .schemas import (
    BacktestAccepted,
    BacktestRequest,
    BacktestResult,
    IndicatorResponse,
    JobStatusResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models()
    yield


app = FastAPI(title="Distributed Backtesting Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------- prices
@app.get("/api/prices/{ticker}")
async def get_prices(
    ticker: str,
    start: str | None = Query(None),
    end: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    df = await load_bars_async(session, ticker, start, end)
    if df.empty:
        raise HTTPException(404, f"no bars for {ticker.upper()} (ingest it first)")
    return {
        "ticker": ticker.upper(),
        "bars": [
            {"ts": ts.isoformat(), **{k: float(df.loc[ts, k]) for k in df.columns}}
            for ts in df.index
        ],
    }


# ---------------------------------------------------------------- indicators
@app.get("/api/indicators/{ticker}", response_model=IndicatorResponse)
async def get_indicator(
    ticker: str,
    indicator: str = Query(..., pattern="^(sma|rsi)$"),
    period: int = Query(20, ge=2, le=400),
    start: str | None = Query(None),
    end: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    r = get_async_redis()
    cache_key = f"indicators:{ticker.upper()}:{indicator}:{period}:{start}:{end}"
    cached = await cache_get_json(r, cache_key)
    if cached is not None:
        await r.aclose()
        return IndicatorResponse(**cached, cached=True)

    df = await load_bars_async(session, ticker, start, end)
    if df.empty:
        await r.aclose()
        raise HTTPException(404, f"no bars for {ticker.upper()} (ingest it first)")

    series = ind.compute(df["close"], indicator, period).dropna()
    points = [{"ts": ts.isoformat(), "value": round(float(v), 6)} for ts, v in series.items()]
    payload = {"ticker": ticker.upper(), "indicator": indicator, "period": period, "points": points}

    await cache_set_json(r, cache_key, payload, settings.indicator_cache_ttl)
    await r.aclose()
    return IndicatorResponse(**payload, cached=False)


# ---------------------------------------------------------------- backtest
@app.post("/api/backtest", response_model=BacktestAccepted, status_code=202)
async def submit_backtest(
    req: BacktestRequest, session: AsyncSession = Depends(get_session)
):
    canonical = req.canonical()
    h = params_hash(canonical)
    job_id = str(uuid.uuid4())

    r = get_async_redis()
    cached = await cache_get_json(r, result_cache_key(h))
    await r.aclose()

    if cached is not None:
        # Cache hit: persist a completed job pointing at the cached result.
        job = BacktestJob(
            id=job_id,
            status=JobStatus.done,
            params_hash=h,
            params=canonical,
            result=cached,
            cache_hit=True,
        )
        session.add(job)
        await session.commit()
        return BacktestAccepted(job_id=job_id, status="done", cache_hit=True)

    job = BacktestJob(id=job_id, status=JobStatus.queued, params_hash=h, params=canonical)
    session.add(job)
    await session.commit()

    backtest_queue.enqueue(run_backtest_job, job_id, job_timeout=settings.backtest_job_timeout)
    return BacktestAccepted(job_id=job_id, status="queued", cache_hit=False)


@app.get("/api/backtest/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(BacktestJob, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        cache_hit=job.cache_hit,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@app.get("/api/backtest/{job_id}/results", response_model=BacktestResult)
async def get_job_results(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(BacktestJob, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != JobStatus.done:
        raise HTTPException(409, f"job is '{job.status.value}', results not ready")
    res = job.result or {}
    return BacktestResult(
        job_id=job.id,
        status=job.status.value,
        metrics=res.get("metrics", {}),
        folds=res.get("folds", []),
        equity_curve=res.get("equity_curve", []),
    )


# ---------------------------------------------------------------- websocket
@app.websocket("/api/backtest/{job_id}/stream")
async def stream_progress(websocket: WebSocket, job_id: str):
    """Stream live progress for a running job.

    Sends the current DB status first (covers jobs that finished before the
    client connected), then forwards Redis pub/sub progress events until the
    job reaches a terminal state.
    """
    await websocket.accept()

    # 1) Send a snapshot of current status (covers jobs already finished).
    from .db import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        job = await session.get(BacktestJob, job_id)
        if job is None:
            await websocket.send_json({"status": "error", "message": "job not found"})
            await websocket.close()
            return
        await websocket.send_json({"status": job.status.value, "pct": 100 if job.status == JobStatus.done else 0})
        if job.status in (JobStatus.done, JobStatus.failed):
            await websocket.close()
            return

    # 2) Subscribe to live progress.
    r = get_async_redis()
    pubsub = r.pubsub()
    await pubsub.subscribe(progress_channel(job_id))
    try:
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
            if msg is None:
                # keepalive ping
                await websocket.send_json({"status": "heartbeat"})
                continue
            data = msg["data"]
            try:
                payload = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                continue
            await websocket.send_json(payload)
            if payload.get("status") in ("done", "failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(progress_channel(job_id))
        await pubsub.aclose()
        await r.aclose()
        try:
            await websocket.close()
        except RuntimeError:
            pass
