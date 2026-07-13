"""The task executed by RQ workers.

Synchronous on purpose: RQ runs plain functions, and the backtest itself is
CPU-bound NumPy/Pandas. Reads/writes use the sync SQLAlchemy session; progress
is published to a Redis pub/sub channel that the API's WebSocket forwards.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

from .backtest.engine import run_backtest
from .cache import params_hash
from .config import settings
from .data import load_bars_sync
from .db import SyncSessionLocal
from .models import BacktestJob, JobStatus
from .redis_client import progress_channel, result_cache_key, sync_redis

log = logging.getLogger("worker")


def _publish(job_id: str, payload: dict) -> None:
    sync_redis.publish(progress_channel(job_id), json.dumps(payload, default=str))


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def run_backtest_job(job_id: str) -> None:
    session = SyncSessionLocal()
    try:
        job = session.get(BacktestJob, job_id)
        if job is None:
            log.error("job %s not found", job_id)
            return

        job.status = JobStatus.running
        job.started_at = _now()
        session.commit()
        _publish(job_id, {"status": "running", "pct": 0, "message": "loading data"})

        p = job.params
        df = load_bars_sync(session, p["ticker"], p["start"], p["end"])

        n_splits = int(p["n_splits"])

        def progress_cb(done: int, total: int, message: str) -> None:
            pct = int(100 * done / total) if total else 0
            _publish(job_id, {"status": "running", "pct": pct, "message": message})

        result = run_backtest(
            df,
            strategy=p["strategy"],
            initial_cash=float(p["initial_cash"]),
            commission_bps=float(p["commission_bps"]),
            slippage_bps=float(p["slippage_bps"]),
            n_splits=n_splits,
            progress_cb=progress_cb,
        )

        job.result = result
        job.status = JobStatus.done
        job.finished_at = _now()
        session.commit()

        # Populate the result cache so identical requests skip recomputation.
        sync_redis.setex(
            result_cache_key(params_hash(p)),
            settings.backtest_cache_ttl,
            json.dumps(result, default=str),
        )

        _publish(
            job_id,
            {"status": "done", "pct": 100, "message": "complete", "metrics": result["metrics"]},
        )
        log.info("job %s done: %s", job_id, result["metrics"])

    except Exception as exc:  # noqa: BLE001
        session.rollback()
        job = session.get(BacktestJob, job_id)
        if job is not None:
            job.status = JobStatus.failed
            job.error = str(exc)
            job.finished_at = _now()
            session.commit()
        _publish(job_id, {"status": "failed", "pct": 0, "message": str(exc)})
        log.exception("job %s failed", job_id)
        raise
    finally:
        session.close()
