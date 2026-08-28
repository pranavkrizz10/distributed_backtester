"""The task executed by RQ workers, with checkpoint/resume support.

If this process is killed mid-run (OOM, deploy, crash) and RQ retries the
job, `run_backtest_job` is called again with the SAME job_id. It loads
whatever checkpoint was last saved on that job row and hands it to the
engine, which then skips the already-completed folds instead of redoing
the whole walk-forward run from fold 1.
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

        resuming = bool(job.checkpoint)
        if resuming:
            log.info("job %s: resuming from checkpoint (%d folds already done)",
                      job_id, len(job.checkpoint))

        job.status = JobStatus.running
        job.started_at = job.started_at or _now()  # keep original start time across resumes
        session.commit()
        _publish(job_id, {
            "status": "running", "pct": 0,
            "message": "resuming from checkpoint" if resuming else "loading data",
        })

        p = job.params
        df = load_bars_sync(session, p["ticker"], p["start"], p["end"])
        n_splits = int(p["n_splits"])

        def progress_cb(done: int, total: int, message: str) -> None:
            pct = int(100 * done / total) if total else 0
            _publish(job_id, {"status": "running", "pct": pct, "message": message})

        def checkpoint_cb(folds: list[dict]) -> None:
            # Persist progress after every fold. This is the checkpoint write.
            job.checkpoint = folds
            session.commit()

        result = run_backtest(
            df,
            strategy=p["strategy"],
            initial_cash=float(p["initial_cash"]),
            commission_bps=float(p["commission_bps"]),
            slippage_bps=float(p["slippage_bps"]),
            n_splits=n_splits,
            progress_cb=progress_cb,
            checkpoint=job.checkpoint,
            checkpoint_cb=checkpoint_cb,
        )

        job.result = result
        job.status = JobStatus.done
        job.checkpoint = None  # terminal: the full result supersedes it
        job.finished_at = _now()
        session.commit()

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
            # NOTE: checkpoint is deliberately NOT cleared here. If this job
            # is retried (RQ retry, or you manually re-enqueue the same
            # job_id), run_backtest_job will pick the checkpoint back up and
            # resume instead of restarting. It's only cleared on success.
            session.commit()
        _publish(job_id, {"status": "failed", "pct": 0, "message": str(exc)})
        log.exception("job %s failed", job_id)
        raise
    finally:
        session.close()
