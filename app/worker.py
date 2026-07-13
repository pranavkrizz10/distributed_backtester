"""RQ worker entrypoint.

Run with:  python -m app.worker
(or the standard `rq worker backtests --url redis://localhost:6379/0`)
"""
import logging

from rq import Worker

from .db import init_models_sync
from .redis_client import backtest_queue, sync_redis

logging.basicConfig(level=logging.INFO)


def main() -> None:
    init_models_sync()  # ensure tables exist even if the worker starts first
    worker = Worker([backtest_queue], connection=sync_redis)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
