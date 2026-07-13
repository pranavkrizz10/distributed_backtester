"""Redis connections and the RQ queue.

Redis plays two roles here:
  * broker for the RQ job queue
  * cache + pub/sub channel for live backtest progress
"""
import redis
import redis.asyncio as aioredis
from rq import Queue

from .config import settings

# Sync client for RQ (must NOT decode responses; RQ expects bytes).
sync_redis = redis.Redis.from_url(settings.redis_url)

# One queue for backtests.
backtest_queue = Queue(
    "backtests", connection=sync_redis, default_timeout=settings.backtest_job_timeout
)


def get_async_redis() -> aioredis.Redis:
    """Async client for the API (cache reads + pub/sub subscriber)."""
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def progress_channel(job_id: str) -> str:
    return f"backtest:progress:{job_id}"


def result_cache_key(params_hash: str) -> str:
    return f"backtest:result:{params_hash}"
