"""Cache key hashing + thin async cache helpers."""
import hashlib
import json
from typing import Any, Optional

import redis.asyncio as aioredis


def params_hash(d: dict) -> str:
    """Deterministic hash of a params dict (order-independent)."""
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


async def cache_get_json(r: aioredis.Redis, key: str) -> Optional[Any]:
    raw = await r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


async def cache_set_json(r: aioredis.Redis, key: str, value: Any, ttl: int) -> None:
    await r.set(key, json.dumps(value, default=str), ex=ttl)
