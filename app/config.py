"""Environment-based configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Async URL is the source of truth; the sync URL is derived for the RQ worker.
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/backtester"
    redis_url: str = "redis://localhost:6379/0"

    # Ingestion tuning
    ingest_max_concurrency: int = 8
    ingest_min_interval_s: float = 0.2  # crude rate-limit between provider calls
    ingest_max_retries: int = 3

    # Cache TTLs (seconds)
    indicator_cache_ttl: int = 3600
    backtest_cache_ttl: int = 86400

    # RQ
    backtest_job_timeout: int = 600

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "+psycopg2")


settings = Settings()
