"""SQLAlchemy engines and session factories.

Two engines on purpose:
  * async (asyncpg)   -> used by the FastAPI app and the async ingestion engine
  * sync  (psycopg2)  -> used by RQ workers, which run synchronous job functions

The ORM models are driver-agnostic, so both share the same Base/metadata.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


# --- async (API + ingestion) ---
async_engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)

# --- sync (RQ worker) ---
sync_engine = create_engine(settings.sync_database_url, pool_pre_ping=True, future=True)
SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False, class_=Session)


async def get_session() -> AsyncSession:
    """FastAPI dependency."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_models() -> None:
    # Import models so they register on the metadata before create_all.
    from . import models  # noqa: F401

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def init_models_sync() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(sync_engine)
