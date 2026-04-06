"""Database engine and session management."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.exc import OperationalError, DBAPIError

from atomicx.config import get_settings

# Naming convention for constraints (Alembic-friendly)
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    metadata = metadata


# Global singletons to prevent multiple engines/pools
_engine = None
_session_factory = None


def get_engine():
    """Get or create async SQLAlchemy engine with optimized connection pooling.

    CRITICAL: Returns a singleton engine to prevent connection pool proliferation.
    Multiple engines = multiple pools = connection exhaustion.

    Pool configuration for high concurrency (DB max_connections=200):
    - pool_size: 20 (persistent pool for baseline operations)
    - max_overflow: 80 (allow bursts up to 100 total connections)
    - pool_timeout: 30s (wait for connection availability)
    - pool_recycle: 3600s (recycle connections every hour to prevent stale)
    - pool_pre_ping: True (validate connections before use)

    This allows learning batches and 50+ agents to operate without exhaustion.
    DB limit: 200, Pool limit: 100, Reserve: 100 for other operations.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_size=20,             # Larger persistent pool
            max_overflow=80,          # Allow bursts to 100 total
            pool_timeout=30,          # Wait 30s for connection
            pool_recycle=3600,        # Recycle after 1 hour
            pool_pre_ping=True,       # Validate before use
            pool_reset_on_return='rollback',  # Clean up connections
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create async session factory (singleton).

    CRITICAL: Returns a singleton factory to prevent multiple engines.
    """
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI — yields an async session with retry logic.

    Handles connection pool exhaustion with exponential backoff:
    - Retry up to 3 times on connection errors
    - Wait 0.5s, 1s, 2s between retries
    - Fail gracefully if pool remains exhausted
    """
    factory = get_session_factory()
    retries = 3
    backoff = 0.5

    for attempt in range(retries):
        try:
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                    return
                except Exception:
                    await session.rollback()
                    raise
        except (OperationalError, DBAPIError) as e:
            error_msg = str(e).lower()
            # Check if it's a connection pool exhaustion error
            if "too many clients" in error_msg or "connection" in error_msg:
                if attempt < retries - 1:
                    # Exponential backoff: 0.5s, 1s, 2s
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    # Final retry failed - raise with helpful message
                    raise RuntimeError(
                        f"Database connection pool exhausted after {retries} retries. "
                        f"Consider reducing concurrent operations or increasing pool size."
                    ) from e
            else:
                # Different error - raise immediately
                raise
