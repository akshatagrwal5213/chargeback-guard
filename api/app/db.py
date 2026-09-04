"""Thin async Postgres layer. No ORM — the queries here are the agent's tools later."""
from __future__ import annotations

import asyncio
import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import asyncpg

from .config import settings

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Everything after this line in schema.sql needs pgvector and is applied
# tolerantly — a Postgres that will not grant the extension should still get
# the other twelve tables.
OPTIONAL_MARKER = "-- === REQUIRES PGVECTOR ==="


CONNECT_TIMEOUT_SECONDS = 10


async def connect() -> None:
    """Never raises. An unreachable database degrades the app, it does not kill it.

    You should be able to clone this repo and run it with no accounts at all,
    and a wrong connection string should tell you so on /health rather than
    crashing at startup with a stack trace.
    """
    global _pool

    if not settings.has_db:
        if settings.database_url.strip():
            log.warning(
                "DATABASE_URL looks like a placeholder — running without a database. "
                "Set a real connection string, or blank the line entirely."
            )
        else:
            log.warning("DATABASE_URL not set — running without a database.")
        return

    # Supabase's transaction pooler (port 6543) multiplexes connections and
    # cannot hold prepared statements, which asyncpg uses by default. Detect it
    # and disable the statement cache rather than failing later with a
    # confusing "prepared statement __asyncpg_stmt_1__ does not exist".
    kwargs: dict[str, Any] = {"min_size": 1, "max_size": 10, "command_timeout": 30}
    if ":6543" in settings.database_url:
        kwargs["statement_cache_size"] = 0
        log.info("Transaction pooler detected (:6543) — statement cache disabled.")

    try:
        _pool = await asyncio.wait_for(
            asyncpg.create_pool(settings.database_url, **kwargs),
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
        log.info("Postgres pool ready.")
    except asyncio.TimeoutError:
        _pool = None
        log.error(
            "Database connection timed out after %ss — continuing without it. "
            "Check the host in DATABASE_URL is reachable.",
            CONNECT_TIMEOUT_SECONDS,
        )
    except socket.gaierror as exc:
        _pool = None
        log.error(
            "Cannot resolve the database host (%s) — continuing without it. "
            "Check the hostname in DATABASE_URL.",
            exc,
        )
    except (asyncpg.PostgresError, OSError) as exc:
        _pool = None
        log.error(
            "Database connection failed (%s: %s) — continuing without it. "
            "The API will report 'degraded' on /health.",
            type(exc).__name__,
            exc,
        )


async def disconnect() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def is_connected() -> bool:
    """Whether a pool actually exists — distinct from whether one was configured."""
    return _pool is not None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL in .env")
    return _pool


async def healthy() -> bool:
    if _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("select 1")
        return True
    except Exception as exc:  # pragma: no cover - depends on live DB
        log.warning("DB health check failed: %s", exc)
        return False


async def fetch(query: str, *args: Any) -> list[dict[str, Any]]:
    async with pool().acquire() as conn:
        rows = await conn.fetch(query, *args)
    return [dict(r) for r in rows]


async def fetchrow(query: str, *args: Any) -> dict[str, Any] | None:
    async with pool().acquire() as conn:
        row = await conn.fetchrow(query, *args)
    return dict(row) if row else None


async def execute(query: str, *args: Any) -> str:
    async with pool().acquire() as conn:
        return await conn.execute(query, *args)


@asynccontextmanager
async def transaction():
    async with pool().acquire() as conn:
        async with conn.transaction():
            yield conn


async def apply_schema() -> None:
    """Idempotent. Safe to run on every boot and from `make schema`.

    Extensions are applied one at a time and tolerated on failure. pgvector is
    not needed until similarity search lands on day 8, and a managed Postgres
    that will not grant it must not take the whole schema down with it.
    """
    sql = SCHEMA_PATH.read_text()

    # Anything below this marker depends on pgvector and is allowed to fail.
    core, _, optional = sql.partition(OPTIONAL_MARKER)

    extension_lines: list[str] = []
    body_lines: list[str] = []
    for line in core.splitlines():
        target = (
            extension_lines
            if line.strip().lower().startswith("create extension")
            else body_lines
        )
        target.append(line)

    async with pool().acquire() as conn:
        for stmt in extension_lines:
            name = stmt.strip().rstrip(";").split()[-1].strip('"')
            try:
                await conn.execute(stmt)
            except Exception as exc:
                log.warning("Extension %s unavailable (%s) — continuing.", name, exc)

        await conn.execute("\n".join(body_lines))
        log.info("Schema applied.")

        if optional.strip():
            try:
                await conn.execute(optional)
                log.info("Optional pgvector objects applied.")
            except Exception as exc:
                log.warning(
                    "Skipped pgvector objects (%s). Only similarity search on "
                    "day 8 depends on these.",
                    exc,
                )
