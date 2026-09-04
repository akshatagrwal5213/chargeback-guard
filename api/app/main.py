from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .config import settings
from .routers import disputes, health, scoring, webhooks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("chargeback-guard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup must never fail on missing or unreachable infrastructure.
    # A misconfigured database should surface on /health, not as a stack trace.
    await db.connect()

    if db.is_connected():
        try:
            await db.apply_schema()
        except Exception as exc:  # pragma: no cover - needs a live DB
            log.error("Schema apply failed (continuing): %s", exc)

    log.info(
        "Started | db=%s stripe=%s razorpay=%s agent=%s",
        "connected" if db.is_connected() else "off",
        "yes" if settings.has_stripe else "no",
        "yes" if settings.razorpay_key_id else "no",
        # Written before Gemini was the default, so it reported agent=no with
        # a working key in .env. `agent_provider` is the one answer.
        settings.agent_provider,
    )
    log.info("Console: http://localhost:8000/console")
    if not db.is_connected():
        log.info("Running without a database. /score and /checklist still work.")

    yield
    await db.disconnect()


app = FastAPI(
    title="chargeback-guard",
    description=(
        "Chargeback propensity scoring and automated representment. "
        "Defense-only: see docs/DEFENSE_ONLY.md"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(scoring.router)
app.include_router(disputes.router)
app.include_router(webhooks.router)
