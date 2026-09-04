from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .. import db
from ..config import settings
from ..evidence.checklist import CHECKLIST

router = APIRouter(tags=["health"])


@router.get("/")
async def root() -> dict:
    return {"service": "chargeback-guard", "version": "0.1.0",
            "docs": "/docs", "console": "/console"}


@router.get("/console", response_class=HTMLResponse, include_in_schema=False)
async def console() -> HTMLResponse:
    """The merchant console.

    One file, served by the API that feeds it. There is no build step and no
    second process: a reviewer clones the repository, runs `make api`, and has
    the screen. A toolchain between them and the demo would be a toolchain
    that can break on their machine rather than mine.
    """
    page = Path(__file__).resolve().parent.parent / "static" / "console.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="console.html is missing")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@router.get("/health")
async def health() -> dict:
    """Degraded-but-up is a valid state. Nothing here should 500 before setup."""
    db_ok = await db.healthy()
    components = {
        "database": "ok" if db_ok else ("unconfigured" if not settings.has_db else "down"),
        "stripe": "configured" if settings.has_stripe else "unconfigured",
        "razorpay": "configured" if settings.razorpay_key_id else "unconfigured",
        "agent": settings.agent_provider,
    }
    ready = db_ok
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "components": components,
        "categories_loaded": len(CHECKLIST),
        "mode": "test-only",
    }


@router.get("/checklist")
async def checklist() -> dict:
    """The category -> required-evidence table, for the UI and the write-up."""
    return {
        cat.value: {
            "label": spec.label,
            "visa_code": spec.visa_code,
            "mastercard_code": spec.mastercard_code,
            "required": [s.value for s in spec.required],
            "supporting": [s.value for s in spec.supporting],
            "base_win_rate": spec.base_win_rate,
            "guidance": spec.guidance,
        }
        for cat, spec in CHECKLIST.items()
    }
