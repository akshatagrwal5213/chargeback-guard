"""Chargeback propensity scoring.

Serves the trained LightGBM model when artifacts are present, and a documented
heuristic otherwise, so a fresh clone works before anyone runs training. The
response shape is identical either way — `model_version` says which you got.

Note the action a high score triggers: evidence_tier = "enhanced". It does NOT
decline the order. See docs/DEFENSE_ONLY.md.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import db
from ..config import settings
from ..ml import predict as ml
from ..ml.features import OrderRecord

log = logging.getLogger(__name__)
router = APIRouter(prefix="/score", tags=["scoring"])

HEURISTIC_VERSION = "heuristic-0"
HEURISTIC_THRESHOLD = 0.35


class ScoreRequest(BaseModel):
    order_id: str
    amount: float
    currency: str = "INR"
    created_at: datetime | None = None
    is_digital: bool = False
    avs_result: str | None = None
    cvv_result: str | None = None
    three_ds_status: str | None = None
    billing_country: str | None = None
    shipping_country: str | None = None
    bin_country: str | None = None
    account_age_days: int | None = None
    prior_disputes: int = 0
    is_guest: bool = False
    txns_last_24h: int = 0
    email_domain: str | None = None


class Reason(BaseModel):
    feature: str
    contribution: float
    detail: str


class ScoreResponse(BaseModel):
    order_id: str
    score: float = Field(ge=0.0, le=1.0)
    model_version: str
    threshold: float
    evidence_tier: str
    capture: list[str]
    reasons: list[Reason]


# What to collect pre-emptively when propensity is high. This is the entire
# point of scoring: a high score changes what we RECORD, not who we serve.
ENHANCED_CAPTURE = [
    "require_signature_on_delivery",
    "archive_listing_snapshot",
    "retain_session_and_device_log",
    "capture_policy_acceptance_receipt",
]

STANDARD_CAPTURE = ["archive_listing_snapshot", "capture_policy_acceptance_receipt"]


def heuristic_score(req: ScoreRequest) -> tuple[float, list[Reason]]:
    """Transparent stand-in for the trained model.

    Weights are stated, not learned. Replaced wholesale on day 2 by LightGBM
    with calibrated outputs; the response shape stays identical so nothing
    downstream has to change.
    """
    reasons: list[Reason] = []
    logit = -2.2  # base rate around 10%

    def bump(weight: float, feature: str, detail: str) -> None:
        nonlocal logit
        logit += weight
        reasons.append(Reason(feature=feature, contribution=weight, detail=detail))

    if req.avs_result and req.avs_result.upper() == "N":
        bump(0.9, "avs_result", "Billing address did not match")
    if req.cvv_result and req.cvv_result.upper() == "N":
        bump(1.1, "cvv_result", "CVV did not match")
    if req.three_ds_status == "authenticated":
        bump(-1.4, "three_ds_status", "3DS authenticated — liability usually shifts")
    if req.billing_country and req.shipping_country and req.billing_country != req.shipping_country:
        bump(0.7, "country_mismatch", "Billing and shipping countries differ")
    if req.bin_country and req.billing_country and req.bin_country != req.billing_country:
        bump(0.5, "bin_mismatch", "Card BIN country differs from billing country")
    if req.prior_disputes > 0:
        bump(min(0.6 * req.prior_disputes, 1.8), "prior_disputes",
             f"Customer has {req.prior_disputes} prior dispute(s)")
    if req.account_age_days is not None and req.account_age_days < 7:
        bump(0.6, "account_age_days", "Account created within the last week")
    if req.is_guest:
        bump(0.35, "is_guest", "Guest checkout — no account history")
    if req.txns_last_24h >= 4:
        bump(0.55, "velocity_24h", f"{req.txns_last_24h} transactions in 24h")
    if req.amount >= 25000:
        bump(0.4, "amount", "High ticket value relative to typical order")
    if req.is_digital:
        bump(0.3, "is_digital", "Digital goods cannot produce a delivery scan")

    score = 1.0 / (1.0 + pow(2.718281828, -logit))
    reasons.sort(key=lambda r: abs(r.contribution), reverse=True)
    return round(score, 4), reasons[:5]


def _to_record(req: ScoreRequest) -> OrderRecord:
    """Scoring request -> the canonical shape the feature code expects."""
    return OrderRecord(
        order_id=req.order_id,
        created_at=req.created_at or datetime.now(tz=timezone.utc),
        amount=req.amount,
        avs_result=req.avs_result,
        cvv_result=req.cvv_result,
        three_ds_status=req.three_ds_status,
        billing_country=req.billing_country,
        shipping_country=req.shipping_country,
        bin_country=req.bin_country,
        account_age_days=req.account_age_days,
        prior_disputes=req.prior_disputes,
        is_guest=req.is_guest,
        txns_card_24h=req.txns_last_24h,
        is_digital=req.is_digital,
        payer_email_domain=req.email_domain,
    )


@router.post("", response_model=ScoreResponse)
async def score_order(req: ScoreRequest) -> ScoreResponse:
    model = ml.load()

    if model is not None:
        record = _to_record(req)
        score, _raw = model.score(record)
        version, threshold = model.version, model.threshold
        reasons = [
            Reason(
                feature=c["feature"],
                contribution=c["contribution"],
                detail=f"{c['feature']} = {c['value']} {c['direction']} the score",
            )
            for c in model.explain(record)
        ]
    else:
        score, reasons = heuristic_score(req)
        version, threshold = HEURISTIC_VERSION, HEURISTIC_THRESHOLD

    tier = "enhanced" if score >= threshold else "standard"
    capture = ENHANCED_CAPTURE if tier == "enhanced" else STANDARD_CAPTURE

    if settings.has_db:
        try:
            await db.execute(
                """
                insert into order_scores (order_id, score, raw_score, model_version, reasons)
                values ($1, $2, $3, $4, $5::jsonb)
                on conflict (order_id) do update
                  set score = excluded.score,
                      raw_score = excluded.raw_score,
                      model_version = excluded.model_version,
                      reasons = excluded.reasons,
                      scored_at = now()
                """,
                req.order_id,
                score,
                score,
                version,
                __import__("json").dumps([r.model_dump() for r in reasons]),
            )
            await db.execute(
                "update orders set evidence_tier = $2 where id = $1", req.order_id, tier
            )
        except Exception as exc:
            log.warning("Could not persist score for %s: %s", req.order_id, exc)

    return ScoreResponse(
        order_id=req.order_id,
        score=score,
        model_version=version,
        threshold=threshold,
        evidence_tier=tier,
        capture=capture,
        reasons=reasons,
    )


@router.get("/{order_id}")
async def get_score(order_id: str) -> dict:
    if not settings.has_db:
        raise HTTPException(status_code=503, detail="Database not configured")
    row = await db.fetchrow("select * from order_scores where order_id = $1", order_id)
    if not row:
        raise HTTPException(status_code=404, detail="No score for that order")
    return row
