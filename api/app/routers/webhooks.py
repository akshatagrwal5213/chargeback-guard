"""Processor webhooks.

Stripe signature verification is implemented locally rather than via the SDK
so the receiver works before you have the stripe package installed, and so the
verification logic is visible to a reviewer.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from .. import db
from ..config import settings
from ..evidence.adapters import category_from_razorpay, category_from_stripe

log = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])

SIGNATURE_TOLERANCE_SECONDS = 300


def verify_stripe_signature(payload: bytes, header: str, secret: str) -> bool:
    """Stripe's scheme: t=<timestamp>,v1=<hex hmac of "timestamp.payload">."""
    if not header or not secret:
        return False
    parts = dict(
        p.split("=", 1) for p in header.split(",") if "=" in p
    )
    timestamp = parts.get("t")
    signature = parts.get("v1")
    if not timestamp or not signature:
        return False

    try:
        age = abs(time.time() - int(timestamp))
    except ValueError:
        return False
    if age > SIGNATURE_TOLERANCE_SECONDS:
        log.warning("Stripe webhook rejected: timestamp outside tolerance (%.0fs)", age)
        return False

    signed = f"{timestamp}.".encode() + payload
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_razorpay_signature(payload: bytes, header: str, secret: str) -> bool:
    """Razorpay signs the raw body with the webhook secret, SHA256."""
    if not header or not secret:
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)


SELFTEST_MARKER = "chargeback_guard_selftest"


async def resolve_order_id(obj: dict) -> str | None:
    """Find the local order a dispute belongs to.

    A real dispute names the charge, not the order. `metadata.order_id` is the
    happy path — set it when the payment is created and the link is free — but
    it is only there if someone remembered, and it is never there on a dispute
    Stripe raised from a test card.

    So the second attempt is the one that matters in practice: match the
    dispute's charge or payment intent against the payment id we recorded on
    the order ourselves. We know our own payment ids; we do not need to be
    told them back.

    The third attempt asks the processor. The metadata we set when the payment
    was created lives on the charge, not on the dispute, and a dispute is the
    one object in the chain that never carries it. One extra call at dispute
    time is cheap, and it is the only resolution available to a merchant whose
    orders predate this system.

    A dispute that matches nothing is returned unlinked. Attaching it to a
    guess would draft a packet citing another customer's delivery.
    """
    explicit = (obj.get("metadata") or {}).get("order_id")
    if explicit:
        return explicit

    candidates = [obj.get("charge"), obj.get("payment_intent")]
    candidates = [c for c in candidates if isinstance(c, str) and c]
    if not candidates:
        return None

    if db.is_connected():
        row = await db.fetchrow(
            "select id from orders where processor_payment_id = any($1::text[]) limit 1",
            candidates,
        )
        if row:
            log.info("Dispute matched order %s by payment id", row["id"])
            return row["id"]

    return await _ask_stripe_for_order(obj)


async def _ask_stripe_for_order(obj: dict,
                                transport: httpx.AsyncBaseTransport | None = None
                                ) -> str | None:
    """Read `metadata.order_id` off the charge, or off its payment intent.

    Best effort by design: a dispute must be recorded whether or not this
    lookup succeeds, so every failure here is logged and swallowed. Losing the
    link costs evidence; letting the call fail the webhook would lose the
    dispute, and Stripe would stop retrying long before anyone noticed.
    """
    if not settings.stripe_secret_key:
        return None

    lookups = []
    if isinstance(obj.get("charge"), str) and obj["charge"]:
        lookups.append(("charges", obj["charge"]))
    if isinstance(obj.get("payment_intent"), str) and obj["payment_intent"]:
        lookups.append(("payment_intents", obj["payment_intent"]))

    try:
        async with httpx.AsyncClient(timeout=8.0, transport=transport) as client:
            for kind, ident in lookups:
                response = await client.get(
                    f"https://api.stripe.com/v1/{kind}/{ident}",
                    auth=(settings.stripe_secret_key, ""),
                )
                if response.status_code != 200:
                    continue
                body = response.json()
                order_id = (body.get("metadata") or {}).get("order_id")
                if order_id:
                    log.info("Dispute matched order %s via %s metadata", order_id, kind)
                    return order_id
                # A charge names its payment intent; follow it once.
                nested = body.get("payment_intent")
                if kind == "charges" and isinstance(nested, str) and nested:
                    lookups.append(("payment_intents", nested))
    except Exception as exc:                      # noqa: BLE001 — see docstring
        log.warning("Could not ask Stripe which order this dispute is for: %s", exc)
    return None


def origin_of(event: dict) -> str:
    """Whether the dispute behind this event exists at the processor.

    `make doctor` posts a correctly signed event of its own to prove the path
    works end to end. It is a real webhook and it writes a real row, but there
    is no dispute at Stripe behind it — so it must not be recorded as filable,
    or the diagnostic leaves behind exactly the trap that a submission against
    a non-existent dispute already sprang once.
    """
    obj = (event.get("data") or {}).get("object") or {}
    metadata = obj.get("metadata") or {}
    return "synthetic" if metadata.get(SELFTEST_MARKER) else "processor"


async def _record_dispute(
    *,
    rail: str,
    processor_dispute_id: str,
    order_id: str | None,
    amount: float,
    currency: str,
    category: str,
    processor_reason: str,
    phase: str,
    respond_by: datetime | None,
    payload: dict,
) -> str | None:
    if not db.is_connected():
        log.info("No DB configured — dispute %s not persisted.", processor_dispute_id)
        return None

    # An order_id we do not hold would violate the foreign key. Record the
    # dispute anyway and leave the link null — losing the dispute because we
    # cannot match it locally is strictly worse than an unlinked row.
    if order_id:
        known = await db.fetchrow("select 1 from orders where id = $1", order_id)
        if not known:
            log.warning(
                "Dispute %s references unknown order %s — recording unlinked.",
                processor_dispute_id,
                order_id,
            )
            order_id = None
    else:
        log.info(
            "Dispute %s carries no order_id in metadata — recording unlinked. "
            "Real orders should set metadata.order_id at payment creation.",
            processor_dispute_id,
        )

    dispute_id = f"dsp_{uuid.uuid4().hex[:16]}"
    await db.execute(
        """
        insert into disputes (
            id, order_id, rail, processor_dispute_id, amount, currency,
            category, processor_reason, phase, status, respond_by, raw_payload,
            origin
        )
        values ($1,$2,$3,$4,$5,$6,$7,$8,$9,'needs_response',$10,$11,$12)
        on conflict (processor_dispute_id) do nothing
        """,
        dispute_id,
        order_id,
        rail,
        processor_dispute_id,
        amount,
        currency,
        category,
        processor_reason,
        phase,
        respond_by,
        json.dumps(payload),
        # `make doctor` posts a correctly signed event of its own to prove the
        # path works. It is a real webhook and a real row, but there is no
        # dispute at Stripe behind it, so it must not be marked filable.
        origin_of(payload),
    )
    if order_id:
        await db.execute(
            "update orders set disputed = true, disputed_at = now() where id = $1",
            order_id,
        )
    log.info("Recorded dispute %s (%s, %s)", dispute_id, rail, category)
    return dispute_id


@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    background: BackgroundTasks,
    stripe_signature: str = Header(default="", alias="Stripe-Signature"),
) -> dict:
    payload = await request.body()

    if settings.stripe_webhook_secret:
        if not verify_stripe_signature(payload, stripe_signature, settings.stripe_webhook_secret):
            raise HTTPException(status_code=400, detail="Invalid signature")
    else:
        log.warning("STRIPE_WEBHOOK_SECRET unset — accepting unverified webhook (dev only).")

    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Malformed JSON")

    event_type = event.get("type", "")
    obj = event.get("data", {}).get("object", {})

    if event_type not in {
        "charge.dispute.created",
        "charge.dispute.updated",
        "charge.dispute.closed",
    }:
        return {"received": True, "handled": False, "type": event_type}

    if obj.get("livemode"):
        raise HTTPException(status_code=400, detail="Live-mode events are refused.")

    reason = obj.get("reason", "general")
    due_by = obj.get("evidence_details", {}).get("due_by")
    respond_by = (
        datetime.fromtimestamp(due_by, tz=timezone.utc) if due_by else None
    )

    dispute_id = await _record_dispute(
        rail="stripe",
        processor_dispute_id=obj.get("id", ""),
        order_id=await resolve_order_id(obj),
        amount=(obj.get("amount") or 0) / 100.0,
        currency=(obj.get("currency") or "inr").upper(),
        category=category_from_stripe(reason).value,
        processor_reason=reason,
        phase="chargeback" if not str(obj.get("status", "")).startswith("warning") else "retrieval",
        respond_by=respond_by,
        payload=event,
    )

    # Triage + evidence agent run happen off the request path — the processor
    # only needs a 200 back, quickly.
    if dispute_id and event_type == "charge.dispute.created":
        background.add_task(_kick_off_pipeline, dispute_id)

    return {"received": True, "handled": True, "dispute_id": dispute_id}


@router.post("/razorpay")
async def razorpay_webhook(
    request: Request,
    background: BackgroundTasks,
    x_razorpay_signature: str = Header(default="", alias="X-Razorpay-Signature"),
) -> dict:
    payload = await request.body()

    if settings.razorpay_key_secret:
        if not verify_razorpay_signature(payload, x_razorpay_signature, settings.razorpay_key_secret):
            raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Malformed JSON")

    event_type = event.get("event", "")
    if not event_type.startswith("payment.dispute"):
        return {"received": True, "handled": False, "type": event_type}

    entity = (
        event.get("payload", {}).get("dispute", {}).get("entity", {})
    )
    respond_by_ts = entity.get("respond_by")
    dispute_id = await _record_dispute(
        rail="razorpay",
        processor_dispute_id=entity.get("id", ""),
        order_id=entity.get("notes", {}).get("order_id"),
        amount=(entity.get("amount") or 0) / 100.0,
        currency=(entity.get("currency") or "inr").upper(),
        category=category_from_razorpay(
            entity.get("reason_code", ""), entity.get("reason_description", "")
        ).value,
        processor_reason=entity.get("reason_code", ""),
        phase=entity.get("phase", "chargeback"),
        respond_by=(
            datetime.fromtimestamp(respond_by_ts, tz=timezone.utc)
            if respond_by_ts
            else None
        ),
        payload=event,
    )

    if dispute_id and event_type == "payment.dispute.created":
        background.add_task(_kick_off_pipeline, dispute_id)

    return {"received": True, "handled": True, "dispute_id": dispute_id}


async def _kick_off_pipeline(dispute_id: str) -> None:
    """Triage, then the evidence agent. Both land in later days' work.

    Runs as a BackgroundTask, so anything raised here is invisible to the
    processor — which is correct (it only needs its 200) but means failures
    must be logged loudly rather than swallowed.
    """
    try:
        log.info(
            "Pipeline queued for %s (triage lands day 4, agent day 8)", dispute_id
        )
    except Exception:
        log.exception("Pipeline failed for %s", dispute_id)
