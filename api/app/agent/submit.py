"""File the packet with the processor.

Three modes, and which one applies is decided by the data, not by a flag:

  stage      a real API call with submit=false. The evidence appears on the
             dispute in Stripe's dashboard and API but is not sent to the
             bank. Real, visible, reversible — the default.
  submit     the same call with submit=true. Irreversible; the issuer sees it.
  dry_run    the dispute exists only in our database, so there is nothing to
             file against. Returns the exact payload that would have been sent.

Most disputes here are generated, so dry_run is the common path. The ones that
arrived through a live `charge.dispute.created` webhook are real objects on
Stripe and take the first two.

The gate that matters: a packet the citation guard marked non-submittable is
refused. Filing a document containing a claim we could not source would
undo the entire point of checking.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .. import db
from ..config import settings
from ..evidence.adapters import to_razorpay, to_stripe
from ..evidence.kinds import KIND_TO_SLOT
from ..evidence.schema import Category, EvidencePacket, SlotValue, Strength

log = logging.getLogger(__name__)

STRIPE_API = "https://api.stripe.com/v1"
RAZORPAY_API = "https://api.razorpay.com/v1"

# Stripe's documented ceiling across every evidence field combined.
STRIPE_EVIDENCE_CHAR_LIMIT = 150_000
TIMEOUT = 20.0


class SubmissionRefused(RuntimeError):
    """The packet is not fit to file. Never a transport error."""


class NothingToSubmit(SubmissionRefused):
    """There is no packet here yet — which is not the same as refusing one.

    Worth separating, because the two answers mean opposite things to whoever
    is reading them. "Draft one first" is a step you have not taken; "this
    packet failed the citation guard" is a decision the system has made and
    will keep making. Collapsing them into one status code hides the second
    behind the first.
    """


def _to_evidence_packet(dispute: dict, packet: dict) -> EvidencePacket:
    """Rebuild the rail-agnostic packet from the stored draft.

    Claims become the narrative; the slots carry the concrete values the
    adapters map onto each processor's field names.
    """
    category = Category(packet.get("category") or dispute["category"])
    ep = EvidencePacket(dispute_id=dispute["id"], category=category)

    # Group the cited records by the kind of evidence they are, so the
    # adapter can place them in the right processor field.
    by_kind: dict[str, list[str]] = {}
    for tool in (packet.get("evidence") or {}).get("tools", []):
        for record in tool.get("records", []):
            by_kind.setdefault(record["kind"], []).append(record["summary"])

    for kind, summaries in by_kind.items():
        slot = KIND_TO_SLOT.get(kind)
        if slot is None:
            continue                      # precedent and anything else: not evidence
        existing = ep.slots.get(slot)
        text = "\n".join(summaries)
        ep.slots[slot] = SlotValue(
            slot=slot,
            value=f"{existing.value}\n{text}" if existing and existing.value else text,
            strength=Strength.FILLED,
        )

    ep.narrative = packet.get("narrative") or ""
    return ep


def _check(packet: dict) -> None:
    """The gate. Refuses before any network call, and says why in one line.

    The verdict is read from the stored packet, not recomputed and not taken
    from the caller. A draft that lost a claim to the guard stays unfilable
    however it is later asked to be filed.
    """
    guard = packet.get("guard") or {}
    rejected = int(guard.get("claims_rejected") or 0)
    if rejected:
        raise SubmissionRefused(
            f"{rejected} claim(s) failed the citation guard. "
            "A packet containing an unsourced assertion is not filed."
        )
    if guard and guard.get("submittable") is False:
        raise SubmissionRefused(
            "The citation guard marked this packet non-submittable."
        )
    if not packet.get("claims"):
        raise SubmissionRefused("The packet has no claims to submit.")


async def _check_still_worth_filing(dispute_id: str) -> None:
    """Refuse to file a dispute the system has decided not to contest.

    The refusal used to live only at drafting: `build_packet` declines to
    write a packet for a dispute triage recommends accepting. That left the
    obvious hole, and `make verify` walked into it on its first real run — a
    packet drafted before the decision changed is still sitting in the
    database, and submission read it out and built a payload from it without
    ever asking what the rule now says.

    A gate that stops a document being written but not sent is not a gate. It
    is checked here, at the point of action, because that is where the action
    is.
    """
    from ..triage import runner as triage_runner

    detail = await triage_runner.detail(dispute_id)
    decision = (detail or {}).get("decision") or {}
    if decision.get("recommendation") == "accept":
        raise SubmissionRefused(
            "Triage recommends accepting this dispute, so it is not filed. "
            + (decision.get("flips_if") or "")
        )


def _fits(body: dict[str, str]) -> dict[str, str]:
    """Trim to Stripe's combined character limit, longest field first.

    Truncating the longest field keeps every field present — losing one
    entirely would drop a whole category of evidence, which is worse than
    shortening the wordiest.
    """
    total = sum(len(v) for v in body.values())
    if total <= STRIPE_EVIDENCE_CHAR_LIMIT:
        return body
    over = total - STRIPE_EVIDENCE_CHAR_LIMIT
    longest = max(body, key=lambda k: len(body[k]))
    log.warning("Evidence is %d chars over the limit; trimming %s", over, longest)
    body[longest] = body[longest][: max(len(body[longest]) - over - 40, 200)] + "\n[trimmed]"
    return body


async def submit(dispute_id: str, mode: str = "stage") -> dict:
    """Stage, submit, or dry-run the latest packet for a dispute.

    `dry_run` forces the no-call path even for a dispute that does exist at
    the processor — useful for showing the payload without touching it. The
    other two degrade to dry_run on their own when there is nothing to file
    against.
    """
    if mode not in ("stage", "submit", "dry_run"):
        raise ValueError("mode must be 'stage', 'submit' or 'dry_run'")

    dispute = await db.fetchrow("select * from disputes where id = $1", dispute_id)
    if not dispute:
        raise NothingToSubmit(f"No dispute {dispute_id}")

    row = await db.fetchrow(
        "select * from evidence_packets where dispute_id = $1 "
        "order by created_at desc limit 1", dispute_id)
    if not row:
        raise NothingToSubmit(
            "No packet has been drafted for this dispute — "
            "POST /disputes/{id}/packet first.")

    packet = dict(row)
    for key in ("evidence", "gaps"):
        if isinstance(packet.get(key), str):
            packet[key] = json.loads(packet[key])
    packet["category"] = dispute["category"]
    packet["claims"] = await db.fetch(
        "select claim from evidence_citations where packet_id = $1", row["id"])
    guard = packet.get("guard")
    packet["guard"] = json.loads(guard) if isinstance(guard, str) else (guard or {})

    _check(packet)
    await _check_still_worth_filing(dispute_id)
    ep = _to_evidence_packet(dispute, packet)
    rail = dispute["rail"]

    if rail == "razorpay":
        body: dict[str, Any] = to_razorpay(ep, action="draft" if mode == "stage" else "submit")
    else:
        body = _fits(to_stripe(ep))

    processor_id = dispute.get("processor_dispute_id") or ""
    live = mode != "dry_run" and _filable(dispute)

    if not live:
        return await _record(dispute_id, row["id"], {
            "mode": "dry_run",
            "rail": rail,
            "reason": (
                "Dry run requested; nothing was sent."
                if mode == "dry_run" else _why_not_filable(dispute)
            ),
            "would_be_mode": "submitted" if mode == "submit" else "staged",
            "endpoint": (f"POST {STRIPE_API}/disputes/{processor_id or '{id}'}"
                         if rail == "stripe"
                         else f"POST {RAZORPAY_API}/disputes/{processor_id or '{id}'}/contest"),
            "payload": body,
            "characters": sum(len(str(v)) for v in body.values()),
        })

    if rail == "stripe":
        result = await _post_stripe(processor_id, body, submit=(mode == "submit"))
    else:
        result = await _post_razorpay(processor_id, body)

    return await _record(dispute_id, row["id"], result)


def _filable(dispute: dict) -> bool:
    """Whether there is really something at the processor to file against.

    This used to guess from the id, which was wrong twice over: the generator
    minted `du_...` ids that looked exactly like Stripe's, and the check
    passed them, so a submission went out against a dispute that did not
    exist. `No such dispute` is a poor way to learn where your data came from.

    Provenance is recorded on the row instead. Credentials are still required
    — without them there is no call to make.
    """
    if dispute.get("origin") != "processor":
        return False
    if not dispute.get("processor_dispute_id"):
        return False
    if dispute.get("rail") == "stripe":
        return bool(settings.stripe_secret_key)
    return bool(settings.razorpay_key_id)


def _why_not_filable(dispute: dict) -> str:
    """The reason, in the terms the reader needs to act on."""
    if dispute.get("origin") != "processor":
        return ("This dispute was generated locally — it has no counterpart at "
                "the processor to file against. Trigger a real one with "
                "`stripe trigger charge.dispute.created` to exercise the live "
                "path. The payload below is exactly what would be sent.")
    if not dispute.get("processor_dispute_id"):
        return ("The dispute arrived over a webhook but carries no processor "
                "id, so there is nothing to address. The payload below is "
                "exactly what would be sent.")
    rail = dispute.get("rail") or "stripe"
    key = "STRIPE_SECRET_KEY" if rail == "stripe" else "RAZORPAY_KEY_ID"
    return (f"This dispute is real, but {key} is not set, so no call can be "
            f"made. The payload below is exactly what would be sent.")


def stripe_form(evidence: dict[str, str], submit: bool) -> dict[str, str]:
    """The form body Stripe expects: nested keys, and the submit flag.

    `submit=false` stages the evidence on the dispute — visible in the API and
    dashboard, not sent to the bank. Getting this one field wrong is the
    difference between a draft and a filed representment, so it is built here
    where a test can read it rather than inline in the request.
    """
    form = {f"evidence[{k}]": v for k, v in evidence.items()}
    form["submit"] = "true" if submit else "false"
    return form


async def _post_stripe(processor_id: str, evidence: dict[str, str],
                       submit: bool, transport: httpx.AsyncBaseTransport | None = None
                       ) -> dict:
    form = stripe_form(evidence, submit)

    async with httpx.AsyncClient(timeout=TIMEOUT, transport=transport) as client:
        response = await client.post(
            f"{STRIPE_API}/disputes/{processor_id}",
            data=form,
            auth=(settings.stripe_secret_key, ""),
        )
    ok = response.status_code == 200
    payload = response.json()
    if not ok:
        log.error("Stripe rejected the submission: %s", payload)
    hint = None
    if response.status_code == 404:
        hint = ("Stripe has no such dispute. The row is marked as having come "
                "from the processor but the object is gone or belongs to "
                "another account — check the key in .env matches the account "
                "that received the webhook.")
    return {
        "hint": hint,
        # A rejected call staged nothing. Reporting the intent as the outcome
        # would leave a packet recorded as filed that the processor never took.
        "mode": ("submitted" if submit else "staged") if ok else "failed",
        "attempted": "submit" if submit else "stage",
        "rail": "stripe",
        "ok": ok,
        "status_code": response.status_code,
        "dispute_id": processor_id,
        "dashboard": f"https://dashboard.stripe.com/test/disputes/{processor_id}",
        "evidence_details": payload.get("evidence_details"),
        "error": payload.get("error"),
        "characters": sum(len(v) for v in evidence.values()),
    }


async def _post_razorpay(processor_id: str, body: dict[str, Any]) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{RAZORPAY_API}/disputes/{processor_id}/contest",
            json=body,
            auth=(settings.razorpay_key_id, settings.razorpay_key_secret),
        )
    ok = response.status_code in (200, 201)
    payload = response.json()
    submitted = body.get("action") == "submit"
    return {
        "mode": ("submitted" if submitted else "staged") if ok else "failed",
        "attempted": "submit" if submitted else "stage",
        "rail": "razorpay",
        "ok": ok,
        "status_code": response.status_code,
        "dispute_id": processor_id,
        "response": payload if ok else None,
        "error": None if ok else payload,
    }


async def _record(dispute_id: str, packet_id: str, result: dict) -> dict:
    """Persist what was sent and what came back. A submission with no record
    of its response cannot be audited later."""
    await db.execute(
        """
        update evidence_packets
           set submitted_at = case when $3 then now() else submitted_at end,
               submission_response = $2::jsonb
         where id = $1
        """,
        packet_id, json.dumps(result, default=str),
        # Staged evidence has not reached the bank, so it is not a submission,
        # and a rejected call is not one either. Kept honest here rather than
        # in the UI.
        result.get("mode") == "submitted",
    )
    result["dispute_id_local"] = dispute_id
    return result
