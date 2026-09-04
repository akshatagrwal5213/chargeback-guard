"""Turn retrieved records into a rebuttal narrative.

The narrative is not free text. It is a list of claims, each carrying the refs
of the records it rests on:

    {"text": "The parcel was delivered on 14 July...", "refs": ["fulfillment_events:881"]}

Structuring it this way is what makes the guarantee checkable. Prose with a
bibliography at the end can be verified only by reading it; prose where every
sentence names its own sources can be verified by a function.

The system prompt tells the model it may only use the supplied records. That
alone would be a request, not a constraint — guard.py is what makes it one, by
checking each claim against the database afterwards and deleting the ones that
do not hold up.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from ..evidence.checklist import required_slots, spec_for
from ..evidence.kinds import slots_filled
from ..evidence.schema import Category, Slot
from .provider import Provider, ProviderError, TemplateProvider
from .records import Evidence

log = logging.getLogger(__name__)

SYSTEM = """You draft representment narratives for a merchant contesting a card \
chargeback. An issuer's analyst reads what you write.

Absolute rules:

1. Every claim must rest on the supplied records. You are given a numbered list \
of records, each with a reference like [fulfillment_events:881]. A claim may \
cite only references from that list.
2. Never state anything the records do not show. If evidence for a point is \
absent, do not make the point. Do not soften, imply, or infer it.
3. Never invent dates, names, amounts, tracking numbers or addresses. Copy them \
exactly from the records.
4. Where the checklist has a gap, say so plainly in `gaps_acknowledged`. An \
issuer trusts a submission that concedes what it cannot show.
5. Write plainly and briefly. Six to ten claims is usually right. No adjectives \
about the customer, no speculation about motive, no rhetoric.
6. Records of kind "precedent" describe OTHER disputes. They are context for \
the merchant only. Never cite them — how a different case resolved is not \
evidence about this transaction, and citing a loss argues against yourself.

Return JSON only, in exactly this shape:

{
  "summary": "One paragraph stating what the merchant asserts and why.",
  "claims": [
    {"text": "A single factual sentence.", "refs": ["table:id", "..."]}
  ],
  "gaps_acknowledged": ["What the records cannot show."]
}

Every entry in "claims" must have at least one ref. A claim you cannot source \
must be omitted, not included with an empty list."""


# Retrieved for the merchant's own decision-making, never put in front of the
# issuer. How comparable disputes resolved says nothing about whether THIS
# parcel was delivered, and a submission that cites the merchant's own losses
# argues against itself. The records stay available to the triage rule and to
# the model's reasoning; they are simply not evidence.
SUBMISSION_EXCLUDED = {"precedent"}

# What the merchant actually asserts, per category. One sentence, addressed to
# the issuer rather than to the drafter.
POSITION = {
    Category.PRODUCT_NOT_RECEIVED:
        "The merchandise was delivered to the cardholder's address.",
    Category.FRAUDULENT:
        "The transaction was authorised by the cardholder.",
    Category.PRODUCT_UNACCEPTABLE:
        "The goods supplied matched their description at the time of purchase.",
    Category.CREDIT_NOT_PROCESSED:
        "No credit was owed to the cardholder, or the credit due was issued.",
    Category.SUBSCRIPTION_CANCELED:
        "The subscription was active and in use for the period charged.",
    Category.DUPLICATE:
        "The two charges are for separate transactions.",
    Category.GENERAL:
        "The charge is valid and corresponds to an order the cardholder placed.",
}


@dataclass
class Claim:
    text: str
    refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"text": self.text, "refs": self.refs}


@dataclass
class Draft:
    summary: str
    claims: list[Claim]
    gaps_acknowledged: list[str] = field(default_factory=list)
    # Set when a model was configured but not used, and says which failure it
    # was. Without it, a template packet is indistinguishable from having no
    # key at all.
    fallback_reason: str = ""
    provider: str = "template"
    raw: str = ""

    def as_dict(self) -> dict:
        return {
            "summary": self.summary,
            "claims": [c.as_dict() for c in self.claims],
            "gaps_acknowledged": self.gaps_acknowledged,
            "provider": self.provider,
        }

    def narrative(self) -> str:
        """Flat prose, for the PDF and for the processor's text field."""
        body = " ".join(c.text for c in self.claims)
        parts = [self.summary.strip(), body.strip()]
        if self.gaps_acknowledged:
            parts.append("The merchant notes: " + " ".join(self.gaps_acknowledged))
        return "\n\n".join(p for p in parts if p)


def build_prompt(dispute: dict, evidence: Evidence, decision: dict | None = None) -> str:
    category = Category(dispute["category"])
    spec = spec_for(category)

    lines = [
        "DISPUTE",
        f"  category   {category.value} ({spec.label})",
        f"  network    Visa {spec.visa_code} / Mastercard {spec.mastercard_code}",
        f"  amount     {dispute.get('currency', 'INR')} {float(dispute['amount']):,.2f}",
        f"  phase      {dispute.get('phase', 'chargeback')}",
        f"  opened     {str(dispute.get('opened_at'))[:10]}",
        "",
        "WHAT THIS CATEGORY TURNS ON",
        f"  {spec.guidance}",
        "",
        "RECORDS YOU MAY CITE",
    ]
    summaries = evidence.summary_lines()
    lines.extend(f"  {line}" for line in summaries) if summaries else \
        lines.append("  (none — do not write a narrative)")

    notes = [r.note for r in evidence.results if r.note]
    if notes:
        lines += ["", "ABSENCES ON RECORD (these are facts too)"]
        lines += [f"  - {n}" for n in notes]

    if decision:
        lines += [
            "",
            "TRIAGE",
            f"  recommendation {decision.get('recommendation')}",
            f"  win estimate   {decision.get('win_probability')}",
        ]

    lines += [
        "",
        "Draft the representment narrative. Cite only the references above, "
        "exactly as written, including the square-bracket table and id.",
    ]
    return "\n".join(lines)


def _parse(text: str) -> dict:
    """Models sometimes wrap JSON in a fence despite being asked not to."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


async def compose_async(dispute: dict, evidence: Evidence, provider: Provider,
                        decision: dict | None = None, feedback: str = "") -> Draft:
    """Draft without blocking the event loop.

    Both vendor SDKs are synchronous, and so is the retry backoff. Calling
    them directly from an async endpoint stalls the whole server for the
    duration — every other request included — and from the outside that is
    indistinguishable from a hang. The work goes to a worker thread instead.
    """
    if isinstance(provider, TemplateProvider):
        return compose_from_template(dispute, evidence)
    return await asyncio.to_thread(
        compose, dispute, evidence, provider, decision, feedback
    )


def compose(dispute: dict, evidence: Evidence, provider: Provider,
            decision: dict | None = None, feedback: str = "") -> Draft:
    """Draft the narrative. Falls back to templates if the model is unusable.

    Synchronous — call `compose_async` from async code.
    """
    if isinstance(provider, TemplateProvider):
        return compose_from_template(dispute, evidence)

    prompt = build_prompt(dispute, evidence, decision)
    if feedback:
        prompt += f"\n\nCORRECTION\n{feedback}"

    try:
        raw = provider.complete(SYSTEM, prompt)
        data = _parse(raw)
    except ProviderError as exc:
        log.warning("Provider failed (%s) — composing from templates", exc)
        return compose_from_template(dispute, evidence,
                                     fallback_reason=f"{provider.name}: {exc}")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        # Usually truncation: the model hit its output ceiling mid-JSON. Worth
        # naming, because "provider: template" on its own looks like a missing
        # key when it is really a prompt that outgrew the response budget.
        log.warning("Unparseable model output (%s) — composing from templates", exc)
        return compose_from_template(
            dispute, evidence,
            fallback_reason=f"{provider.name} returned unparseable output: {exc}")

    claims = [
        Claim(text=str(c.get("text", "")).strip(),
              refs=[str(r).strip() for r in (c.get("refs") or [])])
        for c in data.get("claims", [])
        if str(c.get("text", "")).strip()
    ]
    return Draft(
        summary=str(data.get("summary", "")).strip(),
        claims=claims,
        gaps_acknowledged=[str(g) for g in data.get("gaps_acknowledged", []) if g],
        provider=provider.name,
        raw=raw,
    )


def deterministic_summary(dispute: dict, claim_count: int) -> str:
    """The merchant's position, stated without asserting anything new.

    Used by the template composer, and as the guard's replacement when a
    model-written summary asserts a fact no record supports. It says only what
    the category and the dispute header already say, so it cannot itself be
    ungrounded.
    """
    category = Category(dispute["category"])
    return (
        f"{POSITION[category]} "
        f"The merchant contests this dispute for "
        f"{dispute.get('currency', 'INR')} {float(dispute['amount']):,.2f} and "
        f"submits the {claim_count} record"
        f"{'' if claim_count == 1 else 's'} below in support."
    )


def compose_from_template(dispute: dict, evidence: Evidence,
                          fallback_reason: str = "") -> Draft:
    """Deterministic composition. One claim per record, sourced by construction.

    Less fluent than a model, and every bit as valid: the point of the packet
    is which records support the merchant's position, not the prose. This is
    what lets the repository run with no credentials at all.
    """
    category = Category(dispute["category"])

    claims: list[Claim] = []
    for result in evidence.results:
        for record in result.records:
            if record.kind in SUBMISSION_EXCLUDED:
                continue
            claims.append(Claim(
                text=f"{_lead(record.kind)} {record.summary}.",
                refs=[record.ref],
            ))

    # A gap is a required slot with nothing in it. It is emphatically not a
    # tool's note about its own empty result: `find_similar_disputes` reports
    # how comparable disputes went, which is precedent wearing a different
    # hat, and the access-log tool volunteers that a product "was genuinely
    # never used" — an admission against interest on a physical order. Both
    # were being submitted to the issuer under "The merchant notes:".
    records = [r for result in evidence.results for r in result.records]
    is_digital = any(
        bool(r.fields.get("is_digital")) for r in records if r.kind == "order")
    filled = slots_filled(records)
    gaps = [
        f"{_GAP_LABEL.get(slot, slot.value.replace('_', ' '))} is not on file."
        for slot in required_slots(category, is_digital)
        if slot not in filled
    ]

    # spec.guidance is an instruction to whoever assembles the packet — "show
    # the delivery scan", "prove the refund was not owed". It belongs in the
    # prompt, not in a document an issuer reads, where it looks like the
    # merchant reciting its own homework.
    summary = deterministic_summary(dispute, len(claims))
    return Draft(summary=summary, claims=claims, gaps_acknowledged=gaps,
                 provider="template", fallback_reason=fallback_reason)


# What a missing slot is called in a document an issuer reads.
_GAP_LABEL = {
    Slot.SHIPPING_PROOF: "A carrier delivery scan",
    Slot.SHIPPING_TRACKING: "A tracking number",
    Slot.SHIPPING_DATE: "A shipment date",
    Slot.SHIPPING_ADDRESS: "The delivery address",
    Slot.SHIPPING_CARRIER: "The carrier name",
    Slot.CUSTOMER_SIGNATURE: "A signature on delivery",
    Slot.ACCESS_ACTIVITY_LOG: "An access log",
    Slot.CUSTOMER_COMMUNICATION: "The customer conversation",
    Slot.REFUND_POLICY: "The accepted refund policy",
    Slot.RECEIPT: "The order receipt",
}


def _lead(kind: str) -> str:
    return {
        "order": "The order record shows",
        "customer_history": "The customer account shows",
        "delivery_scan": "The carrier recorded delivery:",
        "carrier_scan": "The carrier recorded",
        "support_message": "Support records show",
        "policy_acceptance": "The customer accepted a policy at checkout:",
        "refund": "The refund ledger shows",
        "access_event": "Access logs record",
        "precedent": "A comparable prior dispute:",
    }.get(kind, "The records show")
