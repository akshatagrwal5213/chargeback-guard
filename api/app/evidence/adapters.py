"""Translate the internal evidence packet onto each processor's own schema.

Two rails, one packet. Stripe is what the demo triggers live; Razorpay is the
Indian rail the same packet has to survive contact with.

Field names verified against:
  docs.stripe.com/api/disputes/object
  razorpay.com/docs/api/disputes/contest/
"""
from __future__ import annotations

from .schema import Category, EvidencePacket, Slot, Strength

# --------------------------------------------------------------------- Stripe

# Stripe splits its evidence hash in two, and the split is not visible in the
# field names. Nine fields take the ID of an uploaded file; the rest take text.
# Posting prose into one of the file fields is rejected with
# `No such file_upload: '<your text>'` — which is how this was found: a real
# staged submission against a real dispute, not a test.
#
# Verified against docs.stripe.com/api/disputes/update, the `evidence` object.
STRIPE_FILE_FIELDS: frozenset[str] = frozenset({
    "cancellation_policy",
    "customer_communication",
    "customer_signature",
    "duplicate_charge_documentation",
    "receipt",
    "refund_policy",
    "service_documentation",
    "shipping_documentation",
    "uncategorized_file",
})

# Most text fields are individually capped at 20,000 characters.
STRIPE_TEXT_LIMIT = 20_000

# Slot -> Stripe *text* field. Only text fields appear here. A slot whose
# natural home is a file field goes into the written statement instead, below.
STRIPE_SLOT_MAP: dict[Slot, str] = {
    Slot.SHIPPING_CARRIER: "shipping_carrier",
    Slot.SHIPPING_TRACKING: "shipping_tracking_number",
    Slot.SHIPPING_DATE: "shipping_date",
    Slot.SHIPPING_ADDRESS: "shipping_address",
    Slot.ACCESS_ACTIVITY_LOG: "access_activity_log",
    Slot.CUSTOMER_NAME: "customer_name",
    Slot.CUSTOMER_EMAIL: "customer_email_address",
    Slot.CUSTOMER_PURCHASE_IP: "customer_purchase_ip",
    Slot.BILLING_ADDRESS: "billing_address",
    Slot.PRODUCT_DESCRIPTION: "product_description",
    # A record that the policy was shown and accepted is disclosure evidence,
    # and disclosure is text. The policy document itself would be the file.
    Slot.REFUND_POLICY: "refund_policy_disclosure",
    Slot.REFUND_POLICY_DISCLOSURE: "refund_policy_disclosure",
    Slot.REFUND_REFUSAL_EXPLANATION: "refund_refusal_explanation",
    Slot.CANCELLATION_POLICY: "cancellation_policy_disclosure",
    Slot.CANCELLATION_POLICY_DISCLOSURE: "cancellation_policy_disclosure",
    Slot.CANCELLATION_REBUTTAL: "cancellation_rebuttal",
    Slot.DUPLICATE_CHARGE_ID: "duplicate_charge_id",
    Slot.DUPLICATE_CHARGE_EXPLANATION: "duplicate_charge_explanation",
    Slot.SERVICE_DATE: "service_date",
    Slot.NARRATIVE: "uncategorized_text",
}

# Everything we hold as text but Stripe expects as a document. It is not
# dropped: it goes into the written statement under a heading, so an analyst
# reads the same facts they would have read off an attachment. Uploading the
# rendered packet as a real file would fill the file fields properly; until
# then, saying it in words beats not saying it.
STRIPE_STATEMENT_SECTION: dict[Slot, str] = {
    Slot.RECEIPT: "Transaction",
    Slot.ORDER_CONFIRMATION: "Order confirmation",
    Slot.SHIPPING_PROOF: "Delivery",
    Slot.CUSTOMER_SIGNATURE: "Signature",
    Slot.CUSTOMER_COMMUNICATION: "Customer contact",
    Slot.SERVICE_DOCUMENTATION: "Service delivered",
    Slot.REFUND_CONFIRMATION: "Refunds",
    Slot.CUSTOMER_HISTORY: "Account history",
    Slot.TERMS_AND_CONDITIONS: "Terms accepted",
    Slot.AUTHORIZATION_PROOF: "Authorisation",
}

STRIPE_REASON_MAP: dict[str, Category] = {
    "fraudulent": Category.FRAUDULENT,
    "unrecognized": Category.GENERAL,
    "product_not_received": Category.PRODUCT_NOT_RECEIVED,
    "product_unacceptable": Category.PRODUCT_UNACCEPTABLE,
    "credit_not_processed": Category.CREDIT_NOT_PROCESSED,
    "subscription_canceled": Category.SUBSCRIPTION_CANCELED,
    "duplicate": Category.DUPLICATE,
    "customer_initiated": Category.GENERAL,
    "general": Category.GENERAL,
    "debit_not_authorized": Category.FRAUDULENT,
    "bank_cannot_process": Category.GENERAL,
    "check_returned": Category.GENERAL,
    "incorrect_account_details": Category.GENERAL,
    "insufficient_funds": Category.GENERAL,
    "noncompliant": Category.GENERAL,
}

# ------------------------------------------------------------------- Razorpay

RAZORPAY_SLOT_MAP: dict[Slot, str] = {
    Slot.SHIPPING_PROOF: "shipping_proof",
    Slot.SHIPPING_TRACKING: "shipping_proof",
    Slot.SHIPPING_DATE: "shipping_proof",
    Slot.SHIPPING_ADDRESS: "shipping_proof",
    Slot.CUSTOMER_SIGNATURE: "shipping_proof",
    Slot.RECEIPT: "billing_proof",
    Slot.ORDER_CONFIRMATION: "billing_proof",
    Slot.BILLING_ADDRESS: "billing_proof",
    Slot.AUTHORIZATION_PROOF: "billing_proof",
    Slot.CANCELLATION_POLICY: "cancellation_proof",
    Slot.CANCELLATION_POLICY_DISCLOSURE: "cancellation_proof",
    Slot.CANCELLATION_REBUTTAL: "cancellation_proof",
    Slot.CUSTOMER_COMMUNICATION: "customer_communication",
    Slot.PRODUCT_DESCRIPTION: "proof_of_service",
    Slot.SERVICE_DOCUMENTATION: "proof_of_service",
    Slot.SERVICE_DATE: "proof_of_service",
    Slot.REFUND_CONFIRMATION: "refund_confirmation",
    Slot.REFUND_POLICY: "refund_cancellation_policy",
    Slot.REFUND_POLICY_DISCLOSURE: "refund_cancellation_policy",
    Slot.REFUND_REFUSAL_EXPLANATION: "explanation_letter",
    Slot.TERMS_AND_CONDITIONS: "term_and_conditions",
    Slot.ACCESS_ACTIVITY_LOG: "access_activity_log",
    Slot.DUPLICATE_CHARGE_ID: "others",
    Slot.DUPLICATE_CHARGE_EXPLANATION: "others",
    Slot.CUSTOMER_PURCHASE_IP: "others",
    Slot.CUSTOMER_HISTORY: "others",
    Slot.NARRATIVE: "explanation_letter",
}

# Razorpay's phase ladder. Catching something at `retrieval` before it becomes
# a chargeback is the cheapest win available anywhere in this system.
RAZORPAY_PHASES = ("fraud", "retrieval", "chargeback", "pre_arbitration", "arbitration")

# Each escalation step carries its own cost, so the downside of contesting a
# weak case compounds. Multipliers feed the triage rule's escalation term.
PHASE_ESCALATION_MULTIPLIER: dict[str, float] = {
    "fraud": 0.5,
    "retrieval": 0.3,
    "chargeback": 1.0,
    "pre_arbitration": 2.2,
    "arbitration": 3.5,
}


def category_from_stripe(reason: str) -> Category:
    return STRIPE_REASON_MAP.get((reason or "").lower(), Category.GENERAL)


def category_from_razorpay(reason_code: str, reason_description: str = "") -> Category:
    """Razorpay passes the network's own code through, so match on description.

    Razorpay does not publish a fixed reason_code enum the way Stripe does —
    the value is issuer/network supplied. We match keywords, and fall back to
    GENERAL rather than guessing.
    """
    text = f"{reason_code} {reason_description}".lower()
    if any(k in text for k in ("fraud", "unauthor", "no cardholder")):
        return Category.FRAUDULENT
    if any(k in text for k in ("not received", "non receipt", "not provided", "non-receipt")):
        return Category.PRODUCT_NOT_RECEIVED
    if any(k in text for k in ("defective", "not as described", "counterfeit", "misrepresent")):
        return Category.PRODUCT_UNACCEPTABLE
    if any(k in text for k in ("credit not processed", "refund not", "credit not")):
        return Category.CREDIT_NOT_PROCESSED
    if any(k in text for k in ("cancel", "recurring", "subscription")):
        return Category.SUBSCRIPTION_CANCELED
    if any(k in text for k in ("duplicate", "paid by other")):
        return Category.DUPLICATE
    return Category.GENERAL


def _collect(packet: EvidencePacket, slot_map: dict[Slot, str]) -> dict[str, str]:
    """Fold internal slots into a rail's fields.

    Several internal slots can share one rail field (Razorpay is coarser than
    Stripe), so values are joined rather than overwritten.
    """
    out: dict[str, list[str]] = {}
    for slot, value in packet.slots.items():
        if value.strength == Strength.MISSING or not value.value:
            continue
        field = slot_map.get(slot)
        if not field:
            continue
        out.setdefault(field, []).append(value.value.strip())
    return {k: "\n\n".join(v) for k, v in out.items()}


def to_stripe(packet: EvidencePacket) -> dict[str, str]:
    """Body for `POST /v1/disputes/:id` — the `evidence[...]` hash.

    Text only. Nothing here may address a field that expects a file upload ID,
    and the result is filtered to guarantee it: a mapping mistake should cost
    a sentence, not the whole submission.
    """
    body = _collect(packet, STRIPE_SLOT_MAP)

    # The written statement: the narrative, then each documentary slot under
    # its own heading, in the order the map declares so it reads the same
    # way twice.
    parts: list[str] = []
    if packet.narrative:
        parts.append(packet.narrative.strip())
    for slot, heading in STRIPE_STATEMENT_SECTION.items():
        value = packet.slots.get(slot)
        if not value or value.strength == Strength.MISSING or not value.value:
            continue
        parts.append(f"{heading.upper()}\n{value.value.strip()}")
    if body.get("uncategorized_text"):
        parts.append(body["uncategorized_text"].strip())
    if parts:
        body["uncategorized_text"] = "\n\n".join(parts)

    return {
        field: text[:STRIPE_TEXT_LIMIT]
        for field, text in body.items()
        if field not in STRIPE_FILE_FIELDS and text
    }


def to_razorpay(packet: EvidencePacket, action: str = "submit") -> dict[str, object]:
    """Body for `POST /v1/disputes/:id/contest`.

    Razorpay expects document ids for most fields when contesting for real;
    the text we produce here is the summary/explanation plus the values that
    ride along as free text. `action` is "draft" while iterating, "submit" to file.
    """
    collected = _collect(packet, RAZORPAY_SLOT_MAP)
    body: dict[str, object] = {"action": action}
    if packet.narrative:
        prior = collected.get("explanation_letter", "")
        collected["explanation_letter"] = (
            f"{prior}\n\n{packet.narrative}".strip() if prior else packet.narrative
        )
    summary = collected.pop("explanation_letter", "")
    if summary:
        body["summary"] = summary[:5000]
        body["explanation_letter"] = summary
    body.update(collected)
    return body


def escalation_cost(base_cost: float, phase: str) -> float:
    """Scale the escalation term by where in the ladder the dispute sits."""
    return base_cost * PHASE_ESCALATION_MULTIPLIER.get(phase, 1.0)
