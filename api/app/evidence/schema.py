"""The rail-agnostic evidence object.

The agent emits exactly this. Adapters translate it to Stripe or Razorpay.
Keeping one internal shape is what lets a single packet serve both rails.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    """Normalised dispute categories.

    These mirror Stripe's seven buckets, which are themselves a mapping over
    the networks' several-hundred reason codes. Razorpay reason codes map onto
    the same set — see adapters.RAZORPAY_REASON_MAP.
    """

    FRAUDULENT = "fraudulent"
    PRODUCT_NOT_RECEIVED = "product_not_received"
    PRODUCT_UNACCEPTABLE = "product_unacceptable"
    CREDIT_NOT_PROCESSED = "credit_not_processed"
    SUBSCRIPTION_CANCELED = "subscription_canceled"
    DUPLICATE = "duplicate"
    GENERAL = "general"


class Slot(StrEnum):
    """Evidence slots. Rail-agnostic names; adapters map these outward."""

    RECEIPT = "receipt"
    ORDER_CONFIRMATION = "order_confirmation"
    SHIPPING_PROOF = "shipping_proof"
    SHIPPING_CARRIER = "shipping_carrier"
    SHIPPING_TRACKING = "shipping_tracking_number"
    SHIPPING_DATE = "shipping_date"
    SHIPPING_ADDRESS = "shipping_address"
    CUSTOMER_SIGNATURE = "customer_signature"
    ACCESS_ACTIVITY_LOG = "access_activity_log"
    CUSTOMER_COMMUNICATION = "customer_communication"
    CUSTOMER_NAME = "customer_name"
    CUSTOMER_EMAIL = "customer_email_address"
    CUSTOMER_PURCHASE_IP = "customer_purchase_ip"
    BILLING_ADDRESS = "billing_address"
    PRODUCT_DESCRIPTION = "product_description"
    REFUND_POLICY = "refund_policy"
    REFUND_POLICY_DISCLOSURE = "refund_policy_disclosure"
    REFUND_CONFIRMATION = "refund_confirmation"
    REFUND_REFUSAL_EXPLANATION = "refund_refusal_explanation"
    CANCELLATION_POLICY = "cancellation_policy"
    CANCELLATION_POLICY_DISCLOSURE = "cancellation_policy_disclosure"
    CANCELLATION_REBUTTAL = "cancellation_rebuttal"
    DUPLICATE_CHARGE_ID = "duplicate_charge_id"
    DUPLICATE_CHARGE_EXPLANATION = "duplicate_charge_explanation"
    SERVICE_DATE = "service_date"
    SERVICE_DOCUMENTATION = "service_documentation"
    CUSTOMER_HISTORY = "customer_history"
    TERMS_AND_CONDITIONS = "terms_and_conditions"
    AUTHORIZATION_PROOF = "authorization_proof"
    NARRATIVE = "narrative"


class Strength(StrEnum):
    FILLED = "filled"
    WEAK = "weak"
    MISSING = "missing"


class Citation(BaseModel):
    """A claim and the record it came from.

    The agent may not assert anything without one of these. See
    agent/guard.py — a packet with an uncited claim does not render.
    """

    claim: str
    source_table: str
    source_id: str
    source_field: str | None = None


class SlotValue(BaseModel):
    slot: Slot
    value: str | None = None
    strength: Strength = Strength.MISSING
    citations: list[Citation] = Field(default_factory=list)
    note: str | None = None


class EvidencePacket(BaseModel):
    dispute_id: str
    category: Category
    slots: dict[Slot, SlotValue] = Field(default_factory=dict)
    narrative: str = ""
    narrative_citations: list[Citation] = Field(default_factory=list)

    @property
    def filled(self) -> list[Slot]:
        return [s for s, v in self.slots.items() if v.strength == Strength.FILLED]

    @property
    def gaps(self) -> list[Slot]:
        return [s for s, v in self.slots.items() if v.strength == Strength.MISSING]

    def completeness(self, required: list[Slot]) -> float:
        """Fraction of the category's required slots that are filled.

        This feeds the triage rule — it is the single strongest input to
        whether contesting is worth it.
        """
        if not required:
            return 0.0
        got = sum(
            1
            for s in required
            if self.slots.get(s) and self.slots[s].strength == Strength.FILLED
        )
        return got / len(required)
