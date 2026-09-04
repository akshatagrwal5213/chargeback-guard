"""Category -> required evidence.

This is hardcoded domain knowledge, deliberately. What each dispute category
needs is settled; asking a model to re-derive it every run is slower, costlier
and wrong more often. The agent's job is to FILL this checklist, not invent it.

Network code references are for the write-up and the UI. Sources:
  Stripe   — docs.stripe.com/disputes/reason-codes-defense-requirements
  Razorpay — razorpay.com/docs/api/disputes/
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Category, Slot


@dataclass(frozen=True)
class CategorySpec:
    category: Category
    label: str
    visa_code: str
    mastercard_code: str
    # Required slots decide completeness. Supporting slots strengthen a packet
    # but their absence is not counted as a gap.
    required: list[Slot]
    supporting: list[Slot] = field(default_factory=list)
    # Digital goods cannot produce a delivery scan. When the order is digital,
    # these slots replace the physical-shipment requirements.
    digital_substitutes: dict[Slot, Slot] = field(default_factory=dict)
    # Base difficulty of winning this category, used as the triage prior.
    # Fraud-code representments are materially harder to win than a
    # not-received case backed by a delivery scan.
    base_win_rate: float = 0.35
    guidance: str = ""


CHECKLIST: dict[Category, CategorySpec] = {
    Category.FRAUDULENT: CategorySpec(
        category=Category.FRAUDULENT,
        label="Fraudulent / no cardholder authorization",
        visa_code="10.4",
        mastercard_code="4837",
        required=[
            Slot.AUTHORIZATION_PROOF,
            Slot.CUSTOMER_PURCHASE_IP,
            Slot.BILLING_ADDRESS,
            Slot.RECEIPT,
        ],
        supporting=[
            Slot.CUSTOMER_NAME,
            Slot.CUSTOMER_EMAIL,
            Slot.SHIPPING_PROOF,
            Slot.CUSTOMER_SIGNATURE,
            Slot.ACCESS_ACTIVITY_LOG,
        ],
        base_win_rate=0.21,
        guidance=(
            "Strongest position is AVS+CVV match plus 3DS authentication, plus "
            "device or IP matching prior undisputed orders from the same customer. "
            "3DS-authenticated transactions usually shift liability to the issuer."
        ),
    ),
    Category.PRODUCT_NOT_RECEIVED: CategorySpec(
        category=Category.PRODUCT_NOT_RECEIVED,
        label="Merchandise or services not received",
        visa_code="13.1",
        mastercard_code="4855",
        required=[
            Slot.SHIPPING_PROOF,
            Slot.SHIPPING_TRACKING,
            Slot.SHIPPING_DATE,
            Slot.SHIPPING_ADDRESS,
        ],
        supporting=[
            Slot.CUSTOMER_SIGNATURE,
            Slot.SHIPPING_CARRIER,
            Slot.CUSTOMER_COMMUNICATION,
            Slot.RECEIPT,
        ],
        digital_substitutes={
            Slot.SHIPPING_PROOF: Slot.ACCESS_ACTIVITY_LOG,
            Slot.SHIPPING_TRACKING: Slot.SERVICE_DOCUMENTATION,
            Slot.SHIPPING_DATE: Slot.SERVICE_DATE,
            Slot.SHIPPING_ADDRESS: Slot.CUSTOMER_EMAIL,
        },
        base_win_rate=0.62,
        guidance=(
            "A carrier delivery scan with date and address is close to decisive. "
            "A signature makes it stronger. Without any delivery scan this is "
            "usually not worth contesting."
        ),
    ),
    Category.PRODUCT_UNACCEPTABLE: CategorySpec(
        category=Category.PRODUCT_UNACCEPTABLE,
        label="Not as described or defective",
        visa_code="13.3",
        mastercard_code="4853",
        required=[
            Slot.PRODUCT_DESCRIPTION,
            Slot.CUSTOMER_COMMUNICATION,
            Slot.REFUND_POLICY,
        ],
        supporting=[Slot.SHIPPING_PROOF, Slot.TERMS_AND_CONDITIONS, Slot.RECEIPT],
        base_win_rate=0.34,
        guidance=(
            "Show the listing as it appeared at purchase, and communications "
            "showing the customer never raised the issue before disputing. "
            "A returns window they did not use is strong."
        ),
    ),
    Category.CREDIT_NOT_PROCESSED: CategorySpec(
        category=Category.CREDIT_NOT_PROCESSED,
        label="Credit not processed",
        visa_code="13.6",
        mastercard_code="4860",
        required=[Slot.REFUND_POLICY, Slot.REFUND_POLICY_DISCLOSURE],
        supporting=[
            Slot.REFUND_CONFIRMATION,
            Slot.REFUND_REFUSAL_EXPLANATION,
            Slot.CUSTOMER_COMMUNICATION,
        ],
        base_win_rate=0.48,
        guidance=(
            "Either the refund was issued — show the transaction record — or it "
            "was not owed, which needs the policy plus proof the customer "
            "accepted it at checkout."
        ),
    ),
    Category.SUBSCRIPTION_CANCELED: CategorySpec(
        category=Category.SUBSCRIPTION_CANCELED,
        label="Cancelled recurring transaction",
        visa_code="13.2",
        mastercard_code="4841",
        required=[Slot.CANCELLATION_POLICY, Slot.ACCESS_ACTIVITY_LOG],
        supporting=[
            Slot.CANCELLATION_POLICY_DISCLOSURE,
            Slot.CANCELLATION_REBUTTAL,
            Slot.CUSTOMER_COMMUNICATION,
        ],
        base_win_rate=0.44,
        guidance=(
            "Usage after the claimed cancellation date is the strongest single "
            "fact. Pair it with proof the cancellation terms were accepted."
        ),
    ),
    Category.DUPLICATE: CategorySpec(
        category=Category.DUPLICATE,
        label="Duplicate processing",
        visa_code="12.6.1",
        mastercard_code="4834",
        required=[Slot.DUPLICATE_CHARGE_ID, Slot.DUPLICATE_CHARGE_EXPLANATION],
        supporting=[Slot.RECEIPT, Slot.ORDER_CONFIRMATION],
        base_win_rate=0.57,
        guidance=(
            "Show the two charges are distinct: different order ids, different "
            "line items, different timestamps."
        ),
    ),
    Category.GENERAL: CategorySpec(
        category=Category.GENERAL,
        label="General / unrecognised",
        visa_code="12.5",
        mastercard_code="4831",
        required=[Slot.RECEIPT, Slot.ORDER_CONFIRMATION],
        supporting=[Slot.CUSTOMER_COMMUNICATION, Slot.CUSTOMER_PURCHASE_IP],
        base_win_rate=0.41,
        guidance=(
            "Often an unrecognised statement descriptor. Show the descriptor as "
            "it appeared alongside an itemised receipt."
        ),
    ),
}


def spec_for(category: Category) -> CategorySpec:
    return CHECKLIST[category]


def required_slots(category: Category, is_digital: bool = False) -> list[Slot]:
    """Required slots, with digital substitutions applied.

    A digital order cannot produce a delivery scan, so demanding one would
    manufacture a permanent gap and push every digital dispute to 'accept'.
    """
    spec = spec_for(category)
    if not is_digital or not spec.digital_substitutes:
        return list(spec.required)
    return [spec.digital_substitutes.get(s, s) for s in spec.required]
