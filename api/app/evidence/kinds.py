"""Which checklist slot a retrieved record fills.

Shared between composition and submission on purpose. These two need to agree
about what a record *is*: the packet says which required slots are still empty,
and the submitter decides which processor field the record belongs in. Two
copies of this mapping would drift, and the packet would acknowledge a gap it
had just filled.
"""
from __future__ import annotations

from .schema import Slot

KIND_TO_SLOT: dict[str, Slot] = {
    "delivery_scan": Slot.SHIPPING_PROOF,
    "carrier_scan": Slot.SHIPPING_PROOF,
    "support_message": Slot.CUSTOMER_COMMUNICATION,
    "policy_acceptance": Slot.REFUND_POLICY,
    "refund": Slot.REFUND_CONFIRMATION,
    "access_event": Slot.ACCESS_ACTIVITY_LOG,
    "order": Slot.RECEIPT,
    "customer_history": Slot.CUSTOMER_HISTORY,
    # `precedent` is deliberately absent. A comparable dispute is an internal
    # signal, not evidence, and it never leaves the building.
}

# A carrier scan is not a delivery scan. Both point at SHIPPING_PROOF, but only
# one of them proves the parcel arrived, so the weaker kind must not be allowed
# to report the slot as filled.
WEAK_KINDS = {"carrier_scan"}

# Slots a record can fill only in the sense that we have *something*; the
# tracking number and dates ride along inside the delivery record rather than
# arriving as records of their own.
IMPLIED_BY_SHIPPING_PROOF = {
    Slot.SHIPPING_TRACKING,
    Slot.SHIPPING_DATE,
    Slot.SHIPPING_ADDRESS,
    Slot.SHIPPING_CARRIER,
}


def slots_filled(records: list) -> set[Slot]:
    """The slots genuinely covered by these records.

    `records` is any iterable of objects with `.kind`. Weak kinds do not count:
    a label that was created but never scanned is not proof of delivery, and
    treating it as such is how a packet ends up asserting something the record
    behind it does not say.
    """
    filled: set[Slot] = set()
    for record in records:
        kind = getattr(record, "kind", None)
        if kind is None or kind in WEAK_KINDS:
            continue
        slot = KIND_TO_SLOT.get(kind)
        if slot is None:
            continue
        filled.add(slot)
        if slot is Slot.SHIPPING_PROOF:
            filled |= IMPLIED_BY_SHIPPING_PROOF
    return filled
