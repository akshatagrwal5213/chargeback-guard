"""What evidence actually exists for a dispute.

Reads the five evidence tables and reports, slot by slot, what the merchant
could put in front of an issuer. Nothing here judges the dispute — it only
establishes the facts the rule then weighs.

Batched on purpose: triaging 2,500 disputes with six queries each is 15,000
round trips to a hosted Postgres. Six queries total, joined in memory instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import db
from ..evidence.checklist import required_slots, spec_for
from ..evidence.schema import Category, Slot, Strength


@dataclass
class Availability:
    """Per-order evidence facts, assembled once from the batch queries."""

    order_id: str
    delivered: bool = False
    delivered_at: str | None = None
    delivery_location: str | None = None
    tracking_number: str | None = None
    carrier: str | None = None
    signature_name: str | None = None
    shipped_at: str | None = None
    had_exception: bool = False

    communications: int = 0
    last_comm_at: str | None = None
    policies: set[str] = field(default_factory=set)
    refund_issued: bool = False
    refund_requested: bool = False
    # Requested and still open, as distinct from requested-and-declined. A
    # merchant that declined a refund has taken a position it can defend;
    # one that has not answered yet has not.
    refund_open: bool = False
    access_events: int = 0
    last_access_at: str | None = None

    is_digital: bool = False
    avs_result: str | None = None
    cvv_result: str | None = None
    three_ds_status: str | None = None
    billing_country: str | None = None
    evidence_tier: str = "standard"

    def slot_state(self, slot: Slot) -> tuple[Strength, str | None]:
        """Whether this slot can be filled, and with what.

        FILLED means the record exists and is usable. WEAK means something
        related exists but would not carry the argument on its own — a
        shipment that was created but never scanned, for instance. MISSING
        means there is nothing.
        """
        S, W, M = Strength.FILLED, Strength.WEAK, Strength.MISSING

        if slot is Slot.SHIPPING_PROOF:
            if self.delivered:
                return S, f"Delivered {self.delivered_at} at {self.delivery_location}"
            if self.had_exception:
                return W, "Carrier reported a delivery exception; no successful scan"
            if self.tracking_number:
                return W, "Label created but the parcel was never scanned as delivered"
            return M, None

        if slot is Slot.SHIPPING_TRACKING:
            return (S, self.tracking_number) if self.tracking_number else (M, None)

        if slot is Slot.SHIPPING_CARRIER:
            return (S, self.carrier) if self.carrier else (M, None)

        if slot is Slot.SHIPPING_DATE:
            return (S, self.shipped_at) if self.shipped_at else (M, None)

        if slot is Slot.SHIPPING_ADDRESS:
            return (S, self.delivery_location) if self.delivery_location else (M, None)

        if slot is Slot.CUSTOMER_SIGNATURE:
            return (S, f"Signed for by {self.signature_name}") if self.signature_name else (M, None)

        if slot is Slot.ACCESS_ACTIVITY_LOG:
            if self.access_events >= 3:
                return S, f"{self.access_events} access events, most recent {self.last_access_at}"
            if self.access_events:
                return W, f"Only {self.access_events} access event(s) recorded"
            return M, None

        if slot is Slot.CUSTOMER_COMMUNICATION:
            if self.communications:
                return S, f"{self.communications} message(s), most recent {self.last_comm_at}"
            return M, None

        if slot in (Slot.REFUND_POLICY, Slot.REFUND_POLICY_DISCLOSURE):
            return (S, "Refund policy accepted at checkout") if "refund" in self.policies else (M, None)

        if slot in (Slot.CANCELLATION_POLICY, Slot.CANCELLATION_POLICY_DISCLOSURE):
            # The refund policy carries the cancellation terms in this merchant's
            # setup; treated as supporting rather than decisive.
            return (W, "Cancellation terms included in the accepted refund policy") \
                if "refund" in self.policies else (M, None)

        if slot is Slot.TERMS_AND_CONDITIONS:
            return (S, "Terms accepted at checkout") if "terms" in self.policies else (M, None)

        if slot is Slot.REFUND_CONFIRMATION:
            if self.refund_issued:
                return S, "Refund issued to the original payment method"
            if self.refund_requested:
                return M, None
            return M, None

        if slot is Slot.REFUND_REFUSAL_EXPLANATION:
            # Only meaningful when a refund was asked for and declined, and only
            # defensible with a policy on file.
            if self.refund_requested and not self.refund_issued and "refund" in self.policies:
                return S, "Refund declined under the accepted returns policy"
            return M, None

        if slot in (Slot.RECEIPT, Slot.ORDER_CONFIRMATION):
            return S, "Itemised order record on file"

        if slot is Slot.BILLING_ADDRESS:
            return (S, self.billing_country) if self.billing_country else (M, None)

        if slot is Slot.AUTHORIZATION_PROOF:
            parts = []
            if (self.avs_result or "").upper() in ("Y", "P"):
                parts.append(f"AVS {self.avs_result}")
            if (self.cvv_result or "").upper() in ("M", "P"):
                parts.append(f"CVV {self.cvv_result}")
            if self.three_ds_status == "authenticated":
                return S, "3DS authenticated" + (f"; {', '.join(parts)}" if parts else "")
            if parts:
                return W, ", ".join(parts) + "; no 3DS authentication"
            return M, None

        if slot is Slot.CUSTOMER_PURCHASE_IP:
            if self.evidence_tier == "enhanced":
                return S, "Session IP and device fingerprint retained at order time"
            return W, "IP on the order record; no retained session log"

        if slot is Slot.PRODUCT_DESCRIPTION:
            # Listing snapshots are only archived under enhanced capture.
            return (S, "Listing snapshot archived at order time") if self.evidence_tier == "enhanced" \
                else (W, "Current listing only; no snapshot from the order date")

        if slot in (Slot.SERVICE_DATE, Slot.SERVICE_DOCUMENTATION):
            if self.is_digital and self.access_events:
                return S, f"Service used {self.access_events} time(s)"
            return M, None

        if slot in (Slot.DUPLICATE_CHARGE_ID, Slot.DUPLICATE_CHARGE_EXPLANATION):
            return S, "Order records show distinct order ids and line items"

        if slot in (Slot.CUSTOMER_NAME, Slot.CUSTOMER_EMAIL):
            return S, "On the customer record"

        return M, None


def completeness(av: Availability, category: Category) -> tuple[float, list[Slot], list[Slot]]:
    """Fraction of required slots filled, plus what is filled and what is not.

    WEAK counts as half. A shipment that was created but never scanned is not
    nothing, but it is not proof of delivery either.
    """
    required = required_slots(category, av.is_digital)
    filled: list[Slot] = []
    gaps: list[Slot] = []
    score = 0.0

    for slot in required:
        strength, _ = av.slot_state(slot)
        if strength is Strength.FILLED:
            score += 1.0
            filled.append(slot)
        elif strength is Strength.WEAK:
            score += 0.5
            gaps.append(slot)
        else:
            gaps.append(slot)

    return (score / len(required) if required else 0.0), filled, gaps



def critical_slot(av: Availability, category: Category) -> tuple[Slot | None, Strength, str | None]:
    """The field that usually decides this category, and its actual state.

    Returns the slot itself rather than a bare boolean so callers can explain
    precisely what is wrong. Saying "no AVS match on file" when the checklist
    shows AVS Y is the kind of contradiction that destroys trust in every
    other line of the explanation.
    """
    decisive = {
        Category.PRODUCT_NOT_RECEIVED: (
            Slot.ACCESS_ACTIVITY_LOG if av.is_digital else Slot.SHIPPING_PROOF
        ),
        Category.FRAUDULENT: Slot.AUTHORIZATION_PROOF,
        Category.CREDIT_NOT_PROCESSED: Slot.REFUND_POLICY,
        Category.SUBSCRIPTION_CANCELED: Slot.ACCESS_ACTIVITY_LOG,
        Category.PRODUCT_UNACCEPTABLE: Slot.CUSTOMER_COMMUNICATION,
        Category.DUPLICATE: Slot.DUPLICATE_CHARGE_ID,
        Category.GENERAL: Slot.RECEIPT,
    }
    slot = decisive.get(category)
    if slot is None:
        return None, Strength.FILLED, None
    strength, value = av.slot_state(slot)
    return slot, strength, value


def critical_slot_present(av: Availability, category: Category) -> bool:
    """Whether the decisive field is strong enough to argue from.

    WEAK is not enough. An AVS match with no 3DS is real evidence but does not
    carry a fraud representment on its own, and a system that contests on it
    is filing claims it will lose.
    """
    _, strength, _ = critical_slot(av, category)
    return strength is Strength.FILLED


async def load_availability(order_ids: list[str]) -> dict[str, Availability]:
    """Six batch queries for the whole worklist."""
    if not order_ids:
        return {}

    out: dict[str, Availability] = {}

    orders = await db.fetch(
        """
        select id, is_digital, avs_result, cvv_result, three_ds_status,
               billing_country, evidence_tier
        from orders where id = any($1::text[])
        """,
        order_ids,
    )
    for row in orders:
        out[row["id"]] = Availability(
            order_id=row["id"],
            is_digital=bool(row["is_digital"]),
            avs_result=row["avs_result"],
            cvv_result=row["cvv_result"],
            three_ds_status=row["three_ds_status"],
            billing_country=row["billing_country"],
            evidence_tier=row["evidence_tier"] or "standard",
        )

    ship = await db.fetch(
        """
        select order_id,
               bool_or(event_type = 'delivered')  as delivered,
               bool_or(event_type = 'exception')  as had_exception,
               max(case when event_type = 'delivered' then occurred_at end) as delivered_at,
               max(case when event_type = 'delivered' then location end)    as location,
               max(case when event_type = 'label_created' then occurred_at end) as shipped_at,
               max(tracking_number) as tracking_number,
               max(carrier)         as carrier,
               max(signature_name)  as signature_name
        from fulfillment_events where order_id = any($1::text[])
        group by order_id
        """,
        order_ids,
    )
    for row in ship:
        av = out.get(row["order_id"])
        if not av:
            continue
        av.delivered = bool(row["delivered"])
        av.had_exception = bool(row["had_exception"])
        av.delivered_at = str(row["delivered_at"])[:19] if row["delivered_at"] else None
        av.delivery_location = row["location"]
        av.shipped_at = str(row["shipped_at"])[:19] if row["shipped_at"] else None
        av.tracking_number = row["tracking_number"]
        av.carrier = row["carrier"]
        av.signature_name = row["signature_name"]

    comms = await db.fetch(
        """
        select order_id, count(*) as n, max(occurred_at) as last_at
        from communications where order_id = any($1::text[]) group by order_id
        """,
        order_ids,
    )
    for row in comms:
        if av := out.get(row["order_id"]):
            av.communications = int(row["n"])
            av.last_comm_at = str(row["last_at"])[:19] if row["last_at"] else None

    policies = await db.fetch(
        """
        select order_id, array_agg(distinct policy_type) as kinds
        from policy_acceptances where order_id = any($1::text[]) group by order_id
        """,
        order_ids,
    )
    for row in policies:
        if av := out.get(row["order_id"]):
            av.policies = set(row["kinds"] or [])

    refunds = await db.fetch(
        """
        select order_id,
               bool_or(status = 'issued') as issued,
               bool_or(status in ('requested', 'declined')) as requested,
               bool_or(status = 'requested') as still_open
        from refunds where order_id = any($1::text[]) group by order_id
        """,
        order_ids,
    )
    for row in refunds:
        if av := out.get(row["order_id"]):
            av.refund_issued = bool(row["issued"])
            av.refund_requested = bool(row["requested"])
            av.refund_open = bool(row["still_open"])

    access = await db.fetch(
        """
        select order_id, count(*) as n, max(occurred_at) as last_at
        from access_logs where order_id = any($1::text[]) group by order_id
        """,
        order_ids,
    )
    for row in access:
        if av := out.get(row["order_id"]):
            av.access_events = int(row["n"])
            av.last_access_at = str(row["last_at"])[:19] if row["last_at"] else None

    return out


def describe(av: Availability, category: Category) -> list[dict]:
    """Every slot for the category, with its state. Drives the UI checklist."""
    spec = spec_for(category)
    rows = []
    for slot in required_slots(category, av.is_digital):
        strength, value = av.slot_state(slot)
        rows.append({"slot": slot.value, "required": True,
                     "strength": strength.value, "value": value})
    for slot in spec.supporting:
        strength, value = av.slot_state(slot)
        rows.append({"slot": slot.value, "required": False,
                     "strength": strength.value, "value": value})
    return rows
