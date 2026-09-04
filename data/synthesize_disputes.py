#!/usr/bin/env python3
"""Generate disputes and the evidence records that answer them.

    python data/synthesize_disputes.py --orders 10000
    python data/synthesize_disputes.py --orders 10000 --load

Reads the order table, materialises an operational history for a sample of it
— shipments, support threads, policy acceptances, refunds, access logs — and
then raises disputes *caused by that history*.

That direction matters. If categories were assigned at random and evidence
generated to match, every dispute would be answerable and the triage rule
would have nothing to decide. Here a parcel with no delivery scan becomes a
not-received dispute that genuinely cannot be defended, and the system has to
recommend accepting it. The gaps are real.

DEFENCE-ONLY: this generates records a merchant would hold in order to answer
a claim — deliveries, conversations, policy acceptances, refunds. It does not
model, generate or parameterise fraud technique. The fraud signal comes from
the label already on the order table. See docs/DEFENSE_ONLY.md.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
from app.evidence.schema import Category  # noqa: E402
from app.ml.tableio import find_table, read_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

CARRIERS = ["Delhivery", "Bluedart", "Ekart", "XpressBees", "India Post"]
CITIES = [
    "Bengaluru 560001", "Mumbai 400001", "New Delhi 110001", "Pune 411001",
    "Hyderabad 500001", "Chennai 600001", "Kolkata 700001", "Jaipur 302001",
]
AGENTS = ["Priya R.", "Arjun M.", "Fatima S.", "Rohit K.", "Neha B."]

# Response windows differ by rail. Stripe surfaces evidence_details.due_by;
# Razorpay surfaces respond_by. Both are short.
RESPOND_DAYS = (7, 21)


def _num(value, default: float = 0.0) -> float:
    """Coerce a possibly-NaN dataframe value to a real number.

    `float(nan) or 0` does not work: NaN is truthy, so it survives. Guests
    have no account age, and that NaN reached timedelta() as an error.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f          # f != f is True only for NaN


@dataclass
class Bundle:
    """Everything generated, ready to load."""

    fulfillment: list[dict] = field(default_factory=list)
    communications: list[dict] = field(default_factory=list)
    policies: list[dict] = field(default_factory=list)
    refunds: list[dict] = field(default_factory=list)
    access: list[dict] = field(default_factory=list)
    disputes: list[dict] = field(default_factory=list)
    outcomes: list[dict] = field(default_factory=list)
    customers: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "orders": len(self.orders),
            "customers": len(self.customers),
            "fulfillment_events": len(self.fulfillment),
            "communications": len(self.communications),
            "policy_acceptances": len(self.policies),
            "refunds": len(self.refunds),
            "access_logs": len(self.access),
            "disputes": len(self.disputes),
            "dispute_outcomes": len(self.outcomes),
        }


# ---------------------------------------------------------------- history

def _ship(order: dict, rng, bundle: Bundle) -> dict:
    """Create a shipment and its scans. Returns what the merchant can prove.

    Three outcomes, and the proportions are the point:
      delivered with signature  — strongest possible evidence
      delivered, no signature   — good, unless the category needs one
      lost or never scanned     — nothing to show, and the dispute is unwinnable

    Whether a signature exists is driven by `evidence_tier`, which is set by
    the propensity model. That is the loop closing: a high score at order time
    changes what the merchant can prove ninety days later.
    """
    order_id = order["order_id"]
    placed = order["created_at"]
    carrier = rng.choice(CARRIERS)
    tracking = f"{rng.choice(['BD', 'DL', 'EK', 'XB'])}{rng.integers(10**9, 10**10)}IN"
    city = rng.choice(CITIES)
    enhanced = order.get("evidence_tier") == "enhanced"

    # Fulfilment failure and disputes are not independent — an undelivered
    # parcel is a leading *cause* of a dispute. Conditioning the failure rate
    # on the label encodes that, and it is what gives the not-received
    # category a realistic share instead of everything falling through to
    # fraud.
    disputed = bool(order.get("disputed"))
    p_never, p_exc = (0.16, 0.12) if disputed else (0.02, 0.025)

    roll = rng.random()
    if roll < p_never:
        outcome = "never_scanned"          # label made, parcel vanished
    elif roll < p_never + p_exc:
        outcome = "exception"              # attempted, failed, returned
    else:
        outcome = "delivered"

    events = [{
        "order_id": order_id, "carrier": carrier, "tracking_number": tracking,
        "event_type": "label_created", "occurred_at": placed + timedelta(hours=float(rng.uniform(2, 20))),
        "location": "Origin hub", "signature_name": None,
    }]

    if outcome == "never_scanned":
        return {"delivered": False, "signature": False, "tracking": tracking,
                "carrier": carrier, "city": city, "delivered_at": None,
                "outcome": outcome}

    transit_h = float(rng.uniform(20, 90))
    events.append({
        "order_id": order_id, "carrier": carrier, "tracking_number": tracking,
        "event_type": "in_transit",
        "occurred_at": placed + timedelta(hours=transit_h * 0.4),
        "location": "In transit", "signature_name": None,
    })

    if outcome == "exception":
        events.append({
            "order_id": order_id, "carrier": carrier, "tracking_number": tracking,
            "event_type": "exception",
            "occurred_at": placed + timedelta(hours=transit_h),
            "location": city, "signature_name": None,
        })
        bundle.fulfillment.extend(events)
        return {"delivered": False, "signature": False, "tracking": tracking,
                "carrier": carrier, "city": city, "delivered_at": None,
                "outcome": outcome}

    delivered_at = placed + timedelta(hours=transit_h)
    # Signature is far more likely when the model asked for enhanced capture.
    signed = rng.random() < (0.88 if enhanced else 0.22)
    events.append({
        "order_id": order_id, "carrier": carrier, "tracking_number": tracking,
        "event_type": "out_for_delivery",
        "occurred_at": delivered_at - timedelta(hours=3),
        "location": city, "signature_name": None,
    })
    events.append({
        "order_id": order_id, "carrier": carrier, "tracking_number": tracking,
        "event_type": "delivered", "occurred_at": delivered_at, "location": city,
        "signature_name": f"{rng.choice(['A.', 'S.', 'R.', 'M.'])} {rng.choice(['Sharma', 'Patel', 'Nair', 'Das', 'Khan'])}"
        if signed else None,
    })
    bundle.fulfillment.extend(events)
    return {"delivered": True, "signature": signed, "tracking": tracking,
            "carrier": carrier, "city": city, "delivered_at": delivered_at,
            "outcome": outcome}


def _policies(order: dict, rng, bundle: Bundle) -> bool:
    """Terms and refund policy acceptance at checkout.

    Guests skip it more often, and its absence is what makes some
    credit-not-processed and cancellation disputes indefensible.
    """
    if rng.random() < (0.72 if order.get("is_guest") else 0.96):
        for kind in ("terms", "refund"):
            bundle.policies.append({
                "order_id": order["order_id"], "policy_type": kind,
                "policy_version": f"v{rng.integers(2, 6)}.{rng.integers(0, 9)}",
                "policy_url": f"https://merchant.example/legal/{kind}",
                "policy_text": (
                    "Returns accepted within 7 days of delivery in original "
                    "condition. Refunds are issued to the original payment "
                    "method within 5–7 business days of receipt."
                    if kind == "refund" else
                    "By completing this purchase you agree to the merchant's "
                    "terms of sale, including the delivery and returns policy."
                ),
                "accepted_at": order["created_at"] - timedelta(seconds=float(rng.uniform(20, 300))),
                "accepted_ip": f"49.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
            })
        return True
    return False


def _support(order: dict, rng, bundle: Bundle, shipment: dict) -> dict:
    """Support conversation, if any. Complaints follow what went wrong."""
    order_id, placed = order["order_id"], order["created_at"]
    customer = order.get("customer_id")
    complained = False
    asked_refund = False

    disputed = bool(order.get("disputed"))

    if not shipment["delivered"] and rng.random() < (0.78 if disputed else 0.45):
        complained = True
        when = placed + timedelta(days=float(rng.uniform(6, 14)))
        bundle.communications.append({
            "customer_id": customer, "order_id": order_id, "channel": "email",
            "direction": "inbound", "occurred_at": when,
            "subject": "Order not received",
            "body": "It has been over a week and my order still has not arrived. "
                    "The tracking has not updated. Please advise.",
            "agent_name": None,
        })
        bundle.communications.append({
            "customer_id": customer, "order_id": order_id, "channel": "email",
            "direction": "outbound", "occurred_at": when + timedelta(hours=float(rng.uniform(1, 20))),
            "subject": "Re: Order not received",
            "body": "Thanks for writing in. We have raised a trace with the "
                    "carrier and will update you within 48 hours.",
            "agent_name": str(rng.choice(AGENTS)),
        })
        asked_refund = rng.random() < 0.55

    elif shipment["delivered"] and rng.random() < (0.34 if disputed else 0.07):
        complained = True
        when = shipment["delivered_at"] + timedelta(days=float(rng.uniform(1, 9)))
        bundle.communications.append({
            "customer_id": customer, "order_id": order_id, "channel": "chat",
            "direction": "inbound", "occurred_at": when,
            "subject": "Item not as described",
            "body": "The item I received does not match the listing photos. "
                    "I would like to return it.",
            "agent_name": None,
        })
        bundle.communications.append({
            "customer_id": customer, "order_id": order_id, "channel": "chat",
            "direction": "outbound", "occurred_at": when + timedelta(minutes=float(rng.uniform(2, 90))),
            "subject": "Re: Item not as described",
            "body": "Sorry to hear that. Our returns window is 7 days from "
                    "delivery — I can start a return for you now.",
            "agent_name": str(rng.choice(AGENTS)),
        })
        asked_refund = rng.random() < 0.7

    return {"complained": complained, "asked_refund": asked_refund}


def _refund(order: dict, rng, bundle: Bundle, asked: bool) -> str | None:
    """Refund lifecycle. 'requested but never issued' is what creates a
    credit-not-processed dispute — and a merchant with no policy on file
    cannot defend it."""
    if not asked:
        # Some cardholders ask for a refund without ever raising a complaint.
        if not (order.get("disputed") and rng.random() < 0.16):
            return None
    requested = order["created_at"] + timedelta(days=float(rng.uniform(7, 20)))
    roll = rng.random()
    status = "issued" if roll < 0.58 else ("declined" if roll < 0.78 else "requested")
    bundle.refunds.append({
        "id": f"rfnd_{uuid.uuid4().hex[:14]}",
        "order_id": order["order_id"],
        "amount": float(order["amount"]),
        "status": status,
        "reason": "customer_request",
        "requested_at": requested,
        "issued_at": requested + timedelta(days=float(rng.uniform(1, 4))) if status == "issued" else None,
        "processor_refund_id": f"re_{uuid.uuid4().hex[:16]}" if status == "issued" else None,
    })
    return status


def _access(order: dict, rng, bundle: Bundle) -> int:
    """Usage logs for digital goods — the substitute for a delivery scan."""
    if not order.get("is_digital"):
        return 0
    n = int(rng.integers(0, 14))
    for i in range(n):
        bundle.access.append({
            "order_id": order["order_id"],
            "customer_id": order.get("customer_id"),
            "occurred_at": order["created_at"] + timedelta(days=float(rng.uniform(0, 45))),
            "ip": f"49.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
            "user_agent": str(rng.choice([
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Mozilla/5.0 (Linux; Android 14)",
            ])),
            "action": str(rng.choice(["login", "stream", "download", "api_call"])),
        })
    return n


# ---------------------------------------------------------------- disputes

def _category_from_history(order: dict, ship: dict, support: dict,
                           refund_status: str | None, accesses: int, rng) -> Category:
    """Pick the category the history actually implies.

    This is the causal direction. Assigning a category first and fabricating
    matching evidence would make every dispute winnable and leave the triage
    rule with nothing to weigh.
    """
    digital = bool(order.get("is_digital"))

    if refund_status in ("requested", "declined") and rng.random() < 0.75:
        return Category.CREDIT_NOT_PROCESSED

    if not digital and not ship["delivered"] and rng.random() < 0.85:
        return Category.PRODUCT_NOT_RECEIVED

    if digital and accesses == 0 and rng.random() < 0.6:
        return Category.PRODUCT_NOT_RECEIVED

    # Non-receipt claimed *despite* a delivery scan. Left out at first, which
    # made every not-received dispute indefensible by construction — and this
    # is in fact the most winnable representment there is: carrier scan,
    # address match, sometimes a signature. A demo without it would never show
    # the system confidently contesting anything.
    if not digital and ship["delivered"] and rng.random() < 0.11:
        return Category.PRODUCT_NOT_RECEIVED

    if digital and accesses > 2 and rng.random() < 0.07:
        return Category.PRODUCT_NOT_RECEIVED

    if support["complained"] and ship["delivered"] and rng.random() < 0.7:
        return Category.PRODUCT_UNACCEPTABLE

    if digital and accesses > 0 and rng.random() < 0.45:
        return Category.SUBSCRIPTION_CANCELED

    if rng.random() < 0.07:
        return Category.DUPLICATE

    # Nothing went wrong operationally and the cardholder still disputes.
    # Usually that is a fraud-category claim, but not overwhelmingly — an
    # unrecognised statement descriptor produces the same silence.
    # rng.choice over enum members returns numpy strings, so map back
    # explicitly rather than relying on the coercion.
    pick = rng.choice(["fraudulent", "general", "product_unacceptable"],
                      p=[0.72, 0.20, 0.08])
    return Category(str(pick))


NETWORK_CODES = {
    Category.FRAUDULENT: ("10.4", "4837"),
    Category.PRODUCT_NOT_RECEIVED: ("13.1", "4855"),
    Category.PRODUCT_UNACCEPTABLE: ("13.3", "4853"),
    Category.CREDIT_NOT_PROCESSED: ("13.6", "4860"),
    Category.SUBSCRIPTION_CANCELED: ("13.2", "4841"),
    Category.DUPLICATE: ("12.6.1", "4834"),
    Category.GENERAL: ("12.5", "4831"),
}

STRIPE_REASON = {
    Category.FRAUDULENT: "fraudulent",
    Category.PRODUCT_NOT_RECEIVED: "product_not_received",
    Category.PRODUCT_UNACCEPTABLE: "product_unacceptable",
    Category.CREDIT_NOT_PROCESSED: "credit_not_processed",
    Category.SUBSCRIPTION_CANCELED: "subscription_canceled",
    Category.DUPLICATE: "duplicate",
    Category.GENERAL: "general",
}


def score_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Set evidence_tier from the trained propensity model.

    This is the architectural claim made concrete: a high score at order time
    does not decline anyone, it raises what gets recorded — and `_ship` then
    makes a signature far more likely for those orders. Without this step the
    two lanes are unconnected and the whole premise is decorative.
    """
    try:
        sys.path.insert(0, str(ROOT / "api"))
        from app.ml import predict as ml
        from app.ml.features import build_matrix, records_from_frame

        model = ml.load()
        if model is None:
            print("  no trained model — evidence_tier defaults to standard "
                  "(run `make train` first to close the loop)")
            return orders.assign(evidence_tier="standard")

        X = build_matrix(records_from_frame(orders), model.stats)
        raw = model.booster.predict(X)
        scores = model.calibrator.predict(raw)
        tier = np.where(scores >= model.threshold, "enhanced", "standard")
        share = float((tier == "enhanced").mean())
        print(f"  scored with {model.version}: {share:.1%} of orders flagged "
              f"for enhanced evidence capture")
        return orders.assign(evidence_tier=tier, propensity=scores)
    except Exception as exc:                      # pragma: no cover
        print(f"  scoring unavailable ({exc}) — evidence_tier defaults to standard")
        return orders.assign(evidence_tier="standard")


def build(orders: pd.DataFrame, rng, now: datetime) -> Bundle:
    bundle = Bundle()
    seen_customers: set[str] = set()

    for order in orders.to_dict(orient="records"):
        bundle.orders.append({
            "id": order["order_id"],
            "customer_id": order.get("customer_id"),
            "created_at": order["created_at"],
            "amount": round(float(order["amount"]), 2),
            "currency": "INR",
            "product_code": order.get("product_code"),
            "is_digital": bool(order.get("is_digital")),
            "billing_country": order.get("billing_country"),
            "shipping_country": order.get("shipping_country"),
            "bin_country": order.get("bin_country"),
            "avs_result": order.get("avs_result"),
            "cvv_result": order.get("cvv_result"),
            "three_ds_status": order.get("three_ds_status"),
            "statement_descriptor": "MERCHANT*ORDER",
            "rail": "stripe",
            "processor_payment_id": None,
            "disputed": bool(order.get("disputed")),
            "disputed_at": None,
            "evidence_tier": order.get("evidence_tier") or "standard",
        })
        cid = order.get("customer_id")
        if cid and cid not in seen_customers:
            seen_customers.add(cid)
            bundle.customers.append({
                "id": cid,
                "email": f"{cid}@{order.get('payer_email_domain') or 'example.com'}",
                "email_domain": order.get("payer_email_domain"),
                "created_at": order["created_at"] - timedelta(days=_num(order.get("account_age_days"))),
                "account_age_days": int(_num(order.get("account_age_days"))),
                "lifetime_value": round(float(order["amount"]) * float(rng.uniform(1.0, 6.0)), 2),
                "prior_disputes": int(_num(order.get("prior_disputes"))),
                "is_guest": bool(order.get("is_guest")),
            })

        digital = bool(order.get("is_digital"))
        ship = ({"delivered": True, "signature": False, "tracking": None,
                 "carrier": None, "city": None, "delivered_at": order["created_at"],
                 "outcome": "digital"} if digital else _ship(order, rng, bundle))
        _policies(order, rng, bundle)
        support = _support(order, rng, bundle, ship)
        refund_status = _refund(order, rng, bundle, support["asked_refund"])
        accesses = _access(order, rng, bundle)

        if not order.get("disputed"):
            continue

        category = _category_from_history(order, ship, support, refund_status, accesses, rng)
        visa, mc = NETWORK_CODES[category]

        opened = order["created_at"] + timedelta(days=float(rng.uniform(12, 75)))
        if opened > now:
            opened = now - timedelta(days=float(rng.uniform(0, 6)))

        rail = "stripe" if rng.random() < 0.7 else "razorpay"
        phase = "chargeback"
        if rail == "razorpay":
            phase = str(rng.choice(["retrieval", "chargeback", "chargeback", "pre_arbitration"],
                                   p=[0.18, 0.6, 0.15, 0.07]))

        dispute_id = f"dsp_{uuid.uuid4().hex[:16]}"
        respond_by = opened + timedelta(days=int(rng.integers(*RESPOND_DAYS)))

        # Disputes older than their deadline already have an outcome; recent
        # ones are still open. That gives the similarity search history to
        # retrieve and the retraining path something to learn from later.
        settled = respond_by < now

        bundle.disputes.append({
            "id": dispute_id,
            "order_id": order["order_id"],
            "rail": rail,
            # Prefixed `sim_` on purpose. These used to be minted as `du_...`,
            # which is exactly what Stripe's own ids look like — so the
            # submitter believed them and posted evidence to a dispute that
            # did not exist. A generated id should be unmistakable.
            "processor_dispute_id": (f"sim_du_{uuid.uuid4().hex[:20]}" if rail == "stripe"
                                     else f"sim_disp_{uuid.uuid4().hex[:14]}"),
            "origin": "synthetic",
            "amount": float(order["amount"]),
            "currency": "INR",
            "category": category.value,
            "processor_reason": STRIPE_REASON[category] if rail == "stripe" else f"{visa} {category.value}",
            "network_code": f"Visa {visa} / MC {mc}",
            "phase": phase,
            "status": ("won" if settled and rng.random() < 0.45 else
                       "lost" if settled else "needs_response"),
            "respond_by": respond_by,
            "opened_at": opened,
        })

        if settled:
            status = bundle.disputes[-1]["status"]
            # Never in the future. Issuers take 10-40 days to rule, but a
            # dispute recorded as already settled cannot close next month —
            # and the agent quotes these dates as precedent.
            closed = min(respond_by + timedelta(days=float(rng.uniform(10, 40))), now)
            bundle.outcomes.append({
                "dispute_id": dispute_id,
                "outcome": status,
                "closed_at": closed,
                "amount_recovered": float(order["amount"]) if status == "won" else 0.0,
                "notes": None,
            })

    return bundle


# ------------------------------------------------------------------ load

async def load(bundle: Bundle, url: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(url, statement_cache_size=0 if ":6543" in url else 100)
    try:
        async with conn.transaction():
            # A dispute that arrived from a processor is not ours to delete.
            # This used to `delete from disputes` outright — the comment said
            # "clear generated rows" and the code cleared everything, so
            # re-running the generator silently destroyed every real dispute
            # received over a webhook, along with the order it pointed at.
            #
            # Protected: processor-origin disputes, the orders they are about,
            # and the evidence those orders carry. Everything else is
            # regenerated.
            protected = [r["order_id"] for r in await conn.fetch(
                "select distinct order_id from disputes "
                "where origin = 'processor' and order_id is not null")]
            keep = set(protected)
            if keep:
                log_note = ", ".join(sorted(keep)[:3])
                print(f"  preserving {len(keep)} live order(s): {log_note}"
                      f"{' …' if len(keep) > 3 else ''}")

            # Citations hang off the packet, not the dispute.
            await conn.execute(
                "delete from evidence_citations where packet_id in "
                "(select p.id from evidence_packets p join disputes d "
                "   on d.id = p.dispute_id where d.origin <> 'processor')")
            await conn.execute(
                "delete from evidence_packets where dispute_id in "
                "(select id from disputes where origin <> 'processor')")
            await conn.execute(
                "delete from dispute_outcomes where dispute_id in "
                "(select id from disputes where origin <> 'processor')")
            await conn.execute("delete from disputes where origin <> 'processor'")

            for table in ("access_logs", "refunds", "policy_acceptances",
                          "communications", "fulfillment_events"):
                await conn.execute(
                    f"delete from {table} where not (order_id = any($1::text[]))",
                    protected)

            # Orders before customers (FK), customers before re-insert —
            # without this a second run duplicates every customer key.
            await conn.execute("delete from order_scores where not (order_id = any($1::text[]))",
                               protected)
            await conn.execute("delete from orders where not (id = any($1::text[]))", protected)
            await conn.execute(
                "delete from customers where id not in (select customer_id from orders "
                "where customer_id is not null)")

            # Anything the generator would re-insert over a preserved row is
            # dropped from the bundle: the live one is the real record.
            live_customers = {r["customer_id"] for r in await conn.fetch(
                "select distinct customer_id from orders where customer_id is not null")}
            customers = [c for c in bundle.customers if c["id"] not in live_customers]
            orders = [o for o in bundle.orders if o["id"] not in keep]
            await _copy(conn, "customers", customers)
            await _copy(conn, "orders", orders)
            def fresh(rows):
                return [r for r in rows if r.get("order_id") not in keep]

            await _copy(conn, "fulfillment_events", fresh(bundle.fulfillment))
            await _copy(conn, "communications", fresh(bundle.communications))
            await _copy(conn, "policy_acceptances", fresh(bundle.policies))
            await _copy(conn, "refunds", fresh(bundle.refunds))
            await _copy(conn, "access_logs", fresh(bundle.access))
            await _copy(conn, "disputes", bundle.disputes)
            await _copy(conn, "dispute_outcomes", bundle.outcomes)
    finally:
        await conn.close()


async def _copy(conn, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    records = [tuple(r[c] for c in columns) for r in rows]
    await conn.copy_records_to_table(table, records=records, columns=columns)


def write_files(bundle: Bundle) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("orders_loaded", bundle.orders),
        ("customers", bundle.customers), ("fulfillment_events", bundle.fulfillment),
        ("communications", bundle.communications), ("policy_acceptances", bundle.policies),
        ("refunds", bundle.refunds), ("access_logs", bundle.access),
        ("disputes", bundle.disputes), ("dispute_outcomes", bundle.outcomes),
    ):
        if rows:
            pd.DataFrame(rows).to_csv(PROCESSED / f"{name}.csv", index=False)


def _defensibility(bundle: Bundle, orders: pd.DataFrame) -> None:
    """How many disputes can actually be answered from the records generated.

    This is the number the whole project turns on. If every dispute had its
    evidence, the triage rule would recommend contesting all of them and the
    agent would never flag a gap — a demo where nothing is ever refused proves
    nothing. A realistic spread is what makes accept-when-weak meaningful.
    """
    delivered = {e["order_id"] for e in bundle.fulfillment if e["event_type"] == "delivered"}
    policy = {p["order_id"] for p in bundle.policies}
    issued = {r["order_id"] for r in bundle.refunds if r["status"] == "issued"}
    used = {a["order_id"] for a in bundle.access}
    talked = {c["order_id"] for c in bundle.communications}
    digital = set(orders.loc[orders["is_digital"] == True, "order_id"])  # noqa: E712

    # The single strongest field per category, per the checklist.
    def defensible(d: dict) -> bool:
        oid, cat = d["order_id"], d["category"]
        if cat == "product_not_received":
            return oid in (used if oid in digital else delivered)
        if cat == "fraudulent":
            return oid in delivered or oid in used
        if cat == "product_unacceptable":
            return oid in talked and oid in policy
        if cat == "credit_not_processed":
            return oid in issued or oid in policy
        if cat == "subscription_canceled":
            return oid in used and oid in policy
        if cat == "duplicate":
            return True
        return oid in policy

    rows: dict[str, list[int]] = {}
    for d in bundle.disputes:
        rows.setdefault(d["category"], [0, 0])
        rows[d["category"]][defensible(d)] += 1

    print("\ndefensibility — can the records answer the claim?")
    total_ok = total = 0
    for cat in sorted(rows, key=lambda c: -sum(rows[c])):
        weak, ok = rows[cat]
        n = weak + ok
        total_ok += ok
        total += n
        print(f"  {cat:<24} {ok:>5,} defensible  {weak:>5,} weak   ({ok / n:.0%})")
    print(f"  {'ALL':<24} {total_ok:>5,} defensible  {total - total_ok:>5,} weak   "
          f"({total_ok / total:.0%})")
    print("\n  Weak cases are the point: the triage rule should recommend")
    print("  accepting them rather than contesting a case it cannot support.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, default=10_000,
                    help="how many orders to give a full operational history")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--load", action="store_true", help="write into Postgres")
    ap.add_argument("--no-score", action="store_true",
                    help="skip model scoring; every order gets the standard tier")
    args = ap.parse_args()

    path = find_table(PROCESSED)
    if path is None:
        print("No order table. Run: make data")
        return 1

    orders = read_table(path)
    rng = np.random.default_rng(args.seed)

    # Every disputed order, plus a sample of clean ones for context and for
    # the similarity search to have neighbours.
    disputed = orders[orders["disputed"] == 1]
    clean = orders[orders["disputed"] == 0]
    take = max(args.orders - len(disputed), 0)
    if take and len(clean) > take:
        clean = clean.sample(take, random_state=args.seed)
    sample = pd.concat([disputed, clean]).sort_values("created_at")

    if "customer_id" not in sample.columns:
        sample = sample.assign(customer_id=sample["card_id"].astype(str).str.replace("card_", "cust_"))

    sample = (sample.assign(evidence_tier="standard") if args.no_score
              else score_orders(sample))

    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    bundle = build(sample, rng, now)

    print(f"\nfrom {len(sample):,} orders ({len(disputed):,} disputed)\n")
    for name, n in bundle.counts().items():
        print(f"  {name:<22} {n:>8,}")

    cats = pd.Series([d["category"] for d in bundle.disputes]).value_counts()
    print("\ndispute categories")
    for cat, n in cats.items():
        print(f"  {cat:<24} {n:>6,}  ({n / len(bundle.disputes):.1%})")

    _defensibility(bundle, sample)

    open_now = sum(1 for d in bundle.disputes if d["status"] == "needs_response")
    print(f"\n  open, awaiting response  {open_now:,}")
    print(f"  already settled          {len(bundle.outcomes):,}")

    write_files(bundle)
    print(f"\nwritten to {PROCESSED.relative_to(ROOT)}/*.csv")

    if args.load:
        url = os.environ.get("DATABASE_URL", "")
        if not url:
            for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
                if line.startswith("DATABASE_URL="):
                    url = line.split("=", 1)[1].strip()
        if not url:
            print("\nDATABASE_URL not set — skipped loading.")
            return 1
        asyncio.run(load(bundle, url))
        print("loaded into Postgres.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
