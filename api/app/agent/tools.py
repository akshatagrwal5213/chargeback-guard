"""The agent's tools. Seven of them, and each one is a parameterised query.

Two rules hold across all of them:

1. **Read-only.** Nothing here writes. An evidence agent that could modify the
   records it cites would be worthless as evidence.

2. **Every returned fact is a Record with a ref.** A tool never returns a bare
   string. If it cannot be pointed at a row, the agent cannot say it.

The JSON schemas are declared beside each function so the two cannot drift —
a tool whose schema promises a parameter the function ignores is a silent
source of wrong answers.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from .. import db
from .records import Record, ToolResult, ref_for

log = logging.getLogger(__name__)

MAX_ROWS = 40          # nothing a representment needs is longer than this
MAX_BODY_CHARS = 1200  # support threads can be long; the agent needs the gist


# ------------------------------------------------------------------ tools

async def get_order(order_id: str) -> ToolResult:
    """Order, authorisation results and the customer behind it."""
    row = await db.fetchrow(
        """
        select o.*, c.email, c.account_age_days, c.lifetime_value,
               c.prior_disputes, c.is_guest
        from orders o left join customers c on c.id = o.customer_id
        where o.id = $1
        """,
        order_id,
    )
    if not row:
        return ToolResult("get_order", note=f"No order {order_id} on file.")

    auth = []
    if row.get("avs_result"):
        auth.append(f"AVS {row['avs_result']}")
    if row.get("cvv_result"):
        auth.append(f"CVV {row['cvv_result']}")
    if row.get("three_ds_status"):
        auth.append(f"3DS {row['three_ds_status']}")

    summary = (
        f"{row['currency']} {float(row['amount']):,.2f} on "
        f"{str(row['created_at'])[:10]}, "
        f"{'digital' if row['is_digital'] else 'physical'} "
        f"({row.get('product_code') or 'uncategorised'})"
        + (f", {', '.join(auth)}" if auth else "")
    )
    return ToolResult("get_order", [
        Record(
            ref=ref_for("orders", row["id"]),
            kind="order",
            summary=summary,
            fields={
                "amount": float(row["amount"]),
                "currency": row["currency"],
                "placed_at": str(row["created_at"]),
                "is_digital": bool(row["is_digital"]),
                "product_code": row.get("product_code"),
                "avs_result": row.get("avs_result"),
                "cvv_result": row.get("cvv_result"),
                "three_ds_status": row.get("three_ds_status"),
                "billing_country": row.get("billing_country"),
                "shipping_country": row.get("shipping_country"),
                "statement_descriptor": row.get("statement_descriptor"),
                "evidence_tier": row.get("evidence_tier"),
                "customer_email": row.get("email"),
            },
        )
    ])


async def get_customer_history(customer_id: str) -> ToolResult:
    """Prior orders and prior disputes. Context for whether this is unusual."""
    row = await db.fetchrow(
        """
        select c.id, c.email, c.account_age_days, c.lifetime_value, c.is_guest,
               count(o.id)                                    as orders,
               count(*) filter (where o.disputed)             as disputed,
               min(o.created_at)                              as first_order,
               max(o.created_at)                              as last_order
        from customers c left join orders o on o.customer_id = c.id
        where c.id = $1
        group by c.id
        """,
        customer_id,
    )
    if not row:
        return ToolResult("get_customer_history",
                          note=f"No customer record for {customer_id}.")

    undisputed = int(row["orders"] or 0) - int(row["disputed"] or 0)
    age_days = int(row.get("account_age_days") or 0)
    # Built outside the f-string: nesting the same quote character inside one
    # is a 3.12 feature, and this project supports 3.11.
    tenure = "guest checkout" if row["is_guest"] else f"{age_days} days old"

    return ToolResult("get_customer_history", [
        Record(
            ref=ref_for("customers", row["id"]),
            kind="customer_history",
            # The prior-dispute count stays in `fields`, where triage and the
            # console can see it, and out of the summary, which is the text
            # that ends up in front of the issuer. It read
            # "2 order(s) since 2026-05-31, 0 never disputed" — both garbled
            # and an argument for the other side: a merchant volunteering that
            # this cardholder has disputed everything they ever bought has
            # made the issuer's case for it.
            summary=(
                f"{row['orders']} order(s) since {str(row['first_order'])[:10]}; "
                f"account {tenure}"
            ),
            fields={
                "orders": int(row["orders"] or 0),
                "undisputed_orders": undisputed,
                "prior_disputes": int(row["disputed"] or 0),
                "account_age_days": age_days,
                "lifetime_value": float(row.get("lifetime_value") or 0),
                "is_guest": bool(row["is_guest"]),
                "first_order": str(row["first_order"])[:19] if row["first_order"] else None,
            },
        )
    ])


async def get_fulfillment(order_id: str) -> ToolResult:
    """Carrier scans for the order, oldest first. The delivery scan is the
    single most decisive record in a not-received dispute."""
    rows = await db.fetch(
        """
        select id, carrier, tracking_number, event_type, occurred_at,
               location, signature_name
        from fulfillment_events where order_id = $1
        order by occurred_at limit $2
        """,
        order_id, MAX_ROWS,
    )
    if not rows:
        return ToolResult("get_fulfillment",
                          note="No shipment records — nothing was ever dispatched, "
                               "or this is a digital order.")

    records = []
    for row in rows:
        bits = [f"{row['event_type']} at {str(row['occurred_at'])[:19]}"]
        if row.get("location"):
            bits.append(row["location"])
        if row.get("signature_name"):
            bits.append(f"signed by {row['signature_name']}")
        records.append(Record(
            ref=ref_for("fulfillment_events", row["id"]),
            kind="delivery_scan" if row["event_type"] == "delivered" else "carrier_scan",
            summary=", ".join(bits) + f" ({row.get('carrier') or 'carrier unknown'}, "
                                      f"{row.get('tracking_number') or 'no tracking'})",
            fields={k: (str(v) if k == "occurred_at" else v)
                    for k, v in row.items() if k != "id"},
        ))
    return ToolResult("get_fulfillment", records)


async def get_communications(order_id: str, include_body: bool = True) -> ToolResult:
    """Support messages for the order. What the customer said, and when."""
    rows = await db.fetch(
        """
        select id, channel, direction, occurred_at, subject, body, agent_name
        from communications where order_id = $1
        order by occurred_at limit $2
        """,
        order_id, MAX_ROWS,
    )
    if not rows:
        return ToolResult("get_communications",
                          note="The customer never contacted support about this order.")

    records = []
    for row in rows:
        body = (row.get("body") or "")[:MAX_BODY_CHARS]
        records.append(Record(
            ref=ref_for("communications", row["id"]),
            kind="support_message",
            summary=(f"{row['direction']} {row['channel']} on "
                     f"{str(row['occurred_at'])[:19]}: {row.get('subject') or '(no subject)'}"),
            fields={
                "direction": row["direction"],
                "channel": row["channel"],
                "occurred_at": str(row["occurred_at"]),
                "subject": row.get("subject"),
                "agent_name": row.get("agent_name"),
                **({"body": body} if include_body else {}),
            },
        ))
    return ToolResult("get_communications", records)


async def get_policy_acceptance(order_id: str) -> ToolResult:
    """Which policies the customer accepted at checkout, when, and from where."""
    rows = await db.fetch(
        """
        select id, policy_type, policy_version, policy_url, policy_text,
               accepted_at, accepted_ip
        from policy_acceptances where order_id = $1 order by policy_type
        """,
        order_id,
    )
    if not rows:
        return ToolResult("get_policy_acceptance",
                          note="No policy acceptance recorded at checkout.")

    return ToolResult("get_policy_acceptance", [
        Record(
            ref=ref_for("policy_acceptances", row["id"]),
            kind="policy_acceptance",
            summary=(f"{row['policy_type']} {row['policy_version']} accepted "
                     f"{str(row['accepted_at'])[:19]} from {row.get('accepted_ip')}"),
            fields={
                "policy_type": row["policy_type"],
                "policy_version": row["policy_version"],
                "policy_url": row.get("policy_url"),
                "policy_text": (row.get("policy_text") or "")[:MAX_BODY_CHARS],
                "accepted_at": str(row["accepted_at"]),
                "accepted_ip": str(row.get("accepted_ip")),
            },
        )
        for row in rows
    ])


async def get_refunds(order_id: str) -> ToolResult:
    """Refund lifecycle. Decides a credit-not-processed dispute outright."""
    rows = await db.fetch(
        """
        select id, amount, status, reason, requested_at, issued_at,
               processor_refund_id
        from refunds where order_id = $1 order by requested_at
        """,
        order_id,
    )
    if not rows:
        return ToolResult("get_refunds",
                          note="No refund was ever requested or issued for this order.")

    return ToolResult("get_refunds", [
        Record(
            ref=ref_for("refunds", row["id"]),
            kind="refund",
            summary=(
                f"refund {row['status']}"
                + (f" on {str(row['issued_at'])[:10]}" if row.get("issued_at") else "")
                + (f", requested {str(row['requested_at'])[:10]}" if row.get("requested_at") else "")
                + f", {float(row['amount']):,.2f}"
            ),
            fields={
                "status": row["status"],
                "amount": float(row["amount"]),
                "reason": row.get("reason"),
                "requested_at": str(row["requested_at"]) if row.get("requested_at") else None,
                "issued_at": str(row["issued_at"]) if row.get("issued_at") else None,
                "processor_refund_id": row.get("processor_refund_id"),
            },
        )
        for row in rows
    ])


async def get_access_log(order_id: str) -> ToolResult:
    """Usage of a digital product. Stands in for a delivery scan when there is
    nothing to ship, and evidences use after a claimed cancellation."""
    rows = await db.fetch(
        """
        select id, occurred_at, ip, user_agent, action
        from access_logs where order_id = $1
        order by occurred_at desc limit $2
        """,
        order_id, MAX_ROWS,
    )
    if not rows:
        return ToolResult("get_access_log",
                          note="No recorded access — either not a digital product, "
                               "or it was genuinely never used.")

    return ToolResult("get_access_log", [
        Record(
            ref=ref_for("access_logs", row["id"]),
            kind="access_event",
            summary=f"{row['action']} at {str(row['occurred_at'])[:19]} from {row.get('ip')}",
            fields={
                "action": row["action"],
                "occurred_at": str(row["occurred_at"]),
                "ip": str(row.get("ip")),
                "user_agent": row.get("user_agent"),
            },
        )
        for row in rows
    ])


async def find_similar_disputes(category: str, limit: int = 5) -> ToolResult:
    """Past disputes in the same category that reached an outcome.

    Precedent, not prediction. It tells the agent what actually happened when
    this merchant argued cases like this one — useful context, and the honest
    basis for a claim like "we have won 7 of 9 comparable cases".

    Matching is on category and outcome for now. pgvector similarity over the
    narratives is the upgrade, and the table is already in the schema.
    """
    rows = await db.fetch(
        """
        select d.id, d.category, d.amount, d.phase, o.outcome,
               o.amount_recovered, o.closed_at
        from dispute_outcomes o join disputes d on d.id = o.dispute_id
        where d.category = $1
        order by o.closed_at desc limit $2
        """,
        category, min(limit, 20),
    )
    if not rows:
        return ToolResult("find_similar_disputes",
                          note=f"No settled {category} disputes yet — no precedent to cite.")

    won = sum(1 for row in rows if row["outcome"] == "won")
    records = [
        Record(
            ref=ref_for("dispute_outcomes", row["id"]),
            kind="precedent",
            summary=(f"{row['category']} for {float(row['amount']):,.2f} at "
                     f"{row['phase']} was {row['outcome']} "
                     f"({str(row['closed_at'])[:10]})"),
            fields={
                "outcome": row["outcome"],
                "amount": float(row["amount"]),
                "amount_recovered": float(row.get("amount_recovered") or 0),
                "phase": row["phase"],
                "closed_at": str(row["closed_at"])[:19],
            },
        )
        for row in rows
    ]
    return ToolResult(
        "find_similar_disputes", records,
        note=f"{won} of the {len(rows)} most recent settled {category} disputes were won.",
    )


# --------------------------------------------------------------- registry

ToolFn = Callable[..., Awaitable[ToolResult]]

REGISTRY: dict[str, ToolFn] = {
    "get_order": get_order,
    "get_customer_history": get_customer_history,
    "get_fulfillment": get_fulfillment,
    "get_communications": get_communications,
    "get_policy_acceptance": get_policy_acceptance,
    "get_refunds": get_refunds,
    "get_access_log": get_access_log,
}

_ORDER_ID = {
    "type": "object",
    "properties": {"order_id": {"type": "string", "description": "The order identifier."}},
    "required": ["order_id"],
}

SCHEMAS: list[dict[str, Any]] = [
    {"name": "get_order",
     "description": "The order itself: amount, date, product, AVS/CVV/3DS results, "
                    "billing and shipping country, and the customer it belongs to.",
     "parameters": _ORDER_ID},
    {"name": "get_customer_history",
     "description": "How many orders this customer has placed, how many were disputed, "
                    "and how long the account has existed.",
     "parameters": {
         "type": "object",
         "properties": {"customer_id": {"type": "string"}},
         "required": ["customer_id"]}},
    {"name": "get_fulfillment",
     "description": "Carrier scans for the order, including whether it was delivered, "
                    "where, when, and whether anyone signed for it.",
     "parameters": _ORDER_ID},
    {"name": "get_communications",
     "description": "Support messages about this order — what the customer said and "
                    "what the merchant replied.",
     "parameters": {
         "type": "object",
         "properties": {
             "order_id": {"type": "string"},
             "include_body": {"type": "boolean",
                              "description": "Include full message text. Default true."}},
         "required": ["order_id"]}},
    {"name": "get_policy_acceptance",
     "description": "Which terms and refund policies the customer accepted at checkout, "
                    "with version, timestamp and IP.",
     "parameters": _ORDER_ID},
    {"name": "get_refunds",
     "description": "Refunds requested, issued or declined for this order.",
     "parameters": _ORDER_ID},
    {"name": "get_access_log",
     "description": "For digital products: when the customer logged in, streamed or "
                    "downloaded. Substitutes for a delivery scan.",
     "parameters": _ORDER_ID},
    {"name": "find_similar_disputes",
     "description": "Past disputes in the same category that reached an outcome, "
                    "so a claim about precedent can be grounded.",
     "parameters": {
         "type": "object",
         "properties": {
             "category": {"type": "string"},
             "limit": {"type": "integer", "description": "Default 5, max 20."}},
         "required": ["category"]}},
]

# find_similar_disputes takes a category rather than an order id, so it is
# registered separately above and added here.
REGISTRY["find_similar_disputes"] = find_similar_disputes


async def call(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Dispatch by name. Unknown tools fail loudly rather than silently."""
    fn = REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"Unknown tool {name!r}. Available: {sorted(REGISTRY)}")
    try:
        return await fn(**arguments)
    except TypeError as exc:
        raise TypeError(f"Bad arguments for {name}: {exc}") from exc
