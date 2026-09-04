#!/usr/bin/env python
"""Raise a real Stripe test dispute against an order we can actually defend.

`stripe trigger charge.dispute.created` makes a dispute out of nothing: no
customer, no order, no delivery scan. The system correctly refuses to draft
for it — there is nothing to cite — which proves the guard and proves nothing
about the submission path.

This picks an order that already carries evidence, pays for it with the card
Stripe disputes on purpose, and records the payment id on the order. When the
dispute arrives over the webhook it resolves to that order, retrieval finds
the scans and the support thread, and the packet has something to say.

    python scripts/live_dispute.py                 # pick an order, pay, wait
    python scripts/live_dispute.py --order ord_x   # a specific one
    python scripts/live_dispute.py --currency usd  # if INR is refused

Test mode only. The card is Stripe's, the dispute is Stripe's, and no money
moves anywhere.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

STRIPE_API = "https://api.stripe.com/v1"

# Stripe's dispute-raising test payment methods. Product-not-received is the
# default because it is the category this system defends best: the evidence is
# a delivery scan, which the generator produces and the checklist demands.
CARDS = {
    "product_not_received": "pm_card_createDisputeProductNotReceived",
    "fraudulent": "pm_card_createDispute",
    "inquiry": "pm_card_createDisputeInquiry",
}

# An order worth defending: delivered, physical, with a support thread and a
# policy acceptance behind it — and one the triage rule will actually agree to
# contest.
#
# The open-refund exclusion is the part that was missing. Picking purely on how
# much evidence an order carries selected one with a full refund requested and
# never issued, which the rule then refused on conduct — correctly, and after
# the payment had already been made. A picker that does not know what the rule
# will decide wastes a real dispute every time it guesses wrong.
PICK = """
select o.id,
       o.amount,
       count(distinct f.id) as scans,
       count(distinct c.id) as messages,
       count(distinct p.id) as policies
  from orders o
  join fulfillment_events f
    on f.order_id = o.id and f.event_type = 'delivered'
  left join communications c on c.order_id = o.id
  left join policy_acceptances p on p.order_id = o.id
 where o.is_digital = false
   and o.processor_payment_id is null
   and not exists (
       select 1 from refunds r
        where r.order_id = o.id and r.status = 'requested')
 group by o.id, o.amount
 order by count(distinct c.id) + count(distinct p.id) desc, o.amount desc
 limit 1
"""


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", help="order id to dispute; default is the best-evidenced")
    parser.add_argument("--category", default="product_not_received", choices=sorted(CARDS))
    parser.add_argument("--currency", default="inr")
    args = parser.parse_args()

    load_env()
    from app import db                                   # noqa: E402
    from app.config import settings                      # noqa: E402

    if not settings.stripe_secret_key:
        print("STRIPE_SECRET_KEY is not set in .env — nothing to call.")
        return 1

    await db.connect()
    if not db.is_connected():
        print("No database. Set DATABASE_URL and try again.")
        return 1

    if args.order:
        row = await db.fetchrow(
            "select id, amount from orders where id = $1", args.order)
        if not row:
            print(f"No order {args.order}.")
            return 1
    else:
        row = await db.fetchrow(PICK)
        if not row:
            print("No delivered physical order that is free of an open refund "
                  "request.\nRun `make seed-disputes` first, or pass --order.")
            return 1
        print(f"Chose {row['id']}: {row['scans']} delivery scan(s), "
              f"{row['messages']} message(s), {row['policies']} policy record(s)")

    order_id = row["id"]
    # Stripe wants the smallest currency unit. INR is paise; USD is cents.
    amount = max(int(round(float(row["amount"]) * 100)), 5000)

    # Created first, confirmed second, with the payment id written onto the
    # order in between. Confirming in one call loses the race: Stripe raises
    # the dispute a second after the charge, and the webhook arrives to find
    # an order that does not yet name the payment it is about.
    body = {
        "amount": str(amount),
        "currency": args.currency,
        "payment_method": CARDS[args.category],
        "payment_method_types[0]": "card",
        "description": f"chargeback-guard live dispute demo for {order_id}",
        "metadata[order_id]": order_id,
    }

    print(f"Paying {amount / 100:,.2f} {args.currency.upper()} with "
          f"{CARDS[args.category]} …")
    auth = (settings.stripe_secret_key, "")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{STRIPE_API}/payment_intents", data=body, auth=auth)
        payload = response.json()
        if response.status_code != 200:
            error = payload.get("error", {})
            print(f"\nStripe refused the payment ({response.status_code}): "
                  f"{error.get('message')}")
            if "currency" in str(error.get("message", "")).lower():
                print("Try again with --currency usd.")
            return 1

        intent_id = payload["id"]

        # The link, written before anything can be disputed.
        await db.execute(
            "update orders set processor_payment_id = $2 where id = $1",
            order_id, intent_id)
        print(f"PaymentIntent {intent_id} linked to {order_id}")

        response = await client.post(
            f"{STRIPE_API}/payment_intents/{intent_id}/confirm", auth=auth)
        payload = response.json()
        if response.status_code != 200:
            print(f"\nStripe refused the confirmation: "
                  f"{payload.get('error', {}).get('message')}")
            return 1

    print(f"Confirmed — {payload.get('status')}")
    await db.disconnect()

    print("""
Done. Stripe raises the dispute a moment later; watch the `make stripe` tab
for charge.dispute.created.

Then:

  ID=$(curl -s "localhost:8000/disputes?origin=processor" | jq -r '.disputes[0].id')
  make packet ID=$ID
  make submit ID=$ID MODE=stage

The packet will cite this order's delivery scans, not an empty database.

If the dispute still shows citable_records: 0, it beat the link into the
database. `make relink` repairs that, and is safe to run at any time.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
