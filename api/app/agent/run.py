"""Retrieve, draft, verify, persist.

Retrieval is deterministic. The checklist already says what a category turns
on, so there is nothing for a model to decide about which records to fetch —
an agent loop here would add latency and nondeterminism to a lookup we can
write down. The tools exist and are individually callable; the orchestration
just does not need to be guessed at.

The model's job is narrow and stated plainly: given these records, write the
narrative. The guard then checks it against the database.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from .. import db
from ..evidence.checklist import required_slots, spec_for
from ..evidence.schema import Category
from ..triage import runner as triage_runner
from . import guard, tools
from .compose import Draft, compose_async, deterministic_summary
from .provider import get_provider
from .records import Evidence

log = logging.getLogger(__name__)

MAX_REPAIRS = 1          # one correction round; beyond that the guard just cuts


async def retrieve(dispute: dict) -> Evidence:
    """Everything the category could possibly need, fetched concurrently.

    These are seven independent queries against a hosted Postgres. Run in
    sequence they cost seven round trips to whichever region the database
    lives in, which from India to Singapore is real wall-clock time for no
    reason — none of them depends on another's result.

    The order of the results is fixed regardless, because gather preserves it.
    That matters: the evidence list is what the model reads, and a narrative
    that reorders itself between runs is harder to review.
    """
    evidence = Evidence()
    order_id = dispute.get("order_id")
    category = Category(dispute["category"])

    if not order_id:
        return evidence

    order_row = await db.fetchrow(
        "select customer_id, is_digital from orders where id = $1", order_id
    )
    if not order_row:
        return evidence

    is_digital = bool(order_row.get("is_digital"))
    customer_id = order_row.get("customer_id")

    jobs = [tools.get_order(order_id)]
    if customer_id:
        jobs.append(tools.get_customer_history(customer_id))
    # Physical goods have scans, digital goods have usage. Both are fetched:
    # the absence of either is itself evidence the agent may cite.
    if not is_digital:
        jobs.append(tools.get_fulfillment(order_id))
    jobs += [
        tools.get_access_log(order_id),
        tools.get_communications(order_id),
        tools.get_policy_acceptance(order_id),
        tools.get_refunds(order_id),
        tools.find_similar_disputes(category.value, limit=5),
    ]

    for result in await asyncio.gather(*jobs):
        evidence.add(result)

    log.info("Retrieved %d citable record(s) for %s across %d tool(s)",
             len(evidence.refs), dispute["id"], len(evidence.results))
    return evidence


async def build_packet(dispute_id: str, persist: bool = True) -> dict | None:
    dispute = await db.fetchrow("select * from disputes where id = $1", dispute_id)
    if not dispute:
        return None

    category = Category(dispute["category"])

    t0 = time.monotonic()
    evidence = await retrieve(dispute)
    t_retrieve = time.monotonic() - t0

    t0 = time.monotonic()
    triage = await triage_runner.detail(dispute_id)
    decision = (triage or {}).get("decision")
    t_triage = time.monotonic() - t0

    # Refusing to draft is itself an outcome, and the honest one. A packet for
    # a dispute the system has already decided not to contest would be a
    # document arguing a case it does not believe.
    if decision and decision.get("recommendation") == "accept":
        return {
            "dispute_id": dispute_id,
            "status": "not_drafted",
            "reason": decision.get("flips_if") or
                      "Triage recommends accepting this dispute.",
            "decision": decision,
            "citable_records": len(evidence.refs),
        }

    provider = get_provider()
    started = time.monotonic()
    draft: Draft = await compose_async(dispute, evidence, provider, decision)
    verdict = guard.verify(draft, evidence, dispute)

    repairs = 0
    while verdict.rejected and repairs < MAX_REPAIRS and provider.name != "template":
        repairs += 1
        log.info("Repair round %d for %s: %d claim(s) failed the guard",
                 repairs, dispute_id, len(verdict.rejected))
        draft = await compose_async(dispute, evidence, provider, decision,
                                    feedback=verdict.feedback())
        verdict = guard.verify(draft, evidence, dispute)

    final = guard.apply(
        draft, verdict,
        fallback_summary=deterministic_summary(dispute, len(verdict.claims)))
    is_digital = bool((await db.fetchrow(
        "select is_digital from orders where id = $1", dispute["order_id"]) or {}
    ).get("is_digital"))
    required = required_slots(category, is_digital)

    packet = {
        "dispute_id": dispute_id,
        "status": "drafted" if final.claims else "empty",
        "category": category.value,
        "provider": final.provider,
        # Empty unless a model was configured and could not be used. "template"
        # with no reason means no key; "template" with a reason means the model
        # was reached and something went wrong, which is a different problem.
        "provider_note": final.fallback_reason,
        "repairs": repairs,
        "timings": {
            "retrieve": round(t_retrieve, 2),
            "triage": round(t_triage, 2),
            "draft": round(time.monotonic() - started, 2),
        },
        "citable_records": len(evidence.refs),
        # The evidence itself, not just the count. The renderer resolves each
        # citation against it to build the source appendix — without it every
        # packet renders with citation chips pointing at nothing.
        "evidence": evidence.as_dict(),
        "summary": final.summary,
        "narrative": final.narrative(),
        "claims": [c.as_dict() for c in final.claims],
        "gaps_acknowledged": final.gaps_acknowledged,
        "guard": verdict.as_dict(),
        "required_slots": [s.value for s in required],
        "guidance": spec_for(category).guidance,
        "decision": decision,
    }

    if persist:
        packet["packet_id"] = await _persist(dispute_id, final, verdict, evidence)
    return packet


async def _persist(dispute_id: str, draft: Draft, verdict: guard.Verdict,
                   evidence: Evidence) -> str:
    packet_id = str(uuid.uuid4())
    async with db.transaction() as conn:
        await conn.execute(
            """
            insert into evidence_packets
                (id, dispute_id, agent_model, evidence, narrative,
                 slots_required, slots_filled, gaps, guard)
            values ($1, $2, $3, $4::jsonb, $5, $6, $7, $8::jsonb, $9::jsonb)
            """,
            packet_id, dispute_id, draft.provider,
            json.dumps(evidence.as_dict(), default=str),
            draft.narrative(),
            len(evidence.results), len(draft.claims),
            json.dumps(draft.gaps_acknowledged),
            # The verdict travels with the packet. Submission reads it back
            # rather than trusting the caller: a draft that lost a claim to
            # the guard must still be unfilable an hour later.
            json.dumps(verdict.as_dict(), default=str),
        )
        # One row per claim per source. `verified` is true because the claim
        # survived the guard — an unverified citation never reaches this table.
        rows = [
            (packet_id, claim.text, ref.split(":", 1)[0], ref.split(":", 1)[1], True)
            for claim in draft.claims for ref in claim.refs
        ]
        if rows:
            await conn.executemany(
                """
                insert into evidence_citations
                    (packet_id, claim, source_table, source_id, verified)
                values ($1, $2, $3, $4, $5)
                """,
                rows,
            )
    log.info("Packet %s stored with %d citation(s)", packet_id, len(rows))
    return packet_id
