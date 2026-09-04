"""Run the triage rule over the dispute worklist and write the results back."""
from __future__ import annotations

import logging

from .. import db
from ..evidence.schema import Category
from .evidence import Availability, describe, load_availability
from .rule import Decision, decide

log = logging.getLogger(__name__)


async def triage_one(dispute_id: str) -> Decision | None:
    dispute = await db.fetchrow("select * from disputes where id = $1", dispute_id)
    if not dispute:
        return None

    availability = await load_availability([dispute["order_id"]]) if dispute["order_id"] else {}
    av = availability.get(dispute["order_id"]) or Availability(order_id=dispute["order_id"] or "")

    decision = decide(dispute, av)
    await _write(decision)
    return decision


async def triage_all(status: str = "needs_response", limit: int | None = None) -> dict:
    """Triage the open worklist in one pass.

    Availability is loaded for every order up front — six queries rather than
    six per dispute, which is the difference between a few seconds and a few
    minutes against a hosted database.
    """
    query = "select * from disputes where status = $1 order by opened_at"
    if limit:
        query += f" limit {int(limit)}"
    disputes = await db.fetch(query, status)
    if not disputes:
        return {"triaged": 0, "contest": 0, "accept": 0}

    order_ids = [d["order_id"] for d in disputes if d["order_id"]]
    availability = await load_availability(order_ids)

    decisions: list[Decision] = []
    for dispute in disputes:
        av = availability.get(dispute["order_id"]) or Availability(order_id=dispute["order_id"] or "")
        decisions.append(decide(dispute, av))

    await _write_many(decisions)

    contest = [d for d in decisions if d.recommendation == "contest"]
    accept = [d for d in decisions if d.recommendation == "accept"]
    return {
        "triaged": len(decisions),
        "contest": len(contest),
        "accept": len(accept),
        "contest_value": round(sum(d.expected_value for d in contest), 2),
        "avoided_effort": round(sum(-d.expected_value for d in accept), 2),
        "by_category": _by_category(decisions),
    }


async def decide_for(disputes: list[dict]) -> list[dict]:
    """Score these disputes without writing anything back.

    The inbox is ordered by expected value and says so on the page, but the
    columns it ordered by are written only by the batch pass — so a database
    nobody had run `make triage` against opened on a screen full of
    "untriaged", under a heading claiming the ordering was the product.

    Scoring is cheap once availability is batched, which it already is for the
    batch pass. The rows carry the decision whether or not anyone has run it.
    """
    if not disputes:
        return []
    order_ids = [d["order_id"] for d in disputes if d.get("order_id")]
    availability = await load_availability(order_ids)

    rows = []
    for dispute in disputes:
        row = dict(dispute)
        if row.get("recommendation") is None:
            av = (availability.get(row["order_id"])
                  or Availability(order_id=row.get("order_id") or ""))
            decision = decide(row, av)
            row["recommendation"] = decision.recommendation
            row["expected_value"] = round(decision.expected_value, 2)
            row["win_probability"] = round(decision.win_probability, 4)
            # Marked, because a score held only in this response is not the
            # same as one the batch pass has committed.
            row["triaged_inline"] = True
        rows.append(row)
    return rows


def _by_category(decisions: list[Decision]) -> list[dict]:
    buckets: dict[str, list[Decision]] = {}
    for d in decisions:
        buckets.setdefault(d.category, []).append(d)
    rows = []
    for category, items in buckets.items():
        contest = [d for d in items if d.recommendation == "contest"]
        rows.append({
            "category": category,
            "disputes": len(items),
            "contest": len(contest),
            "contest_share": round(len(contest) / len(items), 3),
            "mean_p_win": round(sum(d.win_probability for d in items) / len(items), 3),
            "mean_completeness": round(sum(d.completeness for d in items) / len(items), 3),
            "contest_value": round(sum(d.expected_value for d in contest), 2),
        })
    return sorted(rows, key=lambda r: -r["disputes"])


async def _write(d: Decision) -> None:
    await db.execute(
        """
        update disputes
           set win_probability = $2, expected_value = $3,
               recommendation = $4, triage_rule_version = $5, triaged_at = now()
         where id = $1
        """,
        d.dispute_id, d.win_probability, d.expected_value,
        d.recommendation, d.rule_version,
    )


async def _write_many(decisions: list[Decision]) -> None:
    if not decisions:
        return
    async with db.transaction() as conn:
        await conn.executemany(
            """
            update disputes
               set win_probability = $2, expected_value = $3,
                   recommendation = $4, triage_rule_version = $5, triaged_at = now()
             where id = $1
            """,
            [(d.dispute_id, d.win_probability, d.expected_value,
              d.recommendation, d.rule_version) for d in decisions],
        )


async def detail(dispute_id: str) -> dict | None:
    """Everything the dispute-detail screen needs, in one call."""
    dispute = await db.fetchrow("select * from disputes where id = $1", dispute_id)
    if not dispute:
        return None

    availability = await load_availability([dispute["order_id"]]) if dispute["order_id"] else {}
    av = availability.get(dispute["order_id"]) or Availability(order_id=dispute["order_id"] or "")
    category = Category(dispute["category"])
    decision = decide(dispute, av)

    return {
        "decision": {
            "recommendation": decision.recommendation,
            "win_probability": round(decision.win_probability, 4),
            "break_even_probability": round(decision.break_even_probability, 4),
            "margin": round(decision.margin, 4),
            "expected_value": round(decision.expected_value, 2),
            "completeness": round(decision.completeness, 3),
            "critical_evidence": decision.critical_evidence,
            "reasons": decision.reasons,
            "flips_if": decision.flips_if,
            "rule_version": decision.rule_version,
        },
        "evidence": describe(av, category),
        "gaps": decision.gaps,
    }
