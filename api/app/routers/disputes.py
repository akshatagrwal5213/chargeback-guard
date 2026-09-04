"""Dispute inbox and detail. The triage rule lands on day 4 (Tue 25 Aug)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from .. import db
from ..config import settings
from ..evidence.checklist import required_slots, spec_for
from ..evidence.schema import Category
from ..agent import render as agent_render
from ..agent import run as agent_run
from ..agent import submit as agent_submit
from ..routers import webhooks
from ..triage import runner
from ..triage.evidence import Availability, load_availability
from ..triage.rule import sensitivity

router = APIRouter(prefix="/disputes", tags=["disputes"])


# Everything one search box looks through. Listed rather than guessed at,
# because a search that silently does not cover the field you typed is worse
# than no search: you conclude the record is not there.
SEARCHABLE = (
    "d.id",                       # our dispute id
    "d.processor_dispute_id",     # du_… / disp_…
    "d.order_id",
    "d.category",
    "d.status",
    "d.processor_reason",
    "d.network_code",             # "Visa 13.1"
    "d.phase",
    "d.origin",
    "d.amount::text",             # 33636.16 — typing "33,636" also works, see below
    "o.product_code",
    "o.processor_payment_id",     # pi_… / ch_…
    "c.email",
    "c.id",
)

# `d.recommendation` is deliberately absent. It is written only by the batch
# pass, so on a database nobody has run `make triage` against it is null for
# every row — and searching "contest" would return nothing while the screen
# shows a column full of them. A search that silently does not cover what you
# typed is worse than no search: you conclude the record is not there.


@router.get("")
async def list_disputes(
    status: str | None = Query(default=None),
    origin: str | None = Query(default=None, pattern="^(processor|synthetic)$"),
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, le=500),
) -> dict:
    """Inbox. Ordered by expected value descending — that ordering is the product.

    `origin=processor` narrows to disputes that really exist at the processor.
    Without it they are easy to lose: a dispute that has just arrived has not
    been triaged, so it has no expected value, so it sorts to the very bottom
    of a list the demo data fills several times over.

    `q` searches every field a person might have in front of them — a dispute
    id from a log line, a `du_…` from Stripe's dashboard, a customer's email, a
    reason code, an amount. Server-side on purpose: filtering the fifty rows
    already loaded would be a search that cannot find anything not already
    visible, which is the opposite of the point.
    """
    # `has_db` says a URL is configured, not that it connected. The pool
    # raises when it did not, and this endpoint turned that into a 500 with a
    # traceback while every other endpoint answered 503. Same condition, same
    # answer.
    if not db.is_connected():
        # Same shape as a successful answer. A degraded response that drops
        # keys makes every caller write two readers, and the one for the rare
        # path is the one nobody tests.
        return {"disputes": [], "count": 0, "total": 0, "limit": limit,
                "query": q or None,
                "note": "Database not configured or unreachable."}

    clauses, args = [], []
    if status:
        args.append(status)
        clauses.append(f"d.status = ${len(args)}")
    if origin:
        args.append(origin)
        clauses.append(f"d.origin = ${len(args)}")
    if q and q.strip():
        # Commas and the rupee sign come free with copy-paste from this very
        # screen, and neither is in the database.
        term = q.strip().lstrip("₹").replace(",", "")
        args.append(f"%{term}%")
        n = len(args)
        clauses.append(
            "(" + " or ".join(f"{col} ilike ${n}" for col in SEARCHABLE) + ")")
    where = f"where {' and '.join(clauses)}" if clauses else ""
    rows = await db.fetch(
        f"""
        select d.*, o.amount as order_amount, o.is_digital,
               c.email as customer_email,
               (select count(*) from evidence_packets p where p.dispute_id = d.id) as packet_count
        from disputes d
        left join orders o on o.id = d.order_id
        left join customers c on c.id = o.customer_id
        {where}
        order by d.expected_value desc nulls last, d.opened_at desc
        limit {limit}
        """,
        *args,
    )
    # How many there are, not how many fit on the page. A worklist that shows
    # the top sixty of ten thousand and says nothing looks like a database
    # with sixty disputes in it.
    total = await db.fetchrow(
        f"""
        select count(*) as n
        from disputes d
        left join orders o on o.id = d.order_id
        left join customers c on c.id = o.customer_id
        {where}
        """,
        *args,
    )
    # Score anything the batch pass has not, so the ordering the page claims
    # is the ordering it shows. Re-sorted here because the database could only
    # sort by what it had.
    rows = await runner.decide_for(rows)
    rows.sort(key=lambda r: (r.get("expected_value") is None,
                             -float(r.get("expected_value") or 0)))
    return {
        "disputes": rows,
        "count": len(rows),
        "total": int((total or {}).get("n") or 0),
        "limit": limit,
        "query": q or None,
    }


@router.post("/triage")
async def run_triage(
    status: str = Query(default="needs_response"),
    limit: int | None = Query(default=None, le=5000),
) -> dict:
    """Score the whole open worklist and write recommendations back."""
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not configured")
    return await runner.triage_all(status=status, limit=limit)


@router.post("/relink")
async def relink(limit: int = Query(default=500, le=5000)) -> dict:
    """Retry order resolution for disputes recorded without one.

    A dispute can reach us before the order it belongs to is linkable — the
    processor raises it seconds after the charge, and a merchant's own write
    of the payment id may not have landed yet. Recording the dispute unlinked
    is right; leaving it that way forever is not, because an unlinked dispute
    has no evidence behind it and can never be defended.

    Idempotent, and it only ever fills a blank: a dispute that already names
    an order is left alone.
    """
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not configured")

    import json as _json
    rows = await db.fetch(
        "select id, raw_payload from disputes where order_id is null "
        "order by opened_at desc limit $1", limit)

    linked, checked = [], 0
    for row in rows:
        raw = row["raw_payload"]
        if isinstance(raw, str):
            raw = _json.loads(raw)
        obj = ((raw or {}).get("data") or {}).get("object") or {}
        if not obj:
            continue
        checked += 1
        order_id = await webhooks.resolve_order_id(obj)
        if not order_id:
            continue
        known = await db.fetchrow("select 1 from orders where id = $1", order_id)
        if not known:
            continue
        await db.execute(
            "update disputes set order_id = $2 where id = $1 and order_id is null",
            row["id"], order_id)
        linked.append({"dispute_id": row["id"], "order_id": order_id})

    return {"unlinked": len(rows), "checked": checked,
            "linked": len(linked), "disputes": linked}


@router.post("/{dispute_id}/packet")
async def build_packet(dispute_id: str, persist: bool = Query(default=True)) -> dict:
    """Retrieve, draft, verify, store. Returns the packet and the guard's report."""
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not configured")
    packet = await agent_run.build_packet(dispute_id, persist=persist)
    if packet is None:
        raise HTTPException(status_code=404, detail="Dispute not found")
    return packet


@router.get("/{dispute_id}/packet")
async def latest_packet(dispute_id: str) -> dict:
    """The most recent stored packet, with its citations."""
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not configured")
    row = await db.fetchrow(
        """
        select * from evidence_packets where dispute_id = $1
        order by created_at desc limit 1
        """,
        dispute_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="No packet drafted yet")
    citations = await db.fetch(
        "select claim, source_table, source_id, verified from evidence_citations "
        "where packet_id = $1", row["id"],
    )
    return {"packet": row, "citations": citations}


@router.post("/{dispute_id}/submit")
async def submit_packet(
    dispute_id: str,
    mode: str = Query(default="stage", pattern="^(stage|submit|dry_run)$"),
    confirm: bool = Query(default=False),
) -> dict:
    """File the latest packet with the processor.

    `stage` (the default) makes a real API call with submit=false: the evidence
    lands on the dispute and is visible in the processor's dashboard, but is
    not sent to the issuing bank. `submit` sends it, which cannot be undone,
    and therefore requires `confirm=true` — a mistyped mode should not file a
    representment. `dry_run` returns the exact payload and calls nothing.

    A packet the citation guard rejected a claim from is refused here, before
    any network call.
    """
    # Checked before the database, because it is a fact about the request:
    # the answer must not depend on whether this host happens to be connected.
    if mode == "submit" and not confirm:
        raise HTTPException(
            status_code=400,
            detail="mode=submit is irreversible. Repeat with confirm=true.",
        )
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        return await agent_submit.submit(dispute_id, mode=mode)
    except agent_submit.NothingToSubmit as exc:
        # Nothing here yet — same answer, and the same status, as asking for a
        # packet that has not been drafted.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except agent_submit.SubmissionRefused as exc:
        # We have a packet and will not file it. A different thing to be told.
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{dispute_id}/packet.html", response_class=HTMLResponse)
async def packet_html(dispute_id: str) -> HTMLResponse:
    """The submission document. Hover a reference to see its record."""
    dispute, packet = await _latest(dispute_id)
    return HTMLResponse(agent_render.to_html(dispute, packet))


@router.get("/{dispute_id}/packet.pdf")
async def packet_pdf(dispute_id: str):
    """PDF export. Falls back to HTML where the native stack is unavailable —
    a missing system library should not withhold the document."""
    dispute, packet = await _latest(dispute_id)
    html = agent_render.to_html(dispute, packet)
    pdf = agent_render.to_pdf(html)
    if pdf is None:
        return HTMLResponse(
            html,
            headers={"X-Packet-Format": "html; PDF unavailable on this host"},
        )
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition":
                 f'inline; filename="representment-{dispute_id}.pdf"'},
    )


async def _latest(dispute_id: str) -> tuple[dict, dict]:
    """The dispute and its most recent packet, or a 404 explaining which is missing."""
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not configured")
    dispute = await db.fetchrow("select * from disputes where id = $1", dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")
    row = await db.fetchrow(
        "select * from evidence_packets where dispute_id = $1 "
        "order by created_at desc limit 1", dispute_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail="No packet drafted yet — POST to /disputes/{id}/packet first")

    import json as _json
    packet = dict(row)
    for key in ("evidence", "gaps", "guard"):
        if isinstance(packet.get(key), str):
            packet[key] = _json.loads(packet[key])
    # The stored row keeps the narrative and evidence; the claim/guard detail
    # is rebuilt from the citations table so the document always reflects what
    # was actually persisted rather than a re-run.
    citations = await db.fetch(
        "select claim, source_table, source_id from evidence_citations "
        "where packet_id = $1 order by id", row["id"])
    claims: list[dict] = []
    for c in citations:
        ref = f"{c['source_table']}:{c['source_id']}"
        if claims and claims[-1]["text"] == c["claim"]:
            claims[-1]["refs"].append(ref)
        else:
            claims.append({"text": c["claim"], "refs": [ref]})

    narrative = packet.get("narrative") or ""
    packet.update({
        "summary": narrative.split("\n\n")[0] if narrative else "",
        "claims": claims,
        "gaps_acknowledged": packet.get("gaps") or [],
        "provider": packet.get("agent_model") or "template",
        "category": dispute["category"],
        # The verdict as it was at drafting time, not a fresh judgement.
        "guard": packet.get("guard") or {"claims_rejected": 0, "claims_flagged": 0},
    })
    return dispute, packet


@router.get("/{dispute_id}/triage")
async def dispute_triage(dispute_id: str) -> dict:
    """The recommendation, why, and the evidence checklist behind it."""
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not configured")
    result = await runner.detail(dispute_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Dispute not found")
    return result


@router.get("/{dispute_id}/sensitivity")
async def dispute_sensitivity(dispute_id: str, spread: float = Query(default=0.15, le=0.5)) -> dict:
    """How far the win probability can be wrong before the answer changes.

    The estimate is a stated prior, not a fitted model, so this is the honest
    way to present it: not a number to trust, but a decision that either
    survives being wrong or does not.
    """
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not configured")
    dispute = await db.fetchrow("select * from disputes where id = $1", dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")
    av_map = await load_availability([dispute["order_id"]]) if dispute["order_id"] else {}
    av = av_map.get(dispute["order_id"]) or Availability(order_id=dispute["order_id"] or "")
    return sensitivity(dispute, av, spread=spread)


@router.get("/{dispute_id}")
async def get_dispute(dispute_id: str) -> dict:
    if not settings.has_db:
        raise HTTPException(status_code=503, detail="Database not configured")

    dispute = await db.fetchrow("select * from disputes where id = $1", dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")

    order = (
        await db.fetchrow("select * from orders where id = $1", dispute["order_id"])
        if dispute["order_id"]
        else None
    )
    packet = await db.fetchrow(
        """
        select * from evidence_packets
        where dispute_id = $1 order by created_at desc limit 1
        """,
        dispute_id,
    )

    category = Category(dispute["category"])
    spec = spec_for(category)
    is_digital = bool(order and order.get("is_digital"))

    return {
        "dispute": dispute,
        "order": order,
        "packet": packet,
        "checklist": {
            "category": category.value,
            "label": spec.label,
            "visa_code": spec.visa_code,
            "mastercard_code": spec.mastercard_code,
            "required": [s.value for s in required_slots(category, is_digital)],
            "supporting": [s.value for s in spec.supporting],
            "guidance": spec.guidance,
        },
    }
