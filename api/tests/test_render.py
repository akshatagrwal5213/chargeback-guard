"""Packet rendering.

The document is what an issuer's analyst reads and what a reviewer looks at.
These check the properties that make it trustworthy rather than the markup.
"""
from __future__ import annotations

import re

from app.agent import render


def _dispute() -> dict:
    return {
        "id": "dsp_1", "order_id": "ord_1", "rail": "stripe",
        "processor_dispute_id": "du_abc", "amount": 26530.48, "currency": "INR",
        "category": "product_not_received", "phase": "chargeback",
        "opened_at": "2026-07-20T00:00:00Z", "respond_by": "2026-08-05T00:00:00Z",
    }


def _packet(**kw) -> dict:
    base = {
        "category": "product_not_received",
        "provider": "template",
        "summary": "The merchant contests this dispute.",
        "claims": [
            {"text": "The carrier recorded delivery on 2026-07-16.",
             "refs": ["fulfillment_events:881"]},
            {"text": "The order was placed on 2026-07-15.", "refs": ["orders:ord_1"]},
        ],
        "gaps_acknowledged": ["No signature was captured on delivery."],
        "guard": {"claims_rejected": 0, "claims_flagged": 0},
        "evidence": {"tools": [
            {"tool": "get_fulfillment", "records": [
                {"ref": "fulfillment_events:881", "kind": "delivery_scan",
                 "summary": "delivered at 2026-07-16 14:39, New Delhi 110001",
                 "fields": {}},
                {"ref": "fulfillment_events:880", "kind": "carrier_scan",
                 "summary": "in transit", "fields": {}},
            ]},
            {"tool": "get_order", "records": [
                {"ref": "orders:ord_1", "kind": "order",
                 "summary": "INR 26,530.48 on 2026-07-15", "fields": {}},
            ]},
        ]},
    }
    base.update(kw)
    return base


def test_every_citation_resolves_to_a_record():
    """Regression: build_packet persisted the evidence but did not return it,
    so every chip rendered pointing at nothing while still looking correct."""
    html = render.to_html(_dispute(), _packet())
    assert 'title="record not found"' not in html
    assert "delivered at 2026-07-16 14:39" in html


def test_the_appendix_lists_only_records_that_were_cited():
    """Listing everything retrieved pads the document with facts nobody
    argued from, which makes the cited ones harder to check."""
    html = render.to_html(_dispute(), _packet())
    assert "fulfillment_events:881" in html
    assert "orders:ord_1" in html
    # retrieved but never cited
    assert "fulfillment_events:880" not in html


def test_claim_count_matches_what_is_rendered():
    html = render.to_html(_dispute(), _packet())
    assert len(re.findall(r"<li>", html)) >= 2
    assert "2 assertions" in html


def test_acknowledged_gaps_appear_in_the_document():
    """Conceding what the records cannot show is part of the submission, not
    an omission from it."""
    html = render.to_html(_dispute(), _packet())
    assert "No signature was captured" in html
    assert "Acknowledged limitations" in html


def test_a_flagged_packet_says_it_is_not_auto_submitted():
    html = render.to_html(
        _dispute(), _packet(guard={"claims_rejected": 2, "claims_flagged": 1}))
    assert "2 were removed" in html
    assert "not submitted automatically" in html


def test_a_clean_packet_makes_no_removal_claim():
    html = render.to_html(_dispute(), _packet())
    assert "were removed on this submission" not in html
    assert "not submitted automatically" not in html


def test_html_escapes_record_content():
    """Record text comes from a database that a customer's support message
    reached. It must not be able to inject markup into the document."""
    packet = _packet()
    packet["evidence"]["tools"][0]["records"][0]["summary"] = "<script>alert(1)</script>"
    packet["claims"] = [{"text": "Delivery.", "refs": ["fulfillment_events:881"]}]
    html = render.to_html(_dispute(), packet)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_pdf_absence_is_reported_not_raised():
    """A missing system library should not withhold the document."""
    result = render.to_pdf("<html><body>hi</body></html>")
    assert result is None or isinstance(result, bytes)


# ------------------------------------------- what belongs in a submission

def test_precedent_is_never_cited_to_the_issuer():
    """find_similar_disputes is internal decision support. How a different
    case resolved says nothing about whether this parcel was delivered, and a
    submission citing the merchant's own losses argues against itself.

    Found in a rendered document: 'A comparable prior dispute ... was lost.'
    """
    from app.agent.compose import compose_from_template
    from app.agent.records import Evidence, Record, ToolResult

    ev = Evidence()
    ev.add(ToolResult("get_fulfillment", [
        Record(ref="fulfillment_events:1", kind="delivery_scan",
               summary="delivered 2026-07-16"),
    ]))
    ev.add(ToolResult("find_similar_disputes", [
        Record(ref="dispute_outcomes:dsp_x", kind="precedent",
               summary="a comparable dispute was lost"),
    ]))

    draft = compose_from_template(
        {"id": "d", "category": "product_not_received", "amount": 100.0}, ev)

    cited = {ref for c in draft.claims for ref in c.refs}
    assert "fulfillment_events:1" in cited
    assert "dispute_outcomes:dsp_x" not in cited
    assert "lost" not in draft.narrative().lower()


def test_the_position_is_addressed_to_the_issuer_not_the_drafter():
    """spec.guidance says things like 'show the delivery scan' — an
    instruction to whoever assembles the packet. Rendered into the summary it
    read as the merchant reciting its own homework to the adjudicator.
    """
    from app.agent.compose import POSITION, compose_from_template
    from app.agent.records import Evidence, Record, ToolResult
    from app.evidence.checklist import CHECKLIST
    from app.evidence.schema import Category

    for category in Category:
        assert category in POSITION, f"{category} has no stated position"

    ev = Evidence()
    ev.add(ToolResult("get_order", [
        Record(ref="orders:o1", kind="order", summary="INR 100 on 2026-07-01"),
    ]))
    draft = compose_from_template(
        {"id": "d", "category": "product_not_received", "amount": 100.0}, ev)

    assert draft.summary.startswith(POSITION[Category.PRODUCT_NOT_RECEIVED])
    guidance = CHECKLIST[Category.PRODUCT_NOT_RECEIVED].guidance
    assert guidance not in draft.summary


def test_the_rendered_summary_is_not_empty():
    """Regression: build_packet returned 'narrative' but the template read
    'summary', so 'Merchant's position' rendered blank in every document."""
    html = render.to_html(_dispute(), _packet())
    assert "Merchant&#39;s position" in html or "Merchant's position" in html
    assert "The merchant contests this dispute." in html
