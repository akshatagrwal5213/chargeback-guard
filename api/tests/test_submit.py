"""Filing the packet.

The interesting cases are all refusals. A submission path that will not send
a document it cannot stand behind is the whole reason this module has a gate
in front of it, so most of what is checked here is what does *not* happen.
"""
from __future__ import annotations

import pytest

from app.agent import submit as agent_submit
from app.evidence.adapters import to_stripe


def _packet(**kw) -> dict:
    base = {
        "category": "product_not_received",
        "narrative": "The order was delivered.",
        "claims": [{"claim": "The carrier recorded delivery on 2026-07-16."}],
        "guard": {"claims_kept": 1, "claims_rejected": 0,
                  "claims_flagged": 0, "submittable": True},
        "evidence": {"tools": [
            {"tool": "get_fulfillment", "records": [
                {"ref": "fulfillment_events:881", "kind": "delivery_scan",
                 "summary": "delivered at 2026-07-16 14:39, New Delhi 110001",
                 "fields": {}},
            ]},
            {"tool": "get_communications", "records": [
                {"ref": "support_messages:12", "kind": "support_message",
                 "summary": "Customer asked about the parcel on 2026-07-17.",
                 "fields": {}},
            ]},
            {"tool": "find_similar_disputes", "records": [
                {"ref": "disputes:dsp_old", "kind": "precedent",
                 "summary": "A comparable dispute was lost.", "fields": {}},
            ]},
        ]},
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------ the gate

def test_a_rejected_claim_blocks_filing():
    packet = _packet(guard={"claims_rejected": 1, "submittable": False})
    with pytest.raises(agent_submit.SubmissionRefused) as exc:
        agent_submit._check(packet)
    assert "citation guard" in str(exc.value)


def test_non_submittable_verdict_blocks_filing_even_with_no_rejections():
    """`submittable` is false when nothing survived, not only when something
    was cut. Either way there is no document to file."""
    packet = _packet(guard={"claims_rejected": 0, "submittable": False})
    with pytest.raises(agent_submit.SubmissionRefused):
        agent_submit._check(packet)


def test_empty_packet_blocks_filing():
    with pytest.raises(agent_submit.SubmissionRefused):
        agent_submit._check(_packet(claims=[]))


def test_a_clean_packet_passes():
    agent_submit._check(_packet())


def test_a_flagged_but_unrejected_claim_still_files():
    """Flagging is a warning about a value the guard could not corroborate;
    rejection is a claim with no source. Only the second is disqualifying."""
    agent_submit._check(_packet(
        guard={"claims_rejected": 0, "claims_flagged": 2, "submittable": True}))


# ------------------------------------------------------------ what gets sent

def test_precedent_never_reaches_the_processor():
    """Losing a comparable case is an internal signal. Telling the issuer
    about it argues their side for them."""
    ep = agent_submit._to_evidence_packet({"id": "dsp_1"}, _packet())
    body = to_stripe(ep)
    blob = "\n".join(body.values())
    assert "comparable dispute was lost" not in blob
    assert "delivered at 2026-07-16" in blob


def test_records_land_in_the_processor_field_for_their_kind():
    """`shipping_documentation` and `customer_communication` take file IDs, so
    these records go into the written statement under their own headings."""
    ep = agent_submit._to_evidence_packet({"id": "dsp_1"}, _packet())
    body = to_stripe(ep)
    statement = body["uncategorized_text"]
    assert "DELIVERY" in statement and "2026-07-16" in statement
    assert "CUSTOMER CONTACT" in statement and "parcel" in statement


def test_nothing_a_packet_produces_addresses_a_file_field():
    from app.evidence.adapters import STRIPE_FILE_FIELDS

    ep = agent_submit._to_evidence_packet({"id": "dsp_1"}, _packet())
    assert not set(to_stripe(ep)) & STRIPE_FILE_FIELDS


def test_the_narrative_carries_into_the_submission():
    ep = agent_submit._to_evidence_packet({"id": "dsp_1"}, _packet())
    assert "The order was delivered." in to_stripe(ep)["uncategorized_text"]


# ------------------------------------------------------------- Stripe's cap

def test_oversized_evidence_is_trimmed_without_losing_a_field():
    body = {
        "uncategorized_text": "x" * 200_000,
        "shipping_documentation": "y" * 500,
        "customer_communication": "z" * 500,
    }
    fitted = agent_submit._fits(dict(body))
    assert set(fitted) == set(body)                       # nothing dropped
    assert sum(len(v) for v in fitted.values()) <= agent_submit.STRIPE_EVIDENCE_CHAR_LIMIT
    assert fitted["shipping_documentation"] == body["shipping_documentation"]
    assert fitted["uncategorized_text"].endswith("[trimmed]")


def test_evidence_within_the_cap_is_left_alone():
    body = {"uncategorized_text": "x" * 100}
    assert agent_submit._fits(dict(body)) == body


# ----------------------------------------------- deciding what is real

def _dispute(**kw) -> dict:
    base = {"id": "dsp_1", "rail": "stripe", "origin": "synthetic",
            "processor_dispute_id": "sim_du_9f2c1e04dead"}
    base.update(kw)
    return base


def test_a_generated_dispute_is_never_filed_anywhere():
    """Most disputes in this database were synthesised. They have no
    counterpart at the processor, whatever their id looks like."""
    assert agent_submit._filable(_dispute()) is False
    assert agent_submit._filable(_dispute(rail="razorpay")) is False


def test_an_id_that_looks_like_stripes_is_still_not_real(monkeypatch):
    """The check used to read the id, and the generator's ids started with
    `du_`. A submission went out against a dispute that did not exist and came
    back `No such dispute`. Provenance decides now, not shape."""
    from app import config
    monkeypatch.setattr(config.settings, "stripe_secret_key", "sk_test_x")
    assert agent_submit._filable(
        _dispute(processor_dispute_id="du_1MtJUT2eZvKYlo2CNaw2HvEv")) is False


def test_a_webhook_dispute_without_credentials_is_not_filable(monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "stripe_secret_key", "")
    assert agent_submit._filable(
        _dispute(origin="processor", processor_dispute_id="du_abc123")) is False


def test_a_webhook_dispute_with_credentials_is_filable(monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "stripe_secret_key", "sk_test_x")
    assert agent_submit._filable(
        _dispute(origin="processor", processor_dispute_id="du_abc123")) is True


def test_the_refusal_says_which_of_the_two_reasons_applies(monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "stripe_secret_key", "")
    assert "generated locally" in agent_submit._why_not_filable(_dispute())
    assert "STRIPE_SECRET_KEY" in agent_submit._why_not_filable(
        _dispute(origin="processor", processor_dispute_id="du_abc123"))


# --------------------------------------------------------------- the route

def test_submitting_for_real_requires_confirmation():
    """`mode=submit` is irreversible, so it takes a second explicit signal.
    A typo in a query string should not file a representment."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        r = client.post("/disputes/dsp_1/submit?mode=submit")
    assert r.status_code == 400
    assert "confirm=true" in r.json()["detail"]


def test_an_unknown_mode_is_rejected_by_the_schema():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        r = client.post("/disputes/dsp_1/submit?mode=send_it")
    assert r.status_code == 422


# ------------------------------------------------------- the wire format

def test_staging_sends_submit_false():
    """The single field that separates a draft from a filed representment."""
    form = agent_submit.stripe_form({"receipt": "r"}, submit=False)
    assert form["submit"] == "false"
    assert form["evidence[receipt]"] == "r"


def test_submitting_sends_submit_true():
    assert agent_submit.stripe_form({"receipt": "r"}, submit=True)["submit"] == "true"


@pytest.mark.asyncio
async def test_the_stripe_call_is_authenticated_and_form_encoded(monkeypatch):
    """Exercises the real request path against a stub transport, so the
    encoding and auth are checked rather than assumed."""
    import httpx
    from app import config

    monkeypatch.setattr(config.settings, "stripe_secret_key", "sk_test_key")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization", "")
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"id": "du_abc",
                                         "evidence_details": {"has_evidence": True}})

    result = await agent_submit._post_stripe(
        "du_abc", {"shipping_documentation": "delivered 2026-07-16"},
        submit=False, transport=httpx.MockTransport(handler))

    assert seen["url"].endswith("/v1/disputes/du_abc")
    assert seen["auth"].startswith("Basic ")
    assert "evidence%5Bshipping_documentation%5D" in seen["body"]
    assert "submit=false" in seen["body"]
    assert result["mode"] == "staged" and result["ok"] is True


@pytest.mark.asyncio
async def test_a_stripe_error_is_reported_not_swallowed():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "No such dispute"}})

    result = await agent_submit._post_stripe(
        "du_missing", {"receipt": "r"}, submit=False,
        transport=httpx.MockTransport(handler))

    assert result["ok"] is False
    assert result["status_code"] == 400
    assert result["error"]["message"] == "No such dispute"


@pytest.mark.asyncio
async def test_a_rejected_call_is_not_reported_as_staged():
    """A 400 stages nothing. Recording the intent as the outcome would leave
    a packet marked filed that the processor never accepted."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "No such file_upload"}})

    result = await agent_submit._post_stripe(
        "du_x", {"uncategorized_text": "t"}, submit=False,
        transport=httpx.MockTransport(handler))

    assert result["mode"] == "failed"
    assert result["attempted"] == "stage"


def test_nothing_drafted_and_refused_to_file_are_different_answers():
    """A 409 should mean the system has a packet and will not send it. If
    "you have not drafted one yet" shares that status, the refusal that
    matters is indistinguishable from a step the caller simply skipped."""
    assert issubclass(agent_submit.NothingToSubmit, agent_submit.SubmissionRefused)

    with pytest.raises(agent_submit.SubmissionRefused) as refused:
        agent_submit._check(_packet(guard={"claims_rejected": 1, "submittable": False}))
    assert not isinstance(refused.value, agent_submit.NothingToSubmit)


@pytest.mark.asyncio
async def test_a_declined_dispute_cannot_be_filed_from_a_stale_packet(monkeypatch):
    """The refusal used to live only at drafting. A packet written before the
    decision changed stayed in the database, and submission read it out and
    built a payload without asking what the rule now said — which is how a
    dispute the system had declined could still be filed.

    Found by `make verify` on its first real run, not by a unit test.
    """
    from app.triage import runner

    async def declined(dispute_id):
        return {"decision": {"recommendation": "accept",
                             "flips_if": "A refund for this order is open."}}

    monkeypatch.setattr(runner, "detail", declined)
    with pytest.raises(agent_submit.SubmissionRefused) as exc:
        await agent_submit._check_still_worth_filing("dsp_1")
    assert "not filed" in str(exc.value)
    assert "refund" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_a_contested_dispute_passes_the_same_gate(monkeypatch):
    from app.triage import runner

    async def contest(dispute_id):
        return {"decision": {"recommendation": "contest"}}

    monkeypatch.setattr(runner, "detail", contest)
    await agent_submit._check_still_worth_filing("dsp_1")     # does not raise
