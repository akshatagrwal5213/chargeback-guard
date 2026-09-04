"""Day 1 smoke tests. These must stay green for the rest of the build."""
from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from fastapi.testclient import TestClient

from app.evidence.adapters import (
    RAZORPAY_SLOT_MAP,
    STRIPE_SLOT_MAP,
    category_from_razorpay,
    category_from_stripe,
    escalation_cost,
    to_razorpay,
    to_stripe,
)
from app.evidence.checklist import CHECKLIST, required_slots
from app.evidence.schema import Category, Citation, EvidencePacket, Slot, SlotValue, Strength
from app.main import app
from app.routers.scoring import ScoreRequest, heuristic_score
from app.routers.webhooks import verify_stripe_signature

client = TestClient(app)


# ------------------------------------------------------------------ boot

def test_root_and_health_without_config():
    """The app must be usable before any account exists."""
    assert client.get("/").status_code == 200
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["mode"] == "test-only"
    assert body["categories_loaded"] == 7


def test_placeholder_database_url_counts_as_unconfigured():
    """Regression: .env.example once shipped a placeholder host. It is not
    blank, so the app treated it as configured, tried to resolve it and died
    at startup. Template text must read as 'no database'."""
    from app.config import Settings

    assert not Settings(database_url="").has_db
    assert not Settings(database_url="   ").has_db
    assert not Settings(
        database_url="postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres"
    ).has_db
    assert not Settings(database_url="postgresql://u:YOUR_PW@host:5432/db").has_db
    # a real-looking URL must still be accepted
    assert Settings(
        database_url="postgresql://postgres:s3cret@db.abcdxyz.supabase.co:5432/postgres"
    ).has_db


def test_startup_survives_an_unreachable_database(monkeypatch):
    """A wrong connection string must degrade the app, never kill it."""
    import socket as _socket

    from app import db as db_module

    monkeypatch.setattr(
        db_module.settings, "database_url", "postgresql://u:p@nope.invalid:5432/db"
    )

    async def boom(*_a, **_kw):
        raise _socket.gaierror(8, "nodename nor servname provided, or not known")

    monkeypatch.setattr(db_module.asyncpg, "create_pool", boom)

    import asyncio as _asyncio

    _asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        db_module.connect()
    )
    assert db_module.is_connected() is False

    # and the app still serves
    assert client.get("/health").json()["status"] == "degraded"


def test_checklist_endpoint_exposes_all_categories():
    body = client.get("/checklist").json()
    assert set(body) == {c.value for c in Category}
    assert body["product_not_received"]["visa_code"] == "13.1"
    assert body["fraudulent"]["mastercard_code"] == "4837"


# ------------------------------------------------------------- checklist

def test_every_category_has_a_spec_and_required_slots():
    for category in Category:
        spec = CHECKLIST[category]
        assert spec.required, f"{category} has no required slots"
        assert 0.0 < spec.base_win_rate < 1.0
        assert spec.guidance


def test_digital_orders_substitute_shipping_slots():
    """A digital order can't produce a delivery scan; demanding one would
    manufacture a permanent gap and force every digital dispute to 'accept'."""
    physical = required_slots(Category.PRODUCT_NOT_RECEIVED, is_digital=False)
    digital = required_slots(Category.PRODUCT_NOT_RECEIVED, is_digital=True)
    assert Slot.SHIPPING_PROOF in physical
    assert Slot.SHIPPING_PROOF not in digital
    assert Slot.ACCESS_ACTIVITY_LOG in digital
    assert len(physical) == len(digital)


# --------------------------------------------------------------- adapters

def test_stripe_reason_mapping():
    assert category_from_stripe("fraudulent") is Category.FRAUDULENT
    assert category_from_stripe("product_not_received") is Category.PRODUCT_NOT_RECEIVED
    assert category_from_stripe("something_new") is Category.GENERAL


def test_razorpay_reason_mapping_is_keyword_based():
    assert category_from_razorpay("", "No Cardholder Authorization") is Category.FRAUDULENT
    assert category_from_razorpay("4855", "Goods or Services Not Provided") is Category.PRODUCT_NOT_RECEIVED
    assert category_from_razorpay("", "Defective or Not As Described") is Category.PRODUCT_UNACCEPTABLE
    assert category_from_razorpay("9999", "who knows") is Category.GENERAL


def _packet() -> EvidencePacket:
    return EvidencePacket(
        dispute_id="dsp_test",
        category=Category.PRODUCT_NOT_RECEIVED,
        slots={
            Slot.SHIPPING_PROOF: SlotValue(
                slot=Slot.SHIPPING_PROOF,
                value="Delivered 2026-07-14 09:22 IST, Bengaluru 560001",
                strength=Strength.FILLED,
                citations=[Citation(claim="delivered", source_table="fulfillment_events", source_id="881")],
            ),
            Slot.SHIPPING_TRACKING: SlotValue(
                slot=Slot.SHIPPING_TRACKING, value="BD1234567IN", strength=Strength.FILLED
            ),
            Slot.SHIPPING_ADDRESS: SlotValue(
                slot=Slot.SHIPPING_ADDRESS, value="12 MG Road, Bengaluru", strength=Strength.FILLED
            ),
            Slot.SHIPPING_DATE: SlotValue(
                slot=Slot.SHIPPING_DATE, value="2026-07-11", strength=Strength.MISSING
            ),
        },
        narrative="Carrier records show delivery to the cardholder's billing address.",
    )


def test_stripe_adapter_uses_real_field_names():
    body = to_stripe(_packet())
    assert body["shipping_tracking_number"] == "BD1234567IN"
    assert body["shipping_address"] == "12 MG Road, Bengaluru"
    assert "Carrier records" in body["uncategorized_text"]
    # missing slots must not appear at all
    assert "shipping_date" not in body


def test_no_text_is_ever_sent_to_a_file_upload_field():
    """Nine of Stripe's evidence fields take a file upload ID, not prose.
    Sending text to one is rejected outright, and the rejection names only the
    first offender — so this checks the whole hash rather than one field.

    Found in production, not in a test: a staged submission came back with
    `No such file_upload: 'INR 2,803.03 on 2026-05-02, ...'`.
    """
    from app.evidence.adapters import STRIPE_FILE_FIELDS

    packet = _packet()
    # Fill every slot, so a slot that maps somewhere it should not is visible.
    for slot in Slot:
        packet.slots.setdefault(
            slot, SlotValue(slot=slot, value=f"text for {slot.value}",
                            strength=Strength.FILLED))
    body = to_stripe(packet)
    assert not set(body) & STRIPE_FILE_FIELDS, (
        f"text addressed to file field(s): {sorted(set(body) & STRIPE_FILE_FIELDS)}")


def test_documentary_evidence_survives_as_text():
    """A delivery scan has no text field to go to. It must still reach the
    issuer — in the written statement — rather than being silently dropped."""
    body = to_stripe(_packet())
    assert "DELIVERY" in body["uncategorized_text"]
    assert "Delivered" in body["uncategorized_text"]


def test_razorpay_adapter_folds_slots_into_coarser_fields():
    body = to_razorpay(_packet())
    assert body["action"] == "submit"
    # Razorpay groups shipping evidence under one field
    assert "BD1234567IN" in body["shipping_proof"]
    assert "Delivered" in body["shipping_proof"]
    assert "Carrier records" in body["summary"]


def test_every_slot_maps_to_at_least_one_rail():
    for slot in Slot:
        assert slot in STRIPE_SLOT_MAP or slot in RAZORPAY_SLOT_MAP, f"{slot} maps nowhere"


def test_escalation_cost_climbs_with_phase():
    assert escalation_cost(1000, "retrieval") < escalation_cost(1000, "chargeback")
    assert escalation_cost(1000, "chargeback") < escalation_cost(1000, "pre_arbitration")
    assert escalation_cost(1000, "pre_arbitration") < escalation_cost(1000, "arbitration")


def test_completeness_counts_only_required_slots():
    packet = _packet()
    required = required_slots(Category.PRODUCT_NOT_RECEIVED)
    assert packet.completeness(required) == 0.75  # 3 of 4 filled
    assert Slot.SHIPPING_DATE in packet.gaps


# ------------------------------------------------------------- scoring

def test_score_endpoint_returns_capture_actions_not_a_decline():
    """The defense-only guarantee, asserted as a test: the score changes what
    we record, never whether the customer is served.

    Deliberately asserts the invariant rather than a specific tier — the
    heuristic and the trained model put the same order on different sides of
    their thresholds, and pinning one number here would make this test a
    tripwire for retraining instead of a guard on the guarantee.
    """
    r = client.post(
        "/score",
        json={
            "order_id": "ord_1",
            "amount": 42000,
            "cvv_result": "N",
            "avs_result": "N",
            "prior_disputes": 2,
            "is_guest": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["evidence_tier"] in {"standard", "enhanced"}
    assert body["capture"], "every order gets a capture plan, weak or strong"
    assert body["reasons"]
    # no field anywhere in the response may authorise a block
    assert not {"decline", "block", "reject", "action"} & set(body)


def test_higher_risk_never_reduces_evidence_capture():
    """Ordering is the real contract: a riskier order must never come back
    with *less* evidence collected than a safer one."""
    risky = client.post("/score", json={
        "order_id": "ord_risky", "amount": 60000, "cvv_result": "N", "avs_result": "N",
        "is_guest": True, "prior_disputes": 3, "txns_last_24h": 7,
    }).json()
    safe = client.post("/score", json={
        "order_id": "ord_safe", "amount": 900, "cvv_result": "M", "avs_result": "Y",
        "three_ds_status": "authenticated", "account_age_days": 900,
    }).json()

    assert risky["score"] > safe["score"]
    assert len(risky["capture"]) >= len(safe["capture"])


def test_three_ds_lowers_the_score():
    base = ScoreRequest(order_id="a", amount=5000, cvv_result="N")
    with_3ds = ScoreRequest(order_id="a", amount=5000, cvv_result="N", three_ds_status="authenticated")
    assert heuristic_score(with_3ds)[0] < heuristic_score(base)[0]


# ------------------------------------------------------------- webhooks

def _sign(payload: bytes, secret: str, ts: int | None = None) -> str:
    ts = ts or int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def test_stripe_signature_verification():
    secret = "whsec_test"
    payload = json.dumps({"hello": "world"}).encode()
    assert verify_stripe_signature(payload, _sign(payload, secret), secret)
    assert not verify_stripe_signature(payload, _sign(payload, "wrong"), secret)
    assert not verify_stripe_signature(b'{"tampered":1}', _sign(payload, secret), secret)
    # replay outside tolerance
    assert not verify_stripe_signature(payload, _sign(payload, secret, int(time.time()) - 9999), secret)


def test_webhook_refuses_live_mode_events():
    event = {
        "type": "charge.dispute.created",
        "data": {"object": {"id": "du_live", "livemode": True, "reason": "fraudulent", "amount": 1000}},
    }
    r = client.post("/webhooks/stripe", json=event)
    assert r.status_code == 400


def test_webhook_ignores_unrelated_events():
    """With no webhook secret configured, unsigned requests are accepted (dev)."""
    r = client.post("/webhooks/stripe", json={"type": "customer.created", "data": {"object": {}}})
    assert r.status_code == 200
    assert r.json()["handled"] is False


def test_unsigned_request_rejected_once_a_secret_is_configured(webhook_secret):
    """The moment STRIPE_WEBHOOK_SECRET is set, verification is mandatory.

    This is the behaviour that made the suite fail when a real secret landed
    in .env — now it is asserted explicitly instead of depending on the
    developer's local environment.
    """
    event = {"type": "customer.created", "data": {"object": {}}}
    assert client.post("/webhooks/stripe", json=event).status_code == 400

    payload = json.dumps(event).encode()
    signed = client.post(
        "/webhooks/stripe",
        content=payload,
        headers={
            "Content-Type": "application/json",
            "Stripe-Signature": _sign(payload, webhook_secret),
        },
    )
    assert signed.status_code == 200


def test_dispute_with_no_local_order_is_still_accepted():
    """Regression: disputes.order_id was NOT NULL, so `stripe trigger` — which
    sends no metadata.order_id — 500'd on the not-null constraint and the
    dispute was lost. The processor's event is authoritative; an unlinked row
    beats no row."""
    event = {
        "type": "charge.dispute.created",
        "data": {
            "object": {
                "id": "du_no_order",
                "livemode": False,
                "amount": 250000,
                "currency": "inr",
                "reason": "fraudulent",
                "status": "needs_response",
                "evidence_details": {"due_by": int(time.time()) + 86400 * 10},
                "metadata": {},
            }
        },
    }
    r = client.post("/webhooks/stripe", json=event)
    assert r.status_code == 200
    assert r.json()["handled"] is True


def test_schema_declares_dispute_order_id_nullable():
    """Cheap guard on the column that caused the loss."""
    from app.db import SCHEMA_PATH

    sql = SCHEMA_PATH.read_text()
    disputes_block = sql.split("create table if not exists disputes (")[1].split(");")[0]
    order_line = next(
        line for line in disputes_block.splitlines() if "order_id" in line
    )
    assert "not null" not in order_line.lower()
    assert "alter column order_id drop not null" in sql


def test_settings_do_not_read_dotenv_under_pytest():
    """Guards the isolation itself: a developer's .env must not reach tests."""
    from app.config import Settings

    assert Settings.model_config.get("env_file") is None


def test_inbox_can_be_narrowed_to_real_disputes():
    """A dispute that has just arrived has no expected value, so it sorts to
    the bottom of a list the demo data fills several times over. Without a
    filter the one dispute you can actually file is the hardest to find."""
    with TestClient(app) as client:
        assert client.get("/disputes?origin=processor").status_code in (200, 503)
        assert client.get("/disputes?origin=nonsense").status_code == 422


def test_the_selftests_own_event_is_not_recorded_as_real():
    """`make doctor` posts a correctly signed event to prove the path works.
    It is a real webhook and a real row, but nothing exists at Stripe behind
    it — so it must not be marked filable, or the diagnostic leaves a trap."""
    from app.routers.webhooks import SELFTEST_MARKER, origin_of

    def event(**metadata) -> dict:
        return {"id": "evt_1", "type": "charge.dispute.created",
                "data": {"object": {"id": "du_1", "metadata": metadata}}}

    assert origin_of(event()) == "processor"
    assert origin_of(event(**{SELFTEST_MARKER: "true"})) == "synthetic"
    assert origin_of({}) == "processor"          # a malformed event is not ours


async def _resolve(obj: dict):
    from app.routers.webhooks import resolve_order_id
    return await resolve_order_id(obj)


@pytest.mark.asyncio
async def test_explicit_metadata_wins():
    assert await _resolve({"metadata": {"order_id": "ord_7"}, "charge": "ch_1"}) == "ord_7"


@pytest.mark.asyncio
async def test_a_dispute_with_nothing_to_match_resolves_to_nothing():
    """Recorded unlinked rather than guessed at. A dispute attached to the
    wrong order would draft a packet citing another customer's delivery."""
    assert await _resolve({"id": "du_1"}) is None
    assert await _resolve({"metadata": {}, "charge": "ch_unknown"}) is None


@pytest.mark.asyncio
async def test_the_order_id_is_read_off_the_charge_when_the_dispute_lacks_it(monkeypatch):
    """A dispute never carries the metadata set when the payment was created —
    the charge does. One lookup at dispute time recovers the link."""
    import httpx

    from app import config
    from app.routers.webhooks import _ask_stripe_for_order

    monkeypatch.setattr(config.settings, "stripe_secret_key", "sk_test_x")

    def handler(request: httpx.Request) -> httpx.Response:
        if "/charges/ch_1" in str(request.url):
            return httpx.Response(200, json={"id": "ch_1", "metadata": {},
                                             "payment_intent": "pi_1"})
        if "/payment_intents/pi_1" in str(request.url):
            return httpx.Response(200, json={"id": "pi_1",
                                             "metadata": {"order_id": "ord_9"}})
        return httpx.Response(404, json={"error": {"message": "no"}})

    found = await _ask_stripe_for_order(
        {"charge": "ch_1"}, transport=httpx.MockTransport(handler))
    assert found == "ord_9"


@pytest.mark.asyncio
async def test_a_failed_lookup_never_fails_the_webhook(monkeypatch):
    """Losing the link costs evidence. Letting the call fail the webhook would
    lose the dispute itself, and Stripe stops retrying long before anyone
    notices."""
    import httpx

    from app import config
    from app.routers.webhooks import _ask_stripe_for_order

    monkeypatch.setattr(config.settings, "stripe_secret_key", "sk_test_x")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    assert await _ask_stripe_for_order(
        {"charge": "ch_1"}, transport=httpx.MockTransport(handler)) is None


@pytest.mark.asyncio
async def test_no_credentials_means_no_lookup(monkeypatch):
    from app import config
    from app.routers.webhooks import _ask_stripe_for_order

    monkeypatch.setattr(config.settings, "stripe_secret_key", "")
    assert await _ask_stripe_for_order({"charge": "ch_1"}) is None


def test_a_request_is_never_given_a_deadline_the_api_will_reject():
    """Found on a live run. A 504 burned 19.5s of a 25s budget, the remaining
    3.5s became a 4-second deadline, and Gemini answered:

        400 INVALID_ARGUMENT: Manually set deadline 4s is too short.
                              Minimum allowed deadline is 10s.

    The rejection is instant, so it walked all three models in under a second
    and looked exactly like every model being unavailable."""
    from app.agent.provider import (
        CALL_TIMEOUT_SECONDS, MIN_CALL_SECONDS, _deadline_ms)

    for remaining in (0.1, 1.0, 3.5, 9.9):
        assert _deadline_ms(remaining) >= MIN_CALL_SECONDS * 1000
    assert _deadline_ms(12.0) == 12_000
    assert _deadline_ms(999.0) == CALL_TIMEOUT_SECONDS * 1000


def test_a_gateway_timeout_moves_on_rather_than_waiting():
    """Being timed out is not being rate limited. A model that spent the whole
    per-call budget and returned nothing will spend the rest the same way;
    there is nothing to wait out, so the chain advances instead."""
    from app.agent.provider import _is_permanent, _is_timeout, _is_transient

    timeout = Exception("HTTP/1.1 504 Gateway Timeout")
    busy = Exception("429 RESOURCE_EXHAUSTED: high demand")
    gone = Exception("404 NOT_FOUND: no longer available to new users")

    assert _is_timeout(timeout) and not _is_transient(timeout)
    assert _is_transient(busy) and not _is_timeout(busy)
    assert _is_permanent(gone)


def test_the_startup_line_reports_the_provider_actually_in_use():
    """It read `anthropic_api_key` and printed agent=no while Gemini was
    configured and working."""
    from app.config import Settings

    assert Settings(gemini_api_key="x").agent_provider == "gemini"
    assert Settings(anthropic_api_key="x").agent_provider == "anthropic"
    assert Settings().agent_provider == "template"


def test_the_console_is_served_by_the_api_that_feeds_it():
    """One file, no build step. A reviewer clones this, runs `make api`, and
    has the screen — no npm install standing between them and the demo."""
    with TestClient(app) as client:
        r = client.get("/console")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "chargeback" in body
    # It must talk to this API, not a hard-coded host from someone's laptop.
    assert "localhost:8000" not in body.split("<script>")[-1]
    assert "/disputes?limit=" in body


def test_search_covers_the_fields_a_person_would_have_in_front_of_them():
    """A dispute id from a log line, a du_… from Stripe's dashboard, a
    customer's email, an amount read off this very screen. A search that
    silently misses one of those is worse than none: you conclude the record
    is not there."""
    from app.routers.disputes import SEARCHABLE

    for column in ("d.id", "d.processor_dispute_id", "d.order_id",
                   "d.category", "d.amount::text", "c.email",
                   "o.processor_payment_id", "d.network_code"):
        assert column in SEARCHABLE, f"{column} is not searchable"

    # Written only by the batch pass, so null on a database nobody has run
    # `make triage` against — searching it would find nothing while the screen
    # shows a column full of recommendations.
    assert "d.recommendation" not in SEARCHABLE


def test_the_inbox_answers_rather_than_crashing_without_a_database(monkeypatch):
    """`has_db` says a URL is configured, not that it connected. This endpoint
    read the setting and let the pool raise, returning a 500 with a traceback
    where every other endpoint answered cleanly."""
    with TestClient(app) as client:
        r = client.get("/disputes?q=anything")
    assert r.status_code == 200
    assert r.json()["disputes"] == []


def test_the_inbox_reports_how_many_it_is_not_showing():
    """A worklist that shows the top sixty of ten thousand and says nothing
    looks like a database with sixty disputes in it. The count of what matched
    is not the count of what fits on the page."""
    with TestClient(app) as client:
        body = client.get("/disputes?limit=2").json()
    assert "total" in body and "count" in body and "limit" in body
    assert body["count"] <= body["limit"]
