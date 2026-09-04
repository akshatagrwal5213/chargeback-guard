"""Triage rule behaviour.

Pure logic — no database. The rule takes a dispute dict and an Availability,
so every case here is constructed directly.
"""
from __future__ import annotations

import pytest

from app.evidence.schema import Category
from app.triage.evidence import Availability, completeness, critical_slot_present
from app.triage.rule import break_even, decide, sensitivity


def _dispute(**kw) -> dict:
    base = {"id": "dsp_1", "category": "product_not_received",
            "amount": 12000.0, "phase": "chargeback"}
    base.update(kw)
    return base


def _delivered(**kw) -> Availability:
    base = dict(
        order_id="ord_1", delivered=True, delivered_at="2026-07-01 10:00:00",
        delivery_location="Bengaluru 560001", tracking_number="BD123456789IN",
        carrier="Bluedart", shipped_at="2026-06-29 09:00:00",
    )
    base.update(kw)
    return Availability(**base)


def _nothing() -> Availability:
    return Availability(order_id="ord_1")


# ------------------------------------------------------------- economics

def test_break_even_falls_as_the_amount_rises():
    """A larger recovery justifies a longer shot."""
    small = break_even(2_000, "chargeback", effort=250, escalation_base=900)
    large = break_even(80_000, "chargeback", effort=250, escalation_base=900)
    assert small > large
    assert 0.0 < large < small < 1.0


def test_break_even_rises_with_escalation_phase():
    """Contesting at pre-arbitration costs more, so it needs better odds."""
    early = break_even(20_000, "retrieval", 250, 900)
    normal = break_even(20_000, "chargeback", 250, 900)
    late = break_even(20_000, "pre_arbitration", 250, 900)
    assert early < normal < late


# -------------------------------------------------------------- decisions

def test_full_evidence_with_signature_is_contested():
    d = decide(_dispute(), _delivered(signature_name="A. Sharma"))
    assert d.recommendation == "contest"
    assert d.critical_evidence
    assert d.completeness == 1.0
    assert d.win_probability > d.break_even_probability
    assert d.expected_value > 0
    assert any("signature" in r for r in d.reasons)


def test_no_evidence_is_accepted():
    d = decide(_dispute(), _nothing())
    assert d.recommendation == "accept"
    assert not d.critical_evidence
    assert d.contest_blocked


def test_missing_decisive_evidence_blocks_contest_even_when_profitable():
    """The guardrail from docs/DEFENSE_ONLY.md, asserted rather than promised.

    At a large enough amount the arithmetic turns positive even with an empty
    checklist — a system filing a claim it cannot support. The constraint is
    ethical, not economic, so it must outrank expected value.
    """
    d = decide(_dispute(amount=250_000.0), _nothing())
    assert d.expected_value > 0, "fixture no longer exercises the override"
    assert d.recommendation == "accept"
    assert d.contest_blocked
    assert "Refused on evidence, not economics" in (d.flips_if or "")


def test_partial_evidence_without_the_decisive_field_still_blocks():
    """Completeness alone misleads: tracking, carrier and address without a
    delivery scan scores well and proves nothing."""
    av = _delivered(delivered=False, delivered_at=None, delivery_location=None)
    score, _, _ = completeness(av, Category.PRODUCT_NOT_RECEIVED)
    assert score > 0.3, "fixture should have partial evidence"
    assert not critical_slot_present(av, Category.PRODUCT_NOT_RECEIVED)
    assert decide(_dispute(), av).recommendation == "accept"


def test_digital_goods_use_access_logs_instead_of_delivery():
    """A digital order can never produce a delivery scan. If the checklist
    demanded one, every digital dispute would be permanently indefensible."""
    av = Availability(order_id="ord_1", is_digital=True, access_events=9,
                      last_access_at="2026-07-04 22:10:00")
    assert critical_slot_present(av, Category.PRODUCT_NOT_RECEIVED)
    assert decide(_dispute(), av).recommendation == "contest"


def test_three_ds_lifts_a_fraud_dispute():
    plain = Availability(order_id="ord_1", avs_result="Y", cvv_result="M",
                         delivered=True, billing_country="IN")
    authed = Availability(order_id="ord_1", avs_result="Y", cvv_result="M",
                          three_ds_status="authenticated", delivered=True,
                          billing_country="IN")
    d1 = decide(_dispute(category="fraudulent"), plain)
    d2 = decide(_dispute(category="fraudulent"), authed)
    assert d2.win_probability > d1.win_probability
    assert any("3DS" in r for r in d2.reasons)


def test_probability_never_claims_certainty():
    """It is a stated prior. Anything near 0 or 1 would be overclaiming."""
    strong = decide(_dispute(), _delivered(signature_name="A. Sharma"))
    weak = decide(_dispute(), _nothing())
    assert 0.02 < weak.win_probability
    assert strong.win_probability <= 0.90


def test_every_decision_carries_its_reasoning():
    for category in Category:
        d = decide(_dispute(category=category.value), _delivered())
        assert d.reasons, f"{category} produced no explanation"
        assert d.rule_version
        if d.recommendation == "accept":
            assert d.flips_if, f"{category} accepted without saying what would change it"


# ------------------------------------------------------------ sensitivity

def test_sensitivity_reports_whether_the_answer_survives_being_wrong():
    result = sensitivity(_dispute(), _delivered(signature_name="A. Sharma"))
    assert len(result["points"]) == 5
    assert result["stable"] is True
    assert {p["recommendation"] for p in result["points"]} == {"contest"}


def test_sensitivity_flags_a_marginal_case():
    """Find an amount where the decision sits close to break-even, and check
    the analysis says so rather than presenting it as settled."""
    av = _delivered()
    for amount in range(600, 6000, 100):
        result = sensitivity(_dispute(amount=float(amount)), av, spread=0.25)
        if not result["stable"]:
            assert "marginal" in result["note"].lower()
            return
    pytest.skip("no marginal amount found in the scanned range")


def test_escalation_phase_changes_the_recommendation():
    """Razorpay's ladder is not decoration: the same dispute is worth fighting
    at chargeback and not at arbitration."""
    av = _delivered()
    early = decide(_dispute(amount=1500.0, phase="chargeback"), av)
    late = decide(_dispute(amount=1500.0, phase="arbitration"), av)
    assert late.break_even_probability > early.break_even_probability
    assert late.expected_value < early.expected_value


def test_refusal_message_never_contradicts_the_checklist():
    """The explanation must not claim a record is absent when the checklist
    beside it shows a value.

    Found in a live run: a fraud dispute with AVS Y on file was refused with
    'an AVS/CVV match is not on file'. Both lines were rendered on the same
    screen. A contradiction like that discredits every other line of
    reasoning, which for this project is the product.
    """
    from app.evidence.schema import Strength
    from app.triage.evidence import critical_slot

    # AVS matched, no 3DS — thin, not absent.
    av = Availability(order_id="ord_1", avs_result="Y", cvv_result="M",
                      delivered=True, billing_country="SG", evidence_tier="enhanced")
    slot, strength, value = critical_slot(av, Category.FRAUDULENT)
    assert strength is Strength.WEAK and value, "fixture should be thin, not empty"

    d = decide(_dispute(category="fraudulent", amount=32853.0), av)
    assert d.recommendation == "accept"
    assert "is not on file" not in (d.flips_if or ""), d.flips_if
    assert "too thin to carry" in (d.flips_if or "")
    assert "AVS Y" in (d.flips_if or ""), "should quote what IS on file"


def test_refusal_message_says_missing_when_truly_missing():
    d = decide(_dispute(), _nothing())
    assert "is not on file" in (d.flips_if or "")
    assert "too thin" not in (d.flips_if or "")


def test_an_open_refund_request_blocks_contesting():
    """Found by filing. A live dispute scored P(win) 0.90 on a complete
    checklist while the refund ledger showed a full refund requested six days
    earlier and never issued. Contesting there asks the issuer to decide in
    favour of a merchant that has not decided itself."""
    from app.triage.evidence import Availability
    from app.triage.rule import decide

    av = Availability(
        order_id="ord_1", delivered=True, delivered_at="2026-08-07",
        delivery_location="Bengaluru 560001", tracking_number="EK8627922695IN",
        carrier="XpressBees", shipped_at="2026-08-04",
        communications=2, policies={"refund", "terms"},
    )
    dispute = {"id": "dsp_1", "category": "product_not_received",
               "amount": 41652.34, "phase": "chargeback"}

    assert decide(dispute, av).recommendation == "contest"

    av.refund_open = True
    blocked = decide(dispute, av)
    assert blocked.recommendation == "accept"
    assert "refund" in (blocked.flips_if or "").lower()


def test_a_declined_refund_does_not_block_contesting():
    """Declining a refund is a position a merchant can defend — it is what
    Stripe's refund_refusal_explanation field exists for. Only an unanswered
    request is disqualifying."""
    from app.triage.evidence import Availability
    from app.triage.rule import decide

    av = Availability(
        order_id="ord_1", delivered=True, delivered_at="2026-08-07",
        delivery_location="Bengaluru", tracking_number="EK1", carrier="XB",
        shipped_at="2026-08-04", communications=2, policies={"refund"},
        refund_requested=True, refund_open=False,
    )
    dispute = {"id": "dsp_1", "category": "product_not_received",
               "amount": 41652.34, "phase": "chargeback"}
    assert decide(dispute, av).recommendation == "contest"
