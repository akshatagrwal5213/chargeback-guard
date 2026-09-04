"""The citation guard.

The project's central claim is that it cannot assert what it cannot show.
These tests are that claim, written down.
"""
from __future__ import annotations

from app.agent import guard
from app.agent.compose import Claim, Draft
from app.agent.records import Evidence, Record, ToolResult


def _evidence() -> Evidence:
    ev = Evidence()
    ev.add(ToolResult("get_fulfillment", [
        Record(ref="fulfillment_events:881", kind="delivery_scan",
               summary="delivered at 2026-07-16 14:39:36, New Delhi 110001",
               fields={"event_type": "delivered",
                       "occurred_at": "2026-07-16 14:39:36",
                       "location": "New Delhi 110001",
                       "tracking_number": "IP1234567890IN",
                       "signature_name": "A. Sharma"}),
    ]))
    ev.add(ToolResult("get_order", [
        Record(ref="orders:ord_1", kind="order",
               summary="INR 26,530.48 on 2026-07-15",
               fields={"amount": 26530.48, "placed_at": "2026-07-15"}),
    ]))
    ev.add(ToolResult("get_refunds", note="No refund was ever requested."))
    return ev


def _draft(*claims: Claim) -> Draft:
    return Draft(summary="s", claims=list(claims), provider="test")


# ------------------------------------------------------------- sourcing

def test_a_claim_with_no_source_is_deleted():
    v = guard.verify(_draft(Claim("The customer was clearly acting in bad faith.")), _evidence())
    assert v.claims == []
    assert v.rejected[0].reason == "no source cited"
    assert not v.submittable


def test_a_claim_citing_a_record_that_was_never_retrieved_is_deleted():
    """The hallucination case: a plausible ref pointing at nothing we fetched."""
    v = guard.verify(
        _draft(Claim("A signature was obtained.", refs=["fulfillment_events:99999"])),
        _evidence(),
    )
    assert v.claims == []
    assert "was not retrieved" in v.rejected[0].reason


def test_a_properly_sourced_claim_survives():
    v = guard.verify(
        _draft(Claim("The carrier recorded delivery at New Delhi 110001.",
                     refs=["fulfillment_events:881"])),
        _evidence(),
    )
    assert len(v.claims) == 1
    assert v.submittable
    assert v.clean


def test_one_bad_claim_blocks_submission_even_beside_good_ones():
    """A packet is not partially trustworthy. If anything was unsourced, the
    packet does not go out without a human looking at it."""
    v = guard.verify(
        _draft(
            Claim("Delivered to New Delhi 110001.", refs=["fulfillment_events:881"]),
            Claim("The customer has a history of fraud.", refs=[]),
        ),
        _evidence(),
    )
    assert len(v.claims) == 1
    assert len(v.rejected) == 1
    assert not v.submittable


# ------------------------------------------------------------ grounding

def test_a_wrong_date_on_a_real_record_is_flagged():
    """Citing the right row and misreading it. The ref checks out, so sourcing
    passes — grounding is what catches this."""
    v = guard.verify(
        _draft(Claim("The parcel was delivered on 2026-08-02.",
                     refs=["fulfillment_events:881"])),
        _evidence(),
    )
    assert len(v.claims) == 1, "a flag warns, it does not delete"
    assert v.flagged and v.flagged[0].token == "2026-08-02"
    assert not v.clean


def test_the_correct_date_is_not_flagged():
    v = guard.verify(
        _draft(Claim("Delivered on 2026-07-16 to the billing address.",
                     refs=["fulfillment_events:881"])),
        _evidence(),
    )
    assert not v.flagged


def test_a_written_out_date_matches_the_iso_value_in_the_record():
    """'16 July 2026' and '2026-07-16' are the same fact. Flagging the prose
    form would train everyone to ignore the warnings."""
    v = guard.verify(
        _draft(Claim("The parcel was delivered on 16 July 2026.",
                     refs=["fulfillment_events:881"])),
        _evidence(),
    )
    assert not v.flagged, [f.as_dict() for f in v.flagged]


def test_an_invented_tracking_number_is_flagged():
    v = guard.verify(
        _draft(Claim("Tracking XX9999999999IN confirms delivery.",
                     refs=["fulfillment_events:881"])),
        _evidence(),
    )
    assert v.flagged and "XX9999999999IN" in v.flagged[0].token


def test_the_real_amount_is_not_flagged():
    v = guard.verify(
        _draft(Claim("The order totalled 26,530.48.", refs=["orders:ord_1"])),
        _evidence(),
    )
    assert not v.flagged


# -------------------------------------------------------------- feedback

def test_feedback_names_what_failed_and_why():
    """The repair round is only useful if the model is told specifically."""
    v = guard.verify(
        _draft(
            Claim("Bad faith.", refs=[]),
            Claim("Delivered on 2026-08-02.", refs=["fulfillment_events:881"]),
        ),
        _evidence(),
    )
    text = v.feedback()
    assert "Bad faith" in text
    assert "2026-08-02" in text
    assert "exact values" in text


def test_apply_returns_the_draft_without_the_rejected_claims():
    draft = _draft(
        Claim("Delivered.", refs=["fulfillment_events:881"]),
        Claim("Invented.", refs=[]),
    )
    final = guard.apply(draft, guard.verify(draft, _evidence()))
    assert len(final.claims) == 1
    assert "Invented" not in final.narrative()


# ------------------------------------------------------ event loop safety

def test_compose_async_does_not_block_the_event_loop():
    """Regression: POST /packet hung for tens of seconds with nothing served.

    Both vendor SDKs are synchronous and so is the retry backoff, so calling
    them from an async endpoint stalls the entire server — every other request
    included — and from outside it is indistinguishable from a crash.
    """
    import asyncio
    import time as _time

    from app.agent.compose import compose_async
    from app.agent.provider import Provider

    class SlowProvider(Provider):
        name = "slow"

        def complete(self, system: str, user: str) -> str:
            _time.sleep(0.4)                       # blocking, like a real SDK
            return '{"summary":"s","claims":[],"gaps_acknowledged":[]}'

    async def main():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await compose_async({"id": "d", "category": "fraudulent", "amount": 100.0},
                            _evidence(), SlowProvider())
        beat.cancel()
        return ticks

    ticks = asyncio.run(main())
    assert ticks >= 4, (
        f"the loop only ticked {ticks} times during a 0.4s provider call — "
        "the call is running on the event loop instead of a worker thread"
    )


def test_provider_has_a_total_time_budget():
    """Three models x retries x escalating backoff is over a minute of an
    endpoint doing nothing visible. Falling back to templates and saying so
    beats an unbounded wait.

    The budget also has a floor, which the first version did not: it must fit
    two full-length attempts, or one gateway timeout turns every packet into a
    template. That is what happened on a live run — a 504 spent 19.5s of 25,
    and there was no room left to try anything else properly."""
    from app.agent import provider

    assert provider.TOTAL_BUDGET_SECONDS <= 60           # still bounded
    assert provider.TOTAL_BUDGET_SECONDS >= 2 * provider.CALL_TIMEOUT_SECONDS
    assert provider.MIN_CALL_SECONDS >= 10               # the API's own floor
    assert provider.MAX_RETRIES * provider.BACKOFF_SECONDS * len(
        provider.GEMINI_MODELS) > 0


def test_tool_commentary_never_reaches_the_issuer():
    """`gaps_acknowledged` was built from tool notes, so two internal remarks
    were being submitted under "The merchant notes:" — how comparable disputes
    had gone (precedent by another name), and that a product "was genuinely
    never used", which is an admission against interest on a physical order."""
    from app.agent.compose import compose_from_template
    from app.agent.records import Record, ToolResult
    from app.agent.records import Evidence

    evidence = Evidence()
    evidence.add(ToolResult("get_order", [
        Record(ref="orders:ord_1", kind="order", summary="INR 100 on 2026-08-04",
               fields={"is_digital": False})]))
    evidence.add(ToolResult("get_fulfillment", [
        Record(ref="fulfillment_events:1", kind="delivery_scan",
               summary="delivered at 2026-08-07, Bengaluru", fields={})]))
    evidence.add(ToolResult(
        "find_similar_disputes", [
            Record(ref="dispute_outcomes:dsp_x", kind="precedent",
                   summary="a comparable dispute was lost", fields={})],
        note="4 of the 5 most recent settled disputes were won."))
    evidence.add(ToolResult(
        "get_access_log", [],
        note="No recorded access — either not a digital product, or it was "
             "genuinely never used."))

    draft = compose_from_template(
        {"id": "dsp_1", "category": "product_not_received",
         "amount": 100.0, "currency": "INR"}, evidence)
    text = draft.narrative().lower()

    assert "most recent settled" not in text
    assert "genuinely never used" not in text
    assert "comparable dispute" not in text
    assert "delivered at 2026-08-07" in text


def _evidence_with_a_delivery():
    from app.agent.records import Evidence, Record, ToolResult

    evidence = Evidence()
    evidence.add(ToolResult("get_order", [Record(
        ref="orders:ord_1", kind="order",
        summary="INR 33,636.16 on 2026-07-23, physical (electronics)",
        fields={"amount": 33636.16, "is_digital": False})]))
    evidence.add(ToolResult("get_fulfillment", [Record(
        ref="fulfillment_events:9", kind="delivery_scan",
        summary="delivered at 2026-07-26 20:28:46, Mumbai 400001",
        fields={"tracking_number": "EK2222882928IN"})]))
    return evidence


DISPUTE = {"id": "dsp_1", "category": "product_not_received",
           "amount": 33636.16, "currency": "INR",
           "opened_at": "2026-08-29T00:00:00Z"}


def test_the_summary_is_checked_too():
    """The summary leads the packet, is submitted with it, and cites nothing —
    so it was passing through unchecked while every claim beneath it was
    verified. A guarantee with a hole in the first paragraph is not one."""
    from app.agent.compose import Claim, Draft
    from app.agent.guard import verify

    draft = Draft(
        summary="A full refund of INR 99,999.00 was issued on 2026-01-01.",
        claims=[Claim(text="The order was delivered on 2026-07-26.",
                      refs=["fulfillment_events:9"])],
        provider="gemini",
    )
    verdict = verify(draft, _evidence_with_a_delivery(), DISPUTE)
    assert verdict.summary_ungrounded
    assert not verdict.rejected              # the claim itself was fine


def test_an_ungrounded_summary_is_replaced_not_trimmed():
    from app.agent.compose import Claim, Draft, deterministic_summary
    from app.agent.guard import apply, verify

    draft = Draft(
        summary="A full refund of INR 99,999.00 was issued on 2026-01-01.",
        claims=[Claim(text="The order was delivered on 2026-07-26.",
                      refs=["fulfillment_events:9"])],
        provider="gemini",
    )
    verdict = verify(draft, _evidence_with_a_delivery(), DISPUTE)
    final = apply(draft, verdict,
                  fallback_summary=deterministic_summary(DISPUTE, 1))

    assert "99,999.00" not in final.narrative()
    assert "33,636.16" in final.summary


def test_a_grounded_summary_survives():
    """The model's own framing is better than the template's when it holds up.
    The dispute header counts as a record: its amount and opening date are
    facts we hold, not inventions."""
    from app.agent.compose import Claim, Draft
    from app.agent.guard import verify

    draft = Draft(
        summary=("The merchandise was delivered on 2026-07-26, and this "
                 "dispute for INR 33,636.16 was opened on 2026-08-29."),
        claims=[Claim(text="The order was delivered on 2026-07-26.",
                      refs=["fulfillment_events:9"])],
        provider="gemini",
    )
    verdict = verify(draft, _evidence_with_a_delivery(), DISPUTE)
    assert not verdict.summary_ungrounded


def test_a_gap_asserting_an_unsupported_figure_is_dropped():
    from app.agent.compose import Claim, Draft
    from app.agent.guard import apply, verify

    draft = Draft(
        summary="The merchant contests this dispute.",
        claims=[Claim(text="The order was delivered on 2026-07-26.",
                      refs=["fulfillment_events:9"])],
        gaps_acknowledged=["No signature was captured.",
                           "Only INR 12,345.00 of the order shipped."],
        provider="gemini",
    )
    verdict = verify(draft, _evidence_with_a_delivery(), DISPUTE)
    final = apply(draft, verdict)
    assert final.gaps_acknowledged == ["No signature was captured."]
