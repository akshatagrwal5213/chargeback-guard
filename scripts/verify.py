#!/usr/bin/env python
"""Check that the things this project claims are true of this installation.

`make test` proves the logic in isolation. This proves the behaviour against
your database, your key and your running API — which is where every defect
this project has actually had was found.

Each check states the claim it is testing, then passes or fails on evidence.
A check that cannot run says so rather than passing quietly: a green run with
half the checks skipped would be worse than a red one.

    make verify

The API must be running (`make api`). Nothing is sent to any processor: the
submission check uses dry-run mode, which makes no network call at all.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

# The guard logs a warning for every claim it removes. That is right in the
# API and wrong here: this script deliberately feeds it a hostile draft, and
# its own report is the output that matters.
logging.disable(logging.WARNING)

API = "http://localhost:8000"

OK, BAD, SKIP, INFO = "[  ok  ]", "[ FAIL ]", "[ skip ]", "[ .... ]"

# Text that must never appear in anything sent outward. Each of these was in a
# real staged submission before it was caught.
FORBIDDEN = {
    "most recent settled": "the merchant's own win rate",
    "disputes were won": "the merchant's own win rate",
    "was lost (": "a comparable dispute the merchant lost",
    "never disputed": "the cardholder's prior-dispute count",
    "genuinely never used": "an admission that the product was never used",
}

# Stripe evidence fields that take a file upload id, not text.
FILE_FIELDS = {
    "cancellation_policy", "customer_communication", "customer_signature",
    "duplicate_charge_documentation", "receipt", "refund_policy",
    "service_documentation", "shipping_documentation", "uncategorized_file",
}

results: list[tuple[str, str]] = []


def record(mark: str, claim: str, detail: str = "") -> None:
    results.append((mark, claim))
    print(f"{mark} {claim}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def call(path: str, method: str = "GET", timeout: int = 90) -> tuple[int, dict]:
    req = urllib.request.Request(f"{API}{path}", data=b"" if method == "POST" else None,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


# --------------------------------------------------------------- the checks

def check_api() -> dict | None:
    status, health = call("/health")
    if status == 0:
        record(BAD, "the API is reachable", f"{health.get('error')}\nStart it with: make api")
        return None
    components = health.get("components", {})
    if components.get("database") != "ok":
        record(BAD, "the database is connected",
               f"reported '{components.get('database')}' — check DATABASE_URL")
        return None
    record(OK, "the API is up and the database is connected")
    return components


def check_model(components: dict) -> None:
    """Claim: a model writes the narrative, and a missing one is not fatal."""
    provider_name = components.get("agent")
    if provider_name == "template":
        record(SKIP, "a language model is configured",
               "No GEMINI_API_KEY or ANTHROPIC_API_KEY in .env. Packets still\n"
               "compose from templates and every guarantee below still holds —\n"
               "which is the point — but the model path is untested here.")
        return

    from app.agent.provider import get_provider

    provider = get_provider()
    started = time.monotonic()
    try:
        answer = provider.complete(
            "Reply with JSON only.",
            'Return exactly {"ok": true} and nothing else.')
    except Exception as exc:
        record(BAD, f"the configured model ({provider_name}) answers",
               f"{exc}\nThe packet path falls back to templates, so this is not\n"
               f"fatal — but the demo will say provider: template.")
        return
    elapsed = time.monotonic() - started
    if "ok" not in answer.lower():
        record(BAD, f"the configured model ({provider_name}) answers usefully",
               f"replied: {answer[:120]}")
        return
    record(OK, f"the configured model ({provider_name}) answers, in {elapsed:.1f}s")
    if elapsed > 25:
        record(INFO, "", "Slower than the 25s budget — packets will fall back.")


def check_guard_deletes_and_replaces() -> None:
    """Claim: an unsourced assertion cannot survive, wherever it appears."""
    from app.agent.compose import Claim, Draft, deterministic_summary
    from app.agent.guard import apply, verify
    from app.agent.records import Evidence, Record, ToolResult

    evidence = Evidence()
    evidence.add(ToolResult("get_fulfillment", [Record(
        ref="fulfillment_events:1", kind="delivery_scan",
        summary="delivered at 2026-07-26 20:28:46, Mumbai 400001",
        fields={"tracking_number": "EK2222882928IN"})]))
    dispute = {"id": "dsp_x", "category": "product_not_received",
               "amount": 33636.16, "currency": "INR"}

    hostile = Draft(
        summary="A refund of INR 99,999.00 was issued on 2025-01-01.",
        claims=[
            Claim(text="The parcel was delivered on 2026-07-26.",
                  refs=["fulfillment_events:1"]),                 # valid
            Claim(text="The customer disputed 14 prior orders.",
                  refs=["customers:never_retrieved"]),            # bad ref
            Claim(text="The customer admitted receipt by phone.", refs=[]),
            Claim(text="Tracking number ZZ0000000000IN was signed for.",
                  refs=["fulfillment_events:1"]),                 # invented value
        ],
        gaps_acknowledged=["No signature was captured.",
                           "Only INR 12,345.00 of the order shipped."],
        provider="test",
    )
    verdict = verify(hostile, evidence, dispute)
    final = apply(hostile, verdict,
                  fallback_summary=deterministic_summary(dispute, 1))
    text = final.narrative()

    problems = []
    if len(verdict.rejected) != 2:
        problems.append(f"expected 2 deletions, got {len(verdict.rejected)}")
    if not verdict.flagged:
        problems.append("the invented tracking number was not flagged")
    if not verdict.summary_ungrounded:
        problems.append("the invented summary was not caught")
    if "99,999" in text or "14 prior orders" in text or "12,345" in text:
        problems.append("an unsupported figure survived into the document")
    if verdict.submittable:
        problems.append("the packet is still marked submittable")

    if problems:
        record(BAD, "the guard removes what it cannot source", "\n".join(problems))
    else:
        record(OK, "the guard deletes unsourced claims, flags invented values, "
                   "replaces an ungrounded summary and drops an ungrounded gap")


def check_refund_rule() -> None:
    """Claim: an open refund request outranks a profitable contest."""
    from app.triage.evidence import Availability
    from app.triage.rule import decide

    def av(**kw):
        base = dict(order_id="ord_1", delivered=True, delivered_at="2026-07-26",
                    delivery_location="Mumbai 400001", tracking_number="EK1",
                    carrier="Delhivery", shipped_at="2026-07-23",
                    communications=2, policies={"refund", "terms"})
        base.update(kw)
        return Availability(**base)

    dispute = {"id": "dsp_1", "category": "product_not_received",
               "amount": 41652.34, "phase": "chargeback"}

    clean = decide(dispute, av())
    open_refund = decide(dispute, av(refund_open=True))
    declined = decide(dispute, av(refund_requested=True, refund_open=False))

    if clean.recommendation != "contest":
        record(BAD, "a defensible dispute is contested",
               f"got {clean.recommendation}")
    elif open_refund.recommendation != "accept":
        record(BAD, "an open refund request blocks contesting",
               f"got {open_refund.recommendation} on EV Rs.{open_refund.expected_value:,.0f}")
    elif declined.recommendation != "contest":
        record(BAD, "a declined refund does not block contesting",
               f"got {declined.recommendation}")
    else:
        record(OK, f"an open refund blocks a contest worth "
                   f"Rs.{open_refund.expected_value:,.0f}; a declined one does not")


def find_contestable() -> str | None:
    status, body = call("/disputes?limit=50")
    if status != 200:
        return None
    for row in body.get("disputes", []):
        code, detail = call(f"/disputes/{row['id']}/triage")
        if code == 200 and (detail.get("decision") or {}).get(
                "recommendation") == "contest":
            return row["id"]
    return None


def check_packet_and_submission() -> None:
    """Claim: every assertion is sourced, and nothing internal is sent."""
    dispute_id = find_contestable()
    if not dispute_id:
        record(SKIP, "a packet drafts, verifies and stages",
               "No dispute in the first 50 is recommended for contest.\n"
               "Run `make seed-disputes`, or `make live-dispute` for a real one.")
        return

    status, packet = call(f"/disputes/{dispute_id}/packet", method="POST")
    if status != 200 or packet.get("status") != "drafted":
        record(BAD, "a packet drafts", f"{status}: {str(packet)[:200]}")
        return

    guard = packet.get("guard") or {}
    citable = set((packet.get("evidence") or {}).get("citable_refs") or [])
    problems = []

    for claim in packet.get("claims", []):
        if not claim.get("refs"):
            problems.append(f"uncited claim: {claim['text'][:60]}")
        for ref in claim.get("refs", []):
            if ref not in citable:
                problems.append(f"claim cites {ref}, which was not retrieved")
    if guard.get("claims_rejected"):
        problems.append(f"{guard['claims_rejected']} claim(s) failed the guard")

    narrative = (packet.get("narrative") or "").lower()
    for phrase, what in FORBIDDEN.items():
        if phrase in narrative:
            problems.append(f"the narrative reveals {what}")

    if problems:
        record(BAD, "every claim in the packet is sourced and nothing internal "
                    "reaches the narrative", "\n".join(problems))
        return
    record(OK, f"packet {dispute_id}: {len(packet.get('claims', []))} claim(s), "
               f"all sourced, provider {packet.get('provider')}, "
               f"{packet['timings']['draft']:.1f}s")

    # What would actually be sent.
    status, result = call(f"/disputes/{dispute_id}/submit?mode=dry_run", method="POST")
    if status != 200:
        record(BAD, "the submission payload builds", f"{status}: {str(result)[:200]}")
        return

    payload = result.get("payload") or {}
    problems = []
    addressed = set(payload) & FILE_FIELDS
    if addressed:
        problems.append(f"text sent to file-upload field(s): {sorted(addressed)}")
    blob = " ".join(str(v) for v in payload.values()).lower()
    for phrase, what in FORBIDDEN.items():
        if phrase in blob:
            problems.append(f"the payload reveals {what}")
    if result.get("characters", 0) > 150_000:
        problems.append("over Stripe's 150,000-character limit")

    if problems:
        record(BAD, "the submission payload is fit to send", "\n".join(problems))
    else:
        record(OK, f"submission payload: {result.get('characters')} chars across "
                   f"{len(payload)} text field(s), no file field addressed")


def check_refusal_is_not_a_submission() -> None:
    """Claim: a dispute the system declines cannot be filed anyway."""
    status, body = call("/disputes?limit=50")
    if status != 200:
        record(SKIP, "a declined dispute cannot be filed", "no inbox")
        return
    # A declined dispute that *has* a packet is the case worth testing. One
    # without a packet is refused for having nothing to send, which is a
    # different rule and would let this check pass for the wrong reason —
    # as it did on the machine where the gate was written.
    rows = sorted(body.get("disputes", []),
                  key=lambda r: (r.get("packet_count") or 0), reverse=True)
    for row in rows:
        code, detail = call(f"/disputes/{row['id']}/triage")
        if code != 200 or (detail.get("decision") or {}).get(
                "recommendation") != "accept":
            continue
        has_packet = (row.get("packet_count") or 0) > 0
        code, result = call(f"/disputes/{row['id']}/submit?mode=dry_run",
                            method="POST")
        if code == 409 and has_packet:
            record(OK, "a dispute the rule declines is refused at submission, "
                       "even though a packet for it is already stored")
        elif code == 404 and not has_packet:
            record(SKIP, "a declined dispute with a stored packet is refused",
                   "The only declined disputes here have no packet, so this\n"
                   "check exercised the missing-packet path instead. Draft a\n"
                   "packet, then let a refund go unanswered, to test the gate.")
        elif code in (404, 409):
            record(OK, f"a dispute the rule declines is refused at submission "
                       f"({code})")
        else:
            record(BAD, "a dispute the rule declines is refused at submission",
                   f"got {code}: {str(result)[:160]}")
        return
    record(SKIP, "a declined dispute cannot be filed",
           "every dispute in the first 50 is recommended for contest")


def main() -> int:
    print(f"\nchargeback-guard verification — {ROOT}\n" + "-" * 66)
    print("Each line is a claim this project makes, checked against this "
          "installation.\n")

    components = check_api()
    if components is None:
        print("-" * 66)
        print("Cannot verify anything else until the API and database are up.\n")
        return 1

    check_model(components)
    check_guard_deletes_and_replaces()
    check_refund_rule()
    check_packet_and_submission()
    check_refusal_is_not_a_submission()

    print("-" * 66)
    failed = [c for mark, c in results if mark == BAD]
    skipped = [c for mark, c in results if mark == SKIP]
    if failed:
        print(f"{len(failed)} claim(s) not upheld:")
        for claim in failed:
            print(f"  - {claim}")
        print()
        return 1
    if skipped:
        print(f"All checked claims hold. {len(skipped)} could not be checked here:")
        for claim in skipped:
            print(f"  - {claim}")
        print()
        return 0
    print("Every claim checked and upheld.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
