# Defense-only by construction

Track 02 disqualifies anything offense-capable. Three parts of this system sit
near that line. Each one has a design decision behind it, and each is enforced
in code rather than promised in prose.

## 1. Scoring changes what we record, never who we serve

The propensity model's output is an **evidence tier**, not a decision. A high
score sets `orders.evidence_tier = 'enhanced'`, which triggers pre-emptive
collection — signature on delivery, listing snapshot, session and policy-
acceptance logs — so that if a dispute arrives months later, the evidence
already exists.

There is no code path in this repository that declines, blocks, or holds an
order. `POST /score` returns `evidence_tier` and a `capture` list; it has no
allow/deny field. `test_score_endpoint_returns_capture_actions_not_a_decline`
asserts this.

## 2. Model attributions stay inside the merchant console

SHAP contributions describe exactly which signals raise a score, which is also
a map of what to avoid tripping. They are therefore:

- returned only from merchant-authenticated endpoints,
- never included in any customer-facing response, receipt, or notification,
- never exposed through the public order status API.

## 3. The agent cannot argue beyond the evidence

This is the important one. A model that writes persuasive dispute rebuttals
could, pointed the wrong way, help a merchant defeat legitimate consumer
claims. Two mechanisms prevent that.

**The citation guard is a capability limit.** It covers the whole document,
not only the claims: the summary and the acknowledged gaps cite nothing, and
were for a time the only unverified sentences in a packet whose entire premise
is that nothing goes unverified. The agent may only assert facts returned by a
tool call over the merchant's own records. Every sentence in a
narrative carries a `(source_table, source_id)` pair, stored in
`evidence_citations`. A packet containing an uncited claim does not render —
the guard raises before the PDF is produced. The system is structurally unable
to be persuasive beyond what the records support; anti-hallucination is the
side benefit, not the purpose.

**The refusal is enforced where the action is.** Declining to contest was
originally checked only at drafting — `build_packet` will not write a packet
for a dispute triage recommends accepting. `make verify` walked straight into
the hole on its first real run: a packet drafted *before* the decision changed
is still in the database, and submission read it out and built a payload
without ever asking what the rule now said. A gate that stops a document being
written but not sent is not a gate. Submission re-checks triage and refuses
with a 409.

**Insufficient evidence defaults to accept.** When the category checklist
cannot be filled, the triage rule recommends refunding the customer rather
than contesting. This is written into the decision rule, not left to
discretion. It is also the commercially correct answer, which is why it costs
nothing to hold to.

**An open refund request defaults to accept.** The first live submission
surfaced a case the evidence rule was happy with and it should not have been:
P(win) 0.90 on a complete checklist, expected value ₹33,398 — while the refund
ledger for the same order showed a full refund requested six days earlier and
never issued or declined. Contesting there asks an issuer to decide in favour
of a merchant that has not decided itself. The rule now blocks it and says to
answer the refund first. Declining a refund is different and does not block:
that is a position the merchant has taken and can defend, which is what
Stripe's `refund_refusal_explanation` field is for.

**Nothing internal is submitted.** The packet's "merchant notes" were built
from the retrieval tools' own commentary, which meant two things reached the
issuer that should never leave the building: how comparable disputes had gone
(precedent under another name, and an argument for the other side), and the
access-log tool's remark that a product "was genuinely never used." Gaps are
now derived from the checklist — a required slot with nothing in it — rather
than from a tool's opinion of its own empty result. In the same pass, the
customer-history line stopped volunteering the cardholder's prior-dispute
count: that number belongs in triage, not in a document arguing the merchant's
case.

## 4. The synthesizer generates defences, not attacks

The data generator in `data/` produces *defensive records* — delivery scans,
support threads, policy acceptances, refund ledger entries — and dispute
metadata. It does not model, generate, or parameterise fraud technique. The
dispute labels used for training are generated from the evidence gaps
themselves — a parcel with no delivery scan becomes a not-received dispute —
so the model learns which orders are hard to defend, not how to attack one.

## 5. Nothing unsourced is ever filed

The submission path reads the guard's verdict back out of the database rather
than recomputing it or trusting the caller. `POST /disputes/{id}/submit`
refuses — before any network call — a packet from which the guard removed a
claim. The verdict is stored on the packet at drafting time, so a document
that failed verification stays unfilable however it is later asked to be sent.

Two further brakes:

- The default mode is `stage`, which is a real API call with `submit=false`.
  The evidence lands on the dispute and is visible in the dashboard, and *this
  system* does not send it to the issuing bank.

  That is not the same as it never being sent, and the distinction was found
  by looking at a real dispute page rather than by reasoning about our own
  code. Stripe's [Smart Disputes](https://docs.stripe.com/disputes/smart-disputes)
  submits its own packet just before the deadline if the merchant takes no
  action — the dispute page counts the days down. A guarantee about our own
  behaviour was being read as a guarantee about the outcome. Auto-submit is an
  account setting at `dashboard.stripe.com/settings/disputes`; what belongs
  here is the accurate sentence.
- `mode=submit` is irreversible and therefore requires a second explicit
  signal, `confirm=true`. A mistyped query string cannot file a representment.

Precedent is excluded from everything sent outward. A comparable dispute the
merchant lost is a useful internal signal and an argument for the other side;
`SUBMISSION_EXCLUDED` keeps it out of the issuer-facing document.

## 6. Test mode only

`config.py` refuses to start if `STRIPE_SECRET_KEY` begins with `sk_live_`.
The webhook handler rejects any event with `livemode: true`. No real card
data, cardholder PII, or production credentials belong in this project at any
point.
