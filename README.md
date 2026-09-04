# chargeback-guard

Predicts which orders will be charged back, prepares the evidence for them in
advance, and — when a dispute arrives — decides whether contesting it is worth
the money and assembles the packet that does it.

Every factual assertion in a generated evidence packet is traceable to a
specific record in the merchant's own database, and the system refuses to
contest a dispute it cannot support.

> **Hackathon Track 02 — AI Risk Manager.** One class of loss: chargebacks.
> Defense-only by construction, see [`docs/DEFENSE_ONLY.md`](docs/DEFENSE_ONLY.md).
> Stripe **test mode only** — the app refuses to start with a live key.

## How this was built

With AI assistance, over two weeks, and the commit history is the record of
it: what was tried, what broke, and why each fix took the shape it did.

The part worth reading is where the bugs came from. Eight of them were found
by running the system against live Stripe test infrastructure rather than by
testing it, and every one had passed a green suite first:

| Found by | What it was |
|---|---|
| `400 No such file_upload: 'INR 2,803.03 on…'` | nine of Stripe's evidence fields take a file id, not prose |
| `404 No such dispute: 'du_1cca8eaf…'` | the generator minted ids that looked exactly like Stripe's |
| reading a staged packet | it told the issuer the merchant's own win rate |
| reading it again | it volunteered the cardholder's prior-dispute count |
| reading it a third time | it wanted to contest while sitting on an unissued refund |
| `make verify` | a dispute the rule declined could still be filed from a stale packet |
| a 41-second draft | a budget that gated attempts without bounding them |
| a dispute page | "staged" does not mean "not sent" — see below |

None of those came out of a conversation. They came out of a status code, a
log line, and a badge on a dashboard. `make verify` is where those checks now
live, so they stay found.

---

## Status

Complete and running end to end against live Stripe test infrastructure.

| | Component | State |
|---|---|---|
| ✅ | API skeleton, health, OpenAPI docs | done |
| ✅ | Postgres schema (all tables) | done |
| ✅ | Category → evidence checklist, 7 categories | done |
| ✅ | Stripe + Razorpay evidence adapters | done |
| ✅ | Webhook receivers with signature verification | done |
| ✅ | Propensity model (LightGBM, time-split, calibrated) | done |
| ✅ | Leakage audit + false-positive cost curve | done |
| ✅ | SHAP explanations wired to `/score` | done |
| ✅ | Dispute + evidence synthesizer (7 categories, real gaps) | done |
| ✅ | Triage rule — completeness, EV, contest/accept | done |
| ✅ | Agent: 7 tools, drafting, citation guard | done |
| ✅ | Packet as a document (HTML/PDF) | done |
| ✅ | Live submission — stage, submit, dry run | done |
| ✅ | Merchant console, served by the API | done |
| ✅ | `make verify` — every claim checked against a live install | done |

`POST /score` serves the trained model when artifacts are present and a
documented heuristic otherwise, so a fresh clone works before anyone runs
training. `model_version` in the response says which you got.

---

## The console

```bash
make api        # then open http://localhost:8000/console
```

One HTML file, served by the API that feeds it. No build step, no second
process, no CORS: a reviewer clones this repository, runs `make api`, and has
the screen. A toolchain between a reviewer and the demo is a toolchain that
can break on their machine rather than mine.

The inbox is ordered by expected value, and scores anything the batch pass has
not — a database nobody has run `make triage` against used to open on a screen
full of "untriaged" under a heading claiming the ordering was the product.

The list is a worklist, not a table dump: it shows the top of the ordering and
says how many are below it — *60 of 10,412 ordered by expected value* — with a
button for the next sixty. It used to show sixty and say nothing, which on a
database of ten thousand looks exactly like a database of sixty.

A search box sits above the list. It is server-side, across every field a
person might have in front of them — a dispute id from a log line, a `du_…`
copied from Stripe's dashboard, a customer's email, an order, an amount, a
reason code. Filtering the fifty rows already loaded would be a search that
cannot find anything not already visible, which is the opposite of the point.
`₹` and thousands separators are stripped, so an amount pasted from this very
screen matches the number in the database.

`recommendation` is deliberately not searchable: the batch pass writes it, so
on a fresh database it is null for every row, and searching "contest" would
return nothing while the screen shows a column full of them.

Four things the screen is built to show, because they are the things worth
arguing about:

- **the decision, with its reasoning** — P(win) against break-even, the log-odds
  each factor contributed, and what would change the answer
- **the checklist** — every slot the category turns on, filled, weak or missing,
  with the actual value on file
- **every claim's source** — click a citation chip and the database row it came
  from opens underneath it, fields and all
- **the refusals** — a dispute the rule declines renders its reason as an
  outcome, not an error, and staging one comes back refused

## Verifying it

```bash
make test      # the logic, in isolation — 147 tests, no credentials needed
make verify    # the behaviour, against your database and your key
make doctor    # the webhook path, end to end
```

`make verify` states each claim this project makes and checks it against the
running installation: the guard removing what it cannot source, an open refund
outranking a profitable contest, every claim in a real packet resolving to a
retrieved record, and the exact payload that would be sent carrying nothing
internal and addressing no file-upload field. A check that cannot run says so
rather than passing quietly.

The distinction between the two matters here. Every defect this project has
actually had — prose posted to a file-upload field, a dispute id that only
looked real, the merchant's own win rate submitted to the issuer, a contest
recommended over an unissued refund — passed the unit tests and was found by
running the thing. `make verify` is where those checks now live, so they stay
found.

## Metrics

<!-- METRICS:START -->
<!-- generated by scripts/sync_readme.py — do not edit by hand -->

Precision, recall and F1 at the operating threshold, on a **time-based**
holdout. The velocity and tenure features are computed over trailing windows,
so a random split can put a card's future beside its past.

| Metric | Value |
|---|---|
| Precision @ threshold 0.18 | **0.337** |
| Recall @ threshold | **0.343** |
| F1 | **0.340** |
| PR-AUC | 0.289 — 8.2x the 3.53% base rate |
| ROC-AUC | 0.807 |
| Brier score | 0.0600 (isotonic-calibrated) |
| Orders flagged for enhanced capture | 7.5% |

Test set: 30,000 orders, 2,219 disputed. Model
`lgbm-1`, trained 2026-08-26.

Every run also scores a **random** split and reports the gap — 0.258
PR-AUC against 0.289 here — so the choice of split is auditable
rather than asserted. Full report including the false-positive cost curve:
[`api/app/ml/artifacts/METRICS.md`](api/app/ml/artifacts/METRICS.md), or
`make metrics`.

**False positives are cheap here by design.** The model's action is enhanced
evidence capture, not declining the order, so being wrong costs
₹120 — a signature on delivery and some storage. Nobody is
refused service. That asymmetry is what puts the threshold at
0.18 rather than 0.5, subject to an operational ceiling of
15% of orders. Net value on the test slice:
₹212,890.

*This table is regenerated by `make train`. If it disagrees with
`METRICS.md`, the run did not finish.*
<!-- METRICS:END -->

---

## Run it

**Requires Python 3.11+.** On macOS, `python3` outside conda is the system 3.9,
which cannot run this — `make install` checks and tells you so rather than
failing halfway through a dependency resolve.

```bash
conda create -n cbg python=3.12 -y && conda activate cbg   # or any 3.11+ env

cp .env.example .env        # works with every value left blank
make install                # core API only — fast, no compiled wheels
make api                    # http://localhost:8000/docs
```

`make install-ml` adds the model, agent and PDF dependencies. Those are split
out deliberately: they carry compiled wheels and system libraries, and a
failure there must not stop anyone from running the API.

The API boots with no database and no accounts — `/health` reports which
components are unconfigured rather than crashing. That means you can clone
this and see it work in about ninety seconds.

```bash
curl localhost:8000/health
curl localhost:8000/checklist | jq

curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{
  "order_id": "ord_1", "amount": 42000,
  "avs_result": "N", "cvv_result": "N",
  "prior_disputes": 2, "is_guest": true
}' | jq
```

Note what the score returns: an `evidence_tier` and a `capture` list. Not a
decline.

### With a database

Create a Supabase project, put the connection URI in `DATABASE_URL`, restart.
The schema applies itself on boot.

### Train the model

```bash
make install-ml
make data          # ~3s
                   # targets a 3.5% dispute rate; --dispute-rate to change it
make train         # ~90s
make metrics
```

**macOS:** LightGBM's wheel does not bundle the OpenMP runtime, so the first
`make train` fails with a `libomp.dylib` dlopen error. One fix:

```bash
brew install libomp
```

`make train` detects this and prints the fix rather than a ctypes traceback.
The API, the tests and the evidence layer all run without it — only training
and model-backed scoring need OpenMP.

Training and serving share `app/ml/features.py`. That is deliberate: computing
features one way in the notebook and another way in the API is the classic
silent failure, and a test asserts the two paths produce identical vectors.

### Generate disputes and their evidence

```bash
make seed-disputes     # local only; `make trigger` is the one that calls Stripe
```

Materialises an operational history for 10k orders — shipments and scans,
support threads, policy acceptances, refunds, access logs — then raises
disputes **caused by that history**. A parcel with no delivery scan becomes a
not-received dispute that genuinely cannot be defended.

That direction is the point. Assigning categories first and fabricating
matching evidence would make every dispute winnable, and the triage rule would
have nothing to decide. Roughly a quarter of generated disputes are weak, and
those are the ones the system should recommend accepting.

It also closes the loop between the two lanes: orders are scored by the
trained model first, and `evidence_tier = enhanced` makes signature-on-delivery
far more likely. Measured on the generated data:

| Evidence tier | Orders | With signature on delivery |
|---|---|---|
| enhanced | 1,123 | **66%** |
| standard | 6,210 | 19% |

### Triage the worklist

With the API running:

```bash
make triage
```

Scores every open dispute: evidence completeness against the category
checklist, a win probability, and

```
EV = P(win) x amount x recovery - effort - P(lose) x escalation
```

Rearranged, that gives a **break-even win probability** per dispute — the
threshold this case has to clear to be worth fighting. That number is
auditable without trusting the probability estimate itself, which is the right
posture when the estimate is a stated prior rather than a fitted model.

On the generated worklist: **50% contest, 50% accept**. Fraud disputes have
81% evidence completeness but only 25% get contested, because fraud
representments start from a 21% base win rate — high evidence, poor odds.

**The guardrail that outranks the arithmetic.** With no decisive evidence,
expected value still turns positive on a large enough amount: at P(win)=3% a
₹250,000 dispute pencils out with an empty checklist. That is a system filing
a claim it cannot support. `contest_blocked` refuses those regardless of EV —
**165 disputes** in the current run where the numbers said contest and there
was nothing to argue from. The constraint is ethical, not economic, so it is
enforced in code and asserted in a test rather than described in a doc.

```bash
curl localhost:8000/disputes/{id}/triage       # decision + reasoning + checklist
curl localhost:8000/disputes/{id}/sensitivity  # does it survive being wrong?
```

### Draft an evidence packet

```bash
curl -X POST localhost:8000/disputes/{id}/packet | jq
```

Retrieves everything the category could need, drafts a narrative, then checks
it. **No API key required** — with none configured the narrative composes from
templates, and the guarantee below holds identically, because it is enforced
against the database rather than by the model.

Set `GEMINI_API_KEY` (free at [aistudio.google.com](https://aistudio.google.com/apikey))
for a model-written narrative. `ANTHROPIC_API_KEY` works too. The provider
walks a chain of known-good models, distinguishing a permanent 404 (advance)
from a transient capacity error (back off and retry) — both were observed on a
real free-tier key.

#### The citation guard

Two checks, run against the database:

**Sourcing — decidable, so it deletes.** Every claim must cite refs that were
actually retrieved for this dispute. A claim citing anything else is removed,
and a packet that had *any* removal is not submittable.

**The summary and the gaps are checked too.** They cite nothing — the summary
is one sentence stating the merchant's position, the gaps are concessions —
and for a while that meant the paragraph an issuer reads first was the one
paragraph nobody verified. They are now checked against everything retrieved
plus the dispute's own header, which is a record we hold: its amount and
opening date are facts, not inventions. An ungrounded summary is **replaced**
with a deterministic position statement rather than trimmed, because half a
corrected position is worse than a plain one. A gap asserting an unsupported
figure is dropped; a concession does not need a number in it.

**Grounding — heuristic, so it warns.** A valid ref is not enough: a model can
cite the right delivery scan and still state the wrong date on it. Dates,
tracking numbers and amounts in a claim are checked against the cited records'
actual field values. Mismatches are flagged, not deleted, because the sentence
may be about a real fact read badly.

Against a deliberately hostile draft on real evidence:

| Claim | Outcome |
|---|---|
| Delivered and signed for *(valid ref)* | kept |
| "disputed 14 prior orders" *(ref to a row never retrieved)* | **deleted** |
| "admitted receipt by phone" *(no ref)* | **deleted** |
| Tracking `ZZ0000000000IN` *(valid ref, invented number)* | **flagged** |
| "delivered 1999-01-01" *(valid ref, wrong date)* | **flagged** |

Packet marked not submittable. Every surviving claim is stored in
`evidence_citations` as `(claim, source_table, source_id)`.

A dispute the triage rule recommends accepting is **not drafted at all** — a
packet arguing a case the system does not believe would be the wrong artifact.

### File it

```bash
curl -X POST "localhost:8000/disputes/{id}/submit?mode=dry_run" | jq
```

| mode | what happens |
|---|---|
| `dry_run` | nothing is sent. Returns the exact endpoint and payload that would be. |
| `stage` *(default)* | a real API call with `submit=false` — the evidence appears on the dispute in Stripe's dashboard and is **not** sent to the issuing bank by us. See the note below on Stripe's own auto-submit. |
| `submit` | the same call with `submit=true`. Irreversible, and refused without `confirm=true`. |

Which one is available is decided by the data, not by the caller: a dispute
that was generated locally has no object at the processor to file against, so
`stage` and `submit` both degrade to `dry_run` and say why. Only disputes that
arrived through a live `charge.dispute.created` webhook are filed for real.

`disputes.origin` is what decides this — `processor` or `synthetic`, written
when the row is created. It was originally inferred from the id, which failed
in the only way that mattered: the generator minted ids beginning `du_`, the
check believed them, and a submission went out against a dispute Stripe had
never heard of. Generated ids now begin `sim_`, and provenance is recorded
rather than guessed. To find one you can actually file:

```bash
curl -s localhost:8000/disputes | jq '.disputes[] | select(.origin=="processor") | .id'
```

**The guard is the gate.** The verdict is stored on the packet when it is
drafted and read back here — not recomputed, and not taken from the caller. A
packet the guard removed a claim from is refused with a **409** before any
network call is made. A dispute with no packet drafted yet is a **404**: the
two are different answers — one is a step you have not taken, the other is a
decision the system has made and will keep making — and giving them the same
status hides the second behind the first. Precedent is excluded from anything sent outward: a comparable
dispute the merchant lost is a useful internal signal and an argument for the
issuer's side.

Stripe caps combined evidence at 150,000 characters. Over that, the longest
field is trimmed rather than any field dropped — losing one entirely would
lose a whole class of evidence.

#### Text fields and file fields

Nine of Stripe's evidence fields — `receipt`, `shipping_documentation`,
`customer_communication`, `customer_signature`, `refund_policy`,
`cancellation_policy`, `service_documentation`,
`duplicate_charge_documentation`, `uncategorized_file` — take the **ID of an
uploaded file**, not text. The field names do not say so.

We found this the way you would want to: a real staged submission came back
`400 No such file_upload: 'INR 2,803.03 on 2026-05-02, physical (apparel), AVS
N, CVV N, 3DS none'`. The adapter now maps slots onto text fields only, and
`to_stripe()` filters its own output against `STRIPE_FILE_FIELDS` so a future
mapping mistake costs a sentence rather than the submission. Evidence with no
text field to go to — a delivery scan, a support thread — is written into the
statement under a heading instead of being dropped:

```
The order was delivered on 2026-07-16.

DELIVERY
delivered at 2026-07-16 14:39, New Delhi 110001

CUSTOMER CONTACT
Customer asked about the parcel on 2026-07-17.
```

A rejected call reports `mode: "failed"`, never `"staged"` — and
`submitted_at` is stamped only on a submission the processor accepted.

#### Staged is not the same as safe

This README said, flatly, that staged evidence is not sent to the bank. That
was true of *this* system and not true of the outcome, which is a distinction
worth more than the sentence saved.

Stripe ships [Smart Disputes](https://docs.stripe.com/disputes/smart-disputes),
which assembles its own evidence packet and, in their words, "automatically
submits the pre-filled evidence packet just before the dispute times out" if
the merchant does nothing. The dispute page shows the countdown — *7 days until
auto-submit*. Staging leaves a document on a dispute that a third party may
file on your behalf.

Nothing here can change that; it is an account setting, at
`dashboard.stripe.com/settings/disputes`. What this repository can do is stop
claiming otherwise.

### What this is not

Stripe already automates dispute responses, better resourced than this and at
no cost unless you win. Any honest comparison starts there:

|  | Smart Disputes | this |
|---|---|---|
| Data | the whole Stripe network | one merchant's own records |
| Integration | none | a schema and a webhook |
| Cost | a fee only when you win | — |
| Evidence | assembled for you | assembled from records you can open |
| When it declines | it contests what is eligible | it declines on thin evidence, and on conduct |

The narrower claim is the defensible one. This system can show you the
database row behind every sentence it wrote, and it refuses cases a
revenue-maximising responder would take — a dispute with no decisive record,
and a dispute where the merchant is sitting on a refund request it has not
answered. Neither is a feature you would ship to win more chargebacks. Both
are the point.

### With live Stripe test-mode webhooks

```bash
make stripe     # tab 1: forwards events, prints the whsec_ to put in .env
make trigger    # tab 2: asks Stripe for a real test dispute
```

A dispute that arrives this way is recorded with `origin = 'processor'` and is
the only kind the submitter will file against. `make seed-disputes` is the
local generator and never contacts Stripe — the two used to be named `dispute`
and `disputes`, one letter apart, which cost an evening.

Regenerating is non-destructive to anything real. `make seed-disputes` replaces
the synthetic rows and steps around every `origin = 'processor'` dispute, the
order it was raised against, and the evidence and packets hanging off it. It
prints which live orders it preserved. An earlier version deleted the whole
`disputes` table on every run, which quietly destroyed a live dispute mid-build;
a test now reads the loader and fails if any unscoped delete comes back.

#### A live dispute with evidence behind it

`make trigger` disputes a charge that has no order behind it: no customer, no
delivery scan, nothing to cite. The system refuses to draft, correctly, and
proves nothing about the submission path.

```bash
make live-dispute
```

picks the best-evidenced delivered order in the database, pays for it with
`pm_card_createDisputeProductNotReceived` — the card Stripe disputes on
purpose — and records the payment id on the order. The dispute Stripe raises a
moment later resolves back to that order, so retrieval finds the scans and the
support thread and the packet has something to say.

Linking is what makes this work, and it is the part a real integration has to
get right. Three attempts, in order:

1. `metadata.order_id` on the dispute — free when it is there, and it never is
   on a dispute the processor raised itself.
2. the dispute's `payment_intent` or `charge` matched against the payment id
   recorded on the order. We know our own payment ids.
3. ask Stripe for the charge and read the metadata set when the payment was
   created. A dispute is the one object in the chain that never carries it.

A dispute that matches nothing is recorded **unlinked** rather than attached to
a guess — a packet citing another customer's delivery would be worse than no
packet. The lookup in (3) is best effort: every failure is logged and
swallowed, because losing the link costs evidence while failing the webhook
would lose the dispute.

```bash
make relink     # retry resolution for disputes recorded without an order
```

A dispute can beat its own order into the database — the processor raises it
seconds after the charge. Recording it unlinked is right; leaving it that way
is not. `make relink` only ever fills a blank, so it is safe to run at any
time.

Stripe's dispute-generating test cards:

| Card | Creates |
|---|---|
| `4000000000000259` | dispute, fraudulent |
| `4000000000002685` | dispute, product not received |
| `4000000000001976` | inquiry (retrieval phase) |

---

## Architecture

Two lanes over one Postgres instance.

```
Lane A   order ──▶ features ──▶ propensity model ──▶ evidence capture
                        │                                    │
                        └──────────── Postgres ◀──────────────┘
                                    + pgvector
                        ┌──────────────┴──────────────┐
Lane B   webhook ──▶ triage ──▶ evidence agent ──▶ packet ──▶ submit
                                   (tools + citations)
```

The dashed relationship is the load-bearing one: a high propensity score does
not decline the order, it writes *more evidence* into the store, so when the
dispute arrives ninety days later Lane B finds what it needs already there.

```
api/app/
├── config.py            settings; refuses live Stripe keys
├── db.py                asyncpg pool + query helpers
├── schema.sql           every table, idempotent
├── evidence/
│   ├── schema.py        rail-agnostic packet + Citation
│   ├── checklist.py     category → required evidence (hardcoded domain knowledge)
│   └── adapters.py      Stripe + Razorpay field mapping
├── routers/
│   ├── health.py        /health, /checklist
│   ├── scoring.py       POST /score
│   ├── disputes.py      inbox + detail
│   └── webhooks.py      signature-verified receivers
├── ml/
│   ├── features.py      shared by training AND serving — no drift possible
│   ├── train.py         time split, calibration, leakage audit, cost curve
│   ├── predict.py       loads artifacts; falls back to the heuristic
│   └── artifacts/       model.txt, calibrator.pkl, METRICS.md
└── agent/               day 6–8
```

## Two rails

Stripe is what the demo triggers live. Razorpay is the Indian rail the same
packet has to survive contact with — its `phase` ladder
(`fraud → retrieval → chargeback → pre_arbitration → arbitration`) drives the
escalation term in the triage rule, because contesting a weak case at
`pre_arbitration` costs materially more than at `chargeback`.

One internal evidence object, two thin adapters. Field names verified against
[Stripe's dispute object](https://docs.stripe.com/api/disputes/object) and
[Razorpay's contest API](https://razorpay.com/docs/api/disputes/contest/).

## What's real and what's synthetic

| | |
|---|---|
| Real | Stripe test-mode webhooks, signature verification, evidence submission, both processors' field schemas, network reason-code mappings |
| **Synthetic** | **The transactions and their labels.** Every metric on this page comes from a generated dataset, not from real payments — read them as evidence that the pipeline is sound, not as a measurement of the world |
| Synthetic | Evidence artifacts (delivery scans, support threads, policy acceptances) — generated defensively, never fraud technique |
| Generator, disclosed | The order table pins two knobs so results reproduce on any machine: a target dispute rate of 3.5% and a signal-to-noise ratio giving ROC-AUC ≈ 0.81. Both are stated in `data/build_dataset.py` rather than tuned until the numbers looked good |
| Prior, not model | The triage win-probability. No public representment outcomes exist, so it is an explicit calibrated prior with a sensitivity analysis, retrainable from `dispute_outcomes` once real results accumulate |

## Test

```bash
make test    # 147 passing
make lint
```

The tests that matter most are the ones asserting behaviour the write-up
claims: that digital orders substitute their shipping requirements, that
forged and replayed webhook signatures are rejected, that live-mode events are
refused, and that scoring returns capture actions rather than a decline.
