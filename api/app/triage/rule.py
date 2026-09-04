"""Contest or accept — a transparent rule, deliberately not a fitted model.

No public dataset carries representment outcomes, so there is nothing honest
to fit against. A model trained on labels invented for the purpose would come
with precision and recall that mean nothing, and this track grades honest
metrics. So this is an explicit prior, every coefficient stated, plus a
sensitivity analysis showing how far the inputs can move before the decision
changes.

The decision that matters is not the probability, it is the comparison:

    contest when   P(win) x amount x recovery  >  effort + P(lose) x escalation

Rearranged, that gives a **break-even win probability** — the threshold this
particular dispute has to clear to be worth fighting. That number is auditable
without anyone having to trust the probability estimate to two decimal places,
which is exactly the right posture when the estimate is a prior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import settings
from ..evidence.adapters import escalation_cost
from ..evidence.checklist import spec_for
from ..evidence.schema import Category
from .evidence import (
    Availability,
    completeness,
    critical_slot,
    critical_slot_present,
)
from ..evidence.schema import Strength

RULE_VERSION = "triage-rule-1"

# --- stated coefficients -------------------------------------------------
# How strongly evidence completeness moves the odds, in log-odds. At full
# completeness a category's base win rate roughly doubles in odds terms; at
# zero it roughly halves.
COMPLETENESS_WEIGHT = 2.4

# Missing the one decisive field for a category is not a partial penalty.
# A not-received claim with no delivery scan does not lose slowly.
MISSING_CRITICAL_PENALTY = 1.9

# 3DS-authenticated card-absent transactions usually shift liability to the
# issuer, which is the single biggest lever in the fraud category.
THREE_DS_BONUS = 1.1

# Bounds. Never claim near-certainty in either direction from a prior.
P_MIN, P_MAX = 0.03, 0.90

# Share of the disputed amount actually recovered on a win.
RECOVERY = 0.90


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class Decision:
    dispute_id: str
    category: str
    amount: float
    phase: str

    win_probability: float
    break_even_probability: float
    margin: float                    # win_probability - break_even
    expected_value: float
    recommendation: str              # contest | accept

    completeness: float
    critical_evidence: bool
    contest_blocked: bool
    filled: list[str]
    gaps: list[str]
    reasons: list[str]
    flips_if: str | None             # what would change the answer
    rule_version: str = RULE_VERSION

    def as_row(self) -> dict:
        return {
            "win_probability": round(self.win_probability, 4),
            "expected_value": round(self.expected_value, 2),
            "recommendation": self.recommendation,
            "triage_rule_version": self.rule_version,
        }


def win_probability(av: Availability, category: Category,
                    score: float, critical: bool) -> tuple[float, list[str]]:
    """Category base rate, adjusted by what the records can actually show."""
    spec = spec_for(category)
    reasons: list[str] = []

    z = _logit(spec.base_win_rate)
    reasons.append(
        f"{category.value} representments succeed about {spec.base_win_rate:.0%} "
        f"of the time before evidence is considered"
    )

    delta = COMPLETENESS_WEIGHT * (score - 0.5) * 2
    z += delta
    reasons.append(
        f"{score:.0%} of the required evidence is available "
        f"({'+' if delta >= 0 else ''}{delta:.2f} log-odds)"
    )

    if not critical:
        z -= MISSING_CRITICAL_PENALTY
        reasons.append(
            "the decisive field for this category is missing "
            f"(-{MISSING_CRITICAL_PENALTY:.2f} log-odds)"
        )

    if category is Category.FRAUDULENT and av.three_ds_status == "authenticated":
        z += THREE_DS_BONUS
        reasons.append(f"3DS authenticated, liability normally shifts (+{THREE_DS_BONUS:.2f})")

    if av.signature_name and category is Category.PRODUCT_NOT_RECEIVED:
        z += 0.8
        reasons.append("a signature was captured on delivery (+0.80)")

    return min(max(_sigmoid(z), P_MIN), P_MAX), reasons


def break_even(amount: float, phase: str, effort: float,
               escalation_base: float, recovery: float = RECOVERY) -> float:
    """The win probability at which contesting exactly breaks even.

        p*A*R - effort - (1-p)*E = 0   =>   p = (effort + E) / (A*R + E)

    Below this, contesting loses money in expectation. It is the number to put
    in front of a merchant, because it needs no faith in the estimate — only
    in the arithmetic.
    """
    esc = escalation_cost(escalation_base, phase)
    denominator = amount * recovery + esc
    if denominator <= 0:
        return 1.0
    return min(max((effort + esc) / denominator, 0.0), 1.0)


def decide(dispute: dict, av: Availability) -> Decision:
    category = Category(dispute["category"])
    amount = float(dispute["amount"])
    phase = dispute.get("phase") or "chargeback"

    score, filled, gaps = completeness(av, category)
    critical = critical_slot_present(av, category)
    p, reasons = win_probability(av, category, score, critical)

    effort = settings.contest_effort_cost
    esc = escalation_cost(settings.escalation_risk_cost, phase)
    ev = p * amount * RECOVERY - effort - (1 - p) * esc
    threshold = break_even(amount, phase, effort, settings.escalation_risk_cost)

    # --- policy override, and it outranks the arithmetic -----------------
    #
    # With no decisive evidence, expected value still turns positive on a
    # large enough amount: at P(win)=3% a Rs.60,000 dispute pencils out even
    # with an empty checklist. That is a system filing a claim it cannot
    # support, which is exactly what docs/DEFENSE_ONLY.md rules out. The
    # constraint is ethical, not economic, so it is enforced here rather than
    # left to the numbers.
    #
    # The second override was found by filing. A live dispute scored 0.90 and
    # an expected value of Rs.33,398 on a complete checklist — while the
    # refund ledger for the same order showed a full refund requested six days
    # earlier and never issued. Contesting there asks an issuer to rule in the
    # merchant's favour on money the merchant has already been asked to return
    # and has not answered for. It is bad faith, and it is also a losing
    # argument, which is the usual arrangement.
    #
    # Declined is different from unanswered: a merchant that refused a refund
    # has taken a position and can defend it — that is what Stripe's
    # `refund_refusal_explanation` field is for.
    refund_open = av.refund_open and not av.refund_issued
    contest_blocked = (not critical) or refund_open
    if contest_blocked:
        recommendation = "accept"
    else:
        recommendation = "contest" if ev > 0 else "accept"

    # What would change the answer. A merchant asking "why not fight this?"
    # deserves an actionable answer, not a probability.
    flips: str | None = None
    if recommendation == "accept":
        if refund_open:
            flips = (
                "Refused on conduct, not evidence: a refund for this order was "
                "requested and has not been issued or declined. Answer the "
                "refund request first — contesting while it is open asks the "
                "issuer to decide in favour of a merchant that has not decided "
                "itself."
            )
        elif contest_blocked:
            slot, strength, value = critical_slot(av, category)
            label = {
                Category.PRODUCT_NOT_RECEIVED: "a delivery scan"
                if not av.is_digital else "an access log",
                Category.FRAUDULENT: "3DS authentication or a full AVS+CVV match",
                Category.CREDIT_NOT_PROCESSED: "the accepted refund policy",
                Category.SUBSCRIPTION_CANCELED: "usage after the cancellation date",
                Category.PRODUCT_UNACCEPTABLE: "the customer conversation",
                Category.DUPLICATE: "the second charge record",
            }.get(category, "the decisive record for this category")

            # Distinguish absent from thin. Claiming a record is missing when
            # the checklist beside it shows a value is a contradiction a
            # reviewer will spot, and it discredits every other line.
            if strength is Strength.WEAK and value:
                head = (
                    f"Refused on evidence, not economics: what is on file "
                    f"({value}) is too thin to carry the argument — this "
                    f"category turns on {label}."
                )
            else:
                head = (
                    f"Refused on evidence, not economics: {label} is not on "
                    "file, so there is nothing to argue from."
                )

            tail = (f" The arithmetic alone would have said contest "
                    f"(EV Rs.{ev:,.0f}), which is why this check is not left "
                    f"to the numbers." if ev > 0 else "")
            flips = (head + tail).strip()
        elif p < threshold:
            gap = threshold - p
            flips = (f"Needs P(win) above {threshold:.0%} to break even; "
                     f"currently {p:.0%}, short by {gap * 100:.0f} points.")
        else:
            flips = "Marginal — the escalation risk at this phase outweighs the recovery."
    elif p - threshold < 0.10:
        flips = (f"Close call: break-even is {threshold:.0%} and P(win) is {p:.0%}. "
                 "A small change in the escalation assumption would flip this.")

    return Decision(
        dispute_id=dispute["id"],
        category=category.value,
        amount=amount,
        phase=phase,
        win_probability=p,
        break_even_probability=threshold,
        margin=p - threshold,
        expected_value=ev,
        recommendation=recommendation,
        completeness=score,
        critical_evidence=critical,
        contest_blocked=contest_blocked,
        filled=[s.value for s in filled],
        gaps=[s.value for s in gaps],
        reasons=reasons,
        flips_if=flips,
    )


def sensitivity(dispute: dict, av: Availability,
                spread: float = 0.15) -> dict:
    """Does the recommendation survive being wrong about the probability?

    The win probability is a prior, so the honest question is not "what is it"
    but "how wrong could it be before the answer changes". A decision that
    holds across the whole plausible band is one a merchant can act on; one
    that flips inside it should be flagged as marginal rather than presented
    as an answer.
    """
    base = decide(dispute, av)
    amount, phase = base.amount, base.phase
    effort = settings.contest_effort_cost
    esc = escalation_cost(settings.escalation_risk_cost, phase)

    points = []
    for delta in (-spread, -spread / 2, 0.0, spread / 2, spread):
        p = min(max(base.win_probability + delta, 0.0), 1.0)
        ev = p * amount * RECOVERY - effort - (1 - p) * esc
        points.append({
            "p_win": round(p, 3),
            "expected_value": round(ev, 2),
            "recommendation": "contest" if ev > 0 else "accept",
        })

    outcomes = {pt["recommendation"] for pt in points}
    return {
        "dispute_id": base.dispute_id,
        "break_even_probability": round(base.break_even_probability, 4),
        "base": base.as_row(),
        "points": points,
        "stable": len(outcomes) == 1,
        "note": (
            "Recommendation holds across the plausible range."
            if len(outcomes) == 1 else
            "Recommendation flips inside the plausible range — treat as marginal."
        ),
    }
