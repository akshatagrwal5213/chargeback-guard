"""The citation guard.

Two checks, run against the database rather than against the model's own
account of itself.

**Sourcing.** Every claim must cite at least one ref, and every ref must be one
that was actually retrieved for this dispute. A claim citing a record that was
never fetched is deleted. This is the hard rule: it is what makes "the system
cannot assert what it cannot show" a property rather than a promise.

**Grounding.** A valid ref is not enough on its own. A model can cite the right
delivery scan and still state the wrong date on it. So dates, tracking numbers
and amounts appearing in a claim are checked against the actual field values of
the records it cites. A mismatch is flagged rather than deleted — the sentence
may be about a genuine fact the extractor read badly — but a flagged packet
never submits silently.

The distinction matters. Sourcing is decidable, so it deletes. Grounding is
heuristic, so it warns. Pretending a heuristic is decidable is how a guard
starts quietly discarding true statements.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .compose import Claim, Draft
from .records import Evidence

log = logging.getLogger(__name__)

# Dates in several shapes, tracking numbers, and money. Deliberately narrow:
# a broad "any number" rule flags counts the model computed legitimately
# ("2 messages"), which trains everyone to ignore the warnings.
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+\w{3,9}\s+\d{4}\b")
_TRACKING = re.compile(r"\b[A-Z]{2}\d{8,}[A-Z]{0,2}\b")
_AMOUNT = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\b\d+\.\d{2}\b")


@dataclass
class Rejection:
    text: str
    reason: str
    refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"text": self.text, "reason": self.reason, "refs": self.refs}


@dataclass
class Flag:
    text: str
    token: str
    refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"text": self.text, "token": self.token, "refs": self.refs}


@dataclass
class Verdict:
    claims: list[Claim]                      # survivors
    rejected: list[Rejection] = field(default_factory=list)
    flagged: list[Flag] = field(default_factory=list)
    # Facts asserted outside any claim — in the summary, or in an
    # acknowledged gap — that no retrieved record supports.
    summary_ungrounded: list[str] = field(default_factory=list)
    gaps_dropped: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.rejected and not self.flagged

    @property
    def submittable(self) -> bool:
        """Nothing unsourced survived, and something is left to say."""
        return bool(self.claims) and not self.rejected

    def as_dict(self) -> dict:
        return {
            "claims_kept": len(self.claims),
            "claims_rejected": len(self.rejected),
            "claims_flagged": len(self.flagged),
            "rejected": [r.as_dict() for r in self.rejected],
            "flagged": [f.as_dict() for f in self.flagged],
            "summary_ungrounded": self.summary_ungrounded,
            "gaps_dropped": self.gaps_dropped,
            "summary_replaced": bool(self.summary_ungrounded),
            "submittable": self.submittable,
        }

    def feedback(self) -> str:
        """What to tell the model on a repair attempt."""
        lines = []
        for r in self.rejected:
            lines.append(f'- Removed "{r.text[:110]}" — {r.reason}')
        for f in self.flagged:
            lines.append(
                f'- "{f.text[:90]}" states {f.token!r}, which does not appear in '
                f'the records it cites. Use the exact value from the record or '
                f'drop the detail.'
            )
        return ("These claims did not hold up. Rewrite using only the supplied "
                "references and the exact values inside them:\n" + "\n".join(lines))


def _grounding_text(evidence: Evidence, refs: list[str]) -> str:
    """Everything the cited records actually contain, flattened for matching."""
    parts = []
    for ref in refs:
        record = evidence.record(ref)
        if record:
            parts.append(record.summary)
            parts.append(json.dumps(record.fields, default=str))
    return " ".join(parts)


def _corpus(evidence: Evidence, dispute: dict | None) -> str:
    """Every fact the packet is entitled to state, flattened for matching.

    A claim is checked against the records it cites. The summary cites
    nothing — it is the paragraph an analyst reads first, and it was passing
    through unchecked — so it is checked against everything retrieved, plus
    the dispute's own header. The dispute is a record we hold: its amount and
    the date it was opened are facts, and a summary that mentions them is not
    inventing anything.
    """
    parts = []
    for result in evidence.results:
        for record in result.records:
            parts.append(record.summary)
            parts.append(json.dumps(record.fields, default=str))
    if dispute:
        parts.append(json.dumps({
            k: str(v) for k, v in dispute.items()
            if k in ("amount", "currency", "category", "opened_at",
                     "respond_by", "phase", "processor_reason")
        }, default=str))
    return " ".join(parts)


def _ungrounded(text: str, corpus: str) -> list[str]:
    """Dates, amounts and tracking numbers in `text` that the corpus lacks."""
    missing = []
    for pattern in (_DATE, _TRACKING, _AMOUNT):
        for token in pattern.findall(text):
            if not any(v in corpus for v in _normalise(token)):
                missing.append(token)
    return missing


def _normalise(token: str) -> set[str]:
    """Variants a date or amount can legitimately take between record and prose."""
    out = {token, token.replace(",", "")}
    month_names = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    m = re.match(r"^(\d{1,2})\s+(\w+)\s+(\d{4})$", token)
    if m:
        day, month, year = m.groups()
        num = month_names.get(month.lower())
        if num:
            out.add(f"{year}-{num}-{int(day):02d}")
    return out


def verify(draft: Draft, evidence: Evidence,
           dispute: dict | None = None) -> Verdict:
    available = evidence.refs
    kept: list[Claim] = []
    rejected: list[Rejection] = []
    flagged: list[Flag] = []

    for claim in draft.claims:
        if not claim.refs:
            rejected.append(Rejection(claim.text, "no source cited"))
            continue

        unknown = [r for r in claim.refs if r not in available]
        if unknown:
            rejected.append(Rejection(
                claim.text,
                f"cites {', '.join(unknown)}, which was not retrieved for this dispute",
                claim.refs,
            ))
            continue

        grounding = _grounding_text(evidence, claim.refs)
        for pattern in (_DATE, _TRACKING, _AMOUNT):
            for token in pattern.findall(claim.text):
                if not any(v in grounding for v in _normalise(token)):
                    flagged.append(Flag(claim.text, token, claim.refs))

        kept.append(claim)

    # Everything in the document that is not a claim. The summary leads the
    # packet and the gaps close it; both are submitted, and neither cited
    # anything, so neither was ever checked.
    corpus = _corpus(evidence, dispute)
    summary_ungrounded = _ungrounded(draft.summary, corpus)
    gaps_dropped = [g for g in draft.gaps_acknowledged if _ungrounded(g, corpus)]

    if rejected:
        log.warning("Citation guard removed %d unsourced claim(s)", len(rejected))
    if flagged:
        log.warning("Citation guard flagged %d ungrounded detail(s)", len(flagged))
    if summary_ungrounded:
        log.warning("Citation guard replaced the summary: %s unsupported",
                    ", ".join(summary_ungrounded))

    return Verdict(claims=kept, rejected=rejected, flagged=flagged,
                   summary_ungrounded=summary_ungrounded,
                   gaps_dropped=gaps_dropped)


def apply(draft: Draft, verdict: Verdict,
          fallback_summary: str | None = None) -> Draft:
    """The draft as it stands after the guard.

    Rejected claims are gone. A summary asserting something no record supports
    is replaced outright rather than trimmed: it is one sentence setting the
    merchant's position, and a half-corrected position statement is worse than
    a plain one. Gaps that assert unsupported facts are dropped — a gap exists
    to concede something, and a concession does not need a figure in it.
    """
    summary = draft.summary
    if verdict.summary_ungrounded and fallback_summary:
        summary = fallback_summary

    return Draft(
        summary=summary,
        claims=verdict.claims,
        gaps_acknowledged=[g for g in draft.gaps_acknowledged
                           if g not in verdict.gaps_dropped],
        provider=draft.provider,
        fallback_reason=draft.fallback_reason,
        raw=draft.raw,
    )
