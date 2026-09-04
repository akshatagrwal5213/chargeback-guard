"""Render a packet as a document.

HTML is the artifact; PDF is an export of it. That ordering is deliberate:

- WeasyPrint needs Pango and Cairo, which are a system install away on macOS
  and absent on plenty of machines. A reviewer should not need `brew install`
  to see the output of the thing they are reviewing.
- Citations are the point, and in a browser a reference can be hovered to show
  the record behind it. A PDF flattens that to static text.

So `to_html` always works with nothing but Jinja2, and `to_pdf` is attempted
and reported honestly when unavailable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..evidence.checklist import spec_for
from ..evidence.schema import Category

log = logging.getLogger(__name__)

TEMPLATES = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES)),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def to_html(dispute: dict, packet: dict) -> str:
    """The submission document. Every claim carries its refs; hovering one
    shows the record it rests on."""
    category = Category(packet.get("category") or dispute["category"])

    # Flatten the stored evidence back into a lookup so a reference in the
    # narrative can resolve to the record it points at.
    records: list[dict] = []
    for tool in (packet.get("evidence") or {}).get("tools", []):
        records.extend(tool.get("records", []))
    by_ref = {r["ref"]: r for r in records}

    # Only the records actually cited belong in the appendix. Listing
    # everything retrieved pads the document with facts nobody argued from.
    cited = {ref for claim in packet.get("claims", []) for ref in claim.get("refs", [])}
    appendix = [by_ref[ref] for ref in sorted(cited) if ref in by_ref]

    def lookup(ref: str) -> str:
        record = by_ref.get(ref)
        return record["summary"] if record else "record not found"

    return _env.get_template("packet.html").render(
        dispute=dispute,
        packet=packet,
        spec=spec_for(category),
        records=appendix,
        guard=packet.get("guard") or {},
        lookup=lookup,
        generated_at=datetime.now(tz=timezone.utc).strftime("%d %b %Y %H:%M UTC"),
    )


def to_pdf(html: str) -> bytes | None:
    """PDF bytes, or None when the native stack is unavailable.

    Returning None rather than raising: a missing system library is not a
    failure of the packet, and the HTML is already the artifact.
    """
    try:
        from weasyprint import HTML
    except ImportError:
        log.info("WeasyPrint not installed — HTML only. `make install-ml` adds it.")
        return None
    except OSError as exc:
        log.info("WeasyPrint present but its native libraries are missing (%s). "
                 "On macOS: brew install pango libffi", exc)
        return None

    try:
        return HTML(string=html).write_pdf()
    except Exception as exc:                                 # noqa: BLE001
        log.warning("PDF render failed (%s) — serving HTML instead", exc)
        return None
