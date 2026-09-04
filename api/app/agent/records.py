"""Citable records.

Every fact the agent is allowed to state must arrive as a Record with a `ref`
that points at an actual row: `table:primary_key`. The guard later checks each
sentence's citation against the set of refs that were genuinely retrieved, so
a claim can only survive if the row behind it exists.

That is the whole mechanism. Not a prompt asking the model to be careful — a
structural limit on what it is able to assert, checked after the fact against
the database rather than against the model's own account of itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Record:
    """One retrieved row, addressable by `ref`."""

    ref: str                      # "fulfillment_events:881"
    kind: str                     # delivery_scan, support_message, ...
    summary: str                  # one line a human can read
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ref": self.ref, "kind": self.kind,
                "summary": self.summary, "fields": self.fields}


@dataclass
class ToolResult:
    tool: str
    records: list[Record] = field(default_factory=list)
    note: str | None = None       # why a result is empty, when it is

    def as_dict(self) -> dict:
        out: dict[str, Any] = {
            "tool": self.tool,
            "count": len(self.records),
            "records": [r.as_dict() for r in self.records],
        }
        if self.note:
            out["note"] = self.note
        return out

    @property
    def refs(self) -> set[str]:
        return {r.ref for r in self.records}


def ref_for(table: str, key: Any) -> str:
    return f"{table}:{key}"


class Evidence:
    """Everything retrieved for one dispute, and the refs it makes citable."""

    def __init__(self) -> None:
        self.results: list[ToolResult] = []

    def add(self, result: ToolResult) -> None:
        self.results.append(result)

    @property
    def refs(self) -> set[str]:
        refs: set[str] = set()
        for result in self.results:
            refs |= result.refs
        return refs

    def record(self, ref: str) -> Record | None:
        for result in self.results:
            for r in result.records:
                if r.ref == ref:
                    return r
        return None

    def as_dict(self) -> dict:
        return {"tools": [r.as_dict() for r in self.results],
                "citable_refs": sorted(self.refs)}

    def summary_lines(self) -> list[str]:
        """Flat, model-facing view: one line per citable record."""
        return [
            f"[{r.ref}] {r.kind}: {r.summary}"
            for result in self.results
            for r in result.records
        ]
