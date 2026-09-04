"""Agent tool contracts.

These do not need a database — they assert the shape every tool must honour,
because the citation guard depends on it entirely.
"""
from __future__ import annotations

import inspect

import pytest

from app.agent import tools
from app.agent.records import Evidence, Record, ToolResult, ref_for


def test_every_registered_tool_has_a_schema():
    """A tool the model cannot see is dead code; a schema with no tool behind
    it is a runtime failure mid-run."""
    registered = set(tools.REGISTRY)
    described = {s["name"] for s in tools.SCHEMAS}
    assert registered == described, (
        f"only in registry: {registered - described}; "
        f"only in schemas: {described - registered}"
    )


def test_schema_parameters_match_the_function_signature():
    """A schema promising a parameter the function ignores produces confidently
    wrong answers rather than an error."""
    for schema in tools.SCHEMAS:
        fn = tools.REGISTRY[schema["name"]]
        actual = set(inspect.signature(fn).parameters)
        declared = set(schema["parameters"]["properties"])
        assert declared <= actual, (
            f"{schema['name']} declares {declared - actual} which it does not accept"
        )
        required = set(schema["parameters"].get("required", []))
        assert required <= actual, f"{schema['name']} requires a parameter it lacks"


def test_every_tool_is_async_and_returns_a_tool_result():
    for name, fn in tools.REGISTRY.items():
        assert inspect.iscoroutinefunction(fn), f"{name} must be async"
        assert inspect.signature(fn).return_annotation in (ToolResult, "ToolResult")


def test_no_tool_can_write():
    """Read-only is the point. An evidence agent that could modify the records
    it cites would be worthless as evidence."""
    import pathlib

    source = (pathlib.Path(tools.__file__)).read_text().lower()
    for verb in ("insert into", "update ", "delete from", "drop ", "alter "):
        assert verb not in source, f"tools.py contains {verb!r}"


def test_schemas_are_json_serialisable():
    """They go over the wire to the model."""
    import json

    json.dumps(tools.SCHEMAS)


def test_unknown_tool_fails_loudly():
    with pytest.raises(KeyError):
        import asyncio
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            tools.call("get_bank_details", {})
        )


# ------------------------------------------------------------ record shape

def test_refs_are_addressable_rows():
    assert ref_for("fulfillment_events", 881) == "fulfillment_events:881"
    r = Record(ref=ref_for("orders", "ord_1"), kind="order", summary="x")
    table, _, key = r.ref.partition(":")
    assert table and key, "a ref must name a table and a key"


def test_evidence_collects_refs_and_finds_records():
    ev = Evidence()
    ev.add(ToolResult("get_fulfillment", [
        Record(ref="fulfillment_events:1", kind="delivery_scan", summary="delivered"),
        Record(ref="fulfillment_events:2", kind="carrier_scan", summary="in transit"),
    ]))
    ev.add(ToolResult("get_refunds", note="none on file"))

    assert ev.refs == {"fulfillment_events:1", "fulfillment_events:2"}
    assert ev.record("fulfillment_events:1").kind == "delivery_scan"
    assert ev.record("communications:99") is None, "must not invent a record"


def test_empty_results_carry_a_note_not_silence():
    """'No refund on file' is itself evidence in a credit-not-processed
    dispute. An empty list with no explanation loses that."""
    result = ToolResult("get_refunds", note="No refund was ever requested.")
    assert result.records == []
    assert result.as_dict()["note"]
    assert result.as_dict()["count"] == 0


def test_summary_lines_are_prefixed_with_their_ref():
    """This is what the model sees. Every line has to arrive already attached
    to the row it came from, or citing it correctly is guesswork."""
    ev = Evidence()
    ev.add(ToolResult("get_order", [
        Record(ref="orders:ord_1", kind="order", summary="INR 5,000 on 2026-07-01"),
    ]))
    line = ev.summary_lines()[0]
    assert line.startswith("[orders:ord_1]")
    assert "INR 5,000" in line
