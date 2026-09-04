"""The merchant console, checked at the source.

There is no JS runner in this project, so these read `console.html` and assert
the properties that broke in practice. They are deliberately narrow: each one
corresponds to a bug that reached the screen.
"""
from __future__ import annotations

from pathlib import Path

import pytest

CONSOLE = (Path(__file__).resolve().parents[1]
           / "app" / "static" / "console.html")


@pytest.fixture(scope="module")
def source() -> str:
    return CONSOLE.read_text()


def _function(source: str, name: str) -> str:
    """The body of a top-level function, up to the next one."""
    start = source.index(f"function {name}(")
    rest = source[start + 1:]
    end = rest.find("\nfunction ")
    tail = rest.find("\nasync function ")
    if tail != -1 and (end == -1 or tail < end):
        end = tail
    return rest[:end] if end != -1 else rest


def test_the_staged_banner_is_rendered_from_state_not_only_the_dom(source):
    """Staging wrote the Stripe link straight into #result and then called
    select() to refresh. Its fetches landed about two seconds later, re-rendered
    the Actions card and threw the link away before anyone could click it. The
    banner has to come back out of `state` on every render."""
    card = _function(source, "submitCard")
    assert 'id="result"' in card
    assert "state.result" in card, (
        "the Actions card rebuilds #result on every render; if it does not "
        "re-emit state.result, any refresh erases the result of the last action")


def test_showing_a_result_records_it(source):
    assert "state.result = { cls, html }" in _function(source, "showResult")


def test_moving_to_another_dispute_drops_the_previous_banner(source):
    select = _function(source, "select")
    assert "clearResult()" in select
    assert "id !== state.id" in select, (
        "clearing unconditionally would wipe the banner staging itself just "
        "set, because submit() re-selects the same dispute to refresh it")


@pytest.mark.parametrize("action", ["draft", "submit"])
def test_a_new_action_starts_without_the_previous_result(source, action):
    assert "clearResult()" in _function(source, action)


def test_the_stripe_link_opens_in_a_new_tab(source):
    """It sits next to buttons that mutate a real dispute. Navigating the
    console away from the dispute you just staged is a bad trade."""
    line = source[source.index("open the dispute at Stripe") - 400:
                  source.index("open the dispute at Stripe")]
    assert 'target="_blank"' in line
    assert 'rel="noopener"' in line
