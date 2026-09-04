"""Properties the generated dispute data must hold.

These are not tests of randomness — they assert the structural claims the
project makes about its own data. If any of them breaks, a demo or a README
sentence becomes untrue.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas", reason="run `make install-ml`")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def _load(name: str) -> pd.DataFrame:
    path = PROCESSED / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{name}.csv not generated — run `make seed-disputes`")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def disputes() -> pd.DataFrame:
    return _load("disputes")


@pytest.fixture(scope="module")
def fulfillment() -> pd.DataFrame:
    return _load("fulfillment_events")


def test_all_seven_categories_are_represented(disputes):
    """The agent has a distinct checklist per category. If a category never
    appears, that branch is never exercised and the demo cannot show it."""
    from app.evidence.schema import Category

    present = set(disputes["category"].unique())
    expected = {c.value for c in Category}
    assert present == expected, f"missing: {expected - present}"


def test_no_single_category_dominates(disputes):
    """A mix that is 90% one category makes the checklist look decorative."""
    share = disputes["category"].value_counts(normalize=True)
    assert share.max() < 0.45, f"{share.idxmax()} is {share.max():.0%} of disputes"
    assert share.min() > 0.01


def test_some_disputes_are_indefensible(disputes, fulfillment):
    """The core property. If every dispute could be answered, the triage rule
    would contest all of them and 'accept when weak' would never fire — which
    is the behaviour that makes this defense-only rather than adversarial."""
    delivered = set(
        fulfillment.loc[fulfillment["event_type"] == "delivered", "order_id"]
    )
    not_received = disputes[disputes["category"] == "product_not_received"]
    assert len(not_received) > 0

    without_proof = (~not_received["order_id"].isin(delivered)).mean()
    assert 0.15 < without_proof < 0.95, (
        f"{without_proof:.0%} of not-received disputes lack a delivery scan — "
        "expected a genuine mix, not all-or-nothing"
    )


def test_some_disputes_are_defensible(disputes, fulfillment):
    """The mirror of the above: a system that never confidently contests
    anything is equally useless."""
    delivered = set(
        fulfillment.loc[fulfillment["event_type"] == "delivered", "order_id"]
    )
    not_received = disputes[disputes["category"] == "product_not_received"]
    with_proof = not_received["order_id"].isin(delivered).mean()
    assert with_proof > 0.15, "no not-received dispute has a delivery scan to cite"


def test_enhanced_capture_produces_more_signatures(fulfillment):
    """The Lane A -> Lane B claim: a high propensity score changes what gets
    recorded, so the evidence exists months later when the dispute lands.

    Skipped when the orders table was generated without a trained model.
    """
    orders_path = PROCESSED / "orders_loaded.csv"
    if not orders_path.exists():
        pytest.skip("run `make seed-disputes`")
    orders = pd.read_csv(orders_path)
    if orders["evidence_tier"].nunique() < 2:
        pytest.skip("no trained model when generated — every tier is 'standard'")

    signed = set(fulfillment.loc[fulfillment["signature_name"].notna(), "order_id"])
    physical = orders[~orders["is_digital"].astype(bool)]
    rate = physical.assign(signed=physical["id"].isin(signed)).groupby("evidence_tier")["signed"].mean()

    assert rate["enhanced"] > rate["standard"] * 1.5, (
        f"enhanced {rate['enhanced']:.0%} vs standard {rate['standard']:.0%} — "
        "the propensity score is not changing what gets captured"
    )


def test_deadlines_are_in_the_future_of_the_dispute(disputes):
    opened = pd.to_datetime(disputes["opened_at"], utc=True)
    due = pd.to_datetime(disputes["respond_by"], utc=True)
    assert (due > opened).all()
    days = (due - opened).dt.days
    assert days.between(6, 22).all(), "response windows outside the 7-21 day range"


def test_settled_disputes_have_outcomes_and_open_ones_do_not(disputes):
    outcomes = _load("dispute_outcomes")
    settled_ids = set(disputes.loc[disputes["status"].isin(["won", "lost"]), "id"])
    outcome_ids = set(outcomes["dispute_id"])
    assert outcome_ids == settled_ids, "outcome rows must match settled disputes exactly"

    open_ids = set(disputes.loc[disputes["status"] == "needs_response", "id"])
    assert not (open_ids & outcome_ids), "an open dispute cannot have an outcome"


def test_razorpay_disputes_use_the_phase_ladder(disputes):
    """The Indian rail's escalation ladder drives the triage cost term."""
    rz = disputes[disputes["rail"] == "razorpay"]
    if rz.empty:
        pytest.skip("no razorpay disputes in this sample")
    valid = {"fraud", "retrieval", "chargeback", "pre_arbitration", "arbitration"}
    assert set(rz["phase"].unique()) <= valid
    assert rz["phase"].nunique() > 1, "phase ladder is not being exercised"


def test_generator_is_deterministic_for_a_seed():
    """Same seed, same data — or none of the properties above mean anything.

    Runs the generator in-process. The first version shelled out twice, which
    took 137 of the suite's 142 seconds AND overwrote the CSVs the other tests
    read, silently skipping one of them. A test that mutates shared state to
    check determinism is its own counterexample.
    """
    import importlib.util

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "synth", ROOT / "data" / "synthesize_disputes.py"
    )
    synth = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves the defining module through
    # sys.modules, and fails with a bare AttributeError without it.
    sys.modules["synth"] = synth
    spec.loader.exec_module(synth)

    orders = _load("orders_loaded").head(600).rename(columns={"id": "order_id"})
    orders["created_at"] = pd.to_datetime(orders["created_at"], utc=True)
    now = pd.Timestamp("2026-09-01", tz="UTC").to_pydatetime()

    runs = [
        synth.build(orders, np.random.default_rng(5), now).counts()
        for _ in range(2)
    ]
    assert runs[0] == runs[1], f"same seed produced different data: {runs}"
    assert runs[0]["disputes"] > 0, "fixture produced no disputes to compare"


def test_the_loader_never_deletes_a_processor_dispute():
    """The generator's load step used to `delete from disputes` outright. Its
    own comment said "clear generated rows"; the code cleared everything, so
    re-running it destroyed every dispute received over a webhook along with
    the order it pointed at — and the only symptom was an empty Live filter.

    Asserted against the source because the destructive part is SQL, not
    Python: there is no return value to check, and by the time a test could
    observe the effect the rows are gone.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2]
              / "data" / "synthesize_disputes.py").read_text()

    load = source[source.index("async def load("):]
    unscoped = re.findall(r"delete from (\w+)\s*(?:\"|')", load)
    assert "disputes" not in unscoped, (
        "the load step deletes every dispute, including real ones")
    assert "origin <> 'processor'" in load, (
        "disputes must be deleted by origin, not wholesale")
    assert "processor" in load and "protected" in load, (
        "orders behind a processor dispute must be preserved too")
