"""Test isolation.

Runs before any test module imports the app, so the settings singleton is
built from a known environment rather than from whatever is in the
developer's shell or .env file.

Without this, putting a real STRIPE_WEBHOOK_SECRET in .env flips the webhook
tests from "no verification" to "verification enforced" and the suite fails
for reasons that have nothing to do with the code.
"""
from __future__ import annotations

import os

import pytest

# Every setting that changes application behaviour. Cleared before import.
MANAGED_VARS = (
    "DATABASE_URL",
    "STRIPE_SECRET_KEY",
    "STRIPE_WEBHOOK_SECRET",
    "RAZORPAY_KEY_ID",
    "RAZORPAY_KEY_SECRET",
    "ANTHROPIC_API_KEY",
    "DISPUTE_FEE",
    "CONTEST_EFFORT_COST",
    "ESCALATION_RISK_COST",
)

for _var in MANAGED_VARS:
    os.environ.pop(_var, None)


@pytest.fixture
def webhook_secret(monkeypatch):
    """Turn on Stripe signature verification for a single test."""
    from app import config

    secret = "whsec_test_only"
    monkeypatch.setattr(config.settings, "stripe_webhook_secret", secret)
    return secret
