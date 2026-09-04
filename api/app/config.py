"""Settings. Everything optional so the app boots before you have any accounts."""
from __future__ import annotations

import sys
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Tests must not read the developer's .env. Otherwise adding a real
# STRIPE_WEBHOOK_SECRET locally silently changes what the suite asserts, and
# the same tests behave differently on another machine or in CI.
_UNDER_TEST = "pytest" in sys.modules


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None if _UNDER_TEST else (".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "chargeback-guard"
    env: str = "dev"

    # Postgres. Empty is allowed: the API boots and reports degraded health.
    database_url: str = ""

    # Stripe (test mode only — see docs/DEFENSE_ONLY.md)
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # Razorpay (the Indian rail; optional until the adapter is wired live)
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""

    # Evidence agent. Provider-agnostic — configure one, or neither.
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    agent_model: str = ""

    @property
    def agent_provider(self) -> str:
        """Which provider to use. Deterministic templates when none is set."""
        if self.gemini_api_key:
            return "gemini"
        if self.anthropic_api_key:
            return "anthropic"
        return "template"

    # Economics used by the triage rule. Rupees.
    dispute_fee: float = 1200.0
    contest_effort_cost: float = 250.0
    escalation_risk_cost: float = 900.0

    @property
    def has_db(self) -> bool:
        """True only for a URL that could plausibly connect.

        `.env.example` used to ship a placeholder host, which is not blank —
        so the app treated it as configured, tried to resolve it, and died at
        startup. Anything still carrying template text counts as unconfigured.
        """
        url = self.database_url.strip()
        if not url:
            return False
        placeholders = ("PROJECT", "PASSWORD", "YOUR_", "<", "example.com")
        return not any(p in url for p in placeholders)

    @property
    def has_stripe(self) -> bool:
        return bool(self.stripe_secret_key)

    @property
    def live_mode_guard(self) -> bool:
        """True if someone has pointed this at a live Stripe key. We refuse to run."""
        return self.stripe_secret_key.startswith("sk_live_")


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.live_mode_guard:
        raise RuntimeError(
            "Refusing to start with a live Stripe key. This project is test-mode only."
        )
    return s


settings = get_settings()
