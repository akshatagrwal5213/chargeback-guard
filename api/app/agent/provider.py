"""Language model access, deliberately behind a narrow interface.

`complete(system, user) -> str`. Nothing more. The caller asks for JSON and
parses it, so every provider is interchangeable and no provider-specific
structured-output API leaks into the rest of the codebase.

Three implementations, chosen by whatever is in .env:

  gemini     free tier, the default
  anthropic  optional alternative
  template   no model at all — packets still compose, deterministically

That last one is not a stub. Because the citation guarantee is enforced after
the fact against the database, it holds whether a model wrote the narrative or
a template did. Which means this repository can be cloned and run end to end
with no credentials, and the central claim still demonstrably holds.

The model list matters as much as the model. Probing a real free-tier key
found two distinct failure modes:

  404 NOT_FOUND        'no longer available to new users' — permanent, so
                       advance to the next model immediately
  'high demand'        transient capacity — back off and retry the same one

Treating those alike is how a demo dies: retrying a deprecated model forever,
or abandoning a working one after a momentary spike.
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Protocol

from ..config import settings

log = logging.getLogger(__name__)

# The SDK repeats an advisory about automatic function calling on every call.
# It arrives through logging, not warnings, so the filter around the call site
# never touched it. We do not use AFC — retrieval here is deterministic — so
# the advice does not apply and the line is pure noise in a demo.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# Verified working against a live free-tier key. Ordered by preference; the
# client walks the list when a model is permanently gone.
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
]

ANTHROPIC_MODELS = ["claude-sonnet-4-5", "claude-haiku-4-5"]

MAX_RETRIES = 2
BACKOFF_SECONDS = 1.5

# A hard ceiling on the whole call, across every model and retry. Without one,
# three models x three retries x escalating backoff is over a minute of an
# endpoint doing nothing visible — which in a demo is indistinguishable from
# a hang. Better to fall back to templates and say so.
# Room for one full-length attempt to fail and another to succeed. A 504 from
# Gemini's free tier can burn the whole per-call timeout, and a budget with no
# space after that turns one bad gateway into a template packet.
TOTAL_BUDGET_SECONDS = 45.0

# One call's own ceiling, and the least remaining budget worth starting one
# with. The floor is not a matter of taste: Gemini rejects a short deadline
# outright —
#
#   400 INVALID_ARGUMENT: Manually set deadline 4s is too short.
#                         Minimum allowed deadline is 10s.
#
# — and the rejection is instant, so a budget-capped call that dips under the
# floor does not merely fail, it fails three models deep in under a second and
# looks exactly like every model being unavailable.
CALL_TIMEOUT_SECONDS = 20.0
MIN_CALL_SECONDS = 10.0


class ProviderError(RuntimeError):
    """Raised when no model could be reached. Never swallowed silently."""


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


def _is_permanent(exc: Exception) -> bool:
    """A model that is gone stays gone. Retrying it wastes the whole budget."""
    text = str(exc).lower()
    return any(s in text for s in ("not_found", "404", "no longer available",
                                   "not supported", "invalid model"))


def _is_timeout(exc: Exception) -> bool:
    """The request itself did not come back.

    Distinct from being rate limited. A gateway timeout means the model spent
    the whole budget and produced nothing; asking the same one again spends
    the rest. Move to the next model instead, and do not sleep first — there
    is nothing to wait out.
    """
    text = str(exc).lower()
    return any(s in text for s in ("504", "gateway timeout", "deadline exceeded",
                                   "timed out", "timeout"))


def _is_transient(exc: Exception) -> bool:
    """Busy, not broken. Worth waiting for, on the same model."""
    text = str(exc).lower()
    return any(s in text for s in ("high demand", "resource_exhausted", "429",
                                   "rate limit", "overloaded", "unavailable",
                                   "503"))


def _deadline_ms(remaining: float) -> int:
    """The timeout to hand one request, in milliseconds.

    Bounded above by the budget left, and below by what the API will accept.
    The loop refuses to start a call with less than MIN_CALL_SECONDS left, so
    the lower clamp should never bind — it is here because a deadline below
    the floor is rejected instantly, and an instant rejection walks the whole
    model chain in under a second while looking like an outage.
    """
    return int(max(MIN_CALL_SECONDS, min(CALL_TIMEOUT_SECONDS, remaining)) * 1000)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "") -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        # An explicitly configured model is tried first, then the known-good
        # chain, without duplicating it.
        self._models = ([model] if model else []) + [
            m for m in GEMINI_MODELS if m != model
        ]

    def complete(self, system: str, user: str) -> str:
        from google.genai import types

        deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
        last: Exception | None = None
        for model in self._models:
            for attempt in range(MAX_RETRIES):
                remaining = deadline - time.monotonic()
                # The budget has to bound the call, not merely gate it. Checked
                # only before each attempt, a request starting at t=24.9s with
                # a 20s timeout of its own overruns to 45 — which is how a
                # 25-second budget produced a 41-second draft on a live run.
                if remaining < MIN_CALL_SECONDS:
                    raise ProviderError(
                        f"Exceeded the {TOTAL_BUDGET_SECONDS:.0f}s budget. "
                        f"Last error: {last}")
                try:
                    # The SDK advises Chat.send_message for automatic function
                    # calling. We do not use AFC — retrieval here is
                    # deterministic — so the notice is noise on every call.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*automatic function calling.*"
                        )
                        response = self._client.models.generate_content(
                            model=model,
                            contents=user,
                            config=types.GenerateContentConfig(
                                system_instruction=system,
                                temperature=0.2,
                                response_mime_type="application/json",
                                max_output_tokens=4096,
                                http_options=types.HttpOptions(
                                    timeout=_deadline_ms(remaining)),
                            ),
                        )
                    text = (response.text or "").strip()
                    if text:
                        log.info("gemini: %s answered", model)
                        return text
                    last = ProviderError(f"{model} returned an empty response")
                    break
                except Exception as exc:                     # noqa: BLE001
                    last = exc
                    if _is_permanent(exc):
                        log.warning("gemini: %s is gone, trying the next model", model)
                        break
                    if _is_timeout(exc):
                        log.warning("gemini: %s timed out, trying the next model", model)
                        break
                    if _is_transient(exc) and attempt < MAX_RETRIES - 1:
                        wait = BACKOFF_SECONDS * (attempt + 1)
                        if deadline - time.monotonic() < wait + MIN_CALL_SECONDS:
                            break          # no time left to sleep and still try
                        log.info("gemini: %s busy, retrying in %.0fs", model, wait)
                        time.sleep(wait)
                        continue
                    break
        raise ProviderError(f"No Gemini model answered. Last error: {last}")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "") -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._models = ([model] if model else []) + [
            m for m in ANTHROPIC_MODELS if m != model
        ]

    def complete(self, system: str, user: str) -> str:
        deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
        last: Exception | None = None
        for model in self._models:
            for attempt in range(MAX_RETRIES):
                if time.monotonic() > deadline:
                    raise ProviderError(f"Exceeded the budget. Last error: {last}")
                try:
                    message = self._client.messages.create(
                        model=model,
                        max_tokens=4096,
                        temperature=0.2,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    text = "".join(
                        block.text for block in message.content
                        if getattr(block, "type", "") == "text"
                    ).strip()
                    if text:
                        return text
                    last = ProviderError(f"{model} returned an empty response")
                    break
                except Exception as exc:                     # noqa: BLE001
                    last = exc
                    if _is_permanent(exc):
                        break
                    if _is_transient(exc) and attempt < MAX_RETRIES - 1:
                        time.sleep(BACKOFF_SECONDS * (attempt + 1))
                        continue
                    break
        raise ProviderError(f"No Anthropic model answered. Last error: {last}")


class TemplateProvider:
    """No model. Signals the caller to compose deterministically.

    Returning a sentinel rather than raising keeps the decision in one place:
    compose.py owns what a packet looks like, whoever writes it.
    """

    name = "template"

    def complete(self, system: str, user: str) -> str:
        return ""


def get_provider() -> Provider:
    """Whichever provider is configured. Never raises at construction time —
    a missing SDK degrades to templates rather than taking the API down."""
    choice = settings.agent_provider

    if choice == "gemini":
        try:
            return GeminiProvider(settings.gemini_api_key, settings.agent_model)
        except ImportError:
            log.warning("google-genai not installed — composing from templates. "
                        "Run: make install-ml")
        except Exception as exc:                             # noqa: BLE001
            log.warning("Gemini unavailable (%s) — composing from templates", exc)

    elif choice == "anthropic":
        try:
            return AnthropicProvider(settings.anthropic_api_key, settings.agent_model)
        except ImportError:
            log.warning("anthropic not installed — composing from templates")
        except Exception as exc:                             # noqa: BLE001
            log.warning("Anthropic unavailable (%s) — composing from templates", exc)

    return TemplateProvider()
