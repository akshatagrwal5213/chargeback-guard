#!/usr/bin/env python3
"""End-to-end self test. Run it any time the pipeline looks stuck.

    python scripts/selftest.py

Posts a correctly-signed synthetic dispute straight at the API, bypassing the
Stripe CLI entirely. That isolates the two things that can fail: the API
itself, or `stripe listen` not forwarding to it.

Standard library only — no dependencies, works in any environment.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent

OK, BAD, INFO = "  ok  ", " FAIL ", " .... "


def say(mark: str, msg: str) -> None:
    print(f"[{mark}] {msg}")


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode()[:300]}
    except Exception as e:
        return 0, {"error": str(e)}


def post(path: str, timeout: int = 60) -> tuple[int, dict]:
    req = urllib.request.Request(f"{API}{path}", data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode()[:300]}
    except Exception as e:
        return 0, {"error": str(e)}


def post_event(event: dict, secret: str) -> tuple[int, str]:
    payload = json.dumps(event).encode()
    headers = {"Content-Type": "application/json"}
    if secret:
        ts = int(time.time())
        sig = hmac.new(
            secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256
        ).hexdigest()
        headers["Stripe-Signature"] = f"t={ts},v1={sig}"
    req = urllib.request.Request(
        f"{API}/webhooks/stripe", data=payload, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return 0, str(e)


def main() -> int:
    print(f"\nchargeback-guard self test  —  {ROOT}\n" + "-" * 58)
    failures = 0

    # 1 — .env
    env = load_env()
    if not (ROOT / ".env").exists():
        say(BAD, ".env does not exist. Run: cp .env.example .env")
        return 1
    say(OK, f".env found, {len(env)} keys")

    secret = env.get("STRIPE_WEBHOOK_SECRET", "")
    if secret:
        say(OK, f"STRIPE_WEBHOOK_SECRET set ({secret[:11]}…)")
    else:
        say(INFO, "STRIPE_WEBHOOK_SECRET empty — posting unsigned")

    # 2 — API reachable
    status, health = get("/health")
    if status == 0:
        say(BAD, f"API not reachable at {API}. Is `make api` running?")
        say(INFO, f"       {health.get('error')}")
        return 1
    say(OK, f"API up, status={health.get('status')}")

    # 3 — database
    dbstate = health.get("components", {}).get("database")
    if dbstate == "ok":
        say(OK, "database connected")
    else:
        say(BAD, f"database is '{dbstate}' — disputes have nowhere to land")
        say(INFO, "       check DATABASE_URL in .env, then restart make api")
        failures += 1

    # 4 — what is already there. Reported, not compared: the inbox is a page,
    # not a total, so a growing database looks static from here.
    _, before = get("/disputes")
    say(INFO, f"disputes on the first page: {before.get('count', 0)}")

    # 5 — signed synthetic dispute
    marker = f"du_selftest_{int(time.time())}"
    event = {
        "id": f"evt_{marker}",
        "type": "charge.dispute.created",
        "data": {
            "object": {
                "id": marker,
                "object": "dispute",
                "livemode": False,
                "amount": 1899900,
                "currency": "inr",
                "reason": "product_not_received",
                "status": "needs_response",
                "evidence_details": {
                    "due_by": int(time.time()) + 86400 * 11,
                    "has_evidence": False,
                },
                # Marks this as our own event. The webhook records it as
                # synthetic, so a diagnostic never leaves behind a dispute the
                # submitter would treat as real and try to file.
                "metadata": {"chargeback_guard_selftest": "true"},
            }
        },
    }
    code, body = post_event(event, secret)
    if code == 200:
        say(OK, f"webhook accepted the event  ->  {body[:120]}")
    elif code == 400 and "signature" in body.lower():
        say(BAD, "signature rejected (400)")
        say(INFO, "       .env's STRIPE_WEBHOOK_SECRET does not match the API's.")
        say(INFO, "       Did you restart `make api` after pasting the secret?")
        failures += 1
    else:
        say(BAD, f"webhook returned {code}: {body[:200]}")
        failures += 1

    # 6 — did it persist
    #
    # Ask for the dispute the webhook says it created, rather than counting
    # the inbox. Counting was wrong twice over: the inbox is capped at 50, so
    # a database with more than that never appears to grow, and a count says
    # nothing about *which* row landed.
    time.sleep(1)
    stored = None
    try:
        stored = json.loads(body).get("dispute_id")
    except Exception:
        pass

    if code != 200:
        pass                                    # already reported above
    elif not stored:
        say(BAD, "webhook returned 200 without a dispute id")
        failures += 1
    else:
        found, row = get(f"/disputes/{stored}")
        dispute = (row or {}).get("dispute") or {}
        if found == 200 and dispute.get("id") == stored:
            say(OK, f"dispute persisted: {stored}  category={dispute.get('category')}"
                    f"  origin={dispute.get('origin')}")
        else:
            say(BAD, f"webhook returned 200 but {stored} is not in the database")
            say(INFO, "       look at the `make api` tab for the error it logged")
            failures += 1

    # 7 — the submission path, without sending anything
    # A dispute that already has a packet is used rather than drafting a new
    # one: drafting calls the model and takes ~20s, which does not belong in a
    # diagnostic. dry_run makes no network call to the processor.
    _, listing = get("/disputes?limit=200")
    with_packet = next(
        (d for d in listing.get("disputes", []) if (d.get("packet_count") or 0) > 0), None
    )
    if not with_packet:
        say(INFO, "no packet drafted yet — try: make packet ID=<dispute id>")
    else:
        did = with_packet["id"]
        code, result = post(f"/disputes/{did}/submit?mode=dry_run")
        if code == 200 and result.get("mode") == "dry_run":
            fields = list((result.get("payload") or {}).keys())
            say(OK, f"submission dry run for {did}: "
                    f"{result.get('characters')} chars across {len(fields)} field(s)")
            say(INFO, f"       would {result.get('endpoint')}")
        elif code == 409:
            say(OK, f"submission refused, correctly: {result.get('detail')}")
        else:
            say(BAD, f"dry-run submission returned {code}: {str(result)[:200]}")
            failures += 1

    print("-" * 58)
    if failures:
        print(f"{failures} problem(s). The API log is where the detail is.\n")
        return 1

    print("All good — the full webhook path works.\n")
    print("If `stripe trigger` still shows nothing, the API is fine and the")
    print("problem is forwarding: check the `make stripe` tab is running and")
    print("shows `Ready! You are using Stripe API Version...`\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
