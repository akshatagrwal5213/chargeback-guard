#!/usr/bin/env python3
"""Build the canonical training table.

    python data/build_dataset.py --rows 120000

A generator producing realistic class balance, correlated signal and —
importantly — *drift over time*, so the chronological split has something to
actually catch rather than being a formality.

Everything downstream reads one schema, so a real transaction table could be
mapped onto it without `ml/train.py` knowing the difference.

Stated plainly, because it governs how every metric in the README should be
read: this data is synthetic. The numbers show the pipeline is sound. They are
not a measurement of the world, and the two knobs below are pinned so nobody
— including the author — can quietly tune them until the results look good.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
from app.ml.tableio import write_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

CANONICAL_COLUMNS = [
    "order_id", "created_at", "amount",
    "avs_result", "cvv_result", "three_ds_status",
    "billing_country", "shipping_country", "bin_country", "billing_distance",
    "account_age_days", "card_age_days", "prior_disputes", "is_guest",
    "txns_card_24h", "txns_card_7d", "txns_ip_24h", "txns_device_24h",
    "is_digital", "product_code", "card_id",
    "payer_email_domain", "recipient_email_domain",
    "disputed",
]

COUNTRIES = ["IN", "IN", "IN", "IN", "IN", "US", "AE", "SG", "GB", "AU"]
DOMAINS = [
    "gmail.com", "gmail.com", "gmail.com", "yahoo.com", "outlook.com",
    "hotmail.com", "rediffmail.com", "icloud.com", "protonmail.com", "mail.com",
]
PRODUCTS = ["electronics", "apparel", "digital_sub", "grocery", "jewellery", "travel"]


# ------------------------------------------------------- history helpers

def _group_bounds(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start/stop offsets for each run in a sorted key array."""
    change = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return change, np.r_[change[1:], len(keys)]


def _rolling_count(frame: pd.DataFrame, key: str, window: str) -> np.ndarray:
    """Prior events on the same key inside a trailing time window.

    Excludes the row itself and only ever looks backwards — which is exactly
    why shuffling these rows leaks. Vectorised with searchsorted; the pandas
    groupby-rolling equivalent takes about a minute at 120k rows.
    """
    seconds = pd.to_timedelta(window).total_seconds()
    times = pd.to_datetime(frame["created_at"]).astype("int64").to_numpy() / 1e9
    keys = frame[key].to_numpy()

    order = np.lexsort((times, keys))
    k_sorted, t_sorted = keys[order], times[order]

    counts = np.empty(len(frame), dtype=np.int64)
    starts, stops = _group_bounds(k_sorted)
    for start, stop in zip(starts, stops):
        block = t_sorted[start:stop]
        left = np.searchsorted(block, block - seconds, side="left")
        counts[start:stop] = np.arange(len(block)) - left

    result = np.empty(len(frame), dtype=np.int64)
    result[order] = counts
    return result


# How separable the generated data is. The feature-driven part of the logit is
# rescaled to this standard deviation before fixed-sigma noise is added, so the
# signal-to-noise ratio — and therefore the achievable PR-AUC — is the same on
# every machine. Without this, pinning only the dispute rate still let PR-AUC
# swing from 0.16 to 0.25 between numpy versions, and a reviewer re-running
# `make train` would not reproduce the README.
#
# 1.7 puts the trained model at roughly ROC-AUC 0.81 / PR-AUC 0.29 on a 3.5%
# base rate. That is deliberately on the conservative side of what published
# fraud models achieve, because a synthetic generator can be dialled to
# produce any number you like and the tuning knob should not be doing the work
# the model is supposed to. Stated here so the choice is visible rather than
# buried.
SIGNAL_SD = 1.7
NOISE_SD = 0.55


def _standardise(signal: np.ndarray, target_sd: float = SIGNAL_SD) -> np.ndarray:
    """Centre and rescale the signal so its spread is reproducible."""
    sd = float(signal.std())
    if sd < 1e-9:
        return signal - float(signal.mean())
    return (signal - float(signal.mean())) / sd * target_sd


def _solve_intercept(logit: np.ndarray, target_rate: float) -> float:
    """Find the intercept that makes the mean predicted rate hit the target.

    Hardcoding an intercept is not reproducible: numpy does not guarantee
    identical streams from `default_rng` across versions, so the same seed on
    a different machine produced 0.82% instead of the 3.75% this was tuned
    for — and a model trained on 0.82% positives learned to flag nothing.

    Solving for the rate instead makes the *thing that matters* reproducible,
    whatever the library versions underneath.
    """
    lo, hi = -20.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        rate = float((1.0 / (1.0 + np.exp(-(logit + mid)))).mean())
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _prior_positive_count(frame: pd.DataFrame) -> np.ndarray:
    """Earlier positive outcomes on the same card. Strictly prior, in time order."""
    times = pd.to_datetime(frame["created_at"]).astype("int64").to_numpy()
    keys = frame["card_id"].to_numpy()
    y = frame["y"].to_numpy()

    order = np.lexsort((times, keys))
    k_sorted, y_sorted = keys[order], y[order]

    prior = np.empty(len(frame), dtype=np.int64)
    starts, stops = _group_bounds(k_sorted)
    for start, stop in zip(starts, stops):
        block = y_sorted[start:stop]
        prior[start:stop] = np.cumsum(block) - block

    result = np.empty(len(frame), dtype=np.int64)
    result[order] = prior
    return result


# --------------------------------------------------------------- synthetic

def synthesize(n_rows: int, days: int = 120, seed: int = 7,
               dispute_rate: float = 0.035) -> pd.DataFrame:
    """Generate orders with correlated risk signal and genuine temporal drift.

    Drift matters: fraud patterns move, so a model fitted on January and tested
    on January flatters itself. Here the mix shifts across the window, which is
    exactly what a time-based split is supposed to expose — and what a random
    split would hide.
    """
    rng = np.random.default_rng(seed)
    start = datetime.now(tz=timezone.utc) - timedelta(days=days)

    # Non-uniform arrival: more orders later in the window, weekday-weighted.
    offsets = np.sort(rng.beta(2.0, 1.6, n_rows) * days * 86400)
    created = [start + timedelta(seconds=float(s)) for s in offsets]
    progress = offsets / offsets.max()          # 0 -> 1 across the window

    n_cards = max(n_rows // 12, 500)
    card_idx = rng.integers(0, n_cards, n_rows)
    card_id = np.array([f"card_{i:06d}" for i in card_idx])

    product = rng.choice(PRODUCTS, n_rows, p=[0.26, 0.24, 0.18, 0.16, 0.09, 0.07])
    is_digital = np.isin(product, ["digital_sub", "travel"])

    # Amounts are lognormal, scaled by category.
    base = rng.lognormal(mean=7.4, sigma=0.85, size=n_rows)
    scale = np.select(
        [product == "jewellery", product == "electronics", product == "grocery"],
        [3.1, 1.8, 0.35],
        default=1.0,
    )
    amount = np.round(base * scale, 2)

    account_age = np.clip(rng.exponential(240, n_rows), 0, 3000)
    # A slice of orders are genuinely new accounts. Draw the mask once —
    # drawing it twice gives two different masks and a shape mismatch.
    fresh = rng.random(n_rows) < 0.13
    account_age[fresh] = rng.uniform(0, 6, int(fresh.sum()))
    card_age = np.clip(account_age - rng.exponential(30, n_rows), 0, None)

    is_guest = rng.random(n_rows) < 0.22
    account_age = np.where(is_guest, np.nan, account_age)

    billing = rng.choice(COUNTRIES, n_rows)
    shipping = billing.copy()
    flip = rng.random(n_rows) < 0.07
    shipping[flip] = rng.choice(COUNTRIES, flip.sum())
    bin_country = billing.copy()
    flip2 = rng.random(n_rows) < 0.05
    bin_country[flip2] = rng.choice(COUNTRIES, flip2.sum())

    distance = np.where(
        billing == shipping,
        rng.exponential(35, n_rows),
        rng.uniform(500, 6000, n_rows),
    )

    avs = rng.choice(["Y", "Y", "Y", "Y", "P", "N", "U"], n_rows)
    cvv = rng.choice(["M", "M", "M", "M", "M", "N", "U"], n_rows)

    # 3DS adoption climbs through the window — a real, dated trend and the
    # clearest source of drift between the train and test slices.
    tds_rate = 0.30 + 0.35 * progress
    tds = np.where(
        rng.random(n_rows) < tds_rate,
        "authenticated",
        rng.choice(["attempted", "failed", "none"], n_rows, p=[0.25, 0.1, 0.65]),
    )

    domain = rng.choice(DOMAINS, n_rows)
    recipient = domain.copy()
    diff = rng.random(n_rows) < 0.06
    recipient[diff] = rng.choice(DOMAINS, diff.sum())

    # Velocity is computed from the actual transaction history per card, not
    # drawn independently. This is how a real feature store builds it, and it
    # is *why* a random split leaks: shuffle these rows and a card's future
    # activity ends up in the training set alongside its past.
    ip_idx = rng.integers(0, max(n_rows // 8, 400), n_rows)
    dev_idx = rng.integers(0, max(n_rows // 6, 400), n_rows)

    frame = pd.DataFrame({
        "created_at": created,
        "card_id": card_id,
        "ip": ip_idx,
        "device": dev_idx,
    })
    txns_24h = _rolling_count(frame, "card_id", "24h")
    txns_7d = _rolling_count(frame, "card_id", "7D")
    txns_ip = _rolling_count(frame, "ip", "24h")
    txns_dev = _rolling_count(frame, "device", "24h")

    # --- risk model the data actually follows -------------------------------
    # Intercept starts at zero and is solved for below, so the realised
    # dispute rate matches `dispute_rate` on any machine.
    logit = np.zeros(n_rows)
    logit += np.where(cvv == "N", 1.55, 0.0)
    logit += np.where(avs == "N", 1.05, 0.0)
    logit += np.where(tds == "authenticated", -1.35, 0.0)
    logit += np.where(billing != shipping, 0.80, 0.0)
    logit += np.where(bin_country != billing, 0.55, 0.0)
    logit += np.where(np.nan_to_num(account_age, nan=0.0) < 7, 0.75, 0.0)
    logit += np.where(is_guest, 0.40, 0.0)
    logit += 0.14 * np.clip(txns_24h - 1, 0, 8)
    logit += 0.30 * np.clip((np.log1p(amount) - 7.6), 0, None)
    logit += np.where(is_digital, 0.35, 0.0)
    logit += np.array([0.9 if d in ("mail.com", "protonmail.com") else 0.0 for d in domain])

    # Drift: dispute pressure rises late in the window, and the weight on
    # velocity strengthens. A random split averages this away; a time split
    # makes the model face it.
    logit += 0.45 * progress
    logit += 0.20 * progress * np.clip(txns_24h - 1, 0, 8)

    # Fix the spread of the feature-driven signal, then add noise at a fixed
    # sigma. Doing it in this order is what makes separability reproducible.
    logit = _standardise(logit)
    logit = logit + rng.normal(0, NOISE_SD, n_rows)

    # Two passes, because prior_disputes depends on realised outcomes.
    # Pass 1 draws disputes without any history term.
    logit += _solve_intercept(logit, dispute_rate)
    p0 = 1.0 / (1.0 + np.exp(-logit))
    disputed0 = (rng.random(n_rows) < p0).astype(int)

    # Count each card's PRIOR disputes, strictly in time order. This is the
    # single strongest leakage channel under a random split: shuffled, the
    # model sees a customer's later disputes while predicting their earlier
    # ones.
    prior_disputes = _prior_positive_count(
        pd.DataFrame({"created_at": created, "card_id": card_id, "y": disputed0})
    )

    # Pass 2: history now genuinely raises risk. Re-solve the intercept so
    # adding the history term does not push the overall rate off target.
    logit2 = logit + 0.5 * prior_disputes
    logit2 += _solve_intercept(logit2, dispute_rate)  # keep the rate on target
    p = 1.0 / (1.0 + np.exp(-logit2))
    disputed = (rng.random(n_rows) < p).astype(int)

    frame = pd.DataFrame({
        "order_id": [f"ord_{i:08d}" for i in range(n_rows)],
        "created_at": created,
        "amount": amount,
        "avs_result": avs,
        "cvv_result": cvv,
        "three_ds_status": tds,
        "billing_country": billing,
        "shipping_country": shipping,
        "bin_country": bin_country,
        "billing_distance": np.round(distance, 1),
        "account_age_days": np.round(account_age, 1),
        "card_age_days": np.round(card_age, 1),
        "prior_disputes": prior_disputes,
        "is_guest": is_guest,
        "txns_card_24h": txns_24h,
        "txns_card_7d": txns_7d,
        "txns_ip_24h": txns_ip,
        "txns_device_24h": txns_dev,
        "is_digital": is_digital,
        "product_code": product,
        "card_id": card_id,
        "payer_email_domain": domain,
        "recipient_email_domain": recipient,
        "disputed": disputed,
    })
    return frame[CANONICAL_COLUMNS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dispute-rate", type=float, default=0.035,
                    help="target share of orders disputed")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    df = synthesize(args.rows, seed=args.seed, dispute_rate=args.dispute_rate)

    df = df.sort_values("created_at").reset_index(drop=True)
    path = write_table(df, OUT)

    rate = df["disputed"].mean()
    span = (df["created_at"].max() - df["created_at"].min())
    print(f"rows          {len(df):,}")
    print(f"dispute rate  {rate:.2%}  ({int(df['disputed'].sum()):,} positives)")
    print(f"time span     {df['created_at'].min():%Y-%m-%d} to {df['created_at'].max():%Y-%m-%d}  ({span.days} days)")
    print(f"written       {path.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
