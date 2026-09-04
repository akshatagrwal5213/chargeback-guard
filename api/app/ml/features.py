"""Feature engineering. Shared by training and serving — deliberately.

The classic failure in a project like this is training/serving skew: features
computed one way in the notebook and another way in the API, so the live model
quietly behaves nothing like the one you measured. Both paths here go through
`build_matrix`, so they cannot drift.

The bridge is `OrderRecord`: a canonical shape that both a public-dataset row
and a live scoring request map into. Add a feature once, and both sides get it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Feature order is part of the model contract. LightGBM takes positional
# arrays, so this list is written into the artifact and checked at load.
FEATURE_NAMES: list[str] = [
    # value
    "amount",
    "amount_log",
    "amount_z_vs_card",
    "amount_z_vs_product",
    # verification
    "avs_match",
    "cvv_match",
    "three_ds_authenticated",
    "three_ds_attempted",
    # mismatch
    "billing_shipping_mismatch",
    "bin_billing_mismatch",
    "billing_distance",
    # tenure and history
    "account_age_days",
    "account_age_missing",
    "card_age_days",
    "prior_disputes",
    "is_guest",
    # velocity
    "txns_card_24h",
    "txns_card_7d",
    "txns_ip_24h",
    "txns_device_24h",
    # channel
    "is_digital",
    "email_domain_risk",
    "email_domains_differ",
    # timing
    "hour_of_day",
    "is_night",
    "day_of_week",
    "is_weekend",
    # completeness — missingness is itself signal
    "n_missing_fields",
]

CATEGORICAL: list[str] = []  # all numeric after encoding; kept explicit


@dataclass
class OrderRecord:
    """Canonical order. Both the dataset loader and the API produce this."""

    order_id: str
    created_at: datetime
    amount: float

    # verification results
    avs_result: str | None = None          # Y / N / P / U
    cvv_result: str | None = None          # M / N / P / U
    three_ds_status: str | None = None     # authenticated / attempted / failed / none

    # geography
    billing_country: str | None = None
    shipping_country: str | None = None
    bin_country: str | None = None
    billing_distance: float | None = None  # km between billing and shipping

    # tenure / history
    account_age_days: float | None = None
    card_age_days: float | None = None
    prior_disputes: int = 0
    is_guest: bool = False

    # velocity, precomputed upstream
    txns_card_24h: int = 0
    txns_card_7d: int = 0
    txns_ip_24h: int = 0
    txns_device_24h: int = 0

    # channel
    is_digital: bool = False
    product_code: str | None = None
    card_id: str | None = None
    payer_email_domain: str | None = None
    recipient_email_domain: str | None = None

    # label — only present in training data
    disputed: int | None = None

    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# Free consumer domains carry more disposable-account risk than corporate ones.
# Weights are stated rather than learned so they can be reviewed; the model
# learns how much to trust this signal.
EMAIL_DOMAIN_RISK: dict[str, float] = {
    "gmail.com": 0.35,
    "yahoo.com": 0.45,
    "hotmail.com": 0.45,
    "outlook.com": 0.35,
    "aol.com": 0.55,
    "mail.com": 0.70,
    "protonmail.com": 0.60,
    "yandex.ru": 0.75,
    "rediffmail.com": 0.40,
    "live.com": 0.40,
    "icloud.com": 0.25,
}
DEFAULT_DOMAIN_RISK = 0.30      # unknown domain: likely corporate, mildly safe
MISSING_DOMAIN_RISK = 0.55      # absent entirely: worse than unknown


def _avs_match(value: str | None) -> float:
    if not value:
        return -1.0
    return {"Y": 1.0, "P": 0.5, "N": 0.0, "U": -1.0}.get(value.upper(), -1.0)


def _cvv_match(value: str | None) -> float:
    if not value:
        return -1.0
    return {"M": 1.0, "P": 0.5, "N": 0.0, "U": -1.0}.get(value.upper(), -1.0)


def _domain_risk(domain: str | None) -> float:
    if not domain:
        return MISSING_DOMAIN_RISK
    return EMAIL_DOMAIN_RISK.get(domain.lower().strip(), DEFAULT_DOMAIN_RISK)


class GroupStats:
    """Amount baselines per card and per product, learned on the TRAINING slice.

    Fitted once during training and saved with the model. Serving reuses the
    saved values — recomputing them at prediction time would leak information
    that was not available when the model was fitted.
    """

    def __init__(self) -> None:
        self.card: dict[str, tuple[float, float]] = {}
        self.product: dict[str, tuple[float, float]] = {}
        self.global_mean: float = 0.0
        self.global_std: float = 1.0

    def fit(self, df: pd.DataFrame) -> "GroupStats":
        amounts = df["amount"].astype(float)
        self.global_mean = float(amounts.mean())
        self.global_std = float(amounts.std()) or 1.0

        for key, target in (("card_id", self.card), ("product_code", self.product)):
            if key not in df.columns:
                continue
            grouped = df.groupby(df[key].fillna("__na__"))["amount"].agg(["mean", "std"])
            for name, row in grouped.iterrows():
                std = float(row["std"]) if pd.notna(row["std"]) and row["std"] else self.global_std
                target[str(name)] = (float(row["mean"]), std)
        return self

    def z(self, which: str, key: str | None, amount: float) -> float:
        table = self.card if which == "card" else self.product
        mean, std = table.get(str(key) if key else "__na__", (self.global_mean, self.global_std))
        return float((amount - mean) / (std or 1.0))

    def to_dict(self) -> dict:
        return {
            "card": {k: list(v) for k, v in self.card.items()},
            "product": {k: list(v) for k, v in self.product.items()},
            "global_mean": self.global_mean,
            "global_std": self.global_std,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroupStats":
        stats = cls()
        stats.card = {k: (v[0], v[1]) for k, v in data.get("card", {}).items()}
        stats.product = {k: (v[0], v[1]) for k, v in data.get("product", {}).items()}
        stats.global_mean = data.get("global_mean", 0.0)
        stats.global_std = data.get("global_std", 1.0)
        return stats


def _row_features(rec: OrderRecord, stats: GroupStats) -> dict[str, float]:
    created = rec.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)

    amount = float(rec.amount or 0.0)
    three_ds = (rec.three_ds_status or "").lower()

    missing = sum(
        1
        for v in (
            rec.avs_result,
            rec.cvv_result,
            rec.three_ds_status,
            rec.billing_country,
            rec.account_age_days,
            rec.payer_email_domain,
        )
        if v in (None, "")
    )

    return {
        "amount": amount,
        "amount_log": float(np.log1p(max(amount, 0.0))),
        "amount_z_vs_card": stats.z("card", rec.card_id, amount),
        "amount_z_vs_product": stats.z("product", rec.product_code, amount),
        "avs_match": _avs_match(rec.avs_result),
        "cvv_match": _cvv_match(rec.cvv_result),
        "three_ds_authenticated": 1.0 if three_ds == "authenticated" else 0.0,
        "three_ds_attempted": 1.0 if three_ds == "attempted" else 0.0,
        "billing_shipping_mismatch": (
            1.0
            if rec.billing_country and rec.shipping_country
            and rec.billing_country != rec.shipping_country
            else 0.0
        ),
        "bin_billing_mismatch": (
            1.0
            if rec.bin_country and rec.billing_country
            and rec.bin_country != rec.billing_country
            else 0.0
        ),
        "billing_distance": float(rec.billing_distance if rec.billing_distance is not None else -1.0),
        "account_age_days": float(rec.account_age_days if rec.account_age_days is not None else -1.0),
        "account_age_missing": 1.0 if rec.account_age_days is None else 0.0,
        "card_age_days": float(rec.card_age_days if rec.card_age_days is not None else -1.0),
        "prior_disputes": float(rec.prior_disputes or 0),
        "is_guest": 1.0 if rec.is_guest else 0.0,
        "txns_card_24h": float(rec.txns_card_24h or 0),
        "txns_card_7d": float(rec.txns_card_7d or 0),
        "txns_ip_24h": float(rec.txns_ip_24h or 0),
        "txns_device_24h": float(rec.txns_device_24h or 0),
        "is_digital": 1.0 if rec.is_digital else 0.0,
        "email_domain_risk": _domain_risk(rec.payer_email_domain),
        "email_domains_differ": (
            1.0
            if rec.payer_email_domain and rec.recipient_email_domain
            and rec.payer_email_domain != rec.recipient_email_domain
            else 0.0
        ),
        "hour_of_day": float(created.hour),
        "is_night": 1.0 if created.hour < 6 else 0.0,
        "day_of_week": float(created.weekday()),
        "is_weekend": 1.0 if created.weekday() >= 5 else 0.0,
        "n_missing_fields": float(missing),
    }


def build_matrix(records: list[OrderRecord], stats: GroupStats) -> pd.DataFrame:
    """Records -> model matrix, columns in FEATURE_NAMES order."""
    rows = [_row_features(r, stats) for r in records]
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return frame.astype("float64")


def build_one(record: OrderRecord, stats: GroupStats) -> np.ndarray:
    """Single record -> 1×N array. The serving path."""
    return build_matrix([record], stats).to_numpy()


def records_from_frame(df: pd.DataFrame) -> list[OrderRecord]:
    """Canonical dataframe -> records. Used by training."""
    records: list[OrderRecord] = []
    for row in df.to_dict(orient="records"):
        created = row.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        elif isinstance(created, pd.Timestamp):
            created = created.to_pydatetime()
        records.append(
            OrderRecord(
                order_id=str(row.get("order_id", "")),
                created_at=created or datetime.now(tz=timezone.utc),
                amount=float(row.get("amount", 0.0)),
                avs_result=row.get("avs_result"),
                cvv_result=row.get("cvv_result"),
                three_ds_status=row.get("three_ds_status"),
                billing_country=row.get("billing_country"),
                shipping_country=row.get("shipping_country"),
                bin_country=row.get("bin_country"),
                billing_distance=row.get("billing_distance"),
                account_age_days=row.get("account_age_days"),
                card_age_days=row.get("card_age_days"),
                prior_disputes=int(row.get("prior_disputes") or 0),
                is_guest=bool(row.get("is_guest")),
                txns_card_24h=int(row.get("txns_card_24h") or 0),
                txns_card_7d=int(row.get("txns_card_7d") or 0),
                txns_ip_24h=int(row.get("txns_ip_24h") or 0),
                txns_device_24h=int(row.get("txns_device_24h") or 0),
                is_digital=bool(row.get("is_digital")),
                product_code=row.get("product_code"),
                card_id=row.get("card_id"),
                payer_email_domain=row.get("payer_email_domain"),
                recipient_email_domain=row.get("recipient_email_domain"),
                disputed=row.get("disputed"),
            )
        )
    return records
