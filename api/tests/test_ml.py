"""Model tests.

Skipped when the ML extras are absent — `make install` deliberately omits them
so the core API stays fast to install. `make install-ml` turns these on.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("lightgbm", reason="run `make install-ml`")
pytest.importorskip("pandas", reason="run `make install-ml`")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.ml import predict as ml  # noqa: E402
from app.ml.features import (  # noqa: E402
    FEATURE_NAMES,
    GroupStats,
    OrderRecord,
    build_matrix,
    records_from_frame,
)

NOW = datetime(2026, 8, 1, 14, 30, tzinfo=timezone.utc)


def _record(**kw) -> OrderRecord:
    base = dict(order_id="ord_1", created_at=NOW, amount=5000.0)
    base.update(kw)
    return OrderRecord(**base)


# ------------------------------------------------------------- features

def test_matrix_columns_match_the_declared_contract():
    stats = GroupStats()
    X = build_matrix([_record()], stats)
    assert list(X.columns) == FEATURE_NAMES
    assert X.shape == (1, len(FEATURE_NAMES))
    assert X.notna().all().all(), "no NaNs may reach the model"


def test_training_and_serving_paths_agree():
    """The whole reason features.py is shared.

    A dataframe row going through the training path and the same order going
    through the serving path must produce an identical vector. If this drifts,
    live scores stop matching the measured model and nothing else notices.
    """
    row = {
        "order_id": "ord_1",
        "created_at": NOW,
        "amount": 5000.0,
        "avs_result": "N",
        "cvv_result": "M",
        "three_ds_status": "authenticated",
        "billing_country": "IN",
        "shipping_country": "US",
        "bin_country": "IN",
        "billing_distance": 1800.0,
        "account_age_days": 12.0,
        "card_age_days": 8.0,
        "prior_disputes": 1,
        "is_guest": False,
        "txns_card_24h": 3,
        "txns_card_7d": 9,
        "txns_ip_24h": 2,
        "txns_device_24h": 1,
        "is_digital": True,
        "product_code": "digital_sub",
        "card_id": "card_000001",
        "payer_email_domain": "gmail.com",
        "recipient_email_domain": "yahoo.com",
        "disputed": 0,
    }
    stats = GroupStats()

    from_training = build_matrix(records_from_frame(pd.DataFrame([row])), stats)
    from_serving = build_matrix([_record(**{k: v for k, v in row.items() if k not in
                                            ("order_id", "created_at", "amount", "disputed")})], stats)

    np.testing.assert_allclose(from_training.to_numpy(), from_serving.to_numpy())


def test_three_ds_and_cvv_move_the_score_the_right_way():
    stats = GroupStats()
    X = build_matrix(
        [
            _record(cvv_result="N", three_ds_status="none"),
            _record(cvv_result="M", three_ds_status="authenticated"),
        ],
        stats,
    )
    assert X.loc[0, "cvv_match"] == 0.0
    assert X.loc[1, "cvv_match"] == 1.0
    assert X.loc[0, "three_ds_authenticated"] == 0.0
    assert X.loc[1, "three_ds_authenticated"] == 1.0


def test_missing_values_encode_as_sentinels_not_nan():
    """LightGBM tolerates NaN, but a sentinel plus an explicit missing flag
    lets the model learn that absence itself is signal."""
    stats = GroupStats()
    X = build_matrix([_record(account_age_days=None, avs_result=None)], stats)
    assert X.loc[0, "account_age_days"] == -1.0
    assert X.loc[0, "account_age_missing"] == 1.0
    assert X.loc[0, "avs_match"] == -1.0
    assert X.loc[0, "n_missing_fields"] >= 2


def test_group_stats_round_trip():
    df = pd.DataFrame({
        "amount": [100.0, 200.0, 300.0, 400.0],
        "card_id": ["a", "a", "b", "b"],
        "product_code": ["x", "y", "x", "y"],
    })
    stats = GroupStats().fit(df)
    restored = GroupStats.from_dict(stats.to_dict())
    assert restored.z("card", "a", 150.0) == pytest.approx(stats.z("card", "a", 150.0))
    assert restored.global_mean == pytest.approx(stats.global_mean)


# -------------------------------------------------------------- serving

def test_missing_artifacts_fall_back_rather_than_crash(monkeypatch, tmp_path):
    """A fresh clone has no model. /score must still answer."""
    ml.reset()
    monkeypatch.setattr(ml, "ARTIFACTS", tmp_path / "nothing")
    assert ml.load() is None
    ml.reset()


def test_feature_drift_refuses_to_serve(monkeypatch, tmp_path):
    """If the saved feature list no longer matches the code, scoring with
    shuffled columns would be silently wrong. Refusing is the safe failure."""
    import json

    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "model.txt").write_text("stub")
    (art / "calibration.json").write_text(json.dumps({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
    (art / "feature_spec.json").write_text(
        json.dumps({"model_version": "stale", "features": ["only_one"], "threshold": 0.4})
    )

    ml.reset()
    monkeypatch.setattr(ml, "ARTIFACTS", art)
    assert ml.load() is None, "a stale artifact must not be served"
    ml.reset()


def test_scoring_endpoint_works_with_or_without_a_model():
    """Same response shape either way — only model_version differs."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    r = client.post(
        "/score",
        json={"order_id": "ord_x", "amount": 9000, "cvv_result": "N", "is_guest": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["evidence_tier"] in {"standard", "enhanced"}
    assert body["capture"]
    assert not {"decline", "block", "reject"} & set(body)


# ----------------------------------------------------------- calibration

def test_calibration_round_trips_without_pickle():
    """Knots in, knots out, and the same numbers as sklearn.

    Regression: the calibrator used to ship as a .pkl. Loading it under a
    different scikit-learn version raised InconsistentVersionWarning — whose
    own wording is "may lead to invalid results" — and the triage rule
    multiplies these probabilities by rupee amounts.
    """
    import json

    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    from app.ml.calibration import Calibrator

    rng = np.random.default_rng(3)
    raw = rng.random(500)
    y = (rng.random(500) < raw).astype(int)

    sk = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    ours = Calibrator.from_sklearn(sk)

    probe = np.linspace(-0.2, 1.2, 60)          # deliberately outside the range
    np.testing.assert_allclose(ours.predict(probe), sk.predict(probe), atol=1e-12)

    restored = Calibrator(**{k: v for k, v in json.loads(
        json.dumps(ours.to_dict())).items() if k in ("x", "y", "method")})
    np.testing.assert_allclose(restored.predict(probe), ours.predict(probe))


def test_no_pickle_artifacts_are_shipped():
    """A repo meant to be cloned by strangers should not ask them to unpickle."""
    from app.ml.predict import ARTIFACTS

    assert not list(ARTIFACTS.glob("*.pkl")), "pickle artifacts must not be committed"
    assert (ARTIFACTS / "calibration.json").exists()
