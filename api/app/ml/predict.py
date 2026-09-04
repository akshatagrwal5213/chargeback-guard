"""Serving. Loads the trained artifacts and scores one order at a time.

Goes through the same `features.build_matrix` as training — that is the whole
point of the module split. If a feature changes, both paths change together or
neither does.

Falls back to the documented heuristic when no model has been trained, so the
API works on a fresh clone.
"""
from __future__ import annotations

import json
import logging
import threading
import warnings
from pathlib import Path

import numpy as np

from .calibration import Calibrator
from .features import FEATURE_NAMES, GroupStats, OrderRecord, build_matrix

log = logging.getLogger(__name__)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
_lock = threading.Lock()


class Model:
    """Loaded model, calibrator, group stats and threshold."""

    def __init__(self, booster, calibrator, stats: GroupStats, spec: dict, metrics: dict):
        self.booster = booster
        self.calibrator = calibrator
        self.stats = stats
        self.version = spec.get("model_version", "unknown")
        self.threshold = float(spec.get("threshold", 0.5))
        self.metrics = metrics
        self._explainer = None

    # SHAP is expensive to construct, so build it on first use rather than
    # paying for it at import when most requests never need explanations.
    def explainer(self):
        if self._explainer is None:
            with _lock:
                if self._explainer is None:
                    import shap

                    self._explainer = shap.TreeExplainer(self.booster)
        return self._explainer

    @staticmethod
    def _normalise_shap(values):
        """shap returns an array or a per-class list depending on the
        shap/lightgbm pairing. Both shapes are handled, so the warning it
        emits about the change is noise — and four copies of it in a test run
        is the kind of thing that makes a reviewer wonder what else is off."""
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        return values

    def score(self, record: OrderRecord) -> tuple[float, float]:
        X = build_matrix([record], self.stats)
        raw = float(self.booster.predict(X)[0])
        calibrated = float(self.calibrator.predict([raw])[0])
        return calibrated, raw

    def explain(self, record: OrderRecord, top: int = 5) -> list[dict]:
        """Per-order SHAP contributions.

        Merchant-internal only. These describe exactly which signals raise a
        score, so they never go to a customer-facing surface.
        """
        X = build_matrix([record], self.stats)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*TreeExplainer shap values output.*",
                category=UserWarning,
            )
            values = self.explainer().shap_values(X)
        row = self._normalise_shap(values)[0]

        order = np.argsort(np.abs(row))[::-1][:top]
        return [
            {
                "feature": FEATURE_NAMES[i],
                "contribution": round(float(row[i]), 5),
                "value": round(float(X.iloc[0, i]), 4),
                "direction": "raises" if row[i] > 0 else "lowers",
            }
            for i in order
            if abs(row[i]) > 1e-6
        ]


_model: Model | None = None
_tried = False


def load() -> Model | None:
    """Load once, cache. Returns None when nothing has been trained yet."""
    global _model, _tried
    if _model is not None or _tried:
        return _model

    with _lock:
        if _model is not None or _tried:
            return _model
        _tried = True

        model_path = ARTIFACTS / "model.txt"
        spec_path = ARTIFACTS / "feature_spec.json"
        calib_path = ARTIFACTS / "calibration.json"

        if not (model_path.exists() and spec_path.exists() and calib_path.exists()):
            log.info(
                "No trained model in %s — /score uses the documented heuristic. "
                "Train with: python -m app.ml.train",
                ARTIFACTS.name,
            )
            return None

        try:
            try:
                import lightgbm as lgb
            except ImportError:
                log.info("lightgbm not installed — /score uses the heuristic. "
                         "Run: make install-ml")
                return None
            except OSError as exc:
                from ._deps import openmp_hint
                log.warning("lightgbm present but unusable (%s) — /score uses "
                            "the heuristic. %s", exc.__class__.__name__, openmp_hint())
                return None

            spec = json.loads(spec_path.read_text())

            # The feature order is part of the contract. A mismatch here means
            # the artifact predates a features.py change, and silently scoring
            # with shuffled columns would be far worse than refusing.
            saved = spec.get("features", [])
            if saved != FEATURE_NAMES:
                log.error(
                    "Feature mismatch: artifact has %d features, code expects %d. "
                    "Retrain before serving. Falling back to the heuristic.",
                    len(saved),
                    len(FEATURE_NAMES),
                )
                return None

            booster = lgb.Booster(model_file=str(model_path))
            calibrator = Calibrator.load(calib_path)
            stats = GroupStats.from_dict(spec.get("group_stats", {}))

            metrics_path = ARTIFACTS / "metrics.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

            _model = Model(booster, calibrator, stats, spec, metrics)
            log.info(
                "Model %s loaded (threshold %.2f, precision %.3f, recall %.3f)",
                _model.version,
                _model.threshold,
                metrics.get("honest", {}).get("precision", float("nan")),
                metrics.get("honest", {}).get("recall", float("nan")),
            )
            return _model
        except Exception:
            log.exception("Failed to load model — falling back to the heuristic")
            return None


def reset() -> None:
    """Drop the cache so a freshly trained model is picked up. Used by tests."""
    global _model, _tried
    with _lock:
        _model = None
        _tried = False
