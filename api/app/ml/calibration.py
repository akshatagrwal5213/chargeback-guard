"""Probability calibration, stored as data rather than as a pickle.

A fitted `IsotonicRegression` is a step function defined by two arrays of
knots. Predicting is linear interpolation between them — `np.interp` reproduces
sklearn's output exactly. So the artifact can be JSON.

Two reasons that matters here, both of which surfaced on a reviewer's machine
rather than mine:

1. **Version coupling.** A calibrator pickled under scikit-learn 1.8 and
   loaded under 1.6 raises InconsistentVersionWarning — sklearn's own words are
   "might lead to breaking code or invalid results". The triage rule multiplies
   these probabilities by rupee amounts, so silently wrong calibration is worse
   than a crash.

2. **Unpickling executes code.** Shipping a .pkl in a repo asks whoever clones
   it to run arbitrary code on load. For a project meant to be cloned and run
   by strangers, that is the wrong default.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Calibrator:
    """Isotonic calibration as knots. No sklearn required at serving time."""

    def __init__(self, x: list[float], y: list[float], method: str = "isotonic"):
        self.x = np.asarray(x, dtype="float64")
        self.y = np.asarray(y, dtype="float64")
        self.method = method
        if len(self.x) != len(self.y) or len(self.x) < 2:
            raise ValueError("calibration needs at least two matching knots")

    def predict(self, raw) -> np.ndarray:
        """Map raw scores to calibrated probabilities.

        np.interp clamps to the end knots outside the fitted range, which is
        the same behaviour as IsotonicRegression(out_of_bounds="clip").
        """
        values = np.asarray(raw, dtype="float64")
        return np.clip(np.interp(values, self.x, self.y), 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"method": self.method,
                "x": [float(v) for v in self.x],
                "y": [float(v) for v in self.y]}

    @classmethod
    def from_sklearn(cls, model) -> "Calibrator":
        return cls(list(model.X_thresholds_), list(model.y_thresholds_))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        data = json.loads(Path(path).read_text())
        return cls(data["x"], data["y"], data.get("method", "isotonic"))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
