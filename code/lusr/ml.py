"""Training-only preprocessing and signed four-terminal current regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .sampling import CHANNEL_LENGTH, INPUT_COLUMNS


CURRENT_COLUMNS = ("id", "ig", "is", "ib")
ERROR_FLOOR_A = 1e-18
MODEL_FAMILIES = ("random_forest", "extra_trees", "hist_gradient_boosting", "mlp")
DERIVED_FEATURES = (
    "vds", "vgs", "vbs", "abs_vds", "vgd", "vbd",
    "min_vd_vs", "max_vd_vs", "temperature_K", "inverse_temperature_K", "w_over_l",
)


def physical_features(inputs: np.ndarray) -> np.ndarray:
    """Deterministic voltage differences and temperature features; no target data."""
    array = np.asarray(inputs, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(INPUT_COLUMNS) or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected finite (n, {len(INPUT_COLUMNS)}) input array")
    temp, width, vd, vs, vg, vb = (array[:, INPUT_COLUMNS.index(name)] for name in ("temp", "w", "vd", "vs", "vg", "vb"))
    kelvin = temp + 273.15
    if np.any(kelvin <= 0):
        raise ValueError("Absolute temperature must be positive")
    derived = np.column_stack((
        vd - vs, vg - vs, vb - vs, np.abs(vd - vs), vg - vd, vb - vd,
        np.minimum(vd, vs), np.maximum(vd, vs), kelvin, 1 / kelvin, width / CHANNEL_LENGTH,
    ))
    return np.column_stack((array, derived))


@dataclass
class SignedCurrentModel:
    """Serializable wrapper; `predict` returns raw signed Id, Ig, Is, Ib in A.

    Feature scaling and every data-dependent target scale are fitted on the
    training partition only. Floor-scale variants use the explicit transform
    constant 1e-18 A (or S for conductance targets). No KCL correction is silently applied. Circuit integration
    may explicitly close the source current and report the raw KCL residual.
    """

    estimator: Any
    family: str
    case: str
    feature_scaler: StandardScaler | None = None
    current_scale: np.ndarray | None = None
    training_rows: int = 0
    variant: str = "baseline"
    target_mode: str = "current"
    target_scale_mode: str = "median"
    standardize_targets: bool = False
    training_target_scale: np.ndarray | None = None
    target_scaler: StandardScaler | None = None

    def fit(self, inputs: np.ndarray, currents: np.ndarray) -> "SignedCurrentModel":
        targets = np.asarray(currents, dtype=np.float64)
        if targets.ndim != 2 or targets.shape != (len(inputs), 4) or not np.all(np.isfinite(targets)):
            raise ValueError("Expected finite (n, 4) training currents")
        features = physical_features(inputs)
        self.feature_scaler = StandardScaler().fit(features)
        self.current_scale = np.maximum(np.median(np.abs(targets), axis=0), ERROR_FLOOR_A)
        represented = targets.copy()
        if self.target_mode == "conductance":
            vds = inputs[:, INPUT_COLUMNS.index("vd")] - inputs[:, INPUT_COLUMNS.index("vs")]
            if np.any(vds == 0):
                raise ValueError("Conductance training transform requires nonzero sampled Vd-Vs")
            gate_body_half = 0.5 * (targets[:, 1] + targets[:, 3])
            represented[:, 0] = (targets[:, 0] + gate_body_half) / vds
            represented[:, 2] = -(targets[:, 2] + gate_body_half) / vds
        elif self.target_mode != "current":
            raise ValueError(f"Unknown target mode: {self.target_mode}")
        if self.target_scale_mode == "median":
            self.training_target_scale = np.maximum(np.median(np.abs(represented), axis=0), ERROR_FLOOR_A)
        elif self.target_scale_mode == "floor":
            self.training_target_scale = np.full(4, ERROR_FLOOR_A)
        else:
            raise ValueError(f"Unknown scale mode: {self.target_scale_mode}")
        transformed = np.arcsinh(represented / self.training_target_scale)
        if self.standardize_targets:
            self.target_scaler = StandardScaler().fit(transformed)
            transformed = self.target_scaler.transform(transformed)
        self.estimator.fit(self.feature_scaler.transform(features), transformed)
        self.training_rows = len(inputs)
        return self

    def transform_currents(self, currents: np.ndarray) -> np.ndarray:
        if self.current_scale is None:
            raise ValueError("Model is not fitted")
        return np.arcsinh(np.asarray(currents, dtype=np.float64) / self.current_scale)

    def predict_transformed(self, inputs: np.ndarray) -> np.ndarray:
        """Common current-space asinh predictions, independent of training representation."""
        return self.transform_currents(self.predict(inputs))

    def _predict_representation(self, inputs: np.ndarray) -> np.ndarray:
        if self.feature_scaler is None or self.current_scale is None:
            raise ValueError("Model is not fitted")
        prediction = np.asarray(self.estimator.predict(self.feature_scaler.transform(physical_features(inputs))))
        if prediction.shape != (len(inputs), 4) or not np.all(np.isfinite(prediction)):
            raise ValueError("Non-finite or incorrectly shaped transformed predictions")
        if self.target_scaler is not None:
            prediction = self.target_scaler.inverse_transform(prediction)
        with np.errstate(over="raise", invalid="raise"):
            return np.sinh(prediction) * self.training_target_scale

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        currents = self._predict_representation(inputs)
        if self.target_mode == "conductance":
            array = np.asarray(inputs, dtype=np.float64)
            vds = array[:, INPUT_COLUMNS.index("vd")] - array[:, INPUT_COLUMNS.index("vs")]
            gate_body_half = 0.5 * (currents[:, 1] + currents[:, 3])
            currents[:, 0] = vds * currents[:, 0] - gate_body_half
            currents[:, 2] = -vds * currents[:, 2] - gate_body_half
        if not np.all(np.isfinite(currents)):
            raise ValueError("Non-finite predicted currents")
        return currents


def make_candidate(family: str, case: str, seed: int, jobs: int = 2, smoke: bool = False, variant: str = "baseline") -> SignedCurrentModel:
    """Four explicitly configured families; validation chooses the final family."""
    if family == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=12 if smoke else 320, min_samples_leaf=1,
            max_features=1.0, max_depth=None, n_jobs=jobs, random_state=seed,
        )
    elif family == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=12 if smoke else 400, min_samples_leaf=1,
            max_features=1.0, max_depth=None, n_jobs=jobs, random_state=seed,
        )
    elif family == "hist_gradient_boosting":
        estimator = MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=12 if smoke else 500, learning_rate=0.08,
            max_leaf_nodes=63, min_samples_leaf=15, l2_regularization=0.01,
            early_stopping=False, random_state=seed,
        ), n_jobs=1)
    elif family == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=(24, 24) if smoke else (160, 160, 80),
            activation="tanh" if "tanh" in variant else "relu", solver="adam", alpha=1e-4 if variant == "baseline" else 1e-6,
            batch_size=64 if smoke else 256, learning_rate_init=1e-3,
            max_iter=15 if smoke else (1200 if variant == "baseline" else 1800), early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=60 if variant == "baseline" else 100, tol=1e-5 if variant == "baseline" else 1e-7,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model family {family!r}; expected one of {MODEL_FAMILIES}")
    if variant != "baseline" and family != "mlp":
        raise ValueError("Tuning variants are defined for the MLP family only")
    return SignedCurrentModel(
        estimator, family, case, variant=variant,
        target_mode="conductance" if "conductance" in variant else "current",
        target_scale_mode="floor" if "floor" in variant else "median",
        standardize_targets=variant != "baseline",
    )


def regression_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Computational metrics with an explicit 1e-18 A relative-error floor.

    Accuracy = 100*(1-mean(abs(pred-reference)/max(abs(reference),floor))).
    It can be negative and is never clipped. This is not classification accuracy.
    Matrix metrics pool relative errors and average per-terminal raw-space R².
    """
    truth, estimate = np.asarray(reference), np.asarray(predicted)
    if truth.shape != estimate.shape or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise ValueError("Reference and predicted currents must be finite and shape matched")
    absolute_error = np.abs(estimate - truth)
    relative_error = absolute_error / np.maximum(np.abs(truth), ERROR_FLOOR_A)
    mean_relative = float(np.mean(relative_error))
    return {
        "values": int(truth.size),
        "raw_r2": float(r2_score(truth, estimate, multioutput="uniform_average")),
        "mae_A": float(np.mean(absolute_error)),
        "rmse_A": float(np.sqrt(np.mean(np.square(estimate - truth)))),
        "median_relative_error_pct": float(100 * np.median(relative_error)),
        "p95_relative_error_pct": float(100 * np.percentile(relative_error, 95)),
        "mape_pct": 100 * mean_relative,
        "mean_accuracy_pct": 100 * (1 - mean_relative),
        "relative_error_floor_A": ERROR_FLOOR_A,
        "floor_reference_count": int(np.count_nonzero(np.abs(truth) < ERROR_FLOOR_A)),
    }


def kcl_metrics(currents: np.ndarray) -> dict[str, float]:
    residual = np.sum(currents, axis=1)
    return {
        "median_abs_kcl_residual_A": float(np.median(np.abs(residual))),
        "max_abs_kcl_residual_A": float(np.max(np.abs(residual))),
    }
