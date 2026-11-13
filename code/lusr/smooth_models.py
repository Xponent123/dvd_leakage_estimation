"""Isolated smooth surrogate experiments; original submitted models are untouched."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .model_card import read_model_card
from .sampling import INPUT_COLUMNS


def load_train_validation(root: Path, case: str, data_root: Path | None = None):
    """Read only designated training/validation row values; skip test row targets."""
    data_root = root / "data" if data_root is None else Path(data_root)
    with (data_root / "splits" / f"{case}_split.csv").open() as stream:
        split_rows = list(csv.DictReader(stream))
    partitions = {name: [int(row["row_index"]) for row in split_rows if row["split"] == name]
                  for name in ("train", "validation")}
    allowed = set(partitions["train"]) | set(partitions["validation"])
    selected = {}
    with (data_root / "transistors" / f"{case}.csv").open() as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != [*INPUT_COLUMNS, "id", "ig", "is", "ib"]:
            raise ValueError("Unexpected transistor CSV header")
        for row_index, row in enumerate(reader):
            if row_index in allowed:
                selected[row_index] = np.asarray(row, dtype=float)
    result = {}
    for name, indices in partitions.items():
        values = np.asarray([selected[index] for index in indices])
        result[name] = (values[:, :12], values[:, 12:])
    return result


def strict_lbfgs(estimator: MLPRegressor, features: np.ndarray, targets: np.ndarray, iterations: int):
    """Refine existing weights with explicit strict SciPy stopping tolerances.

    Uses the pinned scikit-learn 1.9.1 MLP gradient implementation. Unlike its
    public solver wrapper, this specifies ftol=1e-14 instead of SciPy's default.
    """
    estimator._coef_indptr, estimator._intercept_indptr = [], []
    offset = 0
    for coefficient in estimator.coefs_:
        end = offset + coefficient.size
        estimator._coef_indptr.append((offset, end, coefficient.shape))
        offset = end
    for intercept in estimator.intercepts_:
        end = offset + intercept.size
        estimator._intercept_indptr.append((offset, end))
        offset = end
    packed = np.concatenate([x.ravel() for x in [*estimator.coefs_, *estimator.intercepts_]])
    activations = [features] + [None] * (estimator.n_layers_ - 1)
    deltas = [None] * (estimator.n_layers_ - 1)
    coefficient_gradients = [np.empty_like(x) for x in estimator.coefs_]
    intercept_gradients = [np.empty_like(x) for x in estimator.intercepts_]
    result = minimize(
        estimator._loss_grad_lbfgs, packed, method="L-BFGS-B", jac=True,
        args=(features, targets, None, activations, deltas, coefficient_gradients, intercept_gradients),
        options={"maxiter": iterations, "maxfun": iterations * 3, "ftol": 1e-14,
                 "gtol": 1e-9, "maxcor": 30, "maxls": 40},
    )
    estimator._unpack(result.x)
    estimator.loss_, estimator.n_iter_ = float(result.fun), int(result.nit)
    estimator.refinement_info_ = {"message": str(result.message), "status": int(result.status),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "loss": float(result.fun), "maximum_absolute_gradient": float(np.max(np.abs(result.jac)))}
    return estimator


def smooth_relative_objective(packed, shapes, features, reference, inputs, target_mean, target_scale, case, width_normalize=True, smoothing=1e-3, alpha=1e-8):
    """Smoothed raw-current MAPE and its exact tanh-network gradient.

    Smoothing is 0.001 relative-error units (0.1%), not a current floor.
    The denominator floor remains the declared 1e-18 A evaluation floor.
    """
    matrices, biases, offset = [], [], 0
    for shape in shapes:
        size = int(np.prod(shape)); matrices.append(packed[offset:offset + size].reshape(shape)); offset += size
    for shape in shapes:
        biases.append(packed[offset:offset + shape[1]]); offset += shape[1]
    activations = [features]
    for matrix, bias in zip(matrices[:-1], biases[:-1]):
        activations.append(np.tanh(activations[-1] @ matrix + bias))
    output = activations[-1] @ matrices[-1] + biases[-1]
    log_targets = output * target_scale + target_mean
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = recover_currents(inputs, log_targets, case, width_normalize)
    if not np.all(np.isfinite(prediction)):
        return 1e100, np.zeros_like(packed)
    denominator = np.maximum(np.abs(reference), 1e-18)
    relative = (prediction - reference) / denominator
    smooth_magnitude = np.hypot(relative, smoothing)
    loss = float(np.mean(smooth_magnitude - smoothing))
    loss += .5 * alpha * sum(float(np.sum(w * w)) for w in matrices) / len(features)
    derivative = relative / smooth_magnitude / denominator / reference.size
    vds = inputs[:, 3] - inputs[:, 4]
    conductance = np.exp(log_targets[:, 0])
    if width_normalize:
        conductance *= (inputs[:, 2] - 10e-9) / 100e-9
    dconductance = vds * conductance * (derivative[:, 0] - derivative[:, 2])
    dgate = prediction[:, 1] * (derivative[:, 1] - .5 * (derivative[:, 0] + derivative[:, 2]))
    dbody = prediction[:, 3] * (derivative[:, 3] - .5 * (derivative[:, 0] + derivative[:, 2]))
    delta = np.column_stack((dconductance, dgate, dbody)) * target_scale
    weight_gradients, bias_gradients = [None] * len(matrices), [None] * len(matrices)
    for layer in reversed(range(len(matrices))):
        weight_gradients[layer] = activations[layer].T @ delta + alpha * matrices[layer] / len(features)
        bias_gradients[layer] = delta.sum(axis=0)
        if layer:
            delta = (delta @ matrices[layer].T) * (1 - activations[layer] ** 2)
    return loss, np.concatenate([item.ravel() for item in [*weight_gradients, *bias_gradients]])


def canonical_features(inputs: np.ndarray, case: str, nominal: dict) -> np.ndarray:
    """Gate/body biases relative to the effective source, respecting N/P polarity.

    Swapping original drain and source leaves these features unchanged. Width
    uses the model's 5 nm WINT per edge (10 nm total). VDD is excluded as an
    independent feature: only terminal voltage differences enter DC physics.
    """
    x = np.asarray(inputs, dtype=float)
    vdd, temp, width, vd, vs, vg, vb, vth, toxe, ndep, u0, xj = x.T
    nmos = case.startswith("nmos")
    effective_source = np.minimum(vd, vs) if nmos else np.maximum(vd, vs)
    sign = 1 if nmos else -1
    vgs = sign * (vg - effective_source)
    vbs = sign * (vb - effective_source)
    vds = np.abs(vd - vs)
    kelvin = temp + 273.15
    width_effective = width - 10e-9
    if np.any(width_effective <= 0):
        raise ValueError("Width must exceed the model's 10 nm effective-width reduction")
    base = np.column_stack((
        vgs, vbs, vds, 300.0 / kelvin, np.log(width_effective / 100e-9),
        np.abs(vth) / abs(nominal["vth0"]), np.log(toxe / nominal["toxe"]),
        np.log(ndep / nominal["ndep"]), np.log(u0 / nominal["u0"]), np.log(xj / nominal["xj"]),
    ))
    # The OFF gate and body are tied, so Vgs=Vbs; remove exact redundancy.
    if case.endswith("off"):
        base = np.delete(base, 1, axis=1)
    return base


def positive_targets(inputs: np.ndarray, currents: np.ndarray, width_normalize: bool = True) -> np.ndarray:
    """Three symmetric positive quantities: averaged conductance and |Ig|, |Ib|."""
    vds = inputs[:, 3] - inputs[:, 4]
    if np.any(vds == 0):
        raise ValueError("Training conductance transform needs nonzero Vds")
    conductance = (currents[:, 0] - currents[:, 2]) / (2 * vds)
    targets = np.column_stack((conductance, np.abs(currents[:, 1]), np.abs(currents[:, 3])))
    if np.any(targets <= 0):
        raise ValueError("Positive-target experiment requires positive measured conductance/magnitudes")
    if width_normalize:
        targets = targets / ((inputs[:, 2] - 10e-9) / 100e-9)[:, None]
    return np.log(targets)


def recover_currents(inputs: np.ndarray, log_targets: np.ndarray, case: str, width_normalize: bool = True) -> np.ndarray:
    positive = np.exp(log_targets)
    if width_normalize:
        positive *= ((inputs[:, 2] - 10e-9) / 100e-9)[:, None]
    nmos = case.startswith("nmos")
    on = case.endswith("on")
    gate_sign = 1 if nmos == on else -1
    body_sign = -1 if nmos else 1
    gate, body = gate_sign * positive[:, 1], body_sign * positive[:, 2]
    channel = (inputs[:, 3] - inputs[:, 4]) * positive[:, 0]
    return np.column_stack((channel - .5 * (gate + body), gate, -channel - .5 * (gate + body), body))


@dataclass
class CanonicalSmoothModel:
    """A canonical symmetry-aware log-current surrogate; explicitly enforces KCL."""

    case: str
    nominal: dict
    seed: int = 20261022
    hidden: tuple[int, ...] = (96, 96)
    width_normalize: bool = True
    feature_scaler: StandardScaler | None = None
    target_scaler: StandardScaler | None = None
    estimator: MLPRegressor | None = None

    def initialize(self, inputs: np.ndarray, currents: np.ndarray, max_iter: int = 600):
        features = canonical_features(inputs, self.case, self.nominal)
        target = positive_targets(inputs, currents, self.width_normalize)
        self.feature_scaler = StandardScaler().fit(features)
        self.target_scaler = StandardScaler().fit(target)
        self.estimator = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", solver="adam",
            alpha=1e-7, batch_size=256, max_iter=max_iter, learning_rate_init=.001,
            early_stopping=True, validation_fraction=.1, n_iter_no_change=80, tol=1e-7,
            random_state=self.seed, warm_start=True)
        self.estimator.fit(self.feature_scaler.transform(features), self.target_scaler.transform(target))
        return self

    def refine(self, inputs: np.ndarray, currents: np.ndarray, iterations: int = 300):
        strict_lbfgs(self.estimator,
                     self.feature_scaler.transform(canonical_features(inputs, self.case, self.nominal)),
                     self.target_scaler.transform(positive_targets(inputs, currents, self.width_normalize)), iterations)
        return self

    def adapt_training_scalers(self, inputs: np.ndarray, currents: np.ndarray):
        """Refit train-only scalers to a new training pool, preserving predictions."""
        new_features = StandardScaler().fit(canonical_features(inputs, self.case, self.nominal))
        new_targets = StandardScaler().fit(positive_targets(inputs, currents, self.width_normalize))
        first_weights = self.estimator.coefs_[0].copy()
        self.estimator.intercepts_[0] += ((new_features.mean_ - self.feature_scaler.mean_) / self.feature_scaler.scale_) @ first_weights
        self.estimator.coefs_[0] *= (new_features.scale_ / self.feature_scaler.scale_)[:, None]
        ratio = self.target_scaler.scale_ / new_targets.scale_
        self.estimator.coefs_[-1] *= ratio[None, :]
        self.estimator.intercepts_[-1] = (self.estimator.intercepts_[-1] * self.target_scaler.scale_ + self.target_scaler.mean_ - new_targets.mean_) / new_targets.scale_
        self.feature_scaler, self.target_scaler = new_features, new_targets
        return self

    def refine_relative(self, inputs: np.ndarray, currents: np.ndarray, iterations: int = 300):
        if self.estimator.activation != "tanh":
            raise ValueError("Relative objective currently implements tanh hidden-layer gradients")
        shapes = [matrix.shape for matrix in self.estimator.coefs_]
        initial = np.concatenate([item.ravel() for item in [*self.estimator.coefs_, *self.estimator.intercepts_]])
        features = self.feature_scaler.transform(canonical_features(inputs, self.case, self.nominal))
        result = minimize(smooth_relative_objective, initial, jac=True, method="L-BFGS-B",
            args=(shapes, features, currents, inputs, self.target_scaler.mean_, self.target_scaler.scale_, self.case, self.width_normalize),
            options={"maxiter": iterations, "maxfun": iterations * 3, "ftol": 1e-14, "gtol": 1e-9, "maxcor": 30, "maxls": 40})
        offset = 0
        for i, shape in enumerate(shapes):
            size = int(np.prod(shape)); self.estimator.coefs_[i] = result.x[offset:offset + size].reshape(shape).copy(); offset += size
        for i, shape in enumerate(shapes):
            self.estimator.intercepts_[i] = result.x[offset:offset + shape[1]].copy(); offset += shape[1]
        self.estimator.refinement_info_ = {"objective": "mean(sqrt(relative_error**2+0.001**2)-0.001)",
            "message": str(result.message), "status": int(result.status), "iterations": int(result.nit),
            "function_evaluations": int(result.nfev), "loss": float(result.fun)}
        return self

    def predict_log_targets(self, inputs: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(self.estimator.predict(
            self.feature_scaler.transform(canonical_features(inputs, self.case, self.nominal))))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return recover_currents(inputs, self.predict_log_targets(inputs), self.case, self.width_normalize)


@dataclass
class GateEdgeResidualModel:
    """ON-gate current residual relative to the BSIM diffusion-edge baseline.

    At zero gate-to-source/drain voltage the baseline is exactly zero. Tiny
    gate-body tunneling at that point is omitted by this candidate and is
    explicitly assessed on independent boundary validation, never assumed exact.
    """
    case: str
    neighbors: int = 256
    style: str = "raw"
    regularization: float = 1e-12

    def fit(self, inputs, currents):
        from .physics_surrogate import edge_gate_currents, features
        source, drain = edge_gate_currents(inputs, self.case)
        baseline = np.abs(source + drain)
        if np.any(baseline <= 0):
            raise ValueError("Gate residual training requires nonzero diffusion-edge baseline")
        target = np.log(np.abs(currents[:, 1]) / baseline)
        f = features(inputs, self.case, self.style)
        self.center = f.mean(axis=0)
        self.estimator = RBFInterpolator(f - self.center, target[:, None],
            neighbors=self.neighbors, kernel="quintic", degree=2, smoothing=self.regularization)
        self.training_rows = len(inputs)
        return self

    def predict(self, inputs):
        from .physics_surrogate import edge_gate_currents, features
        source, drain = edge_gate_currents(inputs, self.case)
        residual = self.estimator(features(inputs, self.case, self.style) - self.center)[:, 0]
        sign = 1 if self.case.startswith("nmos") == self.case.endswith("on") else -1
        return sign * np.abs(source + drain) * np.exp(residual)


@dataclass
class GateCorrectedModel:
    """Replace only gate current, retaining learned channel asymmetry and KCL residual."""
    transistor_model: object
    gate_model: GateEdgeResidualModel

    @property
    def training_rows(self):
        base_count = self.transistor_model.training_rows
        gate_count = self.gate_model.training_rows
        if base_count != gate_count:
            raise ValueError("Base and gate models were fitted on different numbers of training rows")
        return base_count

    def predict(self, inputs):
        prediction = self.transistor_model.predict(inputs)
        gate = self.gate_model.predict(inputs)
        correction = gate - prediction[:, 1]
        prediction[:, 0] -= .5 * correction
        prediction[:, 2] -= .5 * correction
        prediction[:, 1] = gate
        return prediction
