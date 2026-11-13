"""Tree-family comparison using the same current representation as smooth RBFs."""
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from .physics_surrogate import features, encode, decode


@dataclass
class CanonicalTreeModel:
    case: str
    family: str
    seed: int = 20261024
    jobs: int = 2
    style: str = 'junction'

    def fit(self, inputs, currents):
        if self.family in ('random_forest', 'extra_trees'):
            cls = RandomForestRegressor if self.family == 'random_forest' else ExtraTreesRegressor
            self.estimator = cls(n_estimators=128, min_samples_leaf=2,
                                 max_features=1., random_state=self.seed, n_jobs=self.jobs)
        elif self.family == 'hist_gradient_boosting':
            self.estimator = MultiOutputRegressor(HistGradientBoostingRegressor(
                max_iter=400, learning_rate=.08, max_leaf_nodes=63,
                min_samples_leaf=15, l2_regularization=.01,
                early_stopping=False, random_state=self.seed), n_jobs=1)
        else:
            raise ValueError(self.family)
        self.estimator.fit(features(inputs, self.case, self.style), encode(inputs, currents, self.case))
        self.training_rows = len(inputs)
        return self

    def predict(self, inputs):
        return decode(inputs, self.estimator.predict(features(inputs, self.case, self.style)), self.case)


def power_equivalent(inputs, currents):
    """Source-closed dissipated power/VDD, in A; identical to final aggregation."""
    from .sampling import INPUT_COLUMNS
    x, y = np.asarray(inputs), np.asarray(currents)
    col = {name: x[:, i] for i, name in enumerate(INPUT_COLUMNS)}
    return ((col['vd']-col['vs'])*y[:, 0] + (col['vg']-col['vs'])*y[:, 1]
            + (col['vb']-col['vs'])*y[:, 3])/col['vdd']


def boundary_metrics(inputs, reference, prediction):
    from .sampling import INPUT_COLUMNS
    x = np.asarray(inputs)
    voltages = x[:, [INPUT_COLUMNS.index(n) for n in ('vd', 'vg', 'vs', 'vb')]]
    zero = np.ptp(voltages, axis=1) == 0
    ref, pred = power_equivalent(x, reference), power_equivalent(x, prediction)
    error = np.abs(pred-ref)/np.maximum(np.abs(ref), 1e-18)
    return {'boundary_power_mape_pct': float(100*np.mean(error[~zero])),
            'boundary_power_median_pct': float(100*np.median(error[~zero])),
            'boundary_power_p95_pct': float(100*np.percentile(error[~zero], 95)),
            'boundary_nonzero_bias_rows': int(np.sum(~zero)),
            'boundary_floor_count': int(np.sum((np.abs(ref)<1e-18)&(~zero))),
            'zero_bias_rows': int(np.sum(zero)),
            'zero_bias_max_abs_predicted_power_A': float(np.max(np.abs(pred[zero]))) if np.any(zero) else 0.}
