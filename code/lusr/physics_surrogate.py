"""Smooth current surrogates with canonical MOS coordinates; no simulator at prediction.

Canonicalization only changes representation of actually simulated labels.
The four learned targets are Gd, Gs, |Ig|, |Ib|; source/drain exchange and PMOS
polarity are inverted exactly after prediction. Targets are natural logarithms.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from .sampling import INPUT_COLUMNS

INDEX = {name: i for i, name in enumerate(INPUT_COLUMNS)}


def canonical(inputs, case):
    x = np.asarray(inputs, dtype=float)
    pol = 1. if case.startswith("nmos") else -1.
    d = pol*(x[:, INDEX['vd']]-x[:, INDEX['vb']])
    s = pol*(x[:, INDEX['vs']]-x[:, INDEX['vb']])
    g = pol*(x[:, INDEX['vg']]-x[:, INDEX['vb']])
    return np.minimum(d, s), np.maximum(d, s), g, d < s, pol


def features(inputs, case, style="physical"):
    x = np.asarray(inputs, dtype=float)
    lo, hi, gate, swap, pol = canonical(x, case)
    kelvin = x[:, INDEX['temp']]+273.15
    vt = kelvin*8.617333262e-5
    width = x[:, INDEX['w']]-1e-8
    vth = np.abs(x[:, INDEX['vth0']])
    tox = x[:, INDEX['toxe']]*1e9
    doping = x[:, INDEX['ndep']]
    mobility = x[:, INDEX['u0']]
    xj = x[:, INDEX['xj']]*1e9
    # These fixed scales are dimensional conditioning constants, not fitted
    # physical parameters. OFF cases have no supply-dependent gate voltage.
    values = [lo, hi, (kelvin-300)/300, np.log(width/1e-7)*.1,
              vth, 1/tox, np.log(doping/1e18)*.1, np.log(mobility/.01)*.1,
              np.log(xj/10)*.1]
    if case.endswith("on"):
        values.insert(2, gate)
    if style == "physical":
        values[0] = lo*300/kelvin
        values[1] = (hi-lo)*300/kelvin
        values[4 if case.endswith("off") else 5] = vth*300/kelvin
    elif style == "junction":
        values[0] = lo-.15*np.exp(-lo/vt)
        values[1] = hi-.15*np.exp(-hi/vt)
    elif style != "raw":
        raise ValueError(style)
    return np.column_stack(values)


def encode(inputs, currents, case):
    x, y = np.asarray(inputs), np.array(currents, dtype=float, copy=True)
    lo, hi, gate, swap, pol = canonical(x, case)
    y *= pol
    y[swap, 0], y[swap, 2] = y[swap, 2].copy(), y[swap, 0].copy()
    delta = hi-lo
    if np.any(delta == 0):
        raise ValueError("Training conductance requires nonzero drain-source voltage")
    common = .5*(y[:, 1]+y[:, 3])
    width = (x[:, INDEX['w']]-1e-8)/1e-7
    z = np.column_stack(((y[:, 0]+common)/delta/width,
                         -(y[:, 2]+common)/delta/width,
                         np.abs(y[:, 1])/width, np.abs(y[:, 3])))
    if np.any(z <= 0) or not np.all(np.isfinite(z)):
        raise ValueError("Positive smooth-target premise does not hold for supplied training data")
    return np.log(z)


def decode(inputs, targets, case):
    x = np.asarray(inputs)
    lo, hi, gate, swap, pol = canonical(x, case)
    z = np.exp(targets)
    width = (x[:, INDEX['w']]-1e-8)/1e-7
    ig = z[:, 2]*width*(1 if case.endswith("on") else -1)
    ib = -z[:, 3]
    common = .5*(ig+ib)
    y = np.column_stack(((hi-lo)*z[:, 0]*width-common, ig,
                         -(hi-lo)*z[:, 1]*width-common, ib))
    y[swap, 0], y[swap, 2] = y[swap, 2].copy(), y[swap, 0].copy()
    return y*pol


@dataclass
class PhysicsSurrogate:
    case: str
    method: str = "polynomial"
    degree: int = 4
    regularization: float = 1e-8
    neighbors: int = 128
    kernel: str = "quintic"
    style: str = "physical"

    def _encode(self, inputs, currents):
        return encode(inputs, currents, self.case)

    def _decode(self, inputs, targets):
        return decode(inputs, targets, self.case)

    def fit(self, inputs, currents):
        f = features(inputs, self.case, self.style)
        target = self._encode(inputs, currents)
        self.center = np.mean(f, axis=0)
        # Polynomial variables span roughly [-1,1]; local RBF retains physical
        # scaling so small process variations are not treated as huge distances.
        self.scale = np.max(np.abs(f-self.center), axis=0) if self.method == "polynomial" else np.ones(f.shape[1])
        self.scale = np.maximum(self.scale, 1e-15)
        f = (f-self.center)/self.scale
        if self.method == "polynomial":
            self.transform = PolynomialFeatures(self.degree)
            p = self.transform.fit_transform(f)
            self.estimator = Ridge(alpha=self.regularization, fit_intercept=False).fit(p, target)
        elif self.method == "rbf":
            self.estimator = RBFInterpolator(f, target, neighbors=self.neighbors,
                                            smoothing=self.regularization, kernel=self.kernel,
                                            degree=self.degree)
        else:
            raise ValueError(self.method)
        self.training_rows = len(inputs)
        return self

    def predict(self, inputs):
        f = (features(inputs, self.case, self.style)-self.center)/self.scale
        if self.method == "polynomial":
            target = self.estimator.predict(self.transform.transform(f))
        else:
            target = self.estimator(f)
        return self._decode(inputs, target)


def edge_gate_currents(inputs, case):
    """BSIM4 gate-to-diffusion baseline (manual4.13–4.18), canonical polarity.

    This is a fixed physics normalization; residuals are learned exclusively
    from transistor training labels. Constants are from assigned32nm_MGK card.
    Flat-band potential uses nominal model temperature, as BSIM4v5 does.
    """
    x=np.asarray(inputs)
    lo,hi,g,swap,pol=canonical(x,case)
    tox=x[:,INDEX['toxe']]
    width=x[:,INDEX['w']]-1e-8
    if case.startswith('nmos'):
        a,b,c=.020014,.0027432,.002
        capital_a,capital_b,reference=4.97232e-7,7.45669e11,7.5e-10
    else:
        a,b,c=.011942,.0012217,.0008
        capital_a,capital_b,reference=3.42537e-7,1.16645e12,7.7e-10
    flat=300.15*8.617333262e-5*np.log(500.)
    def edge(v):
        effective=np.sqrt((v-flat)**2+1e-4)
        return width*1.95e-9*capital_a*(reference/tox)/tox**2*v*effective*np.exp(-capital_b*tox*(a-b*effective)*(1+c*effective))
    return edge(g-lo),edge(g-hi)


def physics_normalizers(inputs,case,baseline_kind='gate'):
    x=np.asarray(inputs)
    lo,hi,g,swap,pol=canonical(x,case)
    width=(x[:,INDEX['w']]-1e-8)/1e-7
    scale=np.ones((len(x),4))
    if case.endswith('off'):
        source,drain=edge_gate_currents(x,case)
        scale[:,2]=np.abs(source+drain)/width
        vt=(x[:,INDEX['temp']]+273.15)*8.617333262e-5
        scale[:,3]=5e-15*(-np.expm1(-lo/vt)-np.expm1(-hi/vt))
        if baseline_kind=='edge':
            delta=hi-lo
            edge_g=np.empty(len(x))
            regular=delta>=1e-6
            edge_g[regular]=(source[regular]-drain[regular])/(2*delta[regular])
            if (~regular).any():
                local=x[~regular].copy()
                center=.5*(lo[~regular]+hi[~regular])
                local[:,INDEX['vd']]=local[:,INDEX['vb']]+pol*(center+.5e-6)
                local[:,INDEX['vs']]=local[:,INDEX['vb']]+pol*(center-.5e-6)
                ls,ld=edge_gate_currents(local,case)
                edge_g[~regular]=(ls-ld)/2e-6
            scale[:,0]=edge_g/width
            scale[:,1]=edge_g/width
        elif baseline_kind!='gate':
            raise ValueError(baseline_kind)
    return scale


@dataclass
class PhysicsNormalizedSurrogate(PhysicsSurrogate):
    """RBF/polynomial prediction of residuals relative to fixed MOS baselines."""
    baseline_kind: str = 'gate'

    def _encode(self,inputs,currents):
        baseline=physics_normalizers(inputs,self.case,self.baseline_kind)
        if np.any(baseline<=0):
            raise ValueError('Nonzero training bias required for log-residual normalization')
        return encode(inputs,currents,self.case)-np.log(baseline)

    def _decode(self,inputs,targets):
        baseline=physics_normalizers(inputs,self.case,self.baseline_kind)
        # log0 occurs only at the physically prescribed exactzero boundary.
        with np.errstate(divide='ignore'):
            result=decode(inputs,targets+np.log(baseline),self.case)
        array=np.asarray(inputs)
        zero=np.ptp(array[:,[INDEX[n] for n in ('vd','vs','vg','vb')]],axis=1)==0
        result[zero]=0.
        return result
