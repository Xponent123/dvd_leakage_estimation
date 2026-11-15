#!/usr/bin/env python3
"""Dataset scatter trend plot from the 100k training CSVs; subsampled, fixed seed.
Run: bash run.sh analysis (or python3 code/scripts/18_dataset_trends.py from the repository root).
Writes reports/figures/analysis/trends_dataset.png and results/analysis/dataset_trends.txt."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/transistors"
FIG = ROOT / "reports/figures/analysis"; FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results/analysis"; OUT.mkdir(parents=True, exist_ok=True)
N = 4000; rng = np.random.RandomState(0)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6)); txt = []
def binned(a, x, y, color):
    b = pd.cut(x, np.quantile(x, np.linspace(0, 1, 13)), duplicates="drop")
    m = pd.DataFrame({"x": x, "y": y, "b": b}).groupby("b", observed=True).agg(x=("x", "median"), y=("y", "median"))
    a.plot(m.x, m.y, "-o", c=color, ms=4, lw=1.8); return m
for i, case in enumerate(["nmos_off", "pmos_off", "nmos_on", "pmos_on"]):
    d = pd.read_csv(D / f"{case}.csv"); s = d.iloc[rng.choice(len(d), N, replace=False)]
    col = f"C{i}"
    if case.endswith("off"):
        for a, x, xl, key in [(ax[0], s.temp, d.temp, "temp"), (ax[1], s.vth0.abs(), d.vth0.abs(), "|vth0|")]:
            ok = d.id.abs() > 0
            a.scatter(x, s.id.abs().clip(lower=1e-18), s=3, alpha=.2, c=col, label=case, rasterized=True)
            m = binned(a, xl[ok], d.id.abs()[ok], col)
            txt.append(f"{case} |Id| vs {key}: binned-median ratio last/first = {m.y.iloc[-1]/m.y.iloc[0]:.3g}")
    else:
        ax[2].scatter(s.toxe * 1e9, s.ig.abs().clip(lower=1e-18), s=3, alpha=.2, c=col, label=case, rasterized=True)
        m = binned(ax[2], d.toxe * 1e9, d.ig.abs(), col)
        txt.append(f"{case} |Ig| vs toxe: binned-median ratio last/first = {m.y.iloc[-1]/m.y.iloc[0]:.3g}")
for a, xl, t in zip(ax, ["Temperature (C)", "|vth0| (V)", "toxe (nm)"],
                    ["OFF |Id| vs temperature", "OFF |Id| vs |vth0|", "ON |Ig| vs toxe"]):
    a.set(yscale="log", xlabel=xl, ylabel="|current| (A)", title=t); a.grid(alpha=.3, which="both"); a.legend(markerscale=4)
plt.suptitle(f"Training-set trends ({N} random points per case, lines = binned medians; other parameters vary randomly)", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "trends_dataset.png", dpi=140); plt.close()
(OUT / "dataset_trends.txt").write_text("\n".join(txt) + "\n"); print("\n".join(txt))
