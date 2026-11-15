#!/usr/bin/env python3
"""Per-case, per-terminal test metrics from stored test predictions (no retraining).
Run: bash run.sh analysis (or python3 code/scripts/17_terminal_metrics.py from the repository root). Output: results/analysis/ml_per_case_terminal_metrics.csv"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "results/analysis"; OUT.mkdir(parents=True, exist_ok=True)
FLOOR = 1e-18  # same relative-error floor as results/transistor_test_terminals.csv
def r2(y, p):
    ss = ((y - p) ** 2).sum(); st = ((y - y.mean()) ** 2).sum()
    return 1 - ss / st
rows = []
for case in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]:
    d = pd.read_csv(R / f"test_predictions/{case}.csv")
    for t in ["id", "ig", "is", "ib"]:
        y, p = d[f"{t}_spice"].values, d[f"{t}_ml"].values
        rel = np.abs(p - y) / np.maximum(np.abs(y), FLOOR) * 100
        ly, lp = np.log10(np.maximum(np.abs(y), FLOOR)), np.log10(np.maximum(np.abs(p), FLOOR))
        rows.append(dict(case=case, terminal=t, n=len(d), median_rel_err_pct=np.median(rel), p95_rel_err_pct=np.percentile(rel, 95),
                         mean_accuracy_pct=100 - rel.mean(), r2_log10_abs=r2(ly, lp), r2_linear=r2(y, p),
                         median_abs_log10_err=np.median(np.abs(ly - lp)), sign_mismatch_frac=float(np.mean(np.sign(y) != np.sign(p)))))
df = pd.DataFrame(rows); df.to_csv(OUT / "ml_per_case_terminal_metrics.csv", index=False)
# consistency check vs stored table
ref = pd.read_csv(R / "transistor_test_terminals.csv").set_index(["case", "terminal"])
m = df.set_index(["case", "terminal"])
print("max |median diff| vs stored table (pct pts):", (m.median_rel_err_pct - ref.median_relative_error_pct).abs().max())
print("max |R2 linear diff| vs stored raw_r2:", (m.r2_linear - ref.raw_r2).abs().max())
print(df.round(5).to_string(index=False))
