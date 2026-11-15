#!/usr/bin/env python3
"""Error analysis (gate + circuit) from stored result CSVs and results/error_analysis.json.
No SPICE, no retraining. Run: bash run.sh analysis (or python3 code/scripts/16_error_analysis.py from the repository root).
Writes results/analysis/*.csv, error_analysis_summary.txt and reports/figures/analysis/*.png."""
from pathlib import Path
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "results/analysis"; FIG = ROOT / "reports/figures/analysis"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
CORN = pd.read_csv(ROOT / "data/corners.csv")
PARAMS = ["vdd", "temp", "n_vth0", "p_vth0", "n_toxe", "p_toxe", "n_ndep", "p_ndep", "n_u0", "p_u0", "n_xj", "p_xj"]
CORN["p_vth0_abs"] = CORN.p_vth0.abs()
PARAMS = [p if p != "p_vth0" else "p_vth0_abs" for p in PARAMS]
CNAME = ["circuit", "input_bits"]
log = []
def P(*a):
    s = " ".join(str(x) for x in a); log.append(s); print(s)

g = pd.read_csv(R / "gate_predictions.csv", dtype={"input_bits": str})
g["gate_row"] = np.arange(len(g)); g["err_pct"] = g.relative_error * 100
g = g.drop(columns=["vdd", "temp"]).merge(CORN, on="corner_id", how="left")
c = pd.read_csv(R / "circuit_predictions.csv", dtype={"input_bits": str})
c["err_pct"] = c.relative_error * 100
c = c.drop(columns=["vdd", "temp"]).merge(CORN, on="corner_id", how="left")
tp = pd.read_csv(R / "gate_transistor_predictions.csv")

def analyse(d, name):
    P(f"\n=== {name}: n={len(d)}  median|err| {d.err_pct.abs().median():.3f}%  p95 {d.err_pct.abs().quantile(.95):.3f}%  "
      f"mean signed {d.err_pct.mean():+.3f}%  max {d.err_pct.abs().max():.3f}%")
    rows = []
    for p in PARAMS:
        rows.append(dict(level=name, parameter=p, spearman_abs_err=spearmanr(d[p], d.err_pct.abs())[0],
                         spearman_signed_err=spearmanr(d[p], d.err_pct)[0]))
    cor = pd.DataFrame(rows).sort_values("spearman_abs_err", key=abs, ascending=False)
    P(cor.round(3).to_string(index=False))
    d = d.copy(); d["tband"] = pd.cut(d.temp, [-40, 0, 25, 50, 75, 100, 125.01])
    tb = d.groupby("tband", observed=True).err_pct.agg(
        n="size", median_abs=lambda x: x.abs().median(), p95_abs=lambda x: x.abs().quantile(.95), mean_signed="mean").reset_index()
    tb["level"] = name; tb["tband"] = tb.tband.astype(str)
    P(tb.round(3).to_string(index=False))
    return cor, tb

cg, tg = analyse(g, "gate"); cc, tc = analyse(c, "circuit")
pd.concat([cg, cc]).to_csv(OUT / "error_param_correlations.csv", index=False)
pd.concat([tg, tc]).to_csv(OUT / "error_vs_temperature_bands.csv", index=False)

# per-gate-case summary
gs = g.groupby(CNAME).err_pct.agg(n="size", median_abs=lambda x: x.abs().median(), p95_abs=lambda x: x.abs().quantile(.95),
                                  mean_signed="mean", max_abs=lambda x: x.abs().max()).reset_index()
gs["corr_signed_vs_temp"] = [spearmanr(x.temp, x.err_pct)[0] for _, x in g.groupby(CNAME)]
gs.sort_values("median_abs", ascending=False).to_csv(OUT / "gate_error_by_case.csv", index=False)
P("\n=== per gate/input-pattern (sorted by median |err|)"); P(gs.sort_values("median_abs", ascending=False).round(3).to_string(index=False))

# worst-N tables
N = 20
keep = ["circuit", "input_bits", "corner_id", "temp", "vdd", "n_vth0", "p_vth0", "n_toxe", "p_toxe", "n_ndep", "p_ndep", "n_u0", "p_u0", "n_xj", "p_xj",
        "spice_leakage_A", "ml_leakage_A", "err_pct"]
wg = g.reindex(g.err_pct.abs().sort_values(ascending=False).index).head(N)[keep + ["gate_row"]]
wg.to_csv(OUT / "worst20_gate_cases.csv", index=False)
wc = c.reindex(c.err_pct.abs().sort_values(ascending=False).index).head(N)[keep]
wc.to_csv(OUT / "worst20_circuit_cases.csv", index=False)
P("\n=== worst 10 gate rows"); P(wg.head(10)[["circuit", "input_bits", "corner_id", "temp", "vdd", "n_vth0", "err_pct"]].round(4).to_string(index=False))
P("worst-20 composition:", wg.groupby(CNAME).size().to_dict())
P("\n=== worst 10 circuit rows"); P(wc.head(10)[["input_bits", "corner_id", "temp", "vdd", "n_vth0", "err_pct"]].round(4).to_string(index=False))

# ---- per-device-case attribution (ML power shares; SPICE per-device currents are not stored)
tp["p"] = tp.power_closed_W.abs()
tot = tp.groupby("gate_row").p.sum().rename("ptot")
tp = tp.join(tot, on="gate_row"); tp["share"] = tp.p / tp.ptot
sh_all = tp.groupby("case").share.mean()
worst_rows = set(g.reindex(g.err_pct.abs().sort_values(ascending=False).index).head(100).gate_row)
sh_w = tp[tp.gate_row.isin(worst_rows)].groupby("case").share.mean()
att = pd.DataFrame({"mean_power_share_all_rows": sh_all, "mean_power_share_worst100_rows": sh_w}).reset_index()
# off-device Vg offset (OFF models are trained only at Vg = Vb; Vg-Vb is constant in training)
off = tp[tp.case.str.endswith("off")].copy(); off["vg_off_V"] = (off.vg - off.vb).abs()
off["id_abs"] = off.id_ml.abs()
# calibrate effective subthreshold slope on one SPICE check at OR2 00, corner 94, device g_pi.
# 4.306e-7 A is Id at the recorded gate bias (= worst_gates[0] g_pi spice_currents_A.id in results/error_analysis.json);
# 3.6725e-7 A is a manual ngspice-42 spot check of the same device with Vg = VDD (not stored in the JSON).
r94 = off[(off.gate == "OR2") & (off.vector == 0) & (off.corner_id == 94) & (off.device == "g_pi")].iloc[0]
vt = 8.617333e-5 * (r94.temp + 273.15)
n_eff = r94.vg_off_V / (vt * np.log(4.306e-7 / 3.6725e-7))
P(f"\n=== OFF-model ignores Vg: spot-check g_pi OR2-00 corner 94: Vg offset {r94.vg_off_V*1e3:.2f} mV, T={r94.temp:.1f}C -> n_eff={n_eff:.2f} (Id ratio 1.172)")
off["extra_A"] = off.id_abs * (np.exp(off.vg_off_V / (n_eff * 8.617333e-5 * (off.temp + 273.15))) - 1)
ex = off.groupby("gate_row").extra_A.sum()
g["extra_A"] = g.gate_row.map(ex).fillna(0.0)
g["max_off_vg_mV"] = g.gate_row.map(off.groupby("gate_row").vg_off_V.max()).fillna(0) * 1e3
g["err_corrected_pct"] = (g.ml_leakage_A + g.extra_A - g.spice_leakage_A) / g.spice_leakage_A * 100
vg = g.groupby(CNAME).agg(median_max_off_vg_mV=("max_off_vg_mV", "median"), mean_err_pct=("err_pct", "mean"),
                          median_abs_err_pct=("err_pct", lambda x: x.abs().median()),
                          mean_est_extra_pct=("extra_A", lambda x: 0), mean_err_corrected_pct=("err_corrected_pct", "mean"),
                          median_abs_err_corrected_pct=("err_corrected_pct", lambda x: x.abs().median())).reset_index()
vg["mean_est_extra_pct"] = (vg.mean_err_corrected_pct - vg.mean_err_pct)
vg.to_csv(OUT / "off_model_vg_effect_by_case.csv", index=False)
P(vg.round(4).to_string(index=False))
rho = spearmanr(g.max_off_vg_mV, g.err_pct)[0]
P(f"Spearman(max OFF-device |Vg-Vb|, signed gate error) over all 3600 rows = {rho:+.3f}")
for gate, pat in [("OR2", "00"), ("AND2", "11")]:
    x = g[(g.circuit == gate) & (g.input_bits == pat)]
    P(f"{gate} {pat}: Pearson(signed err, max OFF Vg offset) = {pearsonr(x.err_pct, x.max_off_vg_mV)[0]:+.3f}; "
      f"corr(err, temp) = {pearsonr(x.err_pct, x.temp)[0]:+.3f}")
w1 = g.loc[g.err_pct.abs().idxmax()]
P(f"Worst row ({w1.circuit} {w1.input_bits} corner {int(w1.corner_id)}): err {w1.err_pct:+.3f}%; est. extra OFF current from Vg offset "
  f"{w1.extra_A:.3e} A = {w1.extra_A/w1.spice_leakage_A*100:.2f}% of SPICE leakage (SPICE check: 6.3e-8 A, 4.0%, ~95% of the error); corrected err {w1.err_corrected_pct:+.3f}%")
allc = g.err_pct.abs(); allcc = g.err_corrected_pct.abs()
P(f"All gate rows: median|err| {allc.median():.3f}% -> {allcc.median():.3f}% with first-order Vg correction; p95 {allc.quantile(.95):.3f}% -> {allcc.quantile(.95):.3f}%; max {allc.max():.3f}% -> {allcc.max():.3f}%")
P("NOTE: first-order exp(dVg/(n*kT/q)) correction, n calibrated on one SPICE spot check; an estimate, not a SPICE result.")
att.to_csv(OUT / "gate_device_case_attribution.csv", index=False)
P("\n=== ML power share by device case (all rows vs worst-100 rows)"); P(att.round(4).to_string(index=False))
# per-device attribution for the 20 worst: the device with the largest Vg offset and its power share
rows = []
for _, w in wg.iterrows():
    d = tp[tp.gate_row == w.gate_row].copy(); d["vg_off_V"] = (d.vg - d.vb).abs()
    d = d[d.case.str.endswith("off")]
    k = d.loc[d.vg_off_V.idxmax()]
    rows.append(dict(circuit=w.circuit, input_bits=w.input_bits, corner_id=int(w.corner_id), err_pct=w.err_pct, top_vg_offset_device=k.device,
                     device_case=k.case, vg_offset_mV=k.vg_off_V * 1e3, device_power_share=k.share,
                     dominant_case=tp[tp.gate_row == w.gate_row].groupby("case").share.sum().idxmax(),
                     est_vg_extra_pct_of_leak=g.loc[g.gate_row == w.gate_row, "extra_A"].iloc[0] / w.spice_leakage_A * 100))
pd.DataFrame(rows).to_csv(OUT / "worst20_gate_device_attribution.csv", index=False)
P("\n=== worst-20 gate rows: OFF device with largest Vg offset"); P(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- circuit error decomposition: composition (ideal standalone gates) vs ML, by temperature band
d = c.copy()
d["comp_pct"] = (d.standalone_spice_sum_A - d.spice_leakage_A) / d.spice_leakage_A * 100      # standalone SPICE gate sum vs full-circuit SPICE
d["ml_vs_sum_pct"] = (d.ml_leakage_A - d.standalone_spice_sum_A) / d.standalone_spice_sum_A * 100  # ML vs standalone SPICE gate sum
d["tband"] = pd.cut(d.temp, [-40, 0, 25, 50, 75, 100, 125.01])
dec = d.groupby("tband", observed=True).agg(n=("err_pct", "size"), mean_ml_vs_full_spice_pct=("err_pct", "mean"),
        mean_standalone_sum_vs_full_spice_pct=("comp_pct", "mean"), mean_ml_vs_standalone_sum_pct=("ml_vs_sum_pct", "mean")).reset_index()
dec["tband"] = dec.tband.astype(str)
tot = pd.DataFrame([dict(tband="all", n=len(d), mean_ml_vs_full_spice_pct=d.err_pct.mean(),
        mean_standalone_sum_vs_full_spice_pct=d.comp_pct.mean(), mean_ml_vs_standalone_sum_pct=d.ml_vs_sum_pct.mean())])
dec = pd.concat([dec, tot]); dec.to_csv(OUT / "circuit_error_decomposition_by_temp.csv", index=False)
w0 = d.loc[d.err_pct.abs().idxmax()]
P("\n=== circuit error decomposition (mean signed %, by temperature band)"); P(dec.round(3).to_string(index=False))
P(f"Worst circuit row (vector {w0.input_bits}, corner {int(w0.corner_id)}): ML vs full SPICE {w0.err_pct:+.3f}%, standalone sum vs full SPICE {w0.comp_pct:+.3f}%, ML vs standalone sum {w0.ml_vs_sum_pct:+.3f}%")
# ---- SPICE-based attribution of the five worst gate rows (from results/error_analysis.json)
ea = json.loads((ROOT / "results/error_analysis.json").read_text())
arows = []
for w in ea["worst_gates"]:
    devs = w["devices"]
    off = sum(x["gate_error_contribution_pp"] for x in devs if x["case"].endswith("_off"))
    on = sum(x["gate_error_contribution_pp"] for x in devs if x["case"].endswith("_on"))
    arows.append(dict(gate=w["gate"], input_bits=w["input_bits"], corner_id=w["corner_id"],
                      temp_C=float(CORN.loc[CORN.corner_id == w["corner_id"], "temp"].iloc[0]),
                      signed_error_percent=w["signed_error_percent"], dominant_case=w["dominant_case"],
                      dominant_contribution_pp=w["dominant_contribution_pp"],
                      other_off_pp=off - w["dominant_contribution_pp"], on_pp=on))
att5 = pd.DataFrame(arows); att5.to_csv(OUT / "worst5_gate_attribution.csv", index=False)
P("\n=== SPICE attribution of the five worst gate rows (results/error_analysis.json)"); P(att5.round(3).to_string(index=False))
# ---- supply-only versus source-power gate error
sup = (g.ml_supply_A - g.spice_supply_A).abs() / g.spice_supply_A.abs()
P(f"\nGate MAPE: supply-only current {100*sup.mean():.3f}% vs source power/VDD {g.err_pct.abs().mean():.3f}%")
(OUT / "error_analysis_summary.txt").write_text("\n".join(log) + "\n")

# ---- plots
def tplot(ax, d, title):
    ax.scatter(d.temp, d.err_pct, s=2, alpha=.25, rasterized=True)
    d = d.assign(b=pd.cut(d.temp, np.linspace(-40, 125, 12)))
    m = d.groupby("b", observed=True).agg(t=("temp", "mean"), e=("err_pct", "median"))
    ax.plot(m.t, m.e, "r-o", ms=3, label="binned median"); ax.axhline(0, c="k", lw=.6)
    ax.set(xlabel="Temperature (C)", ylabel="Signed error (%)", title=title); ax.grid(alpha=.3); ax.legend()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); tplot(ax[0], g, "Gate level"); tplot(ax[1], c, "Circuit level")
plt.tight_layout(); plt.savefig(FIG / "error_vs_temperature.png", dpi=140); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, d, t in [(ax[0], g, "Gate level"), (ax[1], c, "Circuit level")]:
    a.scatter(d.n_vth0, d.err_pct, s=2, alpha=.25, rasterized=True); a.axhline(0, c="k", lw=.6)
    a.set(xlabel="NMOS vth0 (V)", ylabel="Signed error (%)", title=f"{t}: error vs vth0"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "error_vs_vth0.png", dpi=140); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
lab = [f"{r.circuit} {r.input_bits} c{int(r.corner_id)}" for r in wg.itertuples()]
ax[0].barh(lab[::-1], wg.err_pct.values[::-1]); ax[0].set(xlabel="Signed error (%)", title="Worst 20 gate rows")
lab = [f"v{r.input_bits} c{int(r.corner_id)}" for r in wc.itertuples()]
ax[1].barh(lab[::-1], wc.err_pct.values[::-1], color="C1"); ax[1].set(xlabel="Signed error (%)", title="Worst 20 circuit rows")
for a in ax: a.grid(alpha=.3, axis="x"); a.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(FIG / "worst_cases_bar.png", dpi=140); plt.close()
print("done")
