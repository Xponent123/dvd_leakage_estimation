# ML-driven PVT-aware leakage estimation

**Team 8:** Dhimant Bhuva (2023102063), Jal Parikh (2023102066), Garima Mittal (2023102069), Krishna Goel (2023112009).  
**Repository:** https://github.com/Xponent123/dvd_leakage_estimation

Machine-learning leakage estimation for the 32nm_MGK technology (nominal VDD 0.9 V), evaluated hierarchically: four single-FET models (NMOS/PMOS, ON/OFF), five logic gates, and an 8-bit parity checker built from 28 NAND2 cells (112 MOSFETs). All reference data come from ngspice 42. The final dataset has 100,000 uniform samples per transistor case, split 70,000 / 15,000 / 15,000 (seed 20261024). The datasets have 100,000 rows per case; an earlier 10,000-row run with MLP models reached 94.30% circuit accuracy (`results/analysis/baseline_10k_summary.json`), and the model was changed at the same time as the dataset size. Model families compared: random forest, extra trees, histogram gradient boosting, local quintic RBF.

| Evaluation | Operating points | Median error | 95th-percentile error | Mean accuracy |
|---|---:|---:|---:|---:|
| Five gates, all 18 input cases | 3,600 | 0.25604% | 1.11632% | 99.63502% |
| Complete 8-bit parity checker | 51,200 | 0.42348% | 1.21420% | 99.48968% |

Mean accuracy is `100 x (1 - mean absolute relative error)` with a `1e-18 A` denominator floor. Individual terminal currents (notably OFF/body) can have larger errors; see the report.

## Layout

```
run.sh                 entry point: setup | reproduce | analysis
code/lusr/             library (sampling, SPICE, models, circuits)
code/scripts/          numbered pipeline scripts + run_revision.sh
code/experiments/      physics-surrogate training and boundary validation
netlists/              single-FET, gate and parity-checker netlists
data/                  transistor datasets, splits, gate/circuit references
models/                32nm_MGK.pm and four saved models (.joblib)
results/               metrics, predictions, error analysis (results/analysis/)
reports/               PDF report, data checks, figures
```

## Run

Tested with Python 3.12 and ngspice 42 (versions pinned in `code/requirements.lock.txt`).

```bash
bash run.sh setup       # install Python deps and pinned ngspice (Ubuntu 24.04 amd64); or: setup --ngspice /abs/path/ngspice
bash run.sh reproduce   # full regeneration into code/reproduced/ (400,000 transistor + 54,800 gate/circuit SPICE points, training; takes hours)
bash run.sh analysis    # regenerate results/analysis/ and reports/figures/analysis/ from stored results (no SPICE, no training; about a minute)
```

`reproduce` ends by drawing the four parity plots with `code/scripts/09_transistor_parity_plots.py`. A fresh full reproduction was not re-run after the final cleanup; the stored files are the originals.

## Deliverables (project brief, Section 6)

| # | Deliverable | Location |
|---|---|---|
| 1 | Four transistor datasets | `data/transistors/` (100,000 rows per case) |
| 2 | Example netlists | `netlists/` |
| 3 | Kirchhoff check and trend plots | `reports/data_checks/`, `reports/figures/analysis/trends_dataset.png`, `results/analysis/dataset_trends.txt` |
| 4 | Four model families x four cases | `results/transistor_validation_models.csv` |
| 5 | Test metrics, parity plots, saved models | `results/transistor_test_metrics.csv`, `results/transistor_test_terminals.csv`, `reports/figures/*_parity.png` (four cases), `models/` |
| 6 | Gate metrics, all 18 cases | `results/gate_metrics.csv` |
| 7 | Gate parity plot (log axes) | `reports/figures/gate_parity.png` |
| 8 | Circuit metrics and parity plot | `results/hierarchy_summary.json`, `results/circuit_metrics_by_vector.csv`, `reports/figures/circuit_parity.png` |
| 9 | Regeneration code | `run.sh`, `code/scripts/`, `code/lusr/` |
| 10 | Typed report | `reports/ML_Leakage_Estimation_Report.pdf` |

Error-analysis evidence cited in the report: `results/error_analysis.json`, `results/gate_transistor_predictions.csv`, `results/analysis/`, `reports/figures/analysis/` (scripts `15` to `18`).
