# ML-driven PVT-aware leakage estimation

**Team 8 — LUSR:** Dhimant Bhuva, Jal Parikh, Garima Mittal, Krishna Goel.  
**Assignment:** 32nm_MGK, nominal VDD 0.9 V, 8-bit parity checker.

The submission contains the final measured datasets, four selected transistor models, all required netlists, evaluation tables and plots, reproduction code, and an editable LaTeX report.

| Evaluation | Operating points | Median error | 95th-percentile error | Mean accuracy |
|---|---:|---:|---:|---:|
| Five gates, all 18 input cases | 3,600 | 0.25604% | 1.11632% | 99.63502% |
| Complete 8-bit parity checker | 51,200 | 0.42348% | 1.21420% | 99.48968% |

Mean accuracy is `100 × (1 − mean absolute relative error)`, with a `1e-18 A` denominator floor. The report separately gives signed terminal-current errors, including weaker OFF/body-current results. The table does not imply that every prediction or every terminal is within 1%.

## Required deliverables

| Project brief, Section 6 | Submitted location |
|---|---|
| 1. Four transistor datasets | [data/transistors/](data/transistors/) — 100,000 rows per case; all 16 required columns |
| 2. Example netlists | [netlists/](netlists/) — four single-FET cases, five complete gates, and complete parity checker |
| 3. Kirchhoff check and three trend plots | [reports/data_checks/](reports/data_checks/) |
| 4. Four model families × four cases | [results/transistor_validation_models.csv](results/transistor_validation_models.csv) |
| 5. Selected-model test metrics, four parity plots and four saved models | [results/transistor_test_metrics.csv](results/transistor_test_metrics.csv), [results/transistor_test_terminals.csv](results/transistor_test_terminals.csv), [reports/figures/](reports/figures/), [models/](models/) |
| 6. All 18 gate/input-case metrics | [results/gate_metrics.csv](results/gate_metrics.csv) |
| 7. Gate parity plot on log axes | [reports/figures/gate_parity.png](reports/figures/gate_parity.png) |
| 8. Test-circuit metrics and parity plot | [results/hierarchy_summary.json](results/hierarchy_summary.json), [results/circuit_metrics_by_vector.csv](results/circuit_metrics_by_vector.csv), [reports/figures/circuit_parity.png](reports/figures/circuit_parity.png) |
| 9. Ordered fixed-seed regeneration code | [run.sh](run.sh), [code/scripts/](code/scripts/), [code/lusr/](code/lusr/) |
| 10. Typed report | [PDF](reports/LUSR_8bit_Parity_Leakage_Report.pdf) and [editable LaTeX](reports/main.tex) |

The report's worst-error explanation is supported by [evaluation-only checks](results/error_analysis.json): fresh single-FET attribution of the five largest gate errors, a controlled internal-gate-bias check, and decomposition of the worst parity error into surrogate and gate-composition contributions. The reproduction pipeline regenerates this evidence and [the included LaTeX explanation](reports/error_analysis.tex) before building the report.

Part A explicitly allows increasing or decreasing the initial 10,000 samples. The submitted final dataset has **100,000 independently uniform samples per case**, partitioned into **70,000 / 15,000 / 15,000** training/validation/test rows using seed **20261024**. The four earlier 10k datasets are superseded. The four saved models and their original SPICE labels are unchanged by this submission cleanup.

The comparison families are random forest, extra trees, histogram gradient boosting and local quintic RBF. Selection uses independent single-FET boundary validation; gate and circuit reference currents are evaluation data. Part B uses recorded SPICE internal/output voltages as prescribed. Part C propagates Boolean inputs through 28 NAND2 cells (112 MOSFETs) and sums gate predictions at the same 200 corners.

## Run

Tested numerical environment: Python 3.12 and ngspice 42. Exact Python versions are in [code/requirements.lock.txt](code/requirements.lock.txt). On Ubuntu 24.04 amd64, setup installs dependencies into the project and extracts the pinned ngspice package:

```bash
bash run.sh setup
bash run.sh verify
```

For a compatible existing simulator, use `bash run.sh setup --ngspice /absolute/path/to/ngspice` and export `NGSPICE` to that path for subsequent simulation. Verification itself needs Python dependencies and `pdftotext`, but does not run SPICE or retrain models.

```bash
bash run.sh reproduce   # all numerical stages into code/reproduced/
bash run.sh report      # compile reports/main.tex with pdfLaTeX, twice
```

The report command needs a working pdfLaTeX installation with the standard LaTeX, recommended, extra and pictures packages plus Latin Modern fonts (or use Overleaf). Upload the contents of `reports/`, including `main.tex`, `error_analysis.tex`, and its figure directories, to edit and compile the report in Overleaf. To regenerate only the numerical error explanation, run `python3 code/scripts/15_explain_worst_cases.py --ngspice /absolute/path/to/ngspice`; this writes regenerable logs under `logs/` and does not fit or change any model.

Full numerical reproduction runs 400,000 transistor and 54,800 gate/circuit operating points, plus supplemental validation and trend checks. It regenerates sampling inputs, fixed splits, candidate-validation outputs, chosen models, comparisons, plots and a numerical PDF report. The submitted model/result files are preserved. LaTeX editing and compilation are separate from fitting or evaluation.

## Supporting evidence

The final tables retain all 3,600 gate and 51,200 complete-circuit references, 60,000 held-out transistor predictions, per-device gate predictions, recorded node voltages, 256-vector circuit metrics, and worst-error rows. These allow independent recalculation of the reported errors and hierarchical sums.

Original freeze and simulation manifests remain unchanged, including their original paths and hashes. `scripts/verify_submission.py` resolves packaged data, models and code locally; checks their original hashes, split coverage, KCL statistics, model reload predictions, gate energy aggregation, all 256 Boolean gate sums, netlists and report. [CHECKSUMS.sha256](CHECKSUMS.sha256) covers the submitted files.

The verifier also checks the error explanation against its input hashes, per-device power sums, complete-circuit error decomposition and LaTeX text. Detailed candidate-training artifacts referenced at historical paths are generated during reproduction; they are not dependencies of the packaged-artifact verification.

Only the code needed for reproduction is retained from the original experiments. Input-only copies, duplicate candidate models and intermediate training outputs are generated during reproduction. They are not needed to use the four submitted models or audit the final results. The submitted checks verify the recovered artifacts; a fresh full SPICE/training reproduction has not been run as part of the packaging cleanup.
