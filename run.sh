#!/usr/bin/env bash
set -euo pipefail
SUBMISSION_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SUBMISSION_ROOT"
SUBMISSION_PYTHON=${PYTHON:-python3}
if [[ -x code/.venv/bin/python ]]; then SUBMISSION_PYTHON="$SUBMISSION_ROOT/code/.venv/bin/python"; fi
case "${1:-verify}" in
  setup) shift; bash code/scripts/00_setup.sh "$@" ;;
  verify) shift; "$SUBMISSION_PYTHON" scripts/verify_submission.py "$@" ;;
  reproduce) bash code/scripts/run_revision.sh reproduced ;;
  report) cd reports; pdflatex -interaction=nonstopmode -halt-on-error main.tex; pdflatex -interaction=nonstopmode -halt-on-error main.tex; mv main.pdf ML_Leakage_Estimation_Report.pdf ;;
  analysis) shift; for s in 16_error_analysis 17_terminal_metrics 18_dataset_trends; do "$SUBMISSION_PYTHON" code/scripts/$s.py; done ;;
  *) echo 'Usage: bash run.sh [setup|verify|reproduce|analysis|report]' >&2; exit 2 ;;
esac
