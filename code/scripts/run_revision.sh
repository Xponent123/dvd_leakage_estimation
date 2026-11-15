#!/usr/bin/env bash
# Reproduce the revised project; use a new output directory for a fresh rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."
REVISION_ROOT="$(realpath -m "${1:-reproduced}")"
if [[ "$REVISION_ROOT" != "$PWD/"* ]]; then
    echo 'Choose a separate output directory inside lusr_parity (for example improvements_99_reproduced).' >&2
    exit 2
fi
PYTHON="$PWD/.venv/bin/python"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
mkdir -p "$REVISION_ROOT/logs"

# An existing frozen run is immutable. It can be resumed
# after interruption using the identical model, source, data and validation hashes.
if [[ ! -f "$REVISION_ROOT/results/ml_manifest.json" ]]; then
    "$PYTHON" scripts/01_prepare_inputs.py --samples 100000 --seed 20261024 --output "$REVISION_ROOT/data"
    "$PYTHON" scripts/02_simulate_transistors.py \
        --input-dir "$REVISION_ROOT/data/inputs" --output-dir "$REVISION_ROOT/data/transistors" \
        --log-dir "$REVISION_ROOT/logs/transistors" --example-dir "$REVISION_ROOT/netlists/transistors" \
        --workers 4 --chunk-size 500
    if ! "$PYTHON" - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('experiments/physics/boundary_validation')
try:
    protocol=json.loads((root/'protocol.json').read_text())
    for case in ('nmos_on','nmos_off','pmos_on','pmos_off'):
        entry=protocol['datasets'][case]
        for name,key in ((case+'.csv','csv_sha256'),(case+'_inputs.csv','input_sha256')):
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==entry[key]
except (OSError,KeyError,ValueError,AssertionError):
    raise SystemExit(1)
PY
    then
        "$PYTHON" experiments/physics/boundary_validation.py
    fi
    "$PYTHON" scripts/10_compare_revision_trees.py --output-root "$REVISION_ROOT" --jobs 2
    "$PYTHON" scripts/12_train_revision_rbf.py --data-root "$REVISION_ROOT/data" --output "$REVISION_ROOT/physics_selected"
    "$PYTHON" scripts/13_prepare_revision_selection.py --output-root "$REVISION_ROOT"
    "$PYTHON" scripts/11_finalize_revision.py --output-root "$REVISION_ROOT" --selection-file "$REVISION_ROOT/results/revision_selection.json"
fi
REVISION_STATUS="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$REVISION_ROOT/results/ml_manifest.json")"
if [[ "$REVISION_STATUS" != COMPLETED ]]; then
    "$PYTHON" scripts/11_finalize_revision.py --output-root "$REVISION_ROOT" --evaluate-only --jobs 2
fi
if [[ ! -f "$REVISION_ROOT/README.md" ]]; then
    printf '%s\n' '# Regenerated numerical deliverables' > "$REVISION_ROOT/README.md"
fi
REVISION_ROWS="$("$PYTHON" -c 'import json,sys; print(*sorted({d["rows"] for d in json.load(open(sys.argv[1]))["datasets"].values()}))' "$REVISION_ROOT/data/input_manifest.json")"
[[ "$REVISION_ROWS" =~ ^[0-9]+$ ]] || { echo "Revision datasets do not share one row count: $REVISION_ROWS" >&2; exit 1; }
"$PYTHON" scripts/04_data_checks.py --data "$REVISION_ROOT/data/transistors" --output "$REVISION_ROOT/reports/data_checks" --expected-rows "$REVISION_ROWS"
"$PYTHON" scripts/05_simulate_gates_circuit.py --output-root "$REVISION_ROOT" --corners 200 --workers 3
"$PYTHON" scripts/verify_circuit_references.py --output-root "$REVISION_ROOT"
"$PYTHON" scripts/06_evaluate_hierarchy.py --output-root "$REVISION_ROOT"
"$PYTHON" scripts/15_explain_worst_cases.py --output-root "$REVISION_ROOT"
"$PYTHON" scripts/09_transistor_parity_plots.py --output-root "$REVISION_ROOT"
