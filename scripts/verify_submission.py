#!/usr/bin/env python3
"""Check the submitted artifacts without fitting models or running SPICE.

Original freeze manifests remain unchanged. Historical paths refer to the original
run; model/source/data checks below resolve the corresponding packaged files.
Candidate model copies and input-only datasets are intentionally not duplicated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from lusr.circuits import propagate
from lusr.ml import regression_metrics
from lusr.model_card import read_model_card
from lusr.sampling import CASES, INPUT_COLUMNS, validate_transistors, validate_corners
from lusr.spice import CURRENT_COLUMNS, OPTIONS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash(relative, expected):
    require(digest(ROOT/relative) == expected, f'Checksum mismatch: {relative}')


def load(relative):
    return json.loads((ROOT/relative).read_text())


def frame(relative):
    return pd.read_csv(ROOT/relative, float_precision='round_trip', dtype={'input_bits': str})


def close(actual, expected, label, atol=1e-10):
    require(np.isclose(actual, expected, rtol=1e-9, atol=atol), f'Incorrect metric: {label}')


def leakage_metrics(table):
    y, p = table.spice_leakage_A.to_numpy(), table.ml_leakage_A.to_numpy()
    e = (p-y)/np.maximum(np.abs(y), 1e-18)
    return dict(count=len(y), spice_median_A=np.median(y), ml_median_A=np.median(p),
                bias_A=np.mean(p-y), bias_percent=100*np.mean(e),
                median_error_percent=100*np.median(np.abs(e)),
                p95_error_percent=100*np.percentile(np.abs(e), 95),
                mean_error_percent=100*np.mean(np.abs(e)),
                mean_accuracy_percent=100*(1-np.mean(np.abs(e))),
                r2=1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))


def verify():
    checksum_file = ROOT/'CHECKSUMS.sha256'
    require(checksum_file.is_file(), 'Missing CHECKSUMS.sha256')
    listed = set()
    for line in checksum_file.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Invalid checksum path')
        require(relative not in listed, f'Duplicate checksum entry: {relative}')
        listed.add(relative)
        check_hash(relative, expected)
    check_hash('models/32nm_MGK.pm', '56710120d5a2ae9e32a1908e1ee29a98f692a1dec0501eed8976ea6e6bea021e')
    ml, inputs = load('results/ml_manifest.json'), load('data/input_manifest.json')
    sim, checks = load('data/transistors/simulation_metadata.json'), load('reports/data_checks/data_checks.json')
    hierarchy = load('results/hierarchy_summary.json')
    require(ml['status'] == 'COMPLETED', 'Incomplete frozen-model evaluation')
    require(ml['frozen_utc'] <= ml['test_started_utc'] <= ml['test_completed_utc'], 'Invalid freeze/test chronology')
    check_hash('data/input_manifest.json', ml['input_manifest_sha256'])
    check_hash('results/revision_selection.json', ml['selection_file_sha256'])
    check_hash('results/ml_manifest.json', hierarchy['ml_manifest_sha256'])
    check_hash('reports/circuit_reference_validation.json', hierarchy['reference_audit_sha256'])
    for source, expected in {**ml['source_sha256'], **ml['evaluation_source_sha256']}.items():
        check_hash('code/'+source, expected)
    check_hash('code/scripts/06_evaluate_hierarchy.py', hierarchy['hierarchy_source_sha256'])
    nominal = read_model_card(ROOT/'models/32nm_MGK.pm').nominal
    result = {'status': 'PASSED', 'checksummed_files': len(listed), 'transistors': {}, 'models': {}}
    for case in CASES:
        path = f'data/transistors/{case}.csv'
        check_hash(path, sim['cases'][case]['output_sha256'])
        check_hash(path, ml['cases'][case]['provenance']['current_sha256'])
        check_hash(path, checks['transistor_datasets'][case]['sha256'])
        data = frame(path)
        require(tuple(data.columns) == (*INPUT_COLUMNS, *CURRENT_COLUMNS), f'Dataset schema: {case}')
        require(len(data) == 100000 and np.isfinite(data.to_numpy()).all(), f'Dataset count/values: {case}')
        validate_transistors(case, {k:data[k].to_numpy() for k in INPUT_COLUMNS}, nominal)
        # The first 12 columns reconstruct the original input-only CSV exactly.
        h = hashlib.sha256()
        with (ROOT/path).open(newline='') as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                text = io.StringIO(newline='')
                values = row[:12] if index == 0 else [format(float(v),'.17g') for v in row[:12]]
                csv.writer(text,lineterminator='\n').writerow(values)
                h.update(text.getvalue().encode())
        require(h.hexdigest() == inputs['datasets'][case]['input_sha256'], f'Input provenance: {case}')
        splits = frame(f'data/splits/{case}_split.csv')
        check_hash(f'data/splits/{case}_split.csv', inputs['datasets'][case]['split_sha256'])
        require(len(splits) == 100000 and set(splits.row_index) == set(range(100000)), f'Split coverage: {case}')
        sizes = splits.split.value_counts().to_dict()
        require(sizes == {'train':70000, 'validation':15000, 'test':15000}, f'Split proportions: {case}')
        residual = np.abs(data[list(CURRENT_COLUMNS)].to_numpy().sum(axis=1))
        old = checks['transistor_datasets'][case]['absolute_residual_A']
        for key, value in [('max', residual.max()), ('median', np.median(residual)), ('p95', np.percentile(residual,95))]:
            close(value, old[key], f'{case} KCL {key}', atol=1e-28)
        result['transistors'][case] = {'rows':len(data), 'splits':sizes, 'max_KCL_residual_A':float(residual.max())}
        model_path = f'models/{case}.joblib'
        check_hash(model_path, ml['cases'][case]['selected_model_sha256'])
        check_hash(model_path, hierarchy['model_hashes'][case])
        prediction_path = f'results/test_predictions/{case}.csv'
        check_hash(prediction_path, ml['cases'][case]['test_predictions_sha256'])
        test = frame(prediction_path)
        test_ids = splits.loc[splits.split == 'test','row_index'].to_numpy()
        require(np.array_equal(test.row_index, test_ids), f'Test partition: {case}')
        truth = data.iloc[test_ids][list(CURRENT_COLUMNS)].to_numpy()
        require(np.array_equal(truth, test[[t+'_spice' for t in CURRENT_COLUMNS]].to_numpy()), f'Test labels: {case}')
        predicted = test[[t+'_ml' for t in CURRENT_COLUMNS]].to_numpy()
        for name, value in regression_metrics(truth, predicted).items():
            if name in ml['cases'][case]['test_score']:
                close(value, ml['cases'][case]['test_score'][name], f'{case}/{name}', atol=1e-24)
        model = joblib.load(ROOT/model_path)
        require(model.training_rows == 70000, f'Model training count: {case}')
        selected = np.linspace(0,len(test)-1,16,dtype=int)
        with threadpool_limits(limits=1):
            current = model.predict(data.iloc[test_ids[selected]][list(INPUT_COLUMNS)].to_numpy())
        require(np.allclose(current,predicted[selected],rtol=1e-8,atol=1e-23), f'Reloaded model predictions: {case}')
        result['models'][case] = {'saved_model_hash':'PASS', 'reloaded_test_predictions':16}
        print(f'PASS {case}: 100,000 rows, fixed splits, KCL, held-out labels and saved model', flush=True)
    comparison = frame('results/transistor_validation_models.csv')
    check_hash('results/transistor_validation_models.csv', ml['comparison_sha256'])
    require(len(comparison) == 16 and set(zip(comparison.case,comparison.model)) == {(c,f) for c in CASES for f in ml['comparison_families']}, 'Four families by four cases')
    require(len(frame('results/transistor_test_metrics.csv')) == 4 and len(frame('results/transistor_test_terminals.csv')) == 16, 'Chosen-model metric tables')
    corners = frame('data/corners.csv')
    validate_corners({k:corners[k].to_numpy() for k in corners.columns}, nominal)
    require(len(corners) == 200, 'Exactly 200 corners required')
    spice = load('reports/spice_both_manifest.json')
    audit = load('reports/circuit_reference_validation.json')
    require(audit['status'] == 'PASSED' and spice['status'] == 'SIMULATED', 'SPICE audit status')
    check_hash('data/corners.csv', inputs['evaluation_corners']['sha256'])
    check_hash('data/corners.csv', spice['corner_sha256'])
    counts = {'INV':2,'NAND2':4,'NOR2':4,'AND2':4,'OR2':4}
    tables = {}
    for kind, directory, total in [('gate','gates',3600),('circuit','circuit',51200)]:
        reference = frame(f'data/{directory}/reference.csv')
        check_hash(f'data/{directory}/reference.csv', spice['datasets'][directory]['sha256'])
        check_hash(f'data/{directory}/reference.csv', audit['gate_reference_sha256' if kind=='gate' else 'circuit_reference_sha256'])
        predicted = frame(f'results/{kind}_predictions.csv')
        expected = ({(c,g,v) for c in range(200) for g,n in counts.items() for v in range(n)} if kind=='gate'
                    else {(c,'PARITY8',v) for c in range(200) for v in range(256)})
        require(len(reference) == len(predicted) == total and set(zip(predicted.corner_id,predicted.circuit,predicted.vector)) == expected, f'{kind} complete coverage')
        keys = ['corner_id','circuit','vector']
        ref = reference.set_index(keys).sort_index()
        pred = predicted.set_index(keys).sort_index()
        require(ref.index.equals(pred.index), f'{kind} reference coverage')
        for name in reference.columns.difference(keys):
            require(np.array_equal(ref[name].to_numpy(),pred[name].to_numpy()), f'{kind} reference field: {name}')
        require(np.isfinite(predicted[['spice_leakage_A','ml_leakage_A','output_V']].to_numpy()).all(), f'{kind} nonfinite values')
        logic_ratio = predicted.output_V/predicted.vdd
        require(((logic_ratio < .1) | (logic_ratio > .9)).all() and np.array_equal((logic_ratio>.5).astype(int),predicted.expected_output), f'{kind} output logic')
        for name,value in leakage_metrics(predicted).items():
            close(value,hierarchy[kind][name],f'{kind}/{name}',atol=1e-24)
        tables[kind] = predicted
        result[kind] = {'rows':total, **{k:float(v) for k,v in leakage_metrics(predicted).items()}}
    gate = tables['gate']
    for _, row in frame('results/gate_metrics.csv').iterrows():
        subset = gate[(gate.circuit==row.gate)&(gate.input_bits==row.input_bits)]
        require(len(subset)==200,'Gate metrics per input')
        for name,value in leakage_metrics(subset).items(): close(value,row[name],f'gate table/{name}',atol=1e-24)
    vectors = frame('results/circuit_metrics_by_vector.csv')
    require(len(vectors)==256 and set(vectors.vector)==set(range(256)), 'Circuit vector table')
    for _, row in vectors.iterrows():
        for name,value in leakage_metrics(tables['circuit'][tables['circuit'].vector==row.vector]).items():
            close(value,row[name],f'circuit table/{name}',atol=1e-24)
    require((gate.node_voltages_json == gate.ml_node_voltages_json).all(), 'Recorded Part B node voltages')
    fet = frame('results/gate_transistor_predictions.csv')
    require(len(fet)==16800, 'Gate transistor coverage')
    power = ((fet.vd-fet.vs)*fet.id_ml+(fet.vg-fet.vs)*fet.ig_ml+(fet.vb-fet.vs)*fet.ib_ml)
    require(np.allclose(power,fet.power_closed_W,rtol=1e-12,atol=1e-25), 'Transistor energy aggregation')
    summed = (power/fet.vdd).groupby(fet.gate_row).sum()
    require(np.allclose(summed.to_numpy(),gate.ml_leakage_A.to_numpy(),rtol=1e-12,atol=1e-23),'Part B gate leakage aggregation')
    lookup = gate.set_index(['corner_id','circuit','vector']).ml_leakage_A.to_dict()
    for vector in range(256):
        nodes, states = propagate(vector)
        require(len(states)==28 and nodes['out']==vector.bit_count()%2, 'Parity Boolean topology')
        frequencies = {}
        for cell,bits in states:
            key = (cell.gate, bits[0]*2+bits[1])
            frequencies[key] = frequencies.get(key,0)+1
        observed = tables['circuit'][tables['circuit'].vector==vector]
        predicted = np.array([sum(n*lookup[(int(c),g,v)] for (g,v),n in frequencies.items()) for c in observed.corner_id])
        require(np.allclose(predicted,observed.ml_leakage_A,rtol=1e-12,atol=1e-22),'Part C hierarchical gate sum')
    check_hash('reports/data_checks/trends.csv',load('reports/data_checks/trend_manifest.json')['csv_sha256'])
    require(len(frame('reports/data_checks/trends.csv'))==246,'Controlled trend points')
    decks = {**{f'transistors/{c}':1 for c in CASES}, **{f'gates/{g}':n for g,n in [('inv',2),('nand2',4),('nor2',4),('and2',6),('or2',6)]}, 'circuit/parity8':112}
    for name,count in decks.items():
        text = (ROOT/f'netlists/{name}.cir').read_text()
        require(OPTIONS in text and len(re.findall(r'(?im)^M\S+\s',text))==count,f'Netlist devices/options: {name}')
        if name.startswith('transistors/'):
            require(all(re.search(rf'(?im)^V{t}\s+{t}\s+0\s',text) for t in 'dgsb'),f'Terminal voltage sources: {name}')
    tex = (ROOT/'reports/main.tex').read_text()
    for image in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex):
        require((ROOT/'reports'/image).is_file(),f'Missing LaTeX figure: {image}')
    pdf_text = subprocess.run(['pdftotext',str(ROOT/'reports/LUSR_8bit_Parity_Leakage_Report.pdf'),'-'],capture_output=True,text=True,check=True).stdout
    require(len(pdf_text)>10000 and all(t in pdf_text for t in ('400,000','51,200','99.6350','99.4897')), 'Report counts/results')
    require(not re.search(r'\b(TODO|TBD|FIXME)\b|\[placeholder\]', pdf_text), 'Unfinished report')
    result['hierarchy'] = 'PASS: per-transistor gate energy and all 256-vector gate sums'
    result['report'] = 'PASS: LaTeX sources, figures and compiled PDF'
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optionally save validation results')
    args=parser.parse_args()
    try:
        result=verify()
    except Exception as error:
        print(f'FAILED: {error}',file=sys.stderr)
        raise SystemExit(1)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"PASSED: {result['checksummed_files']} checksummed files; 400,000 transistor, 3,600 gate and 51,200 circuit rows; four model reloads; hierarchical sums and report.")
