#!/usr/bin/env python3
"""Assemble the four-family validation table and declare the selected models.

This script does not load test targets, gate currents or circuit currents.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from lusr.sampling import CASES
from lusr.spice import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=ROOT/'improvements_99')
    args=p.parse_args();root=args.output_root.resolve()
    physics=root/'physics_selected/manifest.json'
    selected=json.loads(physics.read_text())
    rows=[];boundary_rows=[];models={}
    evidence={str(physics.relative_to(ROOT)):sha256(physics)}
    for name in ('protocol.json',*(f'{case}.csv' for case in CASES)):
        path=ROOT/'experiments/physics/boundary_validation'/name
        evidence[str(path.relative_to(ROOT))]=sha256(path)
    for case in CASES:
        candidate=selected['cases'][case]
        metric=candidate['boundary']['power_nonzero_bias_metrics']
        boundary_rows.append({'case':case,'boundary_power_mape_pct':metric['mape_pct'],
            'boundary_power_median_pct':metric['median_pct'],'boundary_power_p95_pct':metric['p95_pct']})
        for family in ('random_forest','extra_trees','hist_gradient_boosting'):
            path=root/f'results/tree_comparison/{case}_{family}.json'
            tree=json.loads(path.read_text());evidence[str(path.relative_to(ROOT))]=sha256(path)
            if tree['boundary_power_mape_pct']<=metric['mape_pct']:
                raise ValueError(f'{family} beats the chosen RBF on the declared selection score for {case}; revise selection honestly')
            rows.append({k:v for k,v in tree.items() if k in (
                'case','model','training_rows','validation_rows','raw_r2','median_relative_error_pct',
                'p95_relative_error_pct','mape_pct','mean_accuracy_pct','boundary_power_mape_pct')}
                |{'variant':'canonical_log_targets','selected':False})
        rows.append({'case':case,'model':selected['family'],'training_rows':70000,'validation_rows':15000,
                     **candidate['uniform_validation_metrics'],'boundary_power_mape_pct':metric['mape_pct'],
                     'variant':f"{candidate['parameters']['style']}; {candidate['parameters']['neighbors']} neighbors"+('; ON gate residual' if case.endswith('_on') else '; physics residual'),
                     'selected':True})
        for artifact in candidate['outputs'].values():
            path=Path(artifact['path'])
            if sha256(path)!=artifact['sha256']:raise ValueError(f'Changed candidate artifact: {path}')
            evidence[str(path.relative_to(ROOT))]=artifact['sha256']
        models[case]={'model_path':candidate['outputs']['.joblib']['path'],'family':selected['family'],
                      'validation_summary':{'uniform_currents':candidate['uniform_validation_metrics'],
                          'uniform_power':candidate['uniform_power_metrics'],
                          'boundary_power':metric,'parameters':candidate['parameters']}}
    comparison=root/'results/transistor_validation_models.csv'
    pd.DataFrame(rows).to_csv(comparison,index=False)
    pd.DataFrame(boundary_rows).to_csv(root/'results/selected_boundary_validation.csv',index=False)
    selection={'selection_criterion':'Minimize the predeclared independent single-FET boundary-validation nonzero-bias power/VDD MAPE; use uniform-validation smooth-component log RMSE only to break ties. The complete uniform-validation raw-current statistics are reported separately. No final-test or gate/circuit leakage is used for selection.',
        'comparison_csv':str(comparison),'cases':models,'validation_evidence':evidence,
        'preprocessing':{'features':'Canonical drain/source polarity, voltage coordinates and smooth temperature/geometry/process functions of the original12 inputs.',
            'targets':'Positive conductance and gate/body magnitudes in logarithms; physical baselines normalize selected targets. Every fitted residual uses70000 uniform training rows only.',
            'validation':'15000 uniform rows plus512 independent boundary single-FET rows per case; test15000 rows remain untouched until allfour model files and sources are frozen.'},
        'representation_description':selected['representation_description'],
        'hyperparameter_table':[
            ['Random forest','128 trees; all features; unrestricted depth; minimum leaf size2; seed20261024.'],
            ['Extra trees','128 trees; all features; unrestricted depth; minimum leaf size2; seed20261024.'],
            ['Histogram gradient boosting','Four target regressors;400 iterations; rate0.08;63 leaves; minimum leaf15; L2=0.01; seed20261024.'],
            [selected['family'],'Local quintic RBF, degree2 polynomial, smoothing1e-12.512 neighbors for NMOS OFF;256 for other cases. ON cases additionally learn a gate-edge log residual using256 neighbors.']
        ]}
    path=root/'results/revision_selection.json';path.write_text(json.dumps(selection,indent=2)+'\n')
    print(path,flush=True)


if __name__=='__main__':main()
