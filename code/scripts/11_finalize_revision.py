#!/usr/bin/env python3
"""Freeze validation-selected models, then evaluate the untouched test partition once.

Selection JSON supplies cases with model_path, family, validation_summary and
comparison_csv. Selection is completed before this script opens test targets.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from lusr.sampling import CASES, INPUT_COLUMNS
from lusr.ml import CURRENT_COLUMNS, regression_metrics, kcl_metrics
from lusr.revision_models import power_equivalent
from lusr.spice import sha256

spec=importlib.util.spec_from_file_location('baseline_training_io',ROOT/'scripts/03_train_models.py')
io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)


def source_hashes():
    paths=[Path(__file__),ROOT/'scripts/03_train_models.py',*(ROOT/'lusr').glob('*.py')]
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def freeze(root,selection_path,jobs):
    output=root/'results';output.mkdir(exist_ok=True)
    manifest_path=output/'ml_manifest.json'
    if manifest_path.exists():
        raise ValueError('A model freeze already exists. Preserve it; do not overwrite a finalized evaluation.')
    selection=json.loads(selection_path.read_text())
    if set(selection['cases'])!=set(CASES):raise ValueError('Exactly four selected cases required')
    comparison=pd.read_csv(selection['comparison_csv'])
    families=sorted(comparison.model.unique())
    if len(families)!=4 or len(comparison)!=16 or set(zip(comparison.case,comparison.model))!={(c,f) for c in CASES for f in families}:
        raise ValueError('Comparison must contain four model families × four cases')
    comparison.to_csv(output/'transistor_validation_models.csv',index=False)
    for case in CASES:
        family=selection['cases'][case]['family']
        chosen=comparison[(comparison.case==case)&(comparison.selected.astype(str).str.lower()=='true')]
        if len(chosen)!=1 or chosen.iloc[0].model!=family:
            raise ValueError(f'Selection/comparison mismatch: {case}')
    input_path=root/'data/input_manifest.json';inputs=json.loads(input_path.read_text())
    simulations=json.loads((root/'data/transistors/simulation_metadata.json').read_text())
    manifest={'status':'FROZEN_BEFORE_TEST','frozen_utc':datetime.now(timezone.utc).isoformat(),
              'selection':selection['selection_criterion'], 'comparison_families':families,
              'selection_file':str(selection_path),'selection_file_sha256':sha256(selection_path),
              'input_manifest_sha256':sha256(input_path),'source_sha256':source_hashes(),
              'comparison_sha256':sha256(output/'transistor_validation_models.csv'),
              'preprocessing':selection.get('preprocessing',{}),'cases':{}}
    for key in ('hyperparameter_table','representation_description','validation_evidence'):
        manifest[key]=selection[key]
    for name in ('protocol.json',*(f'{case}.csv' for case in CASES)):
        if 'experiments/physics/boundary_validation/'+name not in manifest['validation_evidence']:
            raise ValueError(f'Missing supplemental validation evidence: {name}')
    for artifact,digest in manifest['validation_evidence'].items():
        if sha256(ROOT/artifact)!=digest:
            raise ValueError(f'Validation evidence changed: {artifact}')
    models=root/'models';models.mkdir(exist_ok=True)
    for case in CASES:
        selected=selection['cases'][case]
        path=Path(selected['model_path']);target=models/f'{case}.joblib'
        source=joblib.load(path)
        if getattr(source,'training_rows',None)!=70000:
            raise ValueError(f'Expected70000 training rows for {case}')
        shutil.copy2(path,target)
        restored=joblib.load(target)
        split=pd.read_csv(root/f'data/splits/{case}_split.csv')
        row_ids=split.loc[split.split=='validation','row_index'].to_numpy()[:32]
        x=pd.read_csv(root/f'data/inputs/{case}_inputs.csv').iloc[row_ids][list(INPUT_COLUMNS)].to_numpy()
        with threadpool_limits(limits=jobs):
            a,b=source.predict(x),restored.predict(x)
        if not np.array_equal(a,b):raise ValueError(f'Reload predictions changed: {case}')
        entry=inputs['datasets'][case]
        current=root/f'data/transistors/{case}.csv'
        if sha256(current)!=simulations['cases'][case]['output_sha256']:
            raise ValueError(f'Transistor data hash mismatch: {case}')
        manifest['cases'][case]={'selected_family':selected['family'],
            'selected_model':str(target),'selected_model_sha256':sha256(target),
            'validation_summary':selected['validation_summary'],
            'reload_validation':'PASS: exact equality on32 validation rows',
            'provenance':{'current_csv':str(current),'current_sha256':sha256(current),
                'input_sha256':entry['input_sha256'],'split_sha256':entry['split_sha256'],
                'partition_counts':{k:int((split.split==k).sum()) for k in ('train','validation','test')}}}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Frozen four models before testing: {manifest_path}',flush=True)


def evaluate(root,jobs):
    output=root/'results';path=output/'ml_manifest.json';manifest=json.loads(path.read_text())
    if manifest['status'] not in ('FROZEN_BEFORE_TEST','TEST_EVALUATION_STARTED'):
        raise ValueError('Expected a frozen model set; completed test evaluation cannot be overwritten')
    inputs=json.loads((root/'data/input_manifest.json').read_text())
    if sha256(root/'data/input_manifest.json')!=manifest['input_manifest_sha256']:
        raise ValueError('Input manifest changed after model freeze')
    for source,digest in manifest['source_sha256'].items():
        if sha256(ROOT/source)!=digest:raise ValueError(f'Source changed after freeze: {source}')
    for artifact,digest in manifest['validation_evidence'].items():
        if sha256(ROOT/artifact)!=digest:raise ValueError(f'Validation evidence changed after freeze: {artifact}')
    # Check every case before exposing any test score. An interruption can only
    # resume the same frozen model/source/data combination, never a new selection.
    prepared={}
    for case in CASES:
        metadata=manifest['cases'][case]
        x,y,partitions,provenance=io.load_case(root/'data',case,inputs)
        if provenance!=metadata['provenance']:raise ValueError(f'Data changed: {case}')
        model_path=root/f'models/{case}.joblib'
        if sha256(model_path)!=metadata['selected_model_sha256']:raise ValueError(f'Model changed: {case}')
        prepared[case]=(x,y,partitions,model_path)
    manifest['status']='TEST_EVALUATION_STARTED'
    manifest.setdefault('test_started_utc',datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    scores=[];terminals=[]
    for case in CASES:
        metadata=manifest['cases'][case]
        if 'test_predictions_sha256' in metadata:
            dest=output/f'test_predictions/{case}.csv'
            if sha256(dest)!=metadata['test_predictions_sha256']:
                raise ValueError(f'Partial test result changed: {case}')
            scores.append(metadata['test_score']);terminals.extend(metadata['test_terminal_scores'])
            continue
        x,y,partitions,model_path=prepared[case]
        model=joblib.load(model_path);rows=partitions['test']
        with threadpool_limits(limits=jobs):prediction=model.predict(x[rows])
        power_score=regression_metrics(power_equivalent(x[rows],y[rows]),power_equivalent(x[rows],prediction))
        scores.append({'case':case,'model':metadata['selected_family'],'rows':len(rows),
                       **regression_metrics(y[rows],prediction),**kcl_metrics(prediction),
                       'power_mape_pct':power_score['mape_pct'],
                       'power_mean_accuracy_pct':power_score['mean_accuracy_pct']})
        for i,name in enumerate(CURRENT_COLUMNS):
            terminals.append({'case':case,'model':metadata['selected_family'],'terminal':name,
                              **regression_metrics(y[rows,i],prediction[:,i])})
        frame=pd.DataFrame({'row_index':rows})
        for i,name in enumerate(CURRENT_COLUMNS):
            frame[name+'_spice']=y[rows,i];frame[name+'_ml']=prediction[:,i]
        dest=output/f'test_predictions/{case}.csv';dest.parent.mkdir(exist_ok=True)
        frame.to_csv(dest,index=False)
        metadata['test_predictions']=str(dest);metadata['test_predictions_sha256']=sha256(dest)
        metadata['test_score']=scores[-1]
        metadata['test_terminal_scores']=terminals[-4:]
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(scores[-1]),flush=True)
    io.write_rows(output/'transistor_test_metrics.csv',scores)
    io.write_rows(output/'transistor_test_terminals.csv',terminals)
    manifest['status']='COMPLETED';manifest['test_completed_utc']=datetime.now(timezone.utc).isoformat()
    manifest['evaluation_source_sha256']=source_hashes()
    path.write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=ROOT/'improvements_99')
    p.add_argument('--selection-file',type=Path)
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args();root=args.output_root.resolve()
    if args.evaluate_only:evaluate(root,args.jobs)
    elif args.selection_file:freeze(root,args.selection_file.resolve(),args.jobs)
    else:p.error('Provide --selection-file to freeze, or --evaluate-only after freezing')


if __name__=='__main__':main()
