#!/usr/bin/env python3
"""Train exactly the declared RBF configs and collect auditable validation artifacts.

No final-test or gate/circuit evaluation occurs. Use --collect-only to package
already completed experiments whose data/split hashes match the specified pool.
"""
from __future__ import annotations
import argparse,csv,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lusr.spice import sha256


def corrected_artifacts(case,base_result,metadata,data_root,output,collect_only,parameters):
    """Train or collect the validation-selected ON gate-current residual."""
    import joblib,numpy as np,pandas as pd
    from threadpoolctl import threadpool_limits
    from lusr.ml import regression_metrics
    from lusr.smooth_models import GateEdgeResidualModel,GateCorrectedModel,load_train_validation
    from experiments.physics.run import metrics,energy
    correction=parameters['gate_correction']
    if collect_only:
        directory=ROOT/'experiments/smooth'/f'gate_residual_{case}_raw_256'
        model_path=directory/'model.joblib'
        validation_path=directory/'full_validation_predictions.csv'
        boundary_path=directory/'boundary_predictions.csv'
        evidence=directory/'full_validation_metrics.json'
        if not evidence.exists():raise FileNotFoundError(f'Full wrapper validation pending: {evidence}')
        selected=joblib.load(model_path)
    else:
        directory=output/'training_runs/wrappers'/case
        directory.mkdir(parents=True,exist_ok=True)
        model_path=directory/'model.joblib'
        validation_path=directory/'full_validation_predictions.csv'
        boundary_path=directory/'boundary_predictions.csv'
        data=load_train_validation(ROOT,case,data_root)
        x,y=data['train'];xv,yv=data['validation']
        base=joblib.load(base_result.with_suffix('.joblib'))
        gate=GateEdgeResidualModel(case,neighbors=correction['neighbors'],style=correction['style'],regularization=correction['regularization'])
        with threadpool_limits(limits=1):gate.fit(x,y)
        selected=GateCorrectedModel(base,gate)
        val=pd.read_csv(base_result.with_suffix('.validation.csv'))
        boundary=pd.read_csv(base_result.with_suffix('.boundary.csv'))
        reference=np.loadtxt(ROOT/'experiments/physics/boundary_validation'/f'{case}.csv',delimiter=',',skiprows=1)
        for table,features,destination in ((val,xv,validation_path),(boundary,reference[:,:12],boundary_path)):
            with threadpool_limits(limits=1):
                predicted_gate=np.concatenate([gate.predict(features[i:i+128]) for i in range(0,len(features),128)])
            change=predicted_gate-table.ig_ml.to_numpy()
            table['id_ml']-=.5*change;table['is_ml']-=.5*change;table['ig_ml']=predicted_gate
            if destination==boundary_path:
                table['ml_power_over_vdd_A']=energy(features,table[[t+'_ml' for t in ('id','ig','is','ib')]].to_numpy())
            table.to_csv(destination,index=False)
        joblib.dump(selected,model_path,compress=3)
    if selected.training_rows!=70000:raise ValueError('Incorrect wrapper training row count')
    val=pd.read_csv(validation_path);boundary=pd.read_csv(boundary_path)
    if len(val)!=15000 or len(boundary)!=512:raise ValueError('Incomplete wrapper validation artifacts')
    split=pd.read_csv(data_root/'splits'/f'{case}_split.csv')
    validation_ids=split.loc[split.split=='validation','row_index'].to_numpy()
    if not np.array_equal(val.row_index.to_numpy(),validation_ids):raise ValueError('Wrapper validation partition mismatch')
    x=pd.read_csv(data_root/'inputs'/f'{case}_inputs.csv').iloc[validation_ids].to_numpy()
    b=np.loadtxt(ROOT/'experiments/physics/boundary_validation'/f'{case}.csv',delimiter=',',skiprows=1)
    terminals=('id','ig','is','ib')
    y=val[[t+'_spice' for t in terminals]].to_numpy();prediction=val[[t+'_ml' for t in terminals]].to_numpy()
    by=boundary[[t+'_spice' for t in terminals]].to_numpy();bp=boundary[[t+'_ml' for t in terminals]].to_numpy()
    # Verify that collected prediction tables actually come from this wrapper.
    with threadpool_limits(limits=1):check=selected.predict(x[:16])
    if not np.allclose(check,prediction[:16],rtol=1e-10,atol=1e-23):raise ValueError('Wrapper artifact/prediction mismatch')
    active=np.ptp(b[:,[3,4,5,6]],axis=1)>0
    metadata=dict(metadata)
    metadata['regression_metrics']=regression_metrics(y,prediction)
    metadata['power_metrics']=metrics(energy(x,y),energy(x,prediction))
    metadata['boundary']={'raw_terminal_metrics':metrics(by,bp),
        'power_nonzero_bias_metrics':metrics(energy(b[:,:12],by)[active],energy(b[:,:12],bp)[active]),
        'zero_bias_rows':int((~active).sum()),'zero_bias_predicted_current_max_A':float(abs(bp[~active]).max()) if (~active).any() else None,'groups':{}}
    for group,label in enumerate(['0_0','0_1','1_0','1_1','eps_0','0_eps','1-eps_1','1_1-eps']):
        sl=slice(group*64,(group+1)*64)
        metadata['boundary']['groups'][label]={'raw_terminal_metrics':metrics(by[sl],bp[sl]),'power_metrics':metrics(energy(b[sl,:12],by[sl]),energy(b[sl,:12],bp[sl]))}
    metadata['gate_correction']=correction
    metadata['gate_correction_note']='Learn ON Ig residual relative to abs(edgeS+edgeD); adjust Id/Is by half the gate change to preserve learned channel difference and raw KCL residual. Tiny gate-body current at zero edge bias is omitted and raw errors retained.'
    path=output/'training_runs'/f'{case}_selected_metadata.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(metadata,indent=2)+'\n')
    return {'.joblib':model_path,'.json':path,'.validation.csv':validation_path,'.boundary.csv':boundary_path},metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'improvements_99/data')
    parser.add_argument('--output',type=Path,default=ROOT/'improvements_99/physics_selected')
    parser.add_argument('--collect-only',action='store_true')
    args=parser.parse_args()
    config_path=ROOT/'experiments/physics/selected_configs.json'
    configs=json.loads(config_path.read_text())
    shared=configs['shared_parameters']
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'status':'VALIDATION_ONLY_SELECTED','family':configs['family'],
              'data_directory':str(args.data_root.resolve()),'config_sha256':sha256(config_path),
              'configuration':configs,'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in
              [ROOT/'lusr/physics_surrogate.py',ROOT/'lusr/smooth_models.py',ROOT/'experiments/physics/run.py',Path(__file__)]},
              'cases':{},'test_policy':'No final-test labels or gate/circuit labels opened for fitting, selection, or evaluation here.'}
    manifest['representation_description']='Canonical PMOS polarity and ordered drain/source; log positive Gd/Gs and |Ig|/|Ib|, with width normalization; OFF log residuals relative to fixed BSIM gate-edge and diode-shaped body baselines. ON uses a separately learned gate-edge residual with compensating half gate-current changes in Id/Is; learned channel difference and KCL residual preserved. All four currents reconstructed from learned residuals; no SPICE at inference. Exact all-terminal-equal zero invariant.'
    manifest['hyperparameter_table']={case:{**shared,**settings} for case,settings in configs['cases'].items()}
    manifest['validation_selected']=configs['validation_selected']
    manifest['validation_evidence']={str(p.relative_to(ROOT)):sha256(p) for p in
        [ROOT/'experiments/physics/boundary_validation'/name for name in ('protocol.json',*(case+'.csv' for case in configs['cases']))]}
    comparison=[]
    for case,settings in configs['cases'].items():
        parameters={**shared,**settings}
        stem=f"{case}_rbf_{parameters['style']}_d2_n{parameters['neighbors']}_r1e-12_quintic_train70000_val15000_normalized"
        result=(ROOT/'experiments/physics' if args.collect_only else args.output/'training_runs')/stem
        if not args.collect_only:
            command=[sys.executable,str(ROOT/'experiments/physics/run.py'),'--case',case,'--method','rbf',
                     '--degree','2','--neighbors',str(parameters['neighbors']),'--regularization','1e-12',
                     '--kernel','quintic','--style',parameters['style'],'--jobs','1',
                     '--data-root',str(args.data_root.resolve()),'--output-dir',str((args.output/'training_runs').resolve()),'--boundary','--save-model','--physics-normalized']
            subprocess.run(command,check=True,cwd=ROOT)
        metadata=json.loads(result.with_suffix('.json').read_text())
        if metadata['training_rows']!=70000 or metadata['validation_rows']!=15000:
            raise ValueError(f'Incomplete training/validation coverage for {case}')
        for filename,field in [(args.data_root/'transistors'/f'{case}.csv','data_sha256'),
                               (args.data_root/'splits'/f'{case}_split.csv','split_sha256')]:
            if sha256(filename)!=metadata['provenance'][field]:
                raise ValueError(f'Changed data or split: {filename}')
        sources={suffix:result.with_suffix(suffix) for suffix in ('.joblib','.json','.validation.csv','.boundary.csv')}
        if parameters.get('gate_correction'):
            sources,metadata=corrected_artifacts(case,result,metadata,args.data_root,args.output,args.collect_only,parameters)
        outputs={}
        for suffix,destination in [('.joblib',args.output/'models'/f'{case}.joblib'),
                                   ('.json',args.output/'metrics'/f'{case}.json'),
                                   ('.validation.csv',args.output/'validation_predictions'/f'{case}.csv'),
                                   ('.boundary.csv',args.output/'boundary_predictions'/f'{case}.csv')]:
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(sources[suffix],destination)
            outputs[suffix]={'path':str(destination.resolve()),'sha256':sha256(destination)}
        manifest['cases'][case]={'parameters':parameters,'outputs':outputs,'provenance':metadata['provenance'],
                                 'uniform_validation_metrics':metadata['regression_metrics'],
                                 'uniform_power_metrics':metadata['power_metrics'],
                                 'boundary':metadata['boundary']}
        for artifact in outputs.values():
            path=Path(artifact['path'])
            manifest['validation_evidence'][str(path.relative_to(ROOT))]=artifact['sha256']
        comparison.append({'case':case,'model':configs['family'],'variant':stem,'rows':15000,**metadata['regression_metrics']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (args.output/'validation_comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    print(args.output/'manifest.json',flush=True)


if __name__=='__main__':main()
