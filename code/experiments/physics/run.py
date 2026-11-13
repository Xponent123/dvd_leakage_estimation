"""Original training/validation-only smooth approximation experiments."""
import argparse,csv,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from lusr.physics_surrogate import PhysicsSurrogate,PhysicsNormalizedSurrogate,encode,decode
from lusr.sampling import INPUT_COLUMNS
from lusr.ml import regression_metrics
from lusr.spice import sha256

def metrics(y,p):
    e=abs(p-y)/np.maximum(abs(y),1e-18)
    return {'mape_pct':float(e.mean()*100),'median_pct':float(np.median(e)*100),'p95_pct':float(np.quantile(e,.95)*100),
            'terminal_mape_pct':(e.mean(axis=0)*100).tolist(),'terminal_median_pct':(np.median(e,axis=0)*100).tolist(),
            'terminal_p95_pct':(np.quantile(e,.95,axis=0)*100).tolist(),'floor_reference_count':int((abs(y)<1e-18).sum())}

def energy(x,y):
    return ((x[:,3]-x[:,4])*y[:,0]+(x[:,5]-x[:,4])*y[:,1]+(x[:,6]-x[:,4])*y[:,3])/x[:,0]

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='nmos_off');p.add_argument('--method',default='polynomial');p.add_argument('--degree',type=int,default=4);p.add_argument('--neighbors',type=int,default=128);p.add_argument('--regularization',type=float,default=1e-8);p.add_argument('--kernel',default='quintic');p.add_argument('--style',default='physical');p.add_argument('--jobs',type=int,default=2);p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/physics');p.add_argument('--validation-limit',type=int);p.add_argument('--boundary',action='store_true');p.add_argument('--save-model',action='store_true');p.add_argument('--physics-normalized',action='store_true');p.add_argument('--edge-normalized',action='store_true');a=p.parse_args()
    frame=pd.read_csv(a.data_root/'transistors'/f'{a.case}.csv')
    split=pd.read_csv(a.data_root/'splits'/f'{a.case}_split.csv')
    train=split[split.split=='train'].row_index.to_numpy();valid=split[split.split=='validation'].row_index.to_numpy()
    if a.validation_limit:valid=valid[:a.validation_limit]
    x=frame[list(INPUT_COLUMNS)].to_numpy();y=frame[['id','ig','is','ib']].to_numpy()
    inverse=decode(x[train],encode(x[train],y[train],a.case),a.case)
    if not np.allclose(inverse,y[train],atol=1e-18,rtol=1e-10):raise AssertionError('Transform inverse mismatch')
    cls=PhysicsNormalizedSurrogate if a.physics_normalized or a.edge_normalized else PhysicsSurrogate
    model=cls(a.case,a.method,a.degree,a.regularization,a.neighbors,a.kernel,a.style)
    if a.edge_normalized:model.baseline_kind='edge'
    start=time.monotonic()
    with threadpool_limits(limits=a.jobs):
        model.fit(x[train],y[train]);fit_s=time.monotonic()-start
        prediction=np.vstack([model.predict(x[valid[i:i+256]]) for i in range(0,len(valid),256)])
    result={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'training_rows':len(train),'validation_rows':len(valid),'fit_s':fit_s,'total_s':time.monotonic()-start,**metrics(y[valid],prediction)}
    result['regression_metrics']=regression_metrics(y[valid],prediction)
    result['provenance']={'data_sha256':sha256(a.data_root/'transistors'/f'{a.case}.csv'),'split_sha256':sha256(a.data_root/'splits'/f'{a.case}_split.csv'),'model_source_sha256':sha256(ROOT/'lusr/physics_surrogate.py'),'runner_source_sha256':sha256(Path(__file__)),'evaluation_partition':'validation only; testindices never selected'}
    result['power_metrics']=metrics(energy(x[valid],y[valid]),energy(x[valid],prediction))
    result['smooth_log_rmse']=np.sqrt(np.mean((encode(x[valid],prediction,a.case)-encode(x[valid],y[valid],a.case))**2,axis=0)).tolist()
    if a.boundary:
        b=pd.read_csv(ROOT/'experiments/physics/boundary_validation'/f'{a.case}.csv')
        bx=b[list(INPUT_COLUMNS)].to_numpy();by=b[['id','ig','is','ib']].to_numpy()
        with threadpool_limits(limits=a.jobs):bp=np.vstack([model.predict(bx[i:i+128]) for i in range(0,len(bx),128)])
        nonzero=np.ptp(bx[:,[3,4,5,6]],axis=1)>0
        result['boundary']={'raw_terminal_metrics':metrics(by,bp),'power_nonzero_bias_metrics':metrics(energy(bx,by)[nonzero],energy(bx,bp)[nonzero]),'zero_bias_rows':int((~nonzero).sum()),'zero_bias_predicted_current_max_A':float(abs(bp[~nonzero]).max()) if (~nonzero).any() else None}
        result['boundary']['groups']={}
        group_names=['0_0','0_1','1_0','1_1','eps_0','0_eps','1-eps_1','1_1-eps']
        for group,label in enumerate(group_names):
            sl=slice(group*64,(group+1)*64)
            result['boundary']['groups'][label]={'raw_terminal_metrics':metrics(by[sl],bp[sl]),'power_metrics':metrics(energy(bx[sl],by[sl]),energy(bx[sl],bp[sl]))}
    name=f'{a.case}_{a.method}_{a.style}_d{a.degree}_n{a.neighbors}_r{a.regularization}_{a.kernel}_train{len(train)}_val{len(valid)}'+('_edge' if a.edge_normalized else '_normalized' if a.physics_normalized else '')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    path=a.output_dir/f'{name}.json';path.write_text(json.dumps(result,indent=2)+'\n')
    if a.save_model:
        import joblib
        joblib.dump(model,path.with_suffix('.joblib'),compress=3)
        data={'row_index':valid}
        for index,t in enumerate(['id','ig','is','ib']):data[t+'_spice']=y[valid,index];data[t+'_ml']=prediction[:,index]
        pd.DataFrame(data).to_csv(path.with_suffix('.validation.csv'),index=False)
        if a.boundary:
            data={'row_index':np.arange(len(bx))}
            for index,t in enumerate(['id','ig','is','ib']):data[t+'_spice']=by[:,index];data[t+'_ml']=bp[:,index]
            data['spice_power_over_vdd_A']=energy(bx,by);data['ml_power_over_vdd_A']=energy(bx,bp)
            pd.DataFrame(data).to_csv(path.with_suffix('.boundary.csv'),index=False)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
