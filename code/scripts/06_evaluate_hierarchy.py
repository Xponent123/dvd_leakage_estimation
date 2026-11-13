#!/usr/bin/env python3
"""Apply four frozen transistor models; gate/circuit data remain evaluation-only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lusr.spice import ROOT,INPUT_COLUMNS,sha256
from lusr.circuits import gate_cell,cell_fets,propagate
SOURCE_ROOT=ROOT


def metrics(reference,prediction):
    y=np.asarray(reference,float);p=np.asarray(prediction,float)
    delta=p-y;rel=delta/np.maximum(np.abs(y),1e-18)
    ss=float(np.sum((y-y.mean())**2))
    return {'count':len(y),'spice_median_A':float(np.median(y)),
            'ml_median_A':float(np.median(p)),'bias_A':float(delta.mean()),
            'bias_percent':float(100*rel.mean()),'median_error_percent':float(100*np.median(np.abs(rel))),
            'p95_error_percent':float(100*np.quantile(np.abs(rel),.95)),
            'mean_error_percent':float(100*np.mean(np.abs(rel))),
            'mean_accuracy_percent':float(100*(1-np.mean(np.abs(rel)))),
            'r2':float(1-np.sum(delta**2)/ss) if ss>0 else None,
            'nonpositive_predictions':int(np.sum(p<=0)),'denominator_floor_count':int(np.sum(np.abs(y)<1e-18))}


def parity_plot(df,path,title,group=None):
    fig,ax=plt.subplots(figsize=(6.5,5.4),layout='constrained')
    positive=(df.spice_leakage_A>0)&(df.ml_leakage_A>0)
    good=df[positive]
    if good.empty:
        ax.text(.5,.5,'No positive pairs available for a log plot.\nAll observations retained in metrics.',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(title)
        fig.savefig(path.with_suffix('.png'),dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
        return
    if group:
        for name,part in good.groupby(group):
            ax.scatter(part.spice_leakage_A,part.ml_leakage_A,s=8,alpha=.45,label=name,rasterized=True)
        ax.legend(fontsize=8)
    else:
        density=ax.hexbin(good.spice_leakage_A,good.ml_leakage_A,gridsize=55,xscale='log',yscale='log',mincnt=1,cmap='viridis')
        fig.colorbar(density,ax=ax,label='Observations per bin')
    lo=min(good.spice_leakage_A.min(),good.ml_leakage_A.min())*.8
    hi=max(good.spice_leakage_A.max(),good.ml_leakage_A.max())*1.2
    ax.plot([lo,hi],[lo,hi],'k--',lw=1,label='Ideal')
    ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),
           xlabel='SPICE leakage (A)',ylabel='Hierarchical ML leakage (A)',title=title)
    excluded=int((~positive).sum())
    if excluded:ax.text(.02,.98,f'{excluded} nonpositive predictions excluded from log plot\nAll retained in metrics',transform=ax.transAxes,va='top',fontsize=8)
    ax.grid(True,which='both',alpha=.15)
    fig.savefig(path.with_suffix('.png'),dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT,help='Root containing data, models, reports and results')
    parser.add_argument('--jobs',type=int,default=2,help='Numerical-library thread limit during prediction')
    args=parser.parse_args()
    ROOT=args.output_root.resolve()
    results=ROOT/'results';results.mkdir(exist_ok=True)
    figures=ROOT/'reports/figures';figures.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((results/'ml_manifest.json').read_text())
    if manifest['status']!='COMPLETED':
        raise ValueError('Freeze the selected models and complete held-out testing before hierarchy evaluation')
    for source,digest in manifest.get('evaluation_source_sha256',{}).items():
        if sha256(SOURCE_ROOT/source)!=digest:
            raise ValueError(f'Model/evaluation source changed after held-out testing: {source}')
    audit=json.loads((ROOT/'reports/circuit_reference_validation.json').read_text())
    if audit['status']!='PASSED':
        raise ValueError('Independent reference verification must pass before hierarchy evaluation')
    for label,key in [('gates','gate_reference_sha256'),('circuit','circuit_reference_sha256')]:
        if sha256(ROOT/f'data/{label}/reference.csv')!=audit[key]:
            raise ValueError(f'Reference changed after verification: {label}')
    if sha256(ROOT/'data/corners.csv')!=audit['corner_sha256']:
        raise ValueError('Corner file changed after verification')
    for case,metadata in manifest['cases'].items():
        if sha256(ROOT/'models'/f'{case}.joblib')!=metadata['selected_model_sha256']:
            raise ValueError(f'Frozen model changed: {case}')
    gates=pd.read_csv(ROOT/'data/gates/reference.csv',dtype={'input_bits':str})
    circuit=pd.read_csv(ROOT/'data/circuit/reference.csv',dtype={'input_bits':str})
    corners=pd.read_csv(ROOT/'data/corners.csv').set_index('corner_id')
    if len(gates)!=3600 or len(circuit)!=51200:
        raise ValueError('Full 200-corner reference runs must finish before final evaluation')
    models={case:joblib.load(ROOT/'models'/f'{case}.joblib') for case in ('nmos_on','nmos_off','pmos_on','pmos_off')}
    devices=[]
    for index,row in gates.iterrows():
        cell=gate_cell(row.circuit)
        corner=corners.loc[row.corner_id]
        voltages={'0':0.,'vdd':row.vdd,**json.loads(row.node_voltages_json)}
        bits=tuple(int(c) for c in row.input_bits)
        voltages.update({name:bit*row.vdd for name,bit in zip(cell.inputs,bits)})
        for fet in cell_fets(cell):
            d,g,s,b=[voltages[n] for n in (fet.d,fet.g,fet.s,fet.b)]
            on=(g>row.vdd/2) if fet.kind=='nmos' else (g<row.vdd/2)
            sample={'vdd':row.vdd,'temp':row.temp,'w':fet.w,'vd':d,'vs':s,'vg':g,'vb':b,
                    **{p:corner[fet.kind[0]+'_'+p] for p in ('vth0','toxe','ndep','u0','xj')}}
            devices.append({'gate_row':index,'corner_id':row.corner_id,'gate':row.circuit,'vector':row.vector,
                'device':fet.name,'case':fet.kind+('_on' if on else '_off'),
                'drain_net':fet.d,'gate_net':fet.g,'source_net':fet.s,'body_net':fet.b,**sample})
    device_df=pd.DataFrame(devices)
    for case,model in models.items():
        select=device_df.case==case
        with threadpool_limits(limits=args.jobs):
            predictions=np.asarray(model.predict(device_df.loc[select,list(INPUT_COLUMNS)].to_numpy()),float)
        if predictions.shape!=(int(select.sum()),4) or not np.isfinite(predictions).all():
            raise ValueError(f'Invalid predictions from {case}')
        device_df.loc[select,['id_ml','ig_ml','is_raw_ml','ib_ml']]=predictions
    d=device_df
    d['is_closed_ml']=-(d.id_ml+d.ig_ml+d.ib_ml)
    d['raw_kcl_A']=d.id_ml+d.ig_ml+d.is_raw_ml+d.ib_ml
    d['power_closed_W']=(d.vd-d.vs)*d.id_ml+(d.vg-d.vs)*d.ig_ml+(d.vb-d.vs)*d.ib_ml
    d['power_raw_W']=d.vd*d.id_ml+d.vg*d.ig_ml+d.vs*d.is_raw_ml+d.vb*d.ib_ml
    d['supply_ml_A']=sum(d[current]*(d[net]=='vdd') for current,net in
        [('id_ml','drain_net'),('ig_ml','gate_net'),('is_closed_ml','source_net'),('ib_ml','body_net')])
    d.to_csv(results/'gate_transistor_predictions.csv',index=False)
    grouped=d.groupby('gate_row')
    gates['ml_leakage_A']=grouped.power_closed_W.sum()/gates.vdd
    gates['ml_raw_power_leakage_A']=grouped.power_raw_W.sum()/gates.vdd
    gates['ml_supply_A']=grouped.supply_ml_A.sum()
    gates['internal_voltage_source']='Recorded standalone-gate SPICE voltages reused by ML; no voltage predictor'
    gates['ml_node_voltages_json']=gates.node_voltages_json
    gates['relative_error']=(gates.ml_leakage_A-gates.spice_leakage_A)/np.maximum(np.abs(gates.spice_leakage_A),1e-18)
    gates.to_csv(results/'gate_predictions.csv',index=False)
    gate_metrics=[]
    for (name,bits),part in gates.groupby(['circuit','input_bits']):
        gate_metrics.append({'gate':name,'input_bits':bits,**metrics(part.spice_leakage_A,part.ml_leakage_A)})
    pd.DataFrame(gate_metrics).to_csv(results/'gate_metrics.csv',index=False)
    # Boolean propagation determines the count of NAND input states per vector.
    # Matrix multiplication is exactly the same 28-cell sum, not a new model.
    counts=np.zeros((256,4),dtype=int)
    for vector in range(256):
        for cell,bits in propagate(vector)[1]:
            if cell.gate!='NAND2':raise AssertionError('Unexpected parity cell')
            counts[vector,2*bits[0]+bits[1]]+=1
    if not np.all(counts.sum(axis=1)==28):raise AssertionError('Incorrect gate count')
    pd.DataFrame(counts,columns=['nand_00','nand_01','nand_10','nand_11']).to_csv(results/'parity_gate_state_counts.csv',index_label='vector')
    corner_ids=sorted(corners.index)
    row_corners=np.array([corner_ids.index(i) for i in circuit.corner_id])
    for column,target in [('ml_leakage_A','ml_leakage_A'),('spice_leakage_A','standalone_spice_sum_A'),('ml_supply_A','ml_supply_sum_A')]:
        table=gates[gates.circuit=='NAND2'].pivot(index='corner_id',columns='vector',values=column).loc[corner_ids,[0,1,2,3]].to_numpy()
        sums=table@counts.T
        circuit[target]=sums[row_corners,circuit.vector.to_numpy()]
    circuit['relative_error']=(circuit.ml_leakage_A-circuit.spice_leakage_A)/np.maximum(np.abs(circuit.spice_leakage_A),1e-18)
    circuit.to_csv(results/'circuit_predictions.csv',index=False)
    vector_metrics=[{'input_bits':format(v,'08b'),'vector':v,**metrics(part.spice_leakage_A,part.ml_leakage_A)}
                    for v,part in circuit.groupby('vector')]
    pd.DataFrame(vector_metrics).to_csv(results/'circuit_metrics_by_vector.csv',index=False)
    summary={'gate':metrics(gates.spice_leakage_A,gates.ml_leakage_A),
             'circuit':metrics(circuit.spice_leakage_A,circuit.ml_leakage_A),
             'ideal_gate_composition_only':metrics(circuit.spice_leakage_A,circuit.standalone_spice_sum_A),
             'ml_vs_standalone_gate_sum':metrics(circuit.standalone_spice_sum_A,circuit.ml_leakage_A),
             'source_closure':{'max_raw_fet_kcl_A':float(d.raw_kcl_A.abs().max()),
                               'median_raw_fet_kcl_A':float(d.raw_kcl_A.abs().median()),
                               'negative_fet_power_count':int((d.power_closed_W<0).sum()),
                               'negative_gate_power_count':int((gates.ml_leakage_A<0).sum()),
                               'policy':'Retain raw four currents; close source current for energy aggregation; no clipping'},
             'voltage_provenance':'Gate SPICE node voltages reused; circuit ML uses only ideal Boolean propagation and the gate lookup at the same corner.',
             'evaluation_isolation':'No fitting or model selection occurs in this script.',
             'ml_manifest_sha256':sha256(results/'ml_manifest.json'),
             'reference_audit_sha256':sha256(ROOT/'reports/circuit_reference_validation.json'),
             'hierarchy_source_sha256':sha256(Path(__file__)),
             'model_hashes':{k:sha256(ROOT/'models'/f'{k}.joblib') for k in models}}
    (results/'hierarchy_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    parity_plot(gates,figures/'gate_parity','Gate leakage: 3,600 evaluations',group='circuit')
    parity_plot(circuit,figures/'circuit_parity','Parity checker: 51,200 evaluations')
    worst=gates.assign(abs_error=gates.relative_error.abs()).nlargest(20,'abs_error')
    worst.to_csv(results/'worst_gate_cases.csv',index=False)
    circuit.assign(abs_error=circuit.relative_error.abs()).nlargest(20,'abs_error').to_csv(results/'worst_circuit_cases.csv',index=False)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
