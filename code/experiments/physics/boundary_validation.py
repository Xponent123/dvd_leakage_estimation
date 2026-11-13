"""Independent single-FET supplemental rail validation; never training data."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from lusr.sampling import CASES,INPUT_COLUMNS,sample_transistors
from lusr.model_card import read_model_card
from lusr.spice import CURRENT_COLUMNS,DEFAULT_MODEL,find_ngspice,simulate_rows,sha256

OUT=ROOT/'experiments/physics/boundary_validation'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'status':'SUPPLEMENTAL_VALIDATION_ONLY','seed':2026102407,'rows_per_case':512,
              'source':'Independent random single-FET PVT/W draws, unrelated to200gate/circuitcorners; no measured circuit nodes used.',
              'bias_groups':['(0,0)','(0,1)','(1,0)','(1,1)','(epsilon,0)','(0,epsilon)','(1-epsilon,1)','(1,1-epsilon)'],
              'group_rows':64,'epsilon_distribution':'Uniform(1e-6,1e-3), ratios relative toVDD',
              'predeclared_selection':'Mean absolute relative error of per-FET dissipated power/VDD over nonzero-bias configurations, denominator floor1e-18A; tie-break uniform-validation pooledlogRMSE of positive smoothcomponents. Always report all rawterminal metrics and allzero-bias invariant residuals separately.',
              'power_formula':'[(Vd-Vs)Id+(Vg-Vs)Ig+(Vb-Vs)Ib]/VDD; explicit sourceKCLclosure, sameprojectaggregation',
              'physical_zero_rule':'Allfourterminalvoltages identical=>exactzero power; excluded only from nonzero-bias selection and listed separately, rawcurrentfloorcounts retained.',
              'prohibited_uses':['Never train on supplementalvalidation','Never use gate/circuit currents or200corners for training/selection','Never relabel this as uniform assigneddataset'],
              'model_sha256':sha256(DEFAULT_MODEL),'datasets':{}}
    protocol_path=OUT/'protocol.json'
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    streams=np.random.SeedSequence(protocol['seed']).spawn(4)
    nominal=read_model_card(DEFAULT_MODEL).nominal
    exe=find_ngspice()
    for case,stream in zip(CASES,streams):
        rng=np.random.default_rng(stream)
        sampled=sample_transistors(case,512,nominal,rng)
        eps=rng.uniform(1e-6,1e-3,size=512)
        d=np.repeat([0,0,1,1,0,0,1,1],64).astype(float)
        s=np.repeat([0,1,0,1,0,0,1,1],64).astype(float)
        d[256:320]=eps[256:320];s[320:384]=eps[320:384]
        d[384:448]=1-eps[384:448];s[448:]=1-eps[448:]
        sampled['vd']=d*sampled['vdd'];sampled['vs']=s*sampled['vdd']
        inputs=[{key:float(sampled[key][i]) for key in INPUT_COLUMNS} for i in range(512)]
        path=OUT/f'{case}_inputs.csv'
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=INPUT_COLUMNS);writer.writeheader();writer.writerows(inputs)
        currents,metadata=simulate_rows(case,inputs,exe,DEFAULT_MODEL,OUT/'logs'/case,chunk_size=128,workers=2)
        result=OUT/f'{case}.csv'
        with result.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=(*INPUT_COLUMNS,*CURRENT_COLUMNS));writer.writeheader();writer.writerows({**x,**y} for x,y in zip(inputs,currents))
        protocol['datasets'][case]={'input_sha256':sha256(path),'csv_sha256':sha256(result),'simulation':metadata}
        protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
        print(case,512,'supplementalrailvalidationrows',flush=True)

if __name__=='__main__':main()
