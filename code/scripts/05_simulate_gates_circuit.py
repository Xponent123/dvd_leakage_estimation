#!/usr/bin/env python3
"""Simulate 18 gate cases and every eight-bit parity vector at shared corners."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lusr.circuits import GATES,WN,WP,LENGTH,gate_cell,cell_fets,parity_cells,propagate,circuit_deck,simulate_corner
from lusr.model_card import read_model_card
from lusr.spice import ROOT,DEFAULT_MODEL,find_ngspice,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corners',type=int,default=200)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--only',choices=('gates','circuit','both'),default='both')
    parser.add_argument('--output-root',type=Path,default=ROOT,
                        help='Isolated experiment root containing data/corners.csv')
    args=parser.parse_args()
    if not 1 <= args.corners <= 200:
        parser.error('--corners must be between 1 and 200')
    if args.workers < 1:
        parser.error('--workers must be positive')
    # Partial smoke runs must never overwrite the complete graded datasets.
    partial = args.corners != 200
    output_root=args.output_root.resolve()
    data_root = output_root/'data'/f'smoke_corners_{args.corners}' if partial else output_root/'data'
    report_root = output_root/'reports'/f'smoke_corners_{args.corners}' if partial else output_root/'reports'
    print(f'Reference output directory: {data_root}', flush=True)
    print(f'Manifest output directory: {report_root}', flush=True)
    exe=find_ngspice()
    cornerfile=output_root/'data/corners.csv'
    with cornerfile.open() as f:
        corners=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)][:args.corners]
    if len(corners)!=args.corners:
        raise ValueError('Requested corner count unavailable')
    for vector in range(256): propagate(vector)
    nominal={'corner_id':0,'vdd':.9,'temp':27.}
    for kind,params in read_model_card(DEFAULT_MODEL).nominal.items():
        nominal.update({kind[0]+'_'+k:v for k,v in params.items()})
    for kind in (*GATES,'PARITY8'):
        path=output_root/'netlists'/('circuit' if kind=='PARITY8' else 'gates')/(kind.lower()+'.cir')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(circuit_deck(nominal,kind,[0])[0])
    start=time.monotonic()
    tasks=[(c,g) for c in corners for g in GATES] if args.only!='circuit' else []
    if args.only!='gates':tasks += [(c,'PARITY8') for c in corners]
    records=[]
    def run(item):
        corner,kind=item
        directory=output_root/'logs'/('circuit' if kind=='PARITY8' else 'gates')
        return simulate_corner(corner,kind,exe,directory)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,item) for item in tasks]
        for i,future in enumerate(as_completed(futures),1):
            records.extend(future.result())
            if i%25==0 or i==len(futures):
                print(f'{i}/{len(futures)} corner/netlist jobs; {len(records)} operating points; {time.monotonic()-start:.1f}s',flush=True)
    manifest={'status':'SMOKE_SIMULATED' if partial else 'SIMULATED','corners':len(corners),'corner_sha256':sha256(cornerfile),
              'corner_ids':[int(c['corner_id']) for c in corners],
              'data_directory':str(data_root.relative_to(output_root)),
              'report_directory':str(report_root.relative_to(output_root)),
              'model_sha256':sha256(DEFAULT_MODEL),'ngspice_sha256':sha256(exe),'wn_m':WN,'wp_m':WP,'l_m':LENGTH,
              'parity':'8 total bits a0=LSB; output1 for odd parity (even-parity error)',
              'topology':'balanced 7 XOR, each4 NAND2;28NAND2,112FET',
              'primary_leakage':'-(I(VDD)+sum currents of input sources at VDD)) = all external DC source power/VDD',
              'secondary_leakage':'-I(VDD)', 'output_logic_checks':'all simulated rows passed 0.2/0.8 rail checks',
              'elapsed_s':time.monotonic()-start,'datasets':{}}
    for label,kinds in [('gates',set(GATES)),('circuit',{'PARITY8'})]:
        chosen=sorted([r for r in records if r['circuit'] in kinds],key=lambda r:(r['corner_id'],r['circuit'],r['vector']))
        if not chosen:continue
        expected=len(corners)*(18 if label=='gates' else 256)
        if len(chosen)!=expected:raise AssertionError('Wrong record count')
        path=data_root/label/'reference.csv'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(chosen[0]));writer.writeheader();writer.writerows(chosen)
        manifest['datasets'][label]={'rows':len(chosen),'sha256':sha256(path),'path':str(path.relative_to(output_root))}
    out=report_root/f'spice_{args.only}_manifest.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
