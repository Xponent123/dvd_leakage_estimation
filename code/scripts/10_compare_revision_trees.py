#!/usr/bin/env python3
"""Compare tree models on fresh uniform/boundary validation; never evaluate test."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from lusr.sampling import CASES, INPUT_COLUMNS
from lusr.ml import CURRENT_COLUMNS, regression_metrics
from lusr.revision_models import CanonicalTreeModel, boundary_metrics
from lusr.physics_surrogate import encode
from lusr.spice import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=ROOT/'improvements_99')
    p.add_argument('--boundary', type=Path, default=ROOT/'experiments/physics/boundary_validation')
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--families', nargs='+', default=['random_forest','extra_trees','hist_gradient_boosting'])
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()
    out = args.output_root/'results/tree_comparison'; out.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        split = pd.read_csv(args.output_root/f'data/splits/{case}_split.csv')
        # pandas filtering prevents retaining or using test targets in fitting/scoring.
        frame = pd.read_csv(args.output_root/f'data/transistors/{case}.csv')
        training = frame.iloc[split.loc[split.split=='train','row_index']]
        validation = frame.iloc[split.loc[split.split=='validation','row_index']]
        del frame
        tx, ty = training[list(INPUT_COLUMNS)].to_numpy(), training[list(CURRENT_COLUMNS)].to_numpy()
        vx, vy = validation[list(INPUT_COLUMNS)].to_numpy(), validation[list(CURRENT_COLUMNS)].to_numpy()
        boundary = pd.read_csv(args.boundary/f'{case}.csv')
        bx, by = boundary[list(INPUT_COLUMNS)].to_numpy(), boundary[list(CURRENT_COLUMNS)].to_numpy()
        for family in args.families:
            path = out/f'{case}_{family}.json'
            if path.exists():
                old = json.loads(path.read_text())
                if (old['source_sha256']==sha256(Path(__file__))
                    and old['model_source_sha256']==sha256(ROOT/'lusr/revision_models.py')
                    and old['physics_source_sha256']==sha256(ROOT/'lusr/physics_surrogate.py')
                    and old['data_sha256']==sha256(args.output_root/f'data/transistors/{case}.csv')
                    and old['boundary_sha256']==sha256(args.boundary/f'{case}.csv')):
                    print(f'Cached {case}/{family}', flush=True); continue
            start=time.monotonic()
            with threadpool_limits(limits=args.jobs):
                model=CanonicalTreeModel(case, family, jobs=args.jobs).fit(tx, ty)
                vp, bp = model.predict(vx), model.predict(bx)
            # Keep all terminal metrics; component RMSE is only a supplemental diagnostic.
            score={'case':case,'model':family,'training_rows':len(tx),'validation_rows':len(vx),
                   'seconds':time.monotonic()-start,**regression_metrics(vy,vp),**boundary_metrics(bx,by,bp),
                   'source_sha256':sha256(Path(__file__)),
                   'model_source_sha256':sha256(ROOT/'lusr/revision_models.py'),
                   'physics_source_sha256':sha256(ROOT/'lusr/physics_surrogate.py'),
                   'data_sha256':sha256(args.output_root/f'data/transistors/{case}.csv'),
                   'boundary_sha256':sha256(args.boundary/f'{case}.csv'),
                   'hyperparameters':str(model.estimator.get_params()),
                   'terminals':{name:regression_metrics(vy[:,i],vp[:,i]) for i,name in enumerate(CURRENT_COLUMNS)}}
            path.write_text(json.dumps(score,indent=2)+'\n')
            print(json.dumps({k:score[k] for k in ['case','model','seconds','mape_pct','boundary_power_mape_pct']}),flush=True)
            del model


if __name__=='__main__':
    main()
