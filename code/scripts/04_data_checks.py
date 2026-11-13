#!/usr/bin/env python3
"""Check measured KCL and run controlled, independently instantiated SPICE trends.

Trend W=L=32nm is an illustrative single-device operating point, not gate sizing.
The 246 trend measurements are never used as training examples.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lusr.model_card import read_model_card
from lusr.sampling import CASES, CHANNEL_LENGTH, INPUT_COLUMNS, NOMINAL_VDD, relative_bounds
from lusr.spice import CURRENT_COLUMNS, DEFAULT_MODEL, find_ngspice, fresh_deck, run_deck

ABSTOL = 1e-18
RELTOL = 1e-6
SWEEPS = {
    'off_current_vs_temperature': ('temp', 'off', 'id', 'Temperature (°C)', 1.0, 'OFF drain current |Id| (A)'),
    'off_current_vs_vth0': ('vth0', 'off', 'id', 'Signed threshold parameter vth0 (V)', 1.0, 'OFF drain current |Id| (A)'),
    'gate_current_vs_toxe': ('toxe', 'on', 'ig', 'Electrical oxide thickness toxe (nm)', 1e9, 'ON gate current |Ig| (A)'),
}
REFERENCES = {
    'ngspice_42_manual': 'https://ngspice.sourceforge.io/docs/ngspice-42-manual.pdf',
    'bsim4_oxide_manual': 'https://ngspice.sourceforge.io/external-documents/models/BSIM480_Manual.pdf',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def kcl_summary(values):
    if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
        raise ValueError('KCL requires a finite nonempty N-by-four matrix')
    residual = np.abs(values.sum(axis=1))
    scale = np.abs(values).sum(axis=1)
    normalized = residual / np.maximum(scale, ABSTOL)
    mixed = ABSTOL + RELTOL * scale
    worst = int(np.argmax(normalized))
    def stats(a):
        return {'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)), 'max': float(np.max(a))}
    return {
        'rows': len(values), 'all_currents_finite': True,
        'absolute_residual_A': stats(residual), 'normalized_residual': stats(normalized),
        'normalization': 'abs(id+ig+is+ib) / max(abs(id)+abs(ig)+abs(is)+abs(ib), 1e-18 A)',
        'above_abstol_count': int(np.count_nonzero(residual > ABSTOL)),
        'mixed_tolerance_definition': '1e-18 A + 1e-6 * (abs(id)+abs(ig)+abs(is)+abs(ib))',
        'above_mixed_tolerance_count': int(np.count_nonzero(residual > mixed)),
        'max_mixed_tolerance_ratio': float(np.max(residual / mixed)),
        'worst_normalized_row_index': worst,
        'worst_normalized_row_currents_A': dict(zip(CURRENT_COLUMNS, map(float, values[worst]))),
    }


def check_datasets(directory, expected_rows):
    report = {}
    for case in CASES:
        path = directory / f'{case}.csv'
        frame = pd.read_csv(path)
        if tuple(frame.columns) != (*INPUT_COLUMNS, *CURRENT_COLUMNS) or len(frame) != expected_rows:
            raise ValueError(f'Unexpected dataset schema/row count in {path}')
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite values in {path}')
        report[case] = {'csv': str(path.resolve()), 'sha256': digest(path),
                        **kcl_summary(frame[list(CURRENT_COLUMNS)].to_numpy(dtype=float))}
    return report


def trend_inputs(nominal):
    rows = []
    for trend, (parameter, state, *_rest) in SWEEPS.items():
        for polarity in ('nmos', 'pmos'):
            pmos = polarity == 'pmos'
            case = f'{polarity}_{state}'
            baseline = {
                'vdd': NOMINAL_VDD, 'temp': 27.0, 'w': CHANNEL_LENGTH,
                'vd': 0.0 if pmos else NOMINAL_VDD, 'vs': NOMINAL_VDD if pmos else 0.0,
                'vg': NOMINAL_VDD if case in ('nmos_on', 'pmos_off') else 0.0,
                'vb': NOMINAL_VDD if pmos else 0.0, **nominal[polarity],
            }
            bounds = (-40.0, 125.0) if parameter == 'temp' else relative_bounds(baseline[parameter])
            for index, value in enumerate(np.linspace(*bounds, 41)):
                rows.append({'trend': trend, 'polarity': polarity, 'case': case,
                             'point_index': index, 'swept_parameter': parameter,
                             'swept_value': float(value), **baseline, parameter: float(value)})
    return rows


def trend_identity(rows, executable, model):
    return {'model_sha256': digest(model), 'executable_sha256': digest(executable),
            'simulation_source_sha256': digest(ROOT / 'lusr/spice.py'),
            'input_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


def run_trends(rows, executable, model, output, workers):
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    def run(row):
        path = logs / f"{row['trend']}_{row['polarity']}_{row['point_index']:02d}.cir"
        sample = {column: row[column] for column in INPUT_COLUMNS}
        values, warnings = run_deck(fresh_deck(row['case'], sample, model), path, executable, 1)
        return {**row, **values[0], 'warning_count': warnings}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, rows))
    frame = pd.DataFrame(results)
    frame['kcl_residual_A'] = frame[list(CURRENT_COLUMNS)].sum(axis=1)
    frame.to_csv(output / 'trends.csv', index=False, float_format='%.17g')
    warning_text = {}
    for path in sorted(logs.glob('*.log')):
        for line in path.read_text().splitlines():
            if re.match(r'\s*Warning\b', line, re.I):
                message = line.strip()
                warning_text[message] = warning_text.get(message, 0) + 1
    metadata = {
        'method': 'Fresh ngspice process and embedded model per point; exactly one swept input.',
        'rows': len(frame), 'points_per_polarity_per_sweep': 41, 'fresh_process_count': len(frame),
        'warning_count': int(frame.warning_count.sum()), 'warning_messages': warning_text,
        **trend_identity(rows, executable, model), 'csv_sha256': digest(output / 'trends.csv'),
    }
    json_write(output / 'trend_manifest.json', metadata)
    return frame, metadata


def reuse_trends(rows, executable, model, output):
    metadata = json.loads((output / 'trend_manifest.json').read_text())
    identity = {**trend_identity(rows, executable, model), 'csv_sha256': digest(output / 'trends.csv')}
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise ValueError('Saved trend provenance differs; run again without --reuse-trends')
    frame = pd.read_csv(output / 'trends.csv')
    if len(frame) != 246 or not np.isfinite(frame[list(CURRENT_COLUMNS)].to_numpy()).all():
        raise ValueError('Invalid saved trend rows')
    return frame, metadata


def plot_trends(frame, output):
    figures = output / 'plots'
    figures.mkdir(parents=True, exist_ok=True)
    details = {}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'figure.dpi': 120})
    for name, (parameter, state, terminal, xlabel, factor, ylabel) in SWEEPS.items():
        fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.2), layout='constrained')
        stats = {}
        for axis, polarity, color in zip(axes, ('nmos', 'pmos'), ('#165da8', '#c75b22'), strict=True):
            selected = frame[(frame.trend == name) & (frame.polarity == polarity)].sort_values(parameter)
            x = selected[parameter].to_numpy() * factor
            y = np.abs(selected[terminal].to_numpy())
            if len(x) != 41 or np.any(y <= 0):
                raise ValueError(f'Incomplete/nonpositive log data for {name}/{polarity}')
            axis.semilogy(x, y, color=color, lw=2)
            axis.set(xlabel=xlabel, ylabel=ylabel, title=polarity.upper())
            axis.grid(True, which='major', color='#dedede', alpha=0.8, linewidth=0.7)
            stats[polarity] = {
                'first_swept_value': float(selected[parameter].iloc[0]),
                'last_swept_value': float(selected[parameter].iloc[-1]),
                'first_current_magnitude_A': float(y[0]), 'last_current_magnitude_A': float(y[-1]),
                'last_over_first_current_ratio': float(y[-1] / y[0]),
                'increasing_steps': int(np.count_nonzero(np.diff(y) > 0)),
                'decreasing_steps': int(np.count_nonzero(np.diff(y) < 0)),
                'kcl': kcl_summary(selected[list(CURRENT_COLUMNS)].to_numpy(dtype=float)),
            }
        title = {'temp': 'OFF current versus temperature', 'vth0': 'OFF current versus threshold parameter',
                 'toxe': 'Gate current versus oxide thickness'}[parameter]
        fig.suptitle(title + '\nVDD = 0.9 V; L = W = 32 nm; all other parameters nominal', fontsize=12)
        for extension in ('png', 'pdf'):
            fig.savefig(figures / f'{name}.{extension}', bbox_inches='tight')
        plt.close(fig)
        details[name] = stats
    return details


def write_audit(output, summary):
    lines = ['# Terminal-current checks and controlled SPICE trends', '', f"Status: {summary['status']}.", '',
        'Terminal currents are signed into the MOSFET, measured as the negatives of four separate voltage-source branch currents. KCL uses measured values without projecting or correcting them. [ngspice current convention](https://ngspice.sourceforge.io/docs/ngspice-42-manual.pdf).', '',
        'The absolute residual is |Id+Ig+Is+Ib|. Its normalization uses max(sum of absolute terminal currents, 1e-18 A). The mixed diagnostic threshold is 1e-18 A + 1e-6 × sum of absolute terminal currents. Exceeding the absolute 1e-18 A tolerance alone does not mean the simulator failed to converge.', '']
    if summary['transistor_datasets']:
        lines += ['| Case | Rows | Max residual (A) | Max normalized residual | Above 1e-18 A | Above mixed threshold |',
                  '|---|---:|---:|---:|---:|---:|']
        for case, v in summary['transistor_datasets'].items():
            lines.append(f"| {case} | {v['rows']} | {v['absolute_residual_A']['max']:.6g} | {v['normalized_residual']['max']:.6g} | {v['above_abstol_count']} | {v['above_mixed_tolerance_count']} |")
    else:
        lines.append('Dataset validation is pending in this trends-only invocation.')
    lines += ['', 'The three plots use 41 independent DC solves per polarity and sweep, 246 fresh processes total. Only temperature, vth0, or toxe changes within a sweep. These are controlled trends, not random-data scatter plots; none enters model fitting.', '',
        'Illustrative baseline: VDD=0.9 V, drawn L=W=32 nm, T=27 °C and polarity-specific nominal process parameters. NMOS drain/source/body are 0.9/0/0 V; PMOS are 0/0.9/0.9 V. The first two plots use OFF gate bias; the oxide plot uses ON gate bias. This W does not set gate/circuit sizing. PMOS vth0 retains its negative sign.', '',
        'The assigned model is unchanged. Derived fresh decks change only the swept entry; supplied TOXP, DTOX, TOXM and TOXREF remain nominal. BSIM4 relates DTOX to TOXE−TOXP. An inconsistent explicit triple warns that DTOX is ignored; the older implementation can also warn on nominal PMOS because of floating-point equality. The measured oxide trend shows that TOXE remains active. [BSIM4 oxide definitions](https://ngspice.sourceforge.io/external-documents/models/BSIM480_Manual.pdf).', '',
        'Warning messages: `' + json.dumps(summary['trend_simulation']['warning_messages'], sort_keys=True) + '`.', '',
        'Raw currents: `trends.csv`. Inputs and logs: `logs/`. PNG/PDF figures: `plots/`. Provenance and numerical summaries: `trend_manifest.json`, `data_checks.json`.', '',
        '| Sweep | Polarity | First magnitude (A) | Last magnitude (A) | Last / first |', '|---|---|---:|---:|---:|']
    for name, polarities in summary['trends'].items():
        for polarity, v in polarities.items():
            lines.append(f"| {name} | {polarity} | {v['first_current_magnitude_A']:.6g} | {v['last_current_magnitude_A']:.6g} | {v['last_over_first_current_ratio']:.6g} |")
    (output / 'DATA_CHECKS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/transistors')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/data_checks')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--ngspice')
    parser.add_argument('--expected-rows', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--trends-only', action='store_true')
    parser.add_argument('--reuse-trends', action='store_true')
    args = parser.parse_args()
    if args.workers <= 0 or args.expected_rows <= 0:
        parser.error('workers and expected-rows must be positive')
    model = read_model_card(args.model)
    executable = find_ngspice(args.ngspice)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    datasets = {} if args.trends_only else check_datasets(args.data, args.expected_rows)
    rows = trend_inputs(model.nominal)
    print(f"{'Reusing' if args.reuse_trends else 'Simulating'} 246 controlled single-FET trend points.", flush=True)
    function = reuse_trends if args.reuse_trends else run_trends
    extras = () if args.reuse_trends else (args.workers,)
    frame, metadata = function(rows, executable, model.path, output, *extras)
    trends = plot_trends(frame, output)
    summary = {
        'schema_version': 1, 'status': 'TRENDS_COMPLETE_DATASETS_PENDING' if args.trends_only else 'COMPLETE',
        'source_model': model.as_manifest(),
        'illustrative_baseline': {'vdd_V': NOMINAL_VDD, 'l_m': CHANNEL_LENGTH, 'w_m': CHANNEL_LENGTH,
                                  'temp_C': 27.0, 'gate_sizing_decision': False},
        'abstol_A': ABSTOL, 'reltol': RELTOL, 'transistor_datasets': datasets,
        'trend_simulation': metadata, 'trends': trends,
        'all_trend_points_kcl': kcl_summary(frame[list(CURRENT_COLUMNS)].to_numpy(dtype=float)),
        'simulator': {'executable': str(executable), 'version': subprocess.run([str(executable), '--version'], capture_output=True, text=True, check=True).stdout.strip()},
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
        'script_sha256': digest(Path(__file__).resolve()), 'references': REFERENCES,
    }
    json_write(output / 'data_checks.json', summary)
    write_audit(output, summary)
    print(f"{summary['status']}: {output / 'data_checks.json'}", flush=True)
    for case, v in datasets.items():
        print(f"{case}: {v['rows']} rows; max KCL={v['absolute_residual_A']['max']:.5g} A; max normalized={v['normalized_residual']['max']:.5g}; above mixed tolerance={v['above_mixed_tolerance_count']}")


if __name__ == '__main__':
    main()
