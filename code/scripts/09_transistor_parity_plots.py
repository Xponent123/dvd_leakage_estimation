#!/usr/bin/env python3
"""Draw the four held-out predicted-vs-SPICE parity plots (one per transistor case)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES = ('nmos_on', 'nmos_off', 'pmos_on', 'pmos_off')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[2])
    root = parser.parse_args().output_root
    out = root / 'reports/figures'
    out.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        frame = pd.read_csv(root / 'results/test_predictions' / f'{case}.csv')
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.1), layout='constrained')
        for terminal, ax in zip(('id', 'ig', 'is', 'ib'), axes.flat):
            y = frame[terminal + '_spice']
            p = frame[terminal + '_ml']
            threshold = max(float(np.median(np.abs(y))) * .01, 1e-18)
            ax.scatter(y, p, s=3, alpha=.35, rasterized=True)
            lo = min(y.min(), p.min())
            hi = max(y.max(), p.max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=.8)
            ax.set(xlabel=f'SPICE {terminal} (A)', ylabel=f'ML {terminal} (A)')
            ax.set_xscale('symlog', linthresh=threshold)
            ax.set_yscale('symlog', linthresh=threshold)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.15)
        fig.suptitle(case.upper().replace('_', ' ') + ' — held-out test, signed currents')
        fig.savefig(out / f'{case}_parity.png', dpi=180)
        plt.close(fig)
        print(out / f'{case}_parity.png')


if __name__ == '__main__':
    main()
