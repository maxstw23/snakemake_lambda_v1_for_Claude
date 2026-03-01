"""
Compare efficiency-corrected vs uncorrected Lambda v1 results.

Outputs a 3-page PDF:
  Page 1 : v1(y) for Lambda       — eff-corrected vs no-eff (3×3 centrality grid)
  Page 2 : v1(y) for Lambdabar    — same
  Page 3 : dv1/dy vs centrality for Lambda and Lambdabar, both versions
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import yaml
from iminuit import cost, Minuit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from uncertainties import unumpy

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)

CENTRALITIES = np.array([75, 65, 55, 45, 35, 25, 15, 7.5, 2.5])
CENTRALITY_BINS = np.array([80, 70, 60, 50, 40, 30, 20, 10, 5, 0])


# ---------------------------------------------------------------------------
# Fit helpers (same approach as plot_v1.py)
# ---------------------------------------------------------------------------

def _linear(x, a):
    return a * x


def fit_dv1dy(y_u, v1_u) -> tuple[float, float]:
    """Fit dv1/dy with a linear function over |y| < y_cut."""
    y_cut = float(config['y_cut'])
    mask = np.abs(unumpy.nominal_values(y_u)) < y_cut
    c = cost.LeastSquares(
        unumpy.nominal_values(y_u[mask]),
        unumpy.nominal_values(v1_u[mask]),
        unumpy.std_devs(v1_u[mask]),
        _linear,
    )
    m = Minuit(c, a=0)
    m.migrad()
    if not m.valid:
        raise RuntimeError('Fit did not converge')
    return float(m.values['a']), float(m.errors['a'])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(paths: list[str], particle: str, flow: str = 'v1') -> dict[int, dict]:
    """Read a fit CSV produced by fit_v1.py and return a dict keyed by centrality bin."""
    for path in paths:
        if f'fit_{particle}_{flow}' in path:
            df = pd.read_csv(path, header=[0, 1], index_col=0)
            return {
                int(cen): {s: df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
                for cen in df.columns.levels[0]
            }
    raise FileNotFoundError(f'No CSV found for {particle}_{flow} among: {paths}')


def v1_arrays(data: dict, cen: int, res) -> tuple:
    """Return (y_u, v1_u) as uarrays, resolution-corrected, NaN bins dropped."""
    d = data[cen]
    n = len(d['values'])
    edges = np.linspace(-1., 1., n + 1)
    yc = 0.5 * (edges[:-1] + edges[1:])
    ye = np.diff(edges) / 2
    good = ~np.isnan(d['values'].astype(float))
    v1_u = unumpy.uarray(d['values'][good], d['errors'][good]) / res
    y_u  = unumpy.uarray(yc[good], ye[good])
    return y_u, v1_u


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_v1_grid(
    ax_grid,
    v1_eff: dict,
    v1_no: dict,
    resolution,
    cen_mask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill a pre-allocated 3×3 axes grid with v1(y) panels.
    Returns (dv1dy_eff, dv1dy_eff_err, dv1dy_no, dv1dy_no_err).
    """
    dv1dy_eff     = np.zeros(9)
    dv1dy_eff_err = np.zeros(9)
    dv1dy_no      = np.zeros(9)
    dv1dy_no_err  = np.zeros(9)
    x_fit = np.linspace(-1, 1, 101)

    for cen in range(1, 10):
        ax = ax_grid[(cen - 1) // 3, (cen - 1) % 3]
        ax.hlines(0, -1, 1, color='k', linestyle='--', lw=0.8)
        ax.annotate(
            f'{CENTRALITY_BINS[cen]}\u2013{CENTRALITY_BINS[cen - 1]}%',
            xy=(0.05, 0.88), xycoords='axes fraction', fontsize=14,
        )
        ax.set_xlabel(r'$y$', fontsize=13)
        ax.set_ylabel(r'$v_1$', fontsize=13)
        ax.tick_params(labelsize=11)

        if not cen_mask[cen - 1]:
            continue

        res = resolution[cen - 1]
        for v1_dict, color, marker, label, store in [
            (v1_eff, 'C0', 'o', 'eff-corrected', (dv1dy_eff, dv1dy_eff_err)),
            (v1_no,  'C1', 's', 'no-eff',         (dv1dy_no,  dv1dy_no_err)),
        ]:
            y_u, v1_u = v1_arrays(v1_dict, cen, res)
            ax.errorbar(
                unumpy.nominal_values(y_u), unumpy.nominal_values(v1_u),
                yerr=unumpy.std_devs(v1_u),
                fmt=marker, color=color, capsize=2, ms=4, label=label,
            )
            try:
                slope, slope_err = fit_dv1dy(y_u, v1_u)
                ax.plot(x_fit, slope * x_fit, '-', color=color, lw=1)
                store[0][cen - 1] = slope
                store[1][cen - 1] = slope_err
            except RuntimeError:
                pass

        if cen == 1:
            ax.legend(fontsize=11)

    return dv1dy_eff, dv1dy_eff_err, dv1dy_no, dv1dy_no_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(paths_eff: list[str], paths_no_eff: list[str],
         fres: str, output: str, energy: str) -> None:

    # Resolution
    fres_root = uproot.open(fres)
    hres = unumpy.uarray(
        fres_root['hEPDEP_ew_cos_1'].values(),
        fres_root['hEPDEP_ew_cos_1'].errors(),
    )
    cen_mask   = unumpy.nominal_values(hres) > 0
    resolution = np.where(cen_mask, np.abs(hres) ** 0.5, 0)

    # Load fit results
    lam_eff  = load_csv(paths_eff,    'Lambda')
    lbar_eff = load_csv(paths_eff,    'Lambdabar')
    lam_no   = load_csv(paths_no_eff, 'Lambda')
    lbar_no  = load_csv(paths_no_eff, 'Lambdabar')

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:

        # Pages 1 & 2 : per-centrality v1(y) grids
        slopes: dict[str, tuple] = {}
        for particle_tex, v1_eff, v1_no, key in [
            (r'$\Lambda$',        lam_eff,  lam_no,  'lambda'),
            (r'$\bar{\Lambda}$',  lbar_eff, lbar_no, 'lambdabar'),
        ]:
            fig = plt.figure(figsize=(24, 18))
            gs  = fig.add_gridspec(3, 3, hspace=0, wspace=0)
            ax_grid = gs.subplots(sharex='col', sharey='row')
            fig.suptitle(
                f'{particle_tex}  $v_1(y)$ at {energy}  —  eff-corrected vs no-eff',
                fontsize=16,
            )
            slopes[key] = plot_v1_grid(ax_grid, v1_eff, v1_no, resolution, cen_mask)
            pdf.savefig(fig)
            plt.close(fig)

        # Page 3 : dv1/dy vs centrality
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(rf'$dv_1/dy$ vs Centrality at {energy}', fontsize=16)

        for ax, particle_tex, key in [
            (axes[0], r'$\Lambda$',       'lambda'),
            (axes[1], r'$\bar{\Lambda}$', 'lambdabar'),
        ]:
            dv1dy_eff, dv1dy_eff_err, dv1dy_no, dv1dy_no_err = slopes[key]
            ax.errorbar(
                CENTRALITIES[cen_mask], dv1dy_eff[cen_mask],
                yerr=dv1dy_eff_err[cen_mask],
                fmt='o', color='C0', capsize=2, label='eff-corrected',
            )
            ax.errorbar(
                CENTRALITIES[cen_mask], dv1dy_no[cen_mask],
                yerr=dv1dy_no_err[cen_mask],
                fmt='s', color='C1', capsize=2, label='no-eff',
            )
            ax.hlines(0, 0, 80, linestyle='--', color='k', lw=0.8)
            ax.set_xlabel('Centrality (%)', fontsize=14)
            ax.set_ylabel(r'$dv_1/dy$', fontsize=14)
            ax.set_title(particle_tex, fontsize=14)
            ax.legend(fontsize=12)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 4 : delta dv1/dy (Lambda - Lambdabar) vs centrality
        dv1dy_lam_eff,  dv1dy_lam_eff_err,  dv1dy_lam_no,  dv1dy_lam_no_err  = slopes['lambda']
        dv1dy_lbar_eff, dv1dy_lbar_eff_err, dv1dy_lbar_no, dv1dy_lbar_no_err = slopes['lambdabar']

        delta_eff     = dv1dy_lam_eff  - dv1dy_lbar_eff
        delta_eff_err = np.sqrt(dv1dy_lam_eff_err**2  + dv1dy_lbar_eff_err**2)
        delta_no      = dv1dy_lam_no   - dv1dy_lbar_no
        delta_no_err  = np.sqrt(dv1dy_lam_no_err**2   + dv1dy_lbar_no_err**2)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(
            CENTRALITIES[cen_mask], delta_eff[cen_mask],
            yerr=delta_eff_err[cen_mask],
            fmt='o', color='C0', capsize=2, label='eff-corrected',
        )
        ax.errorbar(
            CENTRALITIES[cen_mask], delta_no[cen_mask],
            yerr=delta_no_err[cen_mask],
            fmt='s', color='C1', capsize=2, label='no-eff',
        )
        ax.hlines(0, 0, 80, linestyle='--', color='k', lw=0.8)
        ax.set_xlabel('Centrality (%)', fontsize=14)
        ax.set_ylabel(r'$\Delta dv_1/dy$ ($\Lambda - \bar{\Lambda}$)', fontsize=14)
        ax.set_title(rf'$\Delta dv_1/dy$ at {energy}', fontsize=14)
        ax.legend(fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare eff-corrected vs uncorrected v1 results')
    parser.add_argument('--paths_eff',    nargs='+', required=True,
                        help='CSV paths from eff-corrected fits (fit_particle outputs)')
    parser.add_argument('--paths_no_eff', nargs='+', required=True,
                        help='CSV paths from uncorrected fits (fit_no_eff outputs)')
    parser.add_argument('--fres',   required=True,
                        help='ROOT file containing hEPDEP_ew_cos_1 resolution histogram')
    parser.add_argument('--output', required=True, help='Output PDF path')
    parser.add_argument('--energy', required=True, help='Collision energy string (e.g. 19p6GeV)')
    args = parser.parse_args()
    main(args.paths_eff, args.paths_no_eff, args.fres, args.output, args.energy)
