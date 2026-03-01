# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Snakemake-based physics analysis pipeline** for measuring Lambda hyperon directed flow (v1) in heavy-ion collisions at RHIC/BES energies (7.7–27 GeV). The pipeline processes ROOT data files, fits invariant mass spectra, extracts flow coefficients, and generates publication-quality plots.

## Environment Setup

```bash
# Create/update conda environment
conda env create -f environment.yaml
# or update existing:
sh update_env.sh
# run with:
conda activate lambda_v1
```

The environment requires: snakemake, uproot, numpy, matplotlib, iminuit, scipy, uncertainties, mplhep, numba, ROOT (external, must be separately installed).

## Running the Pipeline

```bash
# Run full pipeline (produces plots/paper/report.pdf)
snakemake --cores all

# Run a specific rule for a specific target
snakemake --cores all result/sys_tag_0/fit_Lambda_v1_19p6GeV.csv

# Dry run to check what will be executed
snakemake -n

# Visualize the DAG
sh create_dag.sh   # produces dag.pdf and rulegraph.pdf
```

## Architecture

### Pipeline Flow (Snakefile)

The pipeline is driven by `Snakefile` with parameters in `config.yaml`. Data flows:

```
data/*.root (raw ROOT files)
  → combine_lambda / combine_lambda_with_eff  (C++ ROOT scripts)
      → result/sys_tag_N/combined_{particle}_{flow}_{energy}.root
  → fit_particle (fit_v1.py)
      → result/sys_tag_N/fit_{particle}_{flow}_{energy}.csv
  → plot_v1 (plot_v1.py)
      → result/sys_tag_N/data_{energy}.txt
      → plots/sys_tag_N/paper_yaml/*.yaml
  → combine_sys (combine_sys.py)       [combines systematics]
      → plots/final/paper_yaml/*.yaml
  → generate_paper_plots (generate_paper_plots.py)
      → plots/paper/report.pdf
```

### Systematic Uncertainty Structure

- `sys_tag_0`: default dataset
- `sys_tag_1,2,3`: regular systematics (different subsets of same data)
- `special_sys_tag_5,6`: special systematics (same dataset, varied analysis cuts)
- `combine_sys` rule aggregates all into `plots/final/paper_yaml/`

### Key Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `fit_v1.py` | Core fitting script: reads combined ROOT histograms, fits invariant mass spectra (signal + polynomial background), extracts v1 via profile histograms. Uses iminuit for minimization. |
| `fit_v1_pt.py` | Same as above but bins in pT instead of rapidity. |
| `plot_v1.py` | Reads fit CSVs, computes dv1/dy slopes, generates comparison plots with piKp reference. |
| `combine_sys.py` | Combines systematic uncertainties from multiple `sys_tag` variants using quadrature sum of significant deviations. |
| `generate_paper_plots.py` | Final paper figure assembly from YAML data files. |
| `simple_profile.py` | `SimpleProfile` class: stores and manipulates weighted profile histogram data (values, counts, std_devs). Supports rebinning and addition. |
| `measurement.py` | `Measurement` class: weighted average of `uncertainties` variables. |
| `data_point.py` | `DataPoint` class: holds value with separate statistical and systematic errors; supports arithmetic operations. |
| `param_storage.py` | `ParamStorage`: stores fit parameters with optional freezing of indices between fits. |
| `find_bin_center.py` | `BinCenterFinder`: computes weighted bin centers for non-uniform distributions. |

### C++ ROOT Scripts

| File | Purpose |
|------|---------|
| `combine_lambda_without_eff.cpp` | Reads raw ROOT result files, applies pT and rapidity cuts, combines Lambda/Lambdabar histograms across centralities. |
| `combine_lambda_with_eff.cpp` | Same but applies efficiency corrections from a separate eff ROOT file. |
| `calculate_lambda_eff.cpp` | Computes reconstruction efficiency from MC truth-matched data. |
| `Finish_v1_tof_eff.C` | Processes piKp (pions/kaons/protons) v1 data with TOF efficiency corrections. |
| `FitSlope.C` | Fits the v1 vs. rapidity slope (dv1/dy) for piKp particles. |

### Data Layout

```
data/
  result*_{energy}.root          # default dataset per energy
  sys_tag_{1,2,3}/result*.root   # systematic variation datasets
  eff/result*_lambda_exp_{energy}.root  # efficiency files
  v1_piKp/{energy}/{particle}/   # piKp reference data
  model/{urqmd,ampt}/{energy}.root  # model comparisons
```

### Config Parameters (`config.yaml`)

- `energies`: list of collision energies (e.g., `7p7GeV` through `27GeV`)
- `particles`: `[Lambda, Lambdabar]`
- `flows`: `[v1]`
- `pt_lo/pt_hi`: pT range for Lambda selection (0.4–1.8 GeV/c)
- `y_cut`: rapidity cut (0.6)
- `yrebin`: per-energy, per-particle rapidity rebinning factor
- `fit_order`: polynomial order for v1 slope fit (always 3)

### YAML Output Format

Intermediate results are serialized as YAML files in `plots/sys_tag_N/paper_yaml/`. These contain centrality-binned arrays for: x positions, v1 values, errors, dv1/dy slopes, and systematic breakdown (stat/sys split).

## Workflow Guidelines

- Always run `snakemake -n` (dry run) before any real execution.
- Never run the full workflow on real data without explicit confirmation (or any step that involved `fit_v1.py` or `fit_v1_pt.py`).
- Prefer small, focused commits over large sweeping changes.
- After editing a rule, always verify the DAG hasn't broken.

## Code Style

- Python: follow PEP8, use type hints for new functions.
- No magic numbers — define constants in `config.yaml`.
- Write docstrings for any new function that isn't self-evident.
- Don't refactor working code unless asked — fix only the specific thing requested.

## When Uncertain

- If the physics intent of a change is unclear, ask before implementing.
- If a change would affect more than one rule, summarize the blast radius and confirm.
- Never silently skip a step — if something can't be done, say so explicitly.

## Never

- Never delete or overwrite output ROOT files without explicit instruction.
- Never modify `config.yaml` without confirmation.
- Never run grid jobs or submit to the cluster without being asked.

## Preferred Tools

- Use `uproot` for reading ROOT files in new scripts, not PyROOT.
- Use `pathlib.Path` not `os.path` for file handling.
- Use `subprocess.run(..., check=True)` not `os.system()`.
- Run `git status` before committing to avoid surprises.
