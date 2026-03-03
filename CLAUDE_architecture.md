# Architecture Reference

## Pipeline Flow

Driven by `Snakefile` with parameters in `config.yaml`:

```
data/*.root (raw ROOT files)
  → combine_lambda / combine_lambda_with_eff  (C++ ROOT scripts)
      → result/sys_tag_N/combined_{particle}_{flow}_{energy}.root
  → fit_particle (fit_v1.py)
      → result/sys_tag_N/fit_{particle}_{flow}_{energy}.csv
  → plot_v1 (plot_v1.py)
      → result/sys_tag_N/data_{energy}.txt
      → plots/sys_tag_N/paper_yaml/*.yaml
  → combine_sys (combine_sys.py)
      → plots/final/paper_yaml/*.yaml
  → generate_paper_plots (generate_paper_plots.py)
      → plots/paper/report.pdf
```

## Systematic Uncertainty Structure

- `sys_tag_0`: default dataset
- `sys_tag_1,2,3`: regular systematics (different subsets of same data)
- `special_sys_tag_5,6`: special systematics (same dataset, varied analysis cuts)
- `combine_sys` rule aggregates all into `plots/final/paper_yaml/`
- Systematic uncertainty uses `sys_divisor` from `config.yaml` (currently 3, i.e. half-width uniform distribution assumption)

## Key Python Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `fit_v1.py` | Core fitting: reads combined ROOT histograms, fits invariant mass spectra (signal + polynomial background), extracts v1 via profile histograms. Uses iminuit. |
| `fit_v1_pt.py` | Same as above but bins in pT instead of rapidity. |
| `plot_v1.py` | Reads fit CSVs, computes dv1/dy slopes, generates comparison plots with piKp reference. |
| `combine_sys.py` | Combines systematic uncertainties from multiple `sys_tag` variants using quadrature sum of significant deviations. |
| `generate_paper_plots.py` | Final paper figure assembly from YAML data files. |
| `simple_profile.py` | `SimpleProfile` class: weighted profile histogram data; supports rebinning and addition. |
| `measurement.py` | `Measurement` class: weighted average of `uncertainties` variables. |
| `data_point.py` | `DataPoint` class: value with separate stat/sys errors; supports arithmetic. |
| `param_storage.py` | `ParamStorage`: fit parameters with optional freezing between fits. |
| `find_bin_center.py` | `BinCenterFinder`: weighted bin centers for non-uniform distributions. |

## C++ ROOT Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `combine_lambda_without_eff.cpp` | Reads raw ROOT files, applies pT and rapidity cuts, combines Lambda/Lambdabar histograms across centralities. |
| `combine_lambda_with_eff.cpp` | Same but applies efficiency corrections from a separate eff ROOT file. |
| `calculate_lambda_eff.cpp` | Computes reconstruction efficiency from MC truth-matched data. |
| `Finish_v1_tof_eff.C` | Processes piKp v1 data with TOF efficiency corrections. |
| `FitSlope.C` | Fits the v1 vs. rapidity slope (dv1/dy) for piKp particles. |

## Data Layout

```
data/
  result*_{energy}.root                        # default dataset per energy
  sys_tag_{1,2,3}/result*.root                 # systematic variation datasets
  eff/result*_{lambda,lambdabar}_exp_{energy}.root  # efficiency files (19p6GeV, 27GeV)
  v1_piKp/{energy}/{particle}/                 # piKp reference data
  model/{urqmd,ampt}/{energy}.root             # model comparisons
```

Efficiency corrections are applied for Lambda and Lambdabar at 19p6GeV and 27GeV (when eff files are present in `data/eff/`).

## Config Parameters (`config.yaml`)

- `energies`: collision energies (`7p7GeV` through `27GeV`)
- `particles`: `[Lambda, Lambdabar]`
- `flows`: `[v1]`
- `sys_divisor`: divisor for systematic uncertainty combination (3 = half-width uniform)
- `pt_lo/pt_hi`: pT range for Lambda selection (0.4–1.8 GeV/c)
- `y_cut`: rapidity cut (0.6)
- `yrebin`: per-energy, per-particle rapidity rebinning factor
- `fit_order`: polynomial order for dv1/dy slope fit (always 1 = linear)

## YAML Output Format

Intermediate results in `plots/sys_tag_N/paper_yaml/` contain centrality-binned arrays for: x positions, v1 values, errors, dv1/dy slopes, and stat/sys error breakdown.
