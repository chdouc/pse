# Supplementary code for polarisation-resolved near-inertial wave dynamics

This repository contains the analysis and plotting scripts accompanying the
manuscript:

> **A spinor formalism for polarisation-resolved near-inertial wave dynamics**

The code is organized by background-flow type. Computation and plotting are
kept in separate scripts so that numerical results can be generated once and
then reused to reproduce the figures.

## Repository structure

```text
code/
├── common/
│   └── custom_gradient_32_to_256.mat
├── parallel_shear/
│   ├── compute_eigenanalysis.py
│   ├── plot_spectrum.py
│   └── plot_eigenfunctions.py
├── gaussian_vortex/
│   ├── compute_eigenanalysis.py
│   ├── plot_spectrum.py
│   └── plot_eigenfunctions.py
└── sinusoidal_dipole/
    ├── compute_error_statistics.py
    ├── plot_error_statistics.py
    ├── compute_wave_velocity_fields.py
    └── plot_wave_velocity_fields.py
```

| Background flow | Calculation | Figure generation |
| --- | --- | --- |
| Parallel shear | Eigenvalue spectra and selected eigenfunctions | Figures 4 and 5 |
| Gaussian vortex | Radial eigenvalue spectra and selected eigenfunctions | Figures 6 and 7 |
| Sinusoidal dipole | Model-error statistics and squared wave-velocity fields | Figures 8–10 |

Figure numbers are provided only as manuscript cross-references. Script names
describe the associated background flow and calculation.

## Requirements

The scripts require Python 3.10 or later and the following packages:

- NumPy 2.0 or later
- SciPy
- Matplotlib
- h5py
- pandas

Install the dependencies with:

```bash
python -m pip install "numpy>=2.0" scipy matplotlib h5py pandas
```

The publication-style plots use LaTeX by default. If a LaTeX installation is
not available, pass `--no-tex` to the plotting scripts that provide this
option.

## Quick start

Clone the repository and enter its root directory:

```bash
git clone https://github.com/chdouc/pse.git
cd pse
```

### Parallel shear

Generate the numerical data:

```bash
python code/parallel_shear/compute_eigenanalysis.py
```

Create the spectrum and eigenfunction figures:

```bash
python code/parallel_shear/plot_spectrum.py
python code/parallel_shear/plot_eigenfunctions.py
```

### Gaussian vortex

Generate the numerical data:

```bash
python code/gaussian_vortex/compute_eigenanalysis.py
```

Create the spectrum and eigenfunction figures:

```bash
python code/gaussian_vortex/plot_spectrum.py
python code/gaussian_vortex/plot_eigenfunctions.py
```

The default Gaussian-vortex calculation uses a 512-mode Bessel–Galerkin
discretization and is the most computationally demanding calculation in the
repository.

### Sinusoidal dipole

These scripts require the simulation index tables and MATLAB v7.3/HDF5 output
files used in the manuscript.

Compute and plot the model-error statistics:

```bash
python code/sinusoidal_dipole/compute_error_statistics.py \
    --index path/to/error_index.csv
python code/sinusoidal_dipole/plot_error_statistics.py
```

Extract and plot the squared wave-velocity fields:

```bash
python code/sinusoidal_dipole/compute_wave_velocity_fields.py \
    --index path/to/wave_field_index.csv
python code/sinusoidal_dipole/plot_wave_velocity_fields.py
```

The `data_mat` entries in an index table may be absolute paths or paths
relative to the index file.

The error-statistics index must contain the columns:

```text
Ro, background_velocity_mps, Lv_m, data_mat
```

The wave-field index must contain:

```text
background_velocity_mps, Lv_m, data_mat
```

## Outputs

Calculation scripts write reusable data files to a `data/` directory beside
the corresponding scripts:

- compressed NumPy archives (`.npz`) for eigenanalysis and wave fields;
- comma-separated values (`.csv`) for error statistics;
- JSON metadata files for the eigenanalysis calculations.

Plotting scripts read these files and write publication figures to a
neighboring `figures/` directory. Figure stems and input/output paths can be
changed with command-line options. Run any script with `--help` to see all
available parameters.

## Reproducibility checks

The reorganized scripts were checked against the original calculation and
plotting scripts used for the manuscript:

- eigenvalues, eigenvectors, error statistics, and wave fields agreed
  numerically;
- the parallel-shear and Gaussian-vortex reference figures were reproduced
  pixel for pixel;
- the sinusoidal-dipole error figure was reproduced pixel for pixel.

The numerical calculations were not changed during the separation of
calculation and plotting workflows.
