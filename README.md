# Supplementary code for polarisation-resolved near-inertial wave dynamics

This repository accompanies the manuscript:

> **A spinor formalism for polarisation-resolved near-inertial wave dynamics**

It contains the equations, initial conditions, parameter configuration,
numerical solvers, plotting scripts and movie renderer needed to reproduce the
reported results. No observational data, precomputed simulation files, private
server, or machine-specific path is required.

## Reproduce the results

Python 3.11--3.13 is supported. Create a clean environment and install the
pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Run the fast end-to-end check first:

```bash
python reproduce.py --smoke-test
```

Run the manuscript-resolution sinusoidal-dipole calculation, save the data,
generate Figures 8--10 and Movie 2, and validate the numerical results:

```bash
python reproduce.py --all
```

If a completed `simulation.h5` already exists and only the downstream files
need to be rebuilt, use `python reproduce.py --all --reuse-simulation`. This
option fails when the simulation archive is absent. Omit it for a clean
reproduction from the equations.

The full calculation uses three independent mode workers by default. The
worker count can be reduced on a memory-constrained machine:

```bash
python reproduce.py --all --workers 1
```

No network access is used after the Python dependencies and repository have
been installed. The calculation starts from analytic initial conditions.

## Numerical configuration

The version-controlled configuration is `config/reproduction.json`. The full
calculation uses the manuscript parameters:

- domain depth `H=2000 m`;
- Coriolis frequency `f=10^-4 s^-1` and buoyancy frequency `N=20f`;
- sinusoidal-dipole length scale `L=50 km` and speed `U=0.25 m s^-1`;
- periodic horizontal domain `[-pi L, pi L)^2`;
- `128 x 128` Fourier grid;
- 64 time steps per inertial period and 50 inertial periods;
- forced two-thirds dealiasing and no diffusion;
- identical unit horizontal-velocity initial condition for every model;
- a fixed random seed, although the configured initial condition is analytic.

The modal HBE system is advanced with RK4. YBJ, TSB and YBJ+ use Strang
splitting with RK4 for background advection/refraction and an exact Fourier
dispersion step. PSE uses matrix ETDRK4 for the constant linear operator and
pseudospectral evaluation of the background-flow terms.

## Vertical modes

The rigid-lid, flat-bottom modes are used without internal normalization:

```text
Z_n(z) = cos(n*pi*z/H),  n=1,2,...
h_n = 2H/n
```

For `H=2000 m`, modes `n=1,4,8,16,32` have vertical wavelengths
`4000, 1000, 500, 250, 125 m`, respectively. The physical reconstruction
factor is therefore exactly one. Each run records this convention in the HDF5
attributes.

The tests verify

```text
dZ_n/dz = 0 at z=0 and z=-H,
mean_z(Z_n) = 0 for n >= 1.
```

Run the complete test suite with:

```bash
python -m pytest -q
```

## Outputs

The default full output directory is `artifacts/reproduction/full/`:

```text
artifacts/reproduction/full/
|-- data/
|   |-- simulation.h5
|   |-- sinusoidal_dipole_error_statistics.csv
|   |-- sinusoidal_dipole_wave_velocity_fields.npz
|   `-- sinusoidal_dipole_movie_fields.npz
|-- figures/
|   |-- figure8_error_statistics.{png,pdf}
|   |-- sinusoidal_dipole_wave_velocity_10IP.{png,pdf}
|   `-- sinusoidal_dipole_wave_velocity_50IP.{png,pdf}
|-- movies/
|   `-- movie2.mp4
|-- validation.json
`-- manifest.json
```

The movie renderer also writes caption, accessibility and encoding-quality
sidecars beside its working output. Only `movie2.mp4` is the supplementary
video itself.

`manifest.json` records the command, operating system, Python and dependency
versions, elapsed time, peak resident memory, every output file, its size and
its SHA-256 checksum. `validation.json` records the numerical values used by
the acceptance tests.

The full calculation is CPU intensive. Three workers require approximately
4 GB of available memory; 8 GB total system memory and 3 GB of free disk space
are recommended. Actual time and peak memory are recorded in the manifest.

All files under `artifacts/` are disposable caches. Removing that directory
and rerunning `python reproduce.py --all` regenerates every listed result from
the equations and analytic initial condition.

## Validation

The full workflow exits with a nonzero status if any required check fails. It
checks:

- the 128-by-128 grid, 64 time steps per inertial period and complete mode set;
- vertical-mode Neumann boundary conditions, zero mean, wavelength mapping and
  reconstruction factor;
- finite fields and NRE curves and a consistent initial PSE reconstruction;
- all 136 Figure 8 model/mode/window statistics;
- individual and across-mode mean NRE ranges reported in the manuscript;
- the five `n=4`, 50-IP squared-velocity maxima;
- the maximum pointwise PSE--HBE squared-velocity difference for `n=32` at
  10 IP;
- agreement between NRE recomputed from saved complex fields and the
  step-resolved NRE curves;
- generation of the H.264/yuv420p Movie 2 file;
- a full Movie 2 decode, codec, pixel format, frame count, duration,
  resolution, file-size and representative-frame quality check.

The tolerances and reference quantities are recorded in
`config/reproduction.json`. A failed check reports the quantity and measured
value instead of silently accepting a changed result.

## Repository layout

The source tree is organized by background-flow type rather than figure
number:

```text
code/
|-- parallel_shear/          # Figures 4 and 5
|-- gaussian_vortex/         # Figures 6 and 7
|-- polarisation_geometry/   # Figures 1 and 2 and Movie 1
`-- sinusoidal_dipole/       # Figures 8--10 and Movie 2
```

The parallel-shear, Gaussian-vortex and polarisation-geometry workflows remain
available through `run_workflow.py`. The sinusoidal-dipole component scripts
also remain directly executable, but their only numerical input is the
`simulation.h5` created by `reproduce.py`.

Examples:

```bash
python run_workflow.py parallel_shear --validate
python run_workflow.py gaussian_vortex --validate
python run_workflow.py polarisation_geometry_movie --output-directory artifacts/movie1 --validate
```

## Data availability

All numerical data underlying the figures and supplementary movies are
generated by the equations, initial conditions and parameters in this
repository. Figures 8--10 and Movie 2 are recreated from zero with `python
reproduce.py --all`; Figures 1--7 and Movie 1 use the configured workflows
listed above. No external research data or precomputed numerical results are
required, and generated arrays are excluded from version control.

The historical dependency audit and the corresponding replacements are
documented in `AUDIT.md`.
