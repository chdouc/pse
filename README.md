# Supplementary code for polarisation-resolved near-inertial wave dynamics

This repository accompanies the manuscript *A spinor formalism for
polarisation-resolved near-inertial wave dynamics*. It contains the equations,
analytic initial conditions, parameter files, solvers and rendering scripts
used for the figures and supplementary movies. The calculations do not require
downloaded research data, private storage or machine-specific paths.

## Installation

The verified runs used Python 3.13. Install the pinned packages in a clean
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\activate` in place of the second command.
Network access is not used after the repository and Python packages have been
installed.

Contributors can install the pinned testing and static-analysis tools with:

```bash
python -m pip install -r requirements-dev.txt
```

## Main reproduction

Run the short deterministic integration and mode checks before the full job:

```bash
python reproduce.py --smoke-test
```

The manuscript-resolution command creates and validates Figures 1--10 and
Movie 2. It solves every analytic or numerical workflow in the repository,
including the full sinusoidal-dipole integration, and records every output in
one manifest:

```bash
python reproduce.py --all
```

The full command uses three independent mode workers by default. Use
`--workers 1` on a memory-constrained machine. A valid existing
`simulation.h5` can be used to rebuild downstream products with:

```bash
python reproduce.py --all --reuse-simulation
```

The reuse check compares a SHA-256 signature of all simulation-defining
parameters. A mismatch stops the run. Omit `--reuse-simulation` to recompute
from the equations.

Spatial and temporal refinement are checked with:

```bash
python reproduce.py --convergence-test
```

The refinement levels and acceptance limits are in `config/convergence.json`.

On Windows 11 with Python 3.13, 14 physical CPU cores and 15.7 GB RAM, the full
sinusoidal-dipole integration and its downstream outputs took 35.6 minutes and
reached 2.36 GB combined resident memory. A current complete orchestration run
that reused the compatible simulation archive took 12.7 minutes and reached
3.39 GB while regenerating Figures 1--10 and Movie 2. The configured
convergence test took 14.7 minutes and reached 0.26 GB. Each manifest records
the exact runtime and peak combined memory for its own run.

## Figure and movie workflows

The source tree follows the background-flow cases rather than figure numbers.

| Manuscript item | Workflow or command |
| --- | --- |
| Figures 1--2 | `python run_workflow.py polarisation_geometry --no-tex --validate` |
| Figure 3 | `python run_workflow.py background_flows --no-tex --validate` |
| Figures 4--5 | `python run_workflow.py parallel_shear --no-tex --validate` |
| Figures 6--7 | `python run_workflow.py gaussian_vortex --no-tex --validate` |
| Figures 8--10 and Movie 2 | `python reproduce.py --all` |
| Movie 1 | `python run_workflow.py polarisation_geometry_movie --output-directory artifacts/movie1 --no-tex --validate` |

Standalone workflows retain their case-specific default directories. To place
all configured `artifacts/...` outputs below another disposable root, add
`--artifact-root path/to/run`.

The portable commands above use Matplotlib's bundled math fonts and require no
system LaTeX installation. Omit `--no-tex` only when the optional LaTeX/newtx
toolchain is installed. The figure scripts save vector PDF and high-resolution PNG files. The movie
workflows encode H.264 MP4 files and check codec, pixel format, frame count,
dimensions, duration, file size and representative decoded frames.
The submitted Movie 2 target is 2560 x 1440 pixels at 24 frames per second.

## Numerical specification

`config/reproduction.json` contains the physical and numerical parameters.
`config/reference_metrics.json` contains the eigenanalysis and time-dependent
regression values together with the Movie 2 rendering limits. The manuscript
calculation uses:

- depth `H=2000 m`, Coriolis frequency `f=10^-4 s^-1` and `N=20f`;
- dipole length scale `L=50 km` and speed `U=0.25 m s^-1`;
- periodic horizontal domain `[-pi L, pi L)^2` on a `128 x 128` Fourier grid;
- 64 time steps per inertial period and a 50-period integration;
- two-thirds dealiasing, no numerical diffusion and an analytic unit-amplitude
  initial horizontal velocity.

The rigid-lid, flat-bottom vertical modes use no internal normalization:

```text
Z_n(z) = cos(n*pi*z/H),  n=1,2,...
h_n = 2H/n
```

For `H=2000 m`, modes `n=1,4,8,16,32` have wavelengths `4000, 1000,
500, 250, 125 m`. The physical reconstruction factor is one. Automated tests
verify the Neumann conditions at `z=0,-H`, zero vertical mean for `n>=1`, the
wavelength mapping and the recorded normalization.

The modal HBE system uses RK4. YBJ, TSB and YBJ+ use Strang splitting with RK4
for advection/refraction and an exact Fourier dispersion step. PSE uses matrix
ETDRK4 for its constant linear operator and pseudospectral background-flow
terms. Its initial counter-rotating component follows the frozen-local,
strain-only `O(Ro)` expression stated in the manuscript appendix.

## Outputs and validation

The default full output is:

```text
artifacts/reproduction/full/
|-- config_used.json
|-- workflows/
|   |-- polarisation_geometry/  # Figures 1--2: data and figures
|   |-- background_flows/       # Figure 3: data and figure
|   |-- parallel_shear/         # Figures 4--5: data and figures
|   `-- gaussian_vortex/        # Figures 6--7: data and figures
|-- data/
|   |-- simulation.h5
|   |-- sinusoidal_dipole_error_statistics.csv
|   |-- sinusoidal_dipole_wave_velocity_fields.npz
|   `-- sinusoidal_dipole_movie_fields.npz
|-- figures/
|   |-- figure8_error_statistics.{pdf,png}
|   |-- sinusoidal_dipole_wave_velocity_10IP.{pdf,png}
|   `-- sinusoidal_dipole_wave_velocity_50IP.{pdf,png}
|-- movies/
|   |-- movie2.mp4
|   `-- movie2 validation and caption sidecars
|-- validation.json
`-- manifest.json
```

The renderer keeps its captions, accessibility text and quality reports in the
working output. The journal submission uses the MP4 files and a separate
caption document.

The validator stops with a nonzero exit code if a required check fails. It
checks the manuscript grid and time step, modal boundary conditions, 136 NRE
statistics, field maxima, the maximum pointwise PSE--HBE difference, agreement
between saved fields and step-resolved NRE, the Figure 5 branch frequencies
and order, the selected Figure 7 frequencies, and the complete Movie 2 decode.
Reference values and tolerances remain separate: `reference_metrics.json`
stores measured regression targets and `reproduction.json` stores acceptance
tolerances.

Each run writes a manifest containing the command, repository revision and
worktree state, source-file hashes, configuration snapshots, package versions,
runtime, peak combined resident memory, output inventory and SHA-256 checksums.
A failed run also writes a manifest with the exception type and message.

Run the unit and integration tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check code tests reproduce.py run_workflow.py validate_outputs.py
python -m pytest -q
```

The same style checks, core numerical type checks, unit tests and smoke
reproduction run automatically on each GitHub push and pull request.

Files below `artifacts/` are disposable caches. Removing them does not remove a
source input; the commands above regenerate the listed data and products.

## Repository layout

```text
code/
|-- common/                 # shared hashes, figure style and colour table
|-- background_flows/       # analytic fields for Figure 3
|-- polarisation_geometry/  # Figures 1--2 and Movie 1
|-- parallel_shear/         # Figures 4--5
|-- gaussian_vortex/        # Figures 6--7
`-- sinusoidal_dipole/      # Figures 8--10 and Movie 2
config/                     # numerical and validation specifications
tests/                      # deterministic and mathematical checks
workflows/                  # declarative figure and movie commands
```

`AUDIT.md` records the removed external dependencies and the replacement
generation steps. `CITATION.cff` provides software and manuscript citation
metadata.

## Data availability

All numerical data underlying the figures and supplementary movies can be
generated from the equations, initial conditions and parameters in this
repository. No external research dataset or precomputed numerical result is
required.

## License

The software in this repository is released under the BSD 3-Clause License.
See `LICENSE` for the full terms.
