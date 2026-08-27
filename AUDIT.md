# Reproducibility audit

## Scope

The audit covered every version-controlled workflow, calculation script,
plotting script, movie script and documentation file. It specifically traced
the provenance of the data used by Figures 8--10, Movie 2 and their NRE
statistics.

## Findings before the revision

| Item | Previous state | Reproducibility gap |
| --- | --- | --- |
| Figure 8 statistics | Read error curves from a user-supplied CSV index and MATLAB v7.3 files | The equations and time integration that produced the curves were absent |
| Figures 9--10 fields | Read precomputed complex fields through `data_mat` entries | Results could not be regenerated without private simulation output |
| Movie 2 | Read processed full-field files from an external sweep directory | The movie depended on private paths and precomputed fields |
| Paths | Index entries could contain absolute paths | Commands were machine-specific |
| Parameters | Grid and time-step checks were applied only after loading external files | No repository-owned calculation enforced the manuscript resolution |
| Dependencies | Lower version bounds were used | Environments were not exactly repeatable |
| Validation | Selected values were checked only after external data extraction | A missing or changed generation step could not be identified |
| Run record | No single manifest covered calculation, figures and movie | Runtime, memory, output inventory and checksums were incomplete |
| Plotting colour table | A 256-entry MATLAB colour table was stored in `code/common` | This is a version-controlled rendering asset, not simulation data; it is retained so eigenanalysis plots remain self-contained |

No observational dataset is required by these manuscript cases. The missing
inputs were privately generated numerical results rather than research data.

The parallel-shear and Gaussian-vortex eigensolvers and the
polarisation-geometry calculation were already generated from analytic inputs
inside the repository. Their only binary input is the bundled colour table
listed above; none reads a numerical research dataset.

## Resolution

- `code/sinusoidal_dipole/solver.py` now contains the modal HBE, YBJ, TSB,
  YBJ+ and PSE integrations, the sinusoidal-dipole background flow, analytic
  initial condition, Fourier grid and two-thirds projector.
- `config/reproduction.json` is the single source of physical parameters,
  numerical parameters, seed, saved times and validation tolerances.
- `reproduce.py` generates the HDF5 simulation archive before creating any
  statistic, figure or movie input.
- The three product scripts accept only that repository-generated HDF5 file.
  CSV indexes, `data_mat`, private directory discovery and MATLAB-file readers
  were removed.
- The vertical mode is explicitly `cos(n*pi*z/H)`, with `h=2H/n`, no internal
  normalization and a physical reconstruction factor of one.
- Automated tests cover the Neumann conditions, zero vertical mean, wavelength
  mapping, deterministic integration and initial PSE reconstruction.
- Full validation checks NRE, field maxima and pointwise model differences
  against version-controlled tolerances and fails immediately on disagreement.
- Dependency versions are pinned, and every run writes environment, runtime,
  peak-memory, output-inventory and checksum records.

## Generated files and caches

Everything below `artifacts/` is generated. It may be removed in full without
losing a source input. A clean `python reproduce.py --all` run reconstructs
the data, figures and Movie 2 from the version-controlled equations and
configuration.

## Remaining limitations

- Floating-point roundoff and rendering fonts can differ slightly across
  operating systems. Numerical tolerances are wider than roundoff but narrow
  enough to detect a changed solution.
- The 128-by-128, 50-period calculation is intentionally compute intensive.
  The manifest from each run records the actual cost on that machine.
- The MP4 renderer uses the H.264 encoder bundled with the pinned
  `imageio-ffmpeg` package when a system FFmpeg installation is unavailable.
