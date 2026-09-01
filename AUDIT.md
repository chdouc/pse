# Reproducibility audit

## Scope

The audit traced every version-controlled calculation, plotting and movie
workflow. It checked the provenance of Figures 1--10, Movies 1--2 and the NRE
statistics, with detailed attention to the data previously used by Figures
8--10 and Movie 2.

## Removed dependencies

| Item | State before the revision | Repository replacement |
| --- | --- | --- |
| Figure 8 | CSV index pointing to MATLAB v7.3 error files | Five-model integrations and repository-generated NRE table |
| Figures 9--10 | `data_mat` entries pointing to complex fields | Saved fields from `simulation.h5` |
| Movie 2 | Processed arrays in an external sweep directory | Movie archive generated from `simulation.h5` |
| File paths | Absolute paths allowed in indexes | Paths resolved from the repository or a user-selected output directory |
| Parameters | Checks applied after private files were read | Validated manuscript configuration before integration |
| Environment | Dependency lower bounds | Exact direct and transitive package versions in `requirements.txt` |
| Run record | Separate, incomplete logs | One manifest with source provenance, resource use and checksums |
| Main entry | Figures 8--10 and Movie 2 only | Figures 1--10 and Movie 2 under one output tree |

No observational dataset is used by these cases. The unavailable inputs were
private numerical outputs. The revision replaces them with the equations,
analytic initial conditions and parameter files that generate those outputs.

The retained 256-entry colour table in `code/common` is a rendering asset for
the eigenanalysis figures. It is version controlled and is not numerical
research data. Its format and repository licensing are documented in
`code/common/README.md`.

## Generation chain

`code/sinusoidal_dipole/solver.py` integrates the modal HBE, YBJ, TSB, YBJ+
and PSE systems in the sinusoidal-dipole flow. `reproduce.py` creates the HDF5
archive before any statistics, figures or movie arrays are computed. The
downstream scripts accept that archive as their sole numerical input.

The PSE initialization implements the frozen-local, strain-only `O(Ro)`
formula in the manuscript appendix. HDF5 attributes record the formula, mode
normalization, numerical choices, full configuration, and separate signatures
for the configuration, simulation-defining source files and numerical runtime.
Cache reuse is rejected if any signature is missing or differs.

The symmetric-background solvers use the same `H=2000 m`, `f=10^-4 s^-1`,
`N=20f` and unnormalised cosine-mode convention as the time-dependent solver.
Their metadata records `Z_n(z)=cos(n*pi*z/H)`, `h=2H/n` and a unit physical
reconstruction factor.

Figures 1--2 and Movie 1 share one analytic polarisation-geometry calculation.
Figure 3 is generated from analytic expressions for the parallel shear,
Gaussian vortex and sinusoidal dipole. The Figure 4--7 eigenanalysis workflows
start from analytic background flows and solve their matrix eigenproblems in
the repository. `python reproduce.py --all` runs and validates all four static
figure workflows before creating Figures 8--10 and Movie 2.

## Checks

The automated checks cover:

- rigid-lid and flat-bottom Neumann conditions, zero vertical mean,
  `h=2H/n` and a unit reconstruction factor;
- deterministic integration and exact PSE initial reconstruction;
- Figure 5 branch frequencies and ordering, plus the selected Figure 7
  eigenfrequencies and a reduced-basis Gaussian-vortex CI eigensolve;
- configuration validation and rejection of a cache from different parameters,
  solver source or numerical environment;
- NRE ranges, field maxima and a pointwise PSE--HBE field difference;
- Figure 9--10 colour-limit sidecars containing the true extrema and clipped
  sample fractions for every vertical-mode row, cross-checked against the
  numerical archive;
- consistency between saved complex fields and step-resolved NRE curves;
- direct recomputation of every saved Figure 3 analytic field and checks of the
  Figure 8--10 PNG/PDF products;
- spatial and temporal refinement through `python reproduce.py
  --convergence-test`;
- H.264 movie encoding, complete decoding and representative-frame checks.

The continuous-integration workflow repeats style checks, full-repository type
checks, unit tests and the deterministic smoke reproduction on Python 3.13.
Its third-party Actions are pinned to immutable commits. A manual workflow
dispatch additionally runs the complete reproduction and uploads its
configuration, validation and manifest reports even when the job fails.

Measured regression values are stored in `config/reference_metrics.json`.
Acceptance tolerances and simulation parameters are stored in
`config/reproduction.json`. Keeping these roles separate makes changes to a
target or a tolerance visible in version control.

## Caches and known variability

Everything below `artifacts/` is generated and may be deleted. A clean run can
reconstruct each listed output without a research-data download.

Floating-point roundoff can vary across operating systems and numerical
libraries. Rendering fonts can also differ. The numerical tolerances allow
roundoff while still rejecting a changed solution. The MP4 renderer uses the
H.264 encoder bundled with the pinned `imageio-ffmpeg` package by default. The
Movie 1 metadata records its encoder filename, version string and SHA-256
checksum. The Movie 2 render manifest records the encoder and probe executable
filenames, version strings and SHA-256 checksums; explicit executable overrides
remain available for controlled comparisons.
