# Supplementary code for polarisation-resolved near-inertial wave dynamics

This repository contains the calculation and plotting scripts accompanying
the manuscript:

> **A spinor formalism for polarisation-resolved near-inertial wave dynamics**

The repository is organized by background-flow type rather than manuscript
figure number. Numerical calculations and figure generation are separate, so
calculation outputs can be inspected, validated, and reused without repeating
the expensive eigensystem or simulation-data processing steps.

## Design

Each manuscript case has a version-controlled workflow in `workflows/`.
A workflow records the physical and numerical parameters, calculation step,
plotting steps, expected outputs, and validation method. All steps are invoked
through `run_workflow.py`; no source file is modified when parameters change.

The workflow layer provides:

- one explicit parameter record for each manuscript case;
- controlled comparisons using the same input conditions;
- independent `compute`, `plot`, `render`, `validate`, and `all` stages;
- a dry-run mode that displays commands before execution;
- automated checks of output structure and selected reference values.

The individual scripts remain directly executable and can still be used
without the workflow runner.

## Repository structure

```text
.
|-- README.md
|-- requirements.txt
|-- run_workflow.py
|-- validate_outputs.py
|-- workflows/
|   |-- parallel_shear.json
|   |-- gaussian_vortex.json
|   |-- polarisation_geometry_movie.json
|   |-- sinusoidal_dipole_error.json
|   |-- sinusoidal_dipole_wave.json
|   `-- sinusoidal_dipole_movie.json
`-- code/
    |-- common/
    |   `-- custom_gradient_32_to_256.mat
    |-- parallel_shear/
    |   |-- compute_eigenanalysis.py
    |   |-- plot_spectrum.py
    |   `-- plot_eigenfunctions.py
    |-- gaussian_vortex/
    |   |-- compute_eigenanalysis.py
    |   |-- plot_spectrum.py
    |   `-- plot_eigenfunctions.py
    |-- polarisation_geometry/
    |   |-- compute_polarisation_trajectories.py
    |   |-- render_polarisation_movie.py
    |   `-- README.md
    `-- sinusoidal_dipole/
        |-- compute_error_statistics.py
        |-- plot_error_statistics.py
        |-- compute_wave_velocity_fields.py
        |-- plot_wave_velocity_fields.py
        |-- compute_movie_fields.py
        |-- render_wave_velocity_movie.py
        `-- validate_wave_velocity_movie.py
```

| Workflow | Calculation | Manuscript figures |
| --- | --- | --- |
| `parallel_shear` | Eigenvalue spectra and selected eigenfunctions | 4 and 5 |
| `gaussian_vortex` | Radial eigenvalue spectra and selected eigenfunctions | 6 and 7 |
| `polarisation_geometry_movie` | Stokes-Poincare geometry and exact matrix-generator actions | Figures 1 and 2; supplementary movie 1 |
| `sinusoidal_dipole_error` | Controlled model-error comparison | 8 |
| `sinusoidal_dipole_wave` | Controlled squared wave-velocity comparison | 9 and 10 |
| `sinusoidal_dipole_movie` | Time-resolved wave fields and NRE curves | Figures 9 and 10; supplementary movie 2 |

Figure numbers are manuscript cross-references only. File and directory names
describe the corresponding background flow and calculation.

## Requirements

The scripts require Python 3.10 or later. Install the tested runtime
dependencies with:

```bash
python -m pip install -r requirements.txt
```

On Windows, activate the intended Conda or virtual environment first. If
`python --version` opens the Microsoft Store or prints no version, the
Microsoft Store placeholder is being selected instead of a Python
installation.

The publication-style plots use LaTeX by default. If LaTeX and the required
font packages are unavailable, add `--no-tex` to the workflow command.

## Quick start

Clone the repository and list the available workflows:

```bash
git clone https://github.com/chdouc/pse.git
cd pse
python run_workflow.py --list
```

### Self-contained eigenanalysis workflows

Run the complete parallel-shear calculation, create the figures, and validate
the outputs:

```bash
python run_workflow.py parallel_shear --validate
```

Run the corresponding Gaussian-vortex workflow:

```bash
python run_workflow.py gaussian_vortex --validate
```

The Gaussian-vortex workflow uses the manuscript's 512-mode Bessel-Galerkin
discretization and is the most computationally demanding calculation.

### Sinusoidal-dipole workflows

These workflows require simulation index tables and the MATLAB v7.3/HDF5
files used in the manuscript.

Compute and plot the model-error comparison:

```bash
python run_workflow.py sinusoidal_dipole_error --index path/to/error_index.csv --validate
```

Extract and plot the squared wave-velocity comparison:

```bash
python run_workflow.py sinusoidal_dipole_wave --index path/to/wave_field_index.csv --validate
```

Prepare, render, and validate supplementary movie 2:

```bash
python run_workflow.py sinusoidal_dipole_movie \
  --data-root path/to/processed/sweep \
  --output-directory path/to/movies \
  --ffmpeg path/to/ffmpeg \
  --ffprobe path/to/ffprobe
```

The movie workflow accepts either `--data-root` (containing one
`*_index.csv`) or `--index`. Each selected processed MATLAB file must contain
the complete YBJ, TSB, YBJ+, PSE and HBE complex-velocity fields on the
manuscript's 128-by-128 grid with `fc=64` (64 time steps per inertial period),
together with the step-resolved NRE curves. Local input paths are always
supplied at run time and are not stored in the public workflow.

On systems without LaTeX, for example:

```bash
python run_workflow.py parallel_shear --no-tex --validate
```

### Polarisation-geometry movie

Supplementary movie 1 is self-contained and uses no simulation input. Supply
the artifact directory at run time:

```bash
python run_workflow.py polarisation_geometry_movie \
  --output-directory path/to/movies \
  --validate
```

The calculation stage follows the manuscript spinor convention
`|A> = (A_up, conj(A_down))^T`, saves every spinor, Stokes vector and physical
hodograph used in the movie, and performs the matrix and polarisation-geometry
checks. The rendering stage reads those saved arrays, encodes a silent
MP4/H.264/yuv420p movie, verifies the stream and writes the separate JFM
caption, accessibility description, submission notes and preview.

The initial state is reconstructed from the Figure 2 source parameters:

```text
|S| = 1.21
varphi = pi/2 - 2 atan(1/5)
lambda = pi/3
gamma = -pi/4
```

Chapter 1 uses normalised Stokes directions. Chapter 2 uses unnormalised
Stokes vectors and retains the unit sphere only as a scale reference.

## Workflow controls

Run only the calculation or plotting stage:

```bash
python run_workflow.py parallel_shear --stage compute
python run_workflow.py parallel_shear --stage plot
```

Supplementary movie 2 additionally exposes independent render and validation
stages:

```bash
python run_workflow.py sinusoidal_dipole_movie \
  --output-directory path/to/movies \
  --stage render

python run_workflow.py sinusoidal_dipole_movie \
  --output-directory path/to/movies \
  --stage validate
```

Use `--fps`, `--resolution`, `--frame-stride`, `--hold-frames`,
`--opening-seconds`, `--title-seconds`, `--chapter-end-seconds`, and `--crf`
to override movie-rendering defaults.
Movie 2 holds each terminal case frame for an additional 1.5 seconds. Its
typography, model colours and NRE legend order are aligned with
manuscript figures 8--10; its chapter cards also give the common 2000-m depth
and the corresponding unchanged vertical wavelengths from `h=2H/n`.
The n=4 absolute-field colour scale has a fixed upper limit of 37.5.
Difference colourbars use
the explicit model-minus-HBEs normalised squared-velocity expression, and the
curve panel is labelled as instantaneous NRE to distinguish it from the
time-averaged NRE in manuscript figure 8. Scientific-panel titles use 20-pt
type, with 17-pt type for the longer NRE title. Title-card hierarchy, font scale and
vertical placement are aligned with supplementary movie 1. Like movie 1, the
opening uses an overall movie page followed by the first chapter page. The
overall page lasts 2 seconds, while every Chapter page lasts 4 seconds so its
mode, wavelength and time-range information can be read comfortably. The
submission workflow restricts the frame rate to 20--24 fps and always encodes
H.264/yuv420p with a constant frame rate, fast start, and no audio stream.

For the movie workflow, pass the same output directory to either stage:

```bash
python run_workflow.py polarisation_geometry_movie \
  --output-directory path/to/movies \
  --stage compute

python run_workflow.py polarisation_geometry_movie \
  --output-directory path/to/movies \
  --stage plot
```

Inspect the exact commands without executing them:

```bash
python run_workflow.py parallel_shear --dry-run
```

Validate previously generated results:

```bash
python validate_outputs.py parallel_shear
python validate_outputs.py all
```

Use `--data-only` to validate calculation outputs without requiring figure
files:

```bash
python validate_outputs.py gaussian_vortex --data-only
```

Validate an existing movie output directory:

```bash
python validate_outputs.py polarisation_geometry_movie \
  --output-directory path/to/movies
```

Run any individual script with `--help` to inspect its lower-level numerical
and plotting options.

## Sinusoidal-dipole inputs

The `data_mat` entries in an index table may be absolute paths or paths
relative to the index file.

The error-statistics index must contain:

```text
Ro, background_velocity_mps, Lv_m, data_mat
```

The wave-field index must contain:

```text
background_velocity_mps, Lv_m, data_mat
```

The error comparison holds the background speed and averaging windows fixed
while comparing YBJ, TSB, YBJ+, and PSE. The wave-field comparison uses the
same saved cases and ordering for YBJ, TSB, YBJ+, PSE, and the hydrostatic
Boussinesq equations.

Supplementary movie 2 uses modes `n=4`, `n=16`, and `n=32`. Its intermediate
NPZ records the true selected times, source indices and files, model order,
normalisation, fixed colour limits, NRE curves, PSE reconstruction metadata,
and raw-versus-processed reference checks. Physical fields are never
interpolated; playback speed is controlled by holding or striding true saved
states.

## Outputs

Calculation scripts write reusable data beside the corresponding background
flow:

- compressed NumPy archives (`.npz`) for eigenanalysis and wave fields;
- comma-separated values (`.csv`) for error statistics;
- JSON metadata files for eigenanalysis parameters.

Plotting scripts read these files and write PNG and PDF figures to a
neighboring `figures/` directory. Generated `data/` and `figures/` directories
are excluded from version control because they can be recreated by the
configured workflows.

Movie workflows write archives, metadata, MP4 files, previews and text
sidecars to the output directory supplied on the command line. Supplementary
movie 2 also writes a pre/post-encoding QC contact sheet, render manifest and
quality report. Managed temporary PNG frames are removed after a successful
encode. These external submission artifacts are not committed to this
repository.

## Validation

`validate_outputs.py` checks:

- required arrays, table columns, dimensions, and model ordering;
- finite and physically admissible values;
- consistency between coordinates, modes, fields, and metadata;
- selected Gaussian-vortex eigenfrequencies;
- the reference wave-field maxima for vertical mode 4 at 50 inertial periods;
- the presence of both PNG and PDF figure outputs.
- the manuscript matrix definitions, Clifford relations and exact matrix
  exponentials used in supplementary movie 1;
- the Stokes norm, handedness, ellipticity, orientation, common-phase and
  generator-action identities;
- consistency of every movie sphere vector and hodograph with the same saved
  spinor state;
- exact 0--50 IP source ordering, PSE reconstruction, 10 IP and 50 IP field
  consistency, NRE agreement, fixed colour scales and symmetric difference
  limits for supplementary movie 2;
- representative movie-2 frames before and after encoding, including
  compression PSNR and readable preview/contact-sheet dimensions;
- the MP4 container, H.264 profile, yuv420p pixel format, resolution, constant
  frame rate, frame count, duration, absence of audio, fast-start ordering,
  decodability and 10 MB size limit.

During repository preparation, the reorganized scripts were also compared
with the original manuscript scripts. Eigenvalues, eigenvectors, error
statistics, and wave fields agreed numerically. The parallel-shear,
Gaussian-vortex, and sinusoidal-dipole error reference figures were reproduced
pixel for pixel.
