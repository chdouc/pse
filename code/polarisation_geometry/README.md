# Polarisation geometry

This directory contains the reproducible calculation and rendering stages for
Figures 1--2 and supplementary Movie 1. This is an analytic local-geometry
calculation, separate from the Movie 2 background-flow simulation.

The calculation uses the manuscript convention

```text
|A> = (A_up, conj(A_down))^T
```

and computes

```text
S_x = 2 Re(A_up A_down)
S_y = 2 Im(A_up A_down)
S_z = |A_up|^2 - |A_down|^2
|S| = |A_up|^2 + |A_down|^2
phi(theta) = A_up exp(-i theta) + A_down exp(i theta)
```

`compute_polarisation_trajectories.py` constructs the Figure 1 state,
evaluates the exact positive and negative matrix exponentials for all four
generators, validates the algebra and geometry, and writes a compressed NumPy
archive plus JSON metadata.

`plot_polarisation_figures.py` reads the same archive and creates vector PDF
and high-resolution PNG versions of Figures 1 and 2.

`render_polarisation_movie.py` reads those saved states. It does not repeat the
physical calculation. It uses fixed camera views and axis limits, renders the
two movie chapters, encodes MP4/H.264/yuv420p video, verifies the encoded
stream, and writes the caption, accessibility description and submission
notes.

Create and validate Figures 1--2 from the repository root:

```bash
python run_workflow.py polarisation_geometry --validate
```

Create and validate Movie 1:

```bash
python run_workflow.py polarisation_geometry_movie \
  --output-directory path/to/movies \
  --validate
```

Run only the calculation or rendering stage:

```bash
python run_workflow.py polarisation_geometry_movie \
  --output-directory path/to/movies \
  --stage compute

python run_workflow.py polarisation_geometry_movie \
  --output-directory path/to/movies \
  --stage render
```

The output directory is always supplied at run time. No manuscript path is
stored in source code or configuration.
