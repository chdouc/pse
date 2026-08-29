# Shared utilities and plotting assets

`paper_parameters.py` defines the manuscript-wide physical constants and the
unnormalised cosine-mode wavenumber, wavelength and dispersive coefficient.
The symmetric-background and sinusoidal-dipole workflows import these values
instead of maintaining separate depth conventions.

`files.py`, `video.py` and `spectrum_plotting.py` contain shared hashing,
MP4-container and eigenanalysis-plotting operations. They do not contain
scientific results.

`custom_gradient_32_to_256.mat` stores the 256-entry RGB lookup table used by
the parallel-shear and Gaussian-vortex spectrum figures. It is a plotting
asset, not numerical research data or a precomputed scientific result. The
file is included in the repository so the published colours are reproducible
without a download.

The plotting code reads the MATLAB variable `C`, verifies that it
is an `N x 3` finite array, and clips its entries to the valid RGB interval.
The asset is distributed with the supplementary code under the repository's
BSD 3-Clause License.
