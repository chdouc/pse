"""Scientific regression checks for the symmetric-background eigenproblems."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

from common.paper_parameters import (
    DOMAIN_DEPTH,
    vertical_mode_dispersive_coefficient,
    vertical_mode_wavenumber,
    vertical_wavelength,
)


ROOT = Path(__file__).parents[1]
REFERENCE = json.loads(
    (ROOT / "config" / "reference_metrics.json").read_text(encoding="utf-8")
)


def load_script(name: str, relative_path: str) -> ModuleType:
    """Load one repository script under a unique module name."""
    path = ROOT / relative_path
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_shared_vertical_mode_parameters_match_the_manuscript() -> None:
    """The static and time-dependent workflows must use the same depth."""
    config = json.loads(
        (ROOT / "config" / "reproduction.json").read_text(encoding="utf-8")
    )

    assert DOMAIN_DEPTH == config["physical_parameters"]["domain_depth_m"]
    assert vertical_mode_wavenumber(4) == np.pi / 500.0
    assert vertical_wavelength(4) == 1000.0
    assert np.isclose(
        vertical_mode_dispersive_coefficient(4),
        (20.0e-4) ** 2 * 2000.0**2 / (2.0 * np.pi**2 * 1.0e-4 * 4**2),
    )


def test_parallel_shear_figure5_branch_frequencies() -> None:
    """Recompute the Figure 5 branches and compare every plotted frequency."""
    module = load_script(
        "parallel_shear_compute_test",
        "code/parallel_shear/compute_eigenanalysis.py",
    )
    reference = REFERENCE["parallel_shear_figure5"]
    overlays = module.compute_eigenfunction_overlays(
        truncation=48,
        wavenumbers=reference["wavenumbers_k_y_L"],
        vertical_mode=reference["vertical_mode"],
        selected_count=5,
        skip_count=2,
        frequency_min=-0.2,
        frequency_max=1.5,
        grid_size=32,
    )
    normalized = overlays["overlay_frequencies"].real / module.CORIOLIS_FREQUENCY

    assert np.allclose(
        normalized,
        np.asarray(reference["frequencies_over_f"]),
        rtol=0.0,
        atol=reference["absolute_tolerance"],
    )
    assert np.all(np.diff(normalized, axis=1) > 0.0)


def test_gaussian_vortex_uses_the_shared_manuscript_depth() -> None:
    """The Gaussian-vortex metadata must not reinterpret 2H as the depth."""
    module = load_script(
        "gaussian_vortex_compute_test",
        "code/gaussian_vortex/compute_eigenanalysis.py",
    )

    assert module.DOMAIN_DEPTH == DOMAIN_DEPTH == 2000.0


def test_gaussian_vortex_lowest_trapped_frequency() -> None:
    """Recompute a small Gaussian-vortex eigensystem as a CI regression."""
    module = load_script(
        "gaussian_vortex_regression_test",
        "code/gaussian_vortex/compute_eigenanalysis.py",
    )
    reference = REFERENCE["gaussian_vortex_figure7"]
    frequencies, _ = module.solve_eigensystem(
        0,
        basis_size=32,
        radial_domain=10.0 * module.FLOW_LENGTH,
        vertical_mode=reference["vertical_mode"],
    )
    index = module.lowest_frequency_index(
        frequencies,
        frequency_min=-0.5,
        frequency_max=0.5,
    )

    assert np.isclose(
        frequencies[index].real / module.CORIOLIS_FREQUENCY,
        reference["frequencies_over_f"][0],
        rtol=0.0,
        atol=reference["absolute_tolerance"],
    )
    assert abs(frequencies[index].imag) <= 1.0e-12
