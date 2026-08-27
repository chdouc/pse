"""Rigid-lid, flat-bottom vertical modes used in the manuscript."""

from __future__ import annotations

import numpy as np


def vertical_mode(z: np.ndarray, n: int, depth_m: float) -> np.ndarray:
    """Return the unnormalised mode Z_n(z)=cos(n*pi*z/H)."""
    if n < 1:
        raise ValueError("The baroclinic vertical-mode number must be positive.")
    if depth_m <= 0.0:
        raise ValueError("The domain depth must be positive.")
    return np.cos(n * np.pi * np.asarray(z, dtype=float) / depth_m)


def vertical_mode_derivative(z: np.ndarray, n: int, depth_m: float) -> np.ndarray:
    """Return the analytic vertical derivative of Z_n."""
    return -(n * np.pi / depth_m) * np.sin(
        n * np.pi * np.asarray(z, dtype=float) / depth_m
    )


def vertical_wavelength(n: int, depth_m: float) -> float:
    """Return h=2H/n for the unnormalised cosine mode."""
    if n < 1:
        raise ValueError("The vertical-mode number must be positive.")
    return 2.0 * depth_m / n


def mode_metadata(n: int, depth_m: float) -> dict[str, float | int | str]:
    """Describe the mode convention and its physical reconstruction factor."""
    return {
        "vertical_mode": n,
        "domain_depth_m": depth_m,
        "vertical_wavenumber_m-1": n * np.pi / depth_m,
        "vertical_wavelength_m": vertical_wavelength(n, depth_m),
        "definition": "Z_n(z)=cos(n*pi*z/H)",
        "normalisation": "unnormalised",
        "physical_reconstruction_factor": 1.0,
    }


def validate_vertical_mode(
    n: int,
    depth_m: float,
    *,
    boundary_tolerance: float = 1.0e-12,
    mean_tolerance: float = 1.0e-12,
) -> dict[str, float]:
    """Check Neumann boundaries and the analytic zero vertical mean."""
    boundaries = np.array([0.0, -depth_m])
    boundary_error = float(
        np.max(np.abs(vertical_mode_derivative(boundaries, n, depth_m)))
    )
    # Integral_{-H}^0 cos(n*pi*z/H) dz is evaluated analytically.
    mean = float((np.sin(0.0) - np.sin(-n * np.pi)) / (n * np.pi))
    if boundary_error > boundary_tolerance:
        raise AssertionError(
            f"n={n}: Neumann boundary residual {boundary_error:.3e} exceeds "
            f"{boundary_tolerance:.3e}."
        )
    if abs(mean) > mean_tolerance:
        raise AssertionError(
            f"n={n}: vertical mean {mean:.3e} exceeds {mean_tolerance:.3e}."
        )
    return {
        "boundary_derivative_max_abs": boundary_error,
        "vertical_mean": mean,
    }
