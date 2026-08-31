"""Checks for the manuscript vertical-mode convention."""

from __future__ import annotations

import numpy as np
import pytest

from vertical_modes import (
    mode_metadata,
    validate_vertical_mode,
    vertical_mode,
    vertical_mode_derivative,
    vertical_wavelength,
)


DEPTH_M = 2000.0


@pytest.mark.parametrize(
    ("mode", "wavelength_m"),
    [(1, 4000.0), (4, 1000.0), (8, 500.0), (16, 250.0), (32, 125.0)],
)
def test_vertical_wavelengths(mode: int, wavelength_m: float) -> None:
    assert vertical_wavelength(mode, DEPTH_M) == wavelength_m


@pytest.mark.parametrize("mode", [1, 2, 4, 8, 16, 32])
def test_neumann_boundaries_and_zero_mean(mode: int) -> None:
    result = validate_vertical_mode(mode, DEPTH_M)
    assert result["boundary_derivative_max_abs"] <= 1.0e-12
    assert abs(result["vertical_mean"]) <= 1.0e-12


@pytest.mark.parametrize("mode", [1, 4, 8, 16, 32])
def test_numerical_mean_and_boundary_derivative(mode: int) -> None:
    z = np.linspace(-DEPTH_M, 0.0, 100_001)
    values = vertical_mode(z, mode, DEPTH_M)
    numerical_mean = np.trapezoid(values, z) / DEPTH_M
    derivative = vertical_mode_derivative(np.array([-DEPTH_M, 0.0]), mode, DEPTH_M)
    assert abs(numerical_mean) < 2.0e-15
    assert np.max(np.abs(derivative)) < 1.0e-12


def test_unnormalised_physical_reconstruction_is_documented() -> None:
    metadata = mode_metadata(4, DEPTH_M)
    assert metadata["normalisation"] == "unnormalised"
    assert metadata["physical_reconstruction_factor"] == 1.0


@pytest.mark.parametrize("depth_m", [0.0, -2000.0])
def test_nonpositive_depth_is_rejected(depth_m: float) -> None:
    with pytest.raises(ValueError, match="depth"):
        vertical_wavelength(4, depth_m)
    with pytest.raises(ValueError, match="depth"):
        vertical_mode(np.array([0.0]), 4, depth_m)
