"""Physical constants shared by the manuscript calculations."""

from __future__ import annotations

import math


CORIOLIS_FREQUENCY = 1.0e-4
BUOYANCY_FREQUENCY_RATIO = 20.0
BUOYANCY_FREQUENCY = BUOYANCY_FREQUENCY_RATIO * CORIOLIS_FREQUENCY
DOMAIN_DEPTH = 2000.0
FLOW_SPEED = 0.25


def vertical_mode_wavenumber(
    mode: int,
    depth_m: float = DOMAIN_DEPTH,
) -> float:
    """Return ``n*pi/H`` for the manuscript's unnormalised cosine modes."""
    if mode < 1:
        raise ValueError("The vertical mode must be positive.")
    return mode * math.pi / depth_m


def vertical_wavelength(
    mode: int,
    depth_m: float = DOMAIN_DEPTH,
) -> float:
    """Return the physical vertical wavelength ``2H/n`` in metres."""
    vertical_mode_wavenumber(mode, depth_m)
    return 2.0 * depth_m / mode


def vertical_mode_dispersive_coefficient(
    mode: int,
    *,
    depth_m: float = DOMAIN_DEPTH,
    coriolis_frequency: float = CORIOLIS_FREQUENCY,
    buoyancy_frequency: float = BUOYANCY_FREQUENCY,
) -> float:
    """Return ``N^2 H^2 / (2*pi^2*f*n^2)`` for one vertical mode."""
    vertical_mode_wavenumber(mode, depth_m)
    return (
        buoyancy_frequency**2
        * depth_m**2
        / (2.0 * math.pi**2 * coriolis_frequency * mode**2)
    )
