"""Compute the Gaussian-vortex eigenanalysis used in Figures 6 and 7.

The script solves the Bessel--Galerkin radial eigenvalue problem and stores
the spectra and selected radial eigenfunctions in one compressed NumPy file.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
from scipy import special


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.paper_parameters import (  # noqa: E402
    BUOYANCY_FREQUENCY,
    CORIOLIS_FREQUENCY,
    DOMAIN_DEPTH,
    FLOW_SPEED,
    vertical_mode_dispersive_coefficient,
    vertical_wavelength,
)

FLOW_LENGTH = 25.0e3
DEFAULT_VERTICAL_MODE = 4

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "gaussian_vortex_eigenanalysis.npz"
)


def gaussian_vortex(
    radius: np.ndarray,
    *,
    flow_speed: float = FLOW_SPEED,
    length: float = FLOW_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``U_theta`` and ``dU_theta/dr`` for the anticyclonic vortex."""
    scaled_radius = radius / length
    prefactor = -flow_speed * np.sqrt(np.e)
    envelope = np.exp(-0.5 * scaled_radius**2)
    velocity = prefactor * scaled_radius * envelope
    velocity_gradient = prefactor / length * (1.0 - scaled_radius**2) * envelope
    return velocity, velocity_gradient


@lru_cache(maxsize=None)
def bessel_basis(
    order: int,
    basis_size: int,
    quadrature_size: int,
    radial_domain: float,
) -> tuple[np.ndarray, ...]:
    """Return an orthonormal Bessel basis and radial quadrature."""
    nodes, weights = special.roots_legendre(quadrature_size)
    radius = 0.5 * radial_domain * (nodes + 1.0)
    weights = 0.5 * radial_domain * weights
    roots = special.jn_zeros(order, basis_size)
    normalization = np.sqrt(2.0) / (
        radial_domain * np.abs(special.jv(order + 1, roots))
    )
    argument = np.outer(radius / radial_domain, roots)
    basis = special.jv(order, argument) * normalization
    derivative = (
        special.jvp(order, argument, n=1) * (roots / radial_domain) * normalization
    )
    second_derivative = (
        special.jvp(order, argument, n=2) * (roots / radial_domain) ** 2 * normalization
    )
    return (
        radius,
        weights,
        roots,
        basis,
        derivative,
        second_derivative,
    )


def evaluate_bessel_basis(
    order: int,
    roots: np.ndarray,
    radius: np.ndarray,
    radial_domain: float,
) -> np.ndarray:
    """Evaluate a normalized Bessel basis on arbitrary radii."""
    normalization = np.sqrt(2.0) / (
        radial_domain * np.abs(special.jv(order + 1, roots))
    )
    return special.jv(order, np.outer(radius / radial_domain, roots)) * normalization


def assemble_gaussian_vortex_matrix(
    azimuthal_wavenumber: int,
    *,
    basis_size: int,
    radial_domain: float,
    vertical_mode: int,
    coriolis_frequency: float = CORIOLIS_FREQUENCY,
    buoyancy_frequency: float = BUOYANCY_FREQUENCY,
    length: float = FLOW_LENGTH,
    flow_speed: float = FLOW_SPEED,
) -> np.ndarray:
    """Return the Bessel--Galerkin matrix in ``H c = omega c``."""
    if vertical_mode < 1:
        raise ValueError("The vertical mode must be positive.")
    if basis_size < 8:
        raise ValueError("The Bessel basis must contain at least eight modes.")

    quadrature_size = max(4 * basis_size, 256)
    order_up = abs(azimuthal_wavenumber)
    order_down = abs(azimuthal_wavenumber - 2)

    (
        radius,
        weights,
        roots_up,
        basis_up,
        derivative_up,
        second_derivative_up,
    ) = bessel_basis(
        order_up,
        basis_size,
        quadrature_size,
        radial_domain,
    )
    (
        _,
        _,
        roots_down,
        basis_down,
        derivative_down,
        second_derivative_down,
    ) = bessel_basis(
        order_down,
        basis_size,
        quadrature_size,
        radial_domain,
    )

    dispersive_coefficient = vertical_mode_dispersive_coefficient(
        vertical_mode,
        depth_m=DOMAIN_DEPTH,
        coriolis_frequency=coriolis_frequency,
        buoyancy_frequency=buoyancy_frequency,
    )
    velocity, velocity_gradient = gaussian_vortex(
        radius,
        flow_speed=flow_speed,
        length=length,
    )
    inverse_radius = 1.0 / radius
    inverse_radius_squared = inverse_radius**2

    def project(left_basis: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Project with the polar-area measure ``r dr``."""
        return left_basis.T @ ((radius * weights)[:, None] * field)

    ell = azimuthal_wavenumber
    cross_up = (
        second_derivative_down
        - (2 * ell - 3) * inverse_radius[:, None] * derivative_down
        + ell * (ell - 2) * inverse_radius_squared[:, None] * basis_down
    )
    cross_down = (
        second_derivative_up
        + (2 * ell - 1) * inverse_radius[:, None] * derivative_up
        + ell * (ell - 2) * inverse_radius_squared[:, None] * basis_up
    )

    potential_up = (
        0.5 * velocity_gradient + (2 * ell + 1) * velocity * inverse_radius / 2.0
    )
    potential_down = (
        -0.5 * velocity_gradient
        + (2 * ell - 5) * velocity * inverse_radius / 2.0
        - 2.0 * coriolis_frequency
    )
    cross_potential = 0.5 * (velocity_gradient - velocity * inverse_radius)

    block_up_up = np.diag(
        dispersive_coefficient * (roots_up / radial_domain) ** 2
    ) + project(basis_up, potential_up[:, None] * basis_up)
    block_up_down = project(
        basis_up,
        dispersive_coefficient * cross_up + cross_potential[:, None] * basis_down,
    )
    block_down_down = np.diag(
        -dispersive_coefficient * (roots_down / radial_domain) ** 2
    ) + project(basis_down, potential_down[:, None] * basis_down)
    block_down_up = project(
        basis_down,
        -dispersive_coefficient * cross_down - cross_potential[:, None] * basis_up,
    )

    return np.block(
        [
            [block_up_up, block_up_down],
            [block_down_up, block_down_down],
        ]
    ).astype(complex)


def solve_eigensystem(
    azimuthal_wavenumber: int,
    *,
    basis_size: int,
    radial_domain: float,
    vertical_mode: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve and sort one Gaussian-vortex eigensystem."""
    matrix = assemble_gaussian_vortex_matrix(
        azimuthal_wavenumber,
        basis_size=basis_size,
        radial_domain=radial_domain,
        vertical_mode=vertical_mode,
    )
    frequencies, vectors = np.linalg.eig(matrix)
    order = np.lexsort((frequencies.imag, frequencies.real))
    return frequencies[order], vectors[:, order]


def physical_components(
    vectors: np.ndarray,
    radius: np.ndarray,
    azimuthal_wavenumber: int,
    basis_size: int,
    radial_domain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the two physical components of Bessel eigenvectors."""
    roots_up = special.jn_zeros(abs(azimuthal_wavenumber), basis_size)
    roots_down = special.jn_zeros(
        abs(azimuthal_wavenumber - 2),
        basis_size,
    )
    basis_up = evaluate_bessel_basis(
        abs(azimuthal_wavenumber),
        roots_up,
        radius,
        radial_domain,
    )
    basis_down = evaluate_bessel_basis(
        abs(azimuthal_wavenumber - 2),
        roots_down,
        radius,
        radial_domain,
    )
    return (
        basis_up @ vectors[:basis_size],
        basis_down @ vectors[basis_size:],
    )


def component_amplitude_ratio(
    vectors: np.ndarray,
    *,
    azimuthal_wavenumber: int,
    basis_size: int,
    radial_domain: float,
) -> np.ndarray:
    """Return the radial mean ``|A_down| / |A_up|`` for each eigenvector."""
    radius = np.linspace(
        0.0,
        radial_domain,
        max(4097, 8 * basis_size + 1),
    )[1:]
    component_up, component_down = physical_components(
        vectors,
        radius,
        azimuthal_wavenumber,
        basis_size,
        radial_domain,
    )
    amplitude_up = np.trapezoid(
        radius[:, None] * np.abs(component_up),
        radius,
        axis=0,
    )
    amplitude_down = np.trapezoid(
        radius[:, None] * np.abs(component_down),
        radius,
        axis=0,
    )
    return np.divide(
        amplitude_down,
        amplitude_up,
        out=np.full_like(amplitude_down, np.nan),
        where=amplitude_up > 0.0,
    )


def compute_spectrum(
    values: list[int],
    *,
    coordinate: str,
    fixed_azimuthal_wavenumber: int,
    fixed_vertical_mode: int,
    basis_size: int,
    radial_domain: float,
) -> np.ndarray:
    """Scan either vertical mode ``n`` or azimuthal wavenumber ``ell``."""
    rows: list[tuple[float, int, float, float, float]] = []
    for index, value in enumerate(values, start=1):
        if coordinate == "n":
            azimuthal_wavenumber = fixed_azimuthal_wavenumber
            vertical_mode = value
        elif coordinate == "ell":
            azimuthal_wavenumber = value
            vertical_mode = fixed_vertical_mode
        else:
            raise ValueError(f"Unsupported coordinate: {coordinate}")

        frequencies, vectors = solve_eigensystem(
            azimuthal_wavenumber,
            basis_size=basis_size,
            radial_domain=radial_domain,
            vertical_mode=vertical_mode,
        )
        ratios = component_amplitude_ratio(
            vectors,
            azimuthal_wavenumber=azimuthal_wavenumber,
            basis_size=basis_size,
            radial_domain=radial_domain,
        )
        rows.extend(
            (
                float(value),
                branch,
                frequency.real / CORIOLIS_FREQUENCY,
                frequency.imag / CORIOLIS_FREQUENCY,
                ratio,
            )
            for branch, (frequency, ratio) in enumerate(zip(frequencies, ratios))
        )
        print(
            f"{coordinate} {index:03d}/{len(values)}: "
            f"ell={azimuthal_wavenumber}, n={vertical_mode}"
        )
    return np.asarray(rows, dtype=float)


def lowest_frequency_index(
    frequencies: np.ndarray,
    *,
    frequency_min: float,
    frequency_max: float,
) -> int:
    """Return the lowest real frequency inside the prescribed window."""
    candidates = np.flatnonzero(
        (frequencies.real / CORIOLIS_FREQUENCY >= frequency_min)
        & (frequencies.real / CORIOLIS_FREQUENCY <= frequency_max)
    )
    if candidates.size == 0:
        raise ValueError("No eigenfrequency lies in the selected frequency window.")
    return int(candidates[np.argmin(frequencies.real[candidates])])


def normalize_mode(
    vector: np.ndarray,
    *,
    radius: np.ndarray,
    azimuthal_wavenumber: int,
    basis_size: int,
    radial_domain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fix the common phase and apply unit peak combined normalization."""
    component_up, component_down = physical_components(
        vector[:, None],
        radius,
        azimuthal_wavenumber,
        basis_size,
        radial_domain,
    )
    component_up = component_up[:, 0]
    component_down = component_down[:, 0]

    combined = np.r_[component_up, component_down]
    reference_phase = np.angle(combined[np.argmax(np.abs(combined))])
    phase_factor = np.exp(-1j * reference_phase)
    component_up = component_up * phase_factor
    component_down = component_down * phase_factor

    normalization = np.max(
        np.sqrt(np.abs(component_up) ** 2 + np.abs(component_down) ** 2)
    )
    if normalization == 0.0:
        raise ValueError("The selected eigenvector has zero physical norm.")
    return (
        component_up / normalization,
        component_down / normalization,
    )


def compute_selected_modes(
    azimuthal_wavenumbers: list[int],
    *,
    vertical_mode: int,
    basis_size: int,
    radial_domain: float,
    frequency_min: float,
    frequency_max: float,
    radial_grid_size: int,
) -> dict[str, np.ndarray]:
    """Compute the selected radial eigenfunctions used in the mode maps."""
    radius = np.linspace(0.0, radial_domain, radial_grid_size)
    frequencies_out = np.empty(
        len(azimuthal_wavenumbers),
        dtype=complex,
    )
    branch_indices = np.empty(
        len(azimuthal_wavenumbers),
        dtype=int,
    )
    component_up = np.empty(
        (len(azimuthal_wavenumbers), radial_grid_size),
        dtype=complex,
    )
    component_down = np.empty_like(component_up)

    for index, azimuthal_wavenumber in enumerate(azimuthal_wavenumbers):
        frequencies, vectors = solve_eigensystem(
            azimuthal_wavenumber,
            basis_size=basis_size,
            radial_domain=radial_domain,
            vertical_mode=vertical_mode,
        )
        branch = lowest_frequency_index(
            frequencies,
            frequency_min=frequency_min,
            frequency_max=frequency_max,
        )
        up, down = normalize_mode(
            vectors[:, branch],
            radius=radius,
            azimuthal_wavenumber=azimuthal_wavenumber,
            basis_size=basis_size,
            radial_domain=radial_domain,
        )
        frequencies_out[index] = frequencies[branch]
        branch_indices[index] = branch
        component_up[index] = up
        component_down[index] = down
        print(
            f"selected ell={azimuthal_wavenumber}: "
            f"omega/f={frequencies[branch].real / CORIOLIS_FREQUENCY:.6g}"
        )

    return {
        "mode_radius": radius,
        "mode_azimuthal_wavenumbers": np.asarray(
            azimuthal_wavenumbers,
            dtype=int,
        ),
        "mode_frequencies": frequencies_out,
        "mode_branch_indices": branch_indices,
        "mode_component_up": component_up,
        "mode_component_down": component_down,
    }


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute the axisymmetric Gaussian-vortex eigenanalysis."
    )
    parser.add_argument("--basis-size", type=int, default=512)
    parser.add_argument("--radial-domain-over-length", type=float, default=10.0)
    parser.add_argument("--fixed-azimuthal-wavenumber", type=int, default=0)
    parser.add_argument(
        "--fixed-vertical-mode",
        type=int,
        default=DEFAULT_VERTICAL_MODE,
    )
    parser.add_argument("--vertical-mode-min", type=int, default=1)
    parser.add_argument("--vertical-mode-max", type=int, default=32)
    parser.add_argument("--azimuthal-wavenumber-min", type=int, default=-16)
    parser.add_argument("--azimuthal-wavenumber-max", type=int, default=16)
    parser.add_argument(
        "--mode-azimuthal-wavenumbers",
        type=parse_int_list,
        default=parse_int_list("0,1,2,3"),
    )
    parser.add_argument(
        "--mode-vertical-mode",
        type=int,
        default=DEFAULT_VERTICAL_MODE,
    )
    parser.add_argument("--mode-frequency-min", type=float, default=-0.5)
    parser.add_argument("--mode-frequency-max", type=float, default=0.5)
    parser.add_argument("--mode-radial-grid-size", type=int, default=1201)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the complete Gaussian-vortex calculation."""
    args = parse_args()
    if args.radial_domain_over_length <= 0.0:
        raise ValueError("The radial-domain ratio must be positive.")
    if args.vertical_mode_min < 1:
        raise ValueError("The minimum vertical mode must be positive.")
    if args.vertical_mode_max < args.vertical_mode_min:
        raise ValueError("The vertical-mode range is empty.")
    if args.azimuthal_wavenumber_max < args.azimuthal_wavenumber_min:
        raise ValueError("The azimuthal-wavenumber range is empty.")

    radial_domain = args.radial_domain_over_length * FLOW_LENGTH
    vertical_mode_spectrum = compute_spectrum(
        list(
            range(
                args.vertical_mode_min,
                args.vertical_mode_max + 1,
            )
        ),
        coordinate="n",
        fixed_azimuthal_wavenumber=args.fixed_azimuthal_wavenumber,
        fixed_vertical_mode=args.fixed_vertical_mode,
        basis_size=args.basis_size,
        radial_domain=radial_domain,
    )
    azimuthal_spectrum = compute_spectrum(
        list(
            range(
                args.azimuthal_wavenumber_min,
                args.azimuthal_wavenumber_max + 1,
            )
        ),
        coordinate="ell",
        fixed_azimuthal_wavenumber=args.fixed_azimuthal_wavenumber,
        fixed_vertical_mode=args.fixed_vertical_mode,
        basis_size=args.basis_size,
        radial_domain=radial_domain,
    )
    selected_modes = compute_selected_modes(
        args.mode_azimuthal_wavenumbers,
        vertical_mode=args.mode_vertical_mode,
        basis_size=args.basis_size,
        radial_domain=radial_domain,
        frequency_min=args.mode_frequency_min,
        frequency_max=args.mode_frequency_max,
        radial_grid_size=args.mode_radial_grid_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vertical_mode_spectrum=vertical_mode_spectrum,
        azimuthal_spectrum=azimuthal_spectrum,
        radial_domain=np.asarray(radial_domain),
        **selected_modes,  # type: ignore[arg-type]
    )
    metadata = {
        "background_flow": ("U_theta(r) = -U_ref sqrt(e) (r/L) exp[-r^2/(2L^2)]"),
        "radial_discretization": "Bessel--Galerkin",
        "coriolis_frequency_s-1": CORIOLIS_FREQUENCY,
        "buoyancy_frequency_s-1": BUOYANCY_FREQUENCY,
        "flow_speed_m_s-1": FLOW_SPEED,
        "flow_length_m": FLOW_LENGTH,
        "domain_depth_m": DOMAIN_DEPTH,
        "vertical_mode_definition": "Z_n(z) = cos(n*pi*z/H)",
        "vertical_wavelength_m": vertical_wavelength(args.mode_vertical_mode),
        "normalisation": "unnormalised",
        "physical_reconstruction_factor": 1.0,
        "basis_size_per_component": args.basis_size,
        "radial_domain_m": radial_domain,
        "fixed_azimuthal_wavenumber": args.fixed_azimuthal_wavenumber,
        "fixed_vertical_mode": args.fixed_vertical_mode,
        "selected_mode_azimuthal_wavenumbers": (args.mode_azimuthal_wavenumbers),
        "selected_mode_vertical_mode": args.mode_vertical_mode,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(args.output)
    print(args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
