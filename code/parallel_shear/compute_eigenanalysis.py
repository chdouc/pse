"""Compute the parallel-shear eigenanalysis used in Figures 4 and 5.

The script solves the Fourier--Galerkin eigenvalue problem and stores all
quantities required by the two plotting scripts in one compressed NumPy file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CORIOLIS_FREQUENCY = 1.0e-4
BUOYANCY_FREQUENCY = 20.0 * CORIOLIS_FREQUENCY
FLOW_LENGTH = 10.0e3
FLOW_SPEED = 0.25
DOMAIN_DEPTH = 4000.0
DEFAULT_VERTICAL_MODE = 4

SIGMA_0 = np.array([[1, 0], [0, 1]], dtype=complex)
SIGMA_1 = np.array([[1j, 0], [0, -1j]], dtype=complex)
SIGMA_2 = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_3 = np.array([[0, 1j], [-1j, 0]], dtype=complex)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "parallel_shear_eigenanalysis.npz"
)


def fourier_operators(
    truncation: int,
    length: float = FLOW_LENGTH,
) -> tuple[np.ndarray, ...]:
    """Return Fourier--Galerkin operators on one periodic shear cell."""
    modes = np.arange(-truncation, truncation + 1)
    size = modes.size
    identity = np.eye(size, dtype=complex)
    derivative_x = np.diag(1j * modes / length)
    derivative_xx = np.diag(-((modes / length) ** 2))

    cosine = np.zeros((size, size), dtype=complex)
    sine = np.zeros((size, size), dtype=complex)
    mode_index = {mode: index for index, mode in enumerate(modes)}
    for column, mode in enumerate(modes):
        if mode + 1 in mode_index:
            cosine[mode_index[mode + 1], column] += 0.5
            sine[mode_index[mode + 1], column] += 1.0 / (2.0j)
        if mode - 1 in mode_index:
            cosine[mode_index[mode - 1], column] += 0.5
            sine[mode_index[mode - 1], column] -= 1.0 / (2.0j)

    return modes, identity, derivative_x, derivative_xx, cosine, sine


def assemble_parallel_shear_matrix(
    wavenumber_y: float,
    *,
    truncation: int,
    vertical_wavenumber: float,
    coriolis_frequency: float = CORIOLIS_FREQUENCY,
    buoyancy_frequency: float = BUOYANCY_FREQUENCY,
    length: float = FLOW_LENGTH,
    flow_speed: float = FLOW_SPEED,
) -> np.ndarray:
    """Return the matrix ``H`` in the eigenvalue problem ``H a = omega a``."""
    gamma = -(buoyancy_frequency**2 / (coriolis_frequency**2 * vertical_wavenumber**2))
    _, identity, derivative_x, derivative_xx, cosine, sine = fourier_operators(
        truncation,
        length,
    )

    # V(x) = -U_ref sin(x/L), with xi_1 = xi_3 = dV/dx and xi_2 = 0.
    velocity_y = -flow_speed * sine
    velocity_gradient = -(flow_speed / length) * cosine

    delta_0 = derivative_xx - wavenumber_y**2 * identity
    delta_2 = derivative_xx + wavenumber_y**2 * identity
    delta_3 = 2.0j * wavenumber_y * derivative_x

    generator = (
        np.kron(SIGMA_0, 1.0j * wavenumber_y * velocity_y)
        + np.kron(SIGMA_1, coriolis_frequency * identity)
        + 0.5 * np.kron(SIGMA_1 + SIGMA_3, velocity_gradient)
        + (coriolis_frequency * gamma / 2.0)
        * (
            np.kron(SIGMA_1, delta_0)
            - np.kron(SIGMA_3, delta_2)
            + np.kron(SIGMA_2, delta_3)
        )
    )
    return -coriolis_frequency * np.kron(SIGMA_0, identity) - 1.0j * generator


def solve_eigensystem(
    wavenumber_y: float,
    *,
    truncation: int,
    vertical_mode: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve and sort one parallel-shear eigensystem by complex frequency."""
    vertical_wavenumber = vertical_mode * np.pi / DOMAIN_DEPTH
    matrix = assemble_parallel_shear_matrix(
        wavenumber_y,
        truncation=truncation,
        vertical_wavenumber=vertical_wavenumber,
    )
    frequencies, vectors = np.linalg.eig(matrix)
    order = np.lexsort((frequencies.imag, frequencies.real))
    return frequencies[order], vectors[:, order]


def component_amplitude_ratio(
    vectors: np.ndarray,
    *,
    truncation: int,
    quadrature_points: int = 512,
) -> np.ndarray:
    """Return the cell-mean ``|A_down| / |A_up|`` for each eigenvector."""
    modes = np.arange(-truncation, truncation + 1)
    coordinate = np.linspace(0.0, 2.0 * np.pi, quadrature_points, endpoint=False)
    basis = np.exp(1j * np.outer(coordinate, modes))
    component_size = modes.size
    amplitude_up = np.mean(
        np.abs(basis @ vectors[:component_size, :]),
        axis=0,
    )
    amplitude_down = np.mean(
        np.abs(basis @ vectors[component_size:, :]),
        axis=0,
    )
    return np.divide(
        amplitude_down,
        amplitude_up,
        out=np.full_like(amplitude_down, np.nan, dtype=float),
        where=amplitude_up > 0.0,
    )


def spectrum_rows(
    coordinate: float,
    frequencies: np.ndarray,
    ratios: np.ndarray,
) -> list[tuple[float, int, float, float, float, float]]:
    """Convert one eigensystem to the common six-column spectrum format."""
    return [
        (
            coordinate,
            branch,
            frequency.real / CORIOLIS_FREQUENCY,
            frequency.imag / CORIOLIS_FREQUENCY,
            abs(frequency.imag / CORIOLIS_FREQUENCY),
            ratios[branch],
        )
        for branch, frequency in enumerate(frequencies)
    ]


def compute_along_stream_spectrum(
    *,
    truncation: int,
    wavenumber_min: float,
    wavenumber_max: float,
    sample_count: int,
    vertical_mode: int,
) -> np.ndarray:
    """Scan the dimensionless along-stream wavenumber ``k_y L``."""
    rows: list[tuple[float, int, float, float, float, float]] = []
    values = np.linspace(wavenumber_min, wavenumber_max, sample_count)
    for index, dimensionless_wavenumber in enumerate(values, start=1):
        frequencies, vectors = solve_eigensystem(
            dimensionless_wavenumber / FLOW_LENGTH,
            truncation=truncation,
            vertical_mode=vertical_mode,
        )
        ratios = component_amplitude_ratio(vectors, truncation=truncation)
        rows.extend(
            spectrum_rows(
                dimensionless_wavenumber,
                frequencies,
                ratios,
            )
        )
        print(f"k_y L {index:03d}/{sample_count}: {dimensionless_wavenumber:.4g}")
    return np.asarray(rows, dtype=float)


def compute_vertical_mode_spectrum(
    *,
    truncation: int,
    mode_min: int,
    mode_max: int,
    mode_step: int,
    dimensionless_wavenumber: float,
) -> np.ndarray:
    """Scan the integer vertical mode at fixed ``k_y L``."""
    rows: list[tuple[float, int, float, float, float, float]] = []
    mode_values = range(mode_min, mode_max + 1, mode_step)
    for index, vertical_mode in enumerate(mode_values, start=1):
        frequencies, vectors = solve_eigensystem(
            dimensionless_wavenumber / FLOW_LENGTH,
            truncation=truncation,
            vertical_mode=vertical_mode,
        )
        ratios = component_amplitude_ratio(vectors, truncation=truncation)
        rows.extend(spectrum_rows(float(vertical_mode), frequencies, ratios))
        print(
            f"n {index:03d}/{len(range(mode_min, mode_max + 1, mode_step))}: "
            f"{vertical_mode}"
        )
    return np.asarray(rows, dtype=float)


def reconstruct_mode(
    coefficients: np.ndarray,
    modes: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Transform Fourier coefficients to physical space."""
    return np.exp(1j * np.outer(coordinate, modes)) @ coefficients


def fix_common_phase(
    component_up: np.ndarray,
    component_down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a common phase that makes the spinor as real as possible."""
    combined = np.concatenate([component_up, component_down])
    phase_measure = np.sum(combined * combined)
    if np.abs(phase_measure) > 0.0:
        phase = np.exp(-0.5j * np.angle(phase_measure))
    else:
        phase = np.exp(-1j * np.angle(combined[np.argmax(np.abs(combined))]))

    component_up = phase * component_up
    component_down = phase * component_down
    combined = np.concatenate([component_up, component_down])
    if np.real(combined[np.argmax(np.abs(combined))]) < 0.0:
        component_up = -component_up
        component_down = -component_down
    return component_up, component_down


def representative_indices(
    frequencies: np.ndarray,
    *,
    frequency_min: float,
    frequency_max: float,
    merge_tolerance: float = 1.0e-2,
) -> list[int]:
    """Select one representative from each nearly degenerate frequency."""
    candidates = [
        int(index)
        for index in np.argsort(frequencies.real)
        if frequency_min
        <= frequencies[index].real / CORIOLIS_FREQUENCY
        <= frequency_max
    ]
    representatives: list[int] = []
    for index in candidates:
        if (
            not representatives
            or abs(
                frequencies[index].real / CORIOLIS_FREQUENCY
                - frequencies[representatives[-1]].real / CORIOLIS_FREQUENCY
            )
            > merge_tolerance
        ):
            representatives.append(index)
    return representatives


def display_indices(
    frequencies: np.ndarray,
    *,
    selected_count: int,
    skip_count: int,
    frequency_min: float,
    frequency_max: float,
) -> list[int]:
    """Apply the branch selection used for the eigenfunction overlays."""
    representatives = representative_indices(
        frequencies,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
    )
    if len(representatives) < selected_count:
        raise ValueError("Fewer eigenmode representatives were found than requested.")
    return representatives[:selected_count][::-1][skip_count:]


def compute_eigenfunction_overlays(
    *,
    truncation: int,
    wavenumbers: list[float],
    vertical_mode: int,
    selected_count: int,
    skip_count: int,
    frequency_min: float,
    frequency_max: float,
    grid_size: int,
) -> dict[str, np.ndarray]:
    """Compute the phase-aligned eigenfunctions plotted for several ``k_y L``."""
    coordinate = np.linspace(-np.pi, np.pi, grid_size, endpoint=True)
    modes, *_ = fourier_operators(truncation)
    component_size = modes.size
    result_count = selected_count - skip_count

    component_up = np.empty(
        (len(wavenumbers), result_count, grid_size),
        dtype=float,
    )
    component_down = np.empty_like(component_up)
    frequencies_out = np.empty((len(wavenumbers), result_count), dtype=complex)
    ratios = np.empty((len(wavenumbers), result_count), dtype=float)
    branch_indices = np.empty(
        (len(wavenumbers), result_count),
        dtype=int,
    )

    previous_up: list[np.ndarray] | None = None
    previous_down: list[np.ndarray] | None = None
    for wavenumber_index, dimensionless_wavenumber in enumerate(wavenumbers):
        frequencies, vectors = solve_eigensystem(
            dimensionless_wavenumber / FLOW_LENGTH,
            truncation=truncation,
            vertical_mode=vertical_mode,
        )
        indices = display_indices(
            frequencies,
            selected_count=selected_count,
            skip_count=skip_count,
            frequency_min=frequency_min,
            frequency_max=frequency_max,
        )

        current_up: list[np.ndarray] = []
        current_down: list[np.ndarray] = []
        for mode_index, branch_index in enumerate(indices):
            vector = vectors[:, branch_index]
            up = reconstruct_mode(
                vector[:component_size],
                modes,
                coordinate,
            )
            down = reconstruct_mode(
                vector[component_size:],
                modes,
                coordinate,
            )
            up, down = fix_common_phase(up, down)

            if previous_up is not None and previous_down is not None:
                overlap = np.real(
                    np.vdot(previous_up[mode_index], up)
                    + np.vdot(previous_down[mode_index], down)
                )
                if overlap < 0.0:
                    up = -up
                    down = -down

            scale = np.max(np.sqrt(np.abs(up) ** 2 + np.abs(down) ** 2))
            if scale > 0.0:
                up = up / scale
                down = down / scale

            component_up[wavenumber_index, mode_index] = up.real
            component_down[wavenumber_index, mode_index] = down.real
            frequencies_out[wavenumber_index, mode_index] = frequencies[branch_index]
            ratios[wavenumber_index, mode_index] = float(
                np.linalg.norm(vector[component_size:])
                / np.linalg.norm(vector[:component_size])
            )
            branch_indices[wavenumber_index, mode_index] = branch_index
            current_up.append(up)
            current_down.append(down)

        previous_up = current_up
        previous_down = current_down

    return {
        "overlay_coordinate": coordinate,
        "overlay_wavenumbers": np.asarray(wavenumbers, dtype=float),
        "overlay_component_up": component_up,
        "overlay_component_down": component_down,
        "overlay_frequencies": frequencies_out,
        "overlay_ratios": ratios,
        "overlay_branch_indices": branch_indices,
    }


def save_results(
    output_path: Path,
    *,
    along_stream_spectrum: np.ndarray,
    vertical_mode_spectrum: np.ndarray,
    overlays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    """Write numerical results and a human-readable metadata file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        along_stream_spectrum=along_stream_spectrum,
        vertical_mode_spectrum=vertical_mode_spectrum,
        **overlays,
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floating-point values."""
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute the sinusoidal parallel-shear eigenanalysis."
    )
    parser.add_argument("--truncation", type=int, default=48)
    parser.add_argument("--wavenumber-min", type=float, default=-3.0)
    parser.add_argument("--wavenumber-max", type=float, default=3.0)
    parser.add_argument("--wavenumber-count", type=int, default=161)
    parser.add_argument("--vertical-mode", type=int, default=DEFAULT_VERTICAL_MODE)
    parser.add_argument("--mode-min", type=int, default=1)
    parser.add_argument("--mode-max", type=int, default=32)
    parser.add_argument("--mode-step", type=int, default=1)
    parser.add_argument("--fixed-wavenumber", type=float, default=0.0)
    parser.add_argument(
        "--overlay-wavenumbers",
        type=parse_float_list,
        default=parse_float_list("0,0.25,0.5,0.75,1.0"),
    )
    parser.add_argument("--overlay-selected-count", type=int, default=5)
    parser.add_argument("--overlay-skip-count", type=int, default=2)
    parser.add_argument("--overlay-frequency-min", type=float, default=-0.2)
    parser.add_argument("--overlay-frequency-max", type=float, default=1.5)
    parser.add_argument("--overlay-grid-size", type=int, default=900)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the complete parallel-shear calculation."""
    args = parse_args()
    if args.mode_step <= 0:
        raise ValueError("--mode-step must be positive.")
    if args.overlay_skip_count >= args.overlay_selected_count:
        raise ValueError(
            "--overlay-skip-count must be smaller than the selected count."
        )

    along_stream_spectrum = compute_along_stream_spectrum(
        truncation=args.truncation,
        wavenumber_min=args.wavenumber_min,
        wavenumber_max=args.wavenumber_max,
        sample_count=args.wavenumber_count,
        vertical_mode=args.vertical_mode,
    )
    vertical_mode_spectrum = compute_vertical_mode_spectrum(
        truncation=args.truncation,
        mode_min=args.mode_min,
        mode_max=args.mode_max,
        mode_step=args.mode_step,
        dimensionless_wavenumber=args.fixed_wavenumber,
    )
    overlays = compute_eigenfunction_overlays(
        truncation=args.truncation,
        wavenumbers=args.overlay_wavenumbers,
        vertical_mode=args.vertical_mode,
        selected_count=args.overlay_selected_count,
        skip_count=args.overlay_skip_count,
        frequency_min=args.overlay_frequency_min,
        frequency_max=args.overlay_frequency_max,
        grid_size=args.overlay_grid_size,
    )

    metadata = {
        "background_flow": "V(x) = -U_ref sin(x/L)",
        "coriolis_frequency_s-1": CORIOLIS_FREQUENCY,
        "buoyancy_frequency_s-1": BUOYANCY_FREQUENCY,
        "flow_length_m": FLOW_LENGTH,
        "flow_speed_m_s-1": FLOW_SPEED,
        "domain_depth_m": DOMAIN_DEPTH,
        "fourier_truncation": args.truncation,
        "vertical_mode": args.vertical_mode,
        "fixed_wavenumber_k_y_L": args.fixed_wavenumber,
        "wavenumber_range_k_y_L": [
            args.wavenumber_min,
            args.wavenumber_max,
        ],
        "wavenumber_sample_count": args.wavenumber_count,
        "vertical_mode_range": [
            args.mode_min,
            args.mode_max,
            args.mode_step,
        ],
        "overlay_wavenumbers_k_y_L": args.overlay_wavenumbers,
    }
    save_results(
        args.output,
        along_stream_spectrum=along_stream_spectrum,
        vertical_mode_spectrum=vertical_mode_spectrum,
        overlays=overlays,
        metadata=metadata,
    )
    print(args.output)
    print(args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
