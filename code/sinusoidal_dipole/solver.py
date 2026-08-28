"""Self-contained pseudospectral solver for the sinusoidal-dipole case.

The horizontal equations are integrated in coordinates x/L, y/L and ft.
Input and output metadata retain the dimensional manuscript parameters.  The
model order throughout is YBJ, TSB, YBJ+, PSE and the modal hydrostatic
Boussinesq equations (HBEs).
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import h5py
import numpy as np

from specification import (
    MODEL_NAMES,
    NRE_MODEL_NAMES,
    simulation_configuration,
    simulation_signature,
    validate_config,
)
from vertical_modes import mode_metadata, validate_vertical_mode


@dataclass(frozen=True)
class Parameters:
    """Dimensional inputs and their dimensionless numerical counterparts."""

    f: float
    buoyancy_ratio: float
    depth_m: float
    length_m: float
    background_speed_m_s: float
    amplitude_m_s: float
    grid_points: int
    steps_per_period: int
    total_periods: int
    seed: int
    contour_points: int
    dealiasing: str
    diffusion: str
    output_precision: str

    @property
    def buoyancy_frequency(self) -> float:
        return self.buoyancy_ratio * self.f

    @property
    def rossby_number(self) -> float:
        return self.background_speed_m_s / (self.f * self.length_m)

    @property
    def dt(self) -> float:
        return 2.0 * np.pi / self.steps_per_period

    @property
    def step_count(self) -> int:
        return self.total_periods * self.steps_per_period

    def phase_speed(self, n: int) -> float:
        return self.buoyancy_frequency / (n * np.pi / self.depth_m)

    def dimensionless_phase_speed(self, n: int) -> float:
        return self.phase_speed(n) / (self.f * self.length_m)


def parameters_from_config(config: dict[str, Any]) -> Parameters:
    """Construct the numerical parameter set from a reproduction config."""
    validate_config(config)
    physical = config["physical_parameters"]
    numerical = config["numerical_parameters"]
    grid = int(numerical["horizontal_grid"])
    steps = int(numerical["time_steps_per_inertial_period"])
    return Parameters(
        f=float(physical["coriolis_frequency_s-1"]),
        buoyancy_ratio=float(physical["buoyancy_frequency_ratio"]),
        depth_m=float(physical["domain_depth_m"]),
        length_m=float(physical["background_length_scale_m"]),
        background_speed_m_s=float(physical["background_velocity_m_s"]),
        amplitude_m_s=float(physical["initial_velocity_amplitude_m_s"]),
        grid_points=grid,
        steps_per_period=steps,
        total_periods=int(numerical["total_inertial_periods"]),
        seed=int(config["random_seed"]),
        contour_points=int(numerical["pse_etdrk4_contour_points"]),
        dealiasing=str(numerical["dealiasing"]),
        diffusion=str(numerical["diffusion"]),
        output_precision=str(numerical["output_precision"]),
    )


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON reproduction configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


class Grid:
    """Periodic dimensionless grid and common background fields."""

    def __init__(self, parameters: Parameters):
        n = parameters.grid_points
        self.n = n
        self.coordinate = np.linspace(-np.pi, np.pi, n, endpoint=False)
        self.x, self.y = np.meshgrid(self.coordinate, self.coordinate)
        wave_numbers = np.fft.fftfreq(n, d=1.0 / n)
        self.kx, self.ky = np.meshgrid(wave_numbers, wave_numbers)
        self.k2 = self.kx**2 + self.ky**2
        cutoff = (2.0 / 3.0) * np.max(np.abs(wave_numbers))
        self.mask = ((np.abs(self.kx) <= cutoff) & (np.abs(self.ky) <= cutoff)).astype(
            float
        )

        speed = parameters.rossby_number / np.sqrt(2.0)
        self.u = -speed * np.cos(self.y)
        self.v = -speed * np.cos(self.x)
        self.xi0 = np.zeros_like(self.x)
        self.xi1 = speed * (np.sin(self.x) - np.sin(self.y))
        self.xi2 = np.zeros_like(self.x)
        self.xi3 = speed * (np.sin(self.x) + np.sin(self.y))
        self.ux = np.zeros_like(self.x)
        self.uy = speed * np.sin(self.y)
        self.vx = speed * np.sin(self.x)
        self.vy = np.zeros_like(self.x)


def fft2(values: np.ndarray) -> np.ndarray:
    return np.fft.fft2(values, axes=(-2, -1))


def ifft2(values: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(values, axes=(-2, -1))


def spectral_filter(values: np.ndarray, grid: Grid) -> np.ndarray:
    """Apply the forced two-thirds projector on the last two axes."""
    return ifft2(fft2(values) * grid.mask)


def relative_l2(reference: np.ndarray, model: np.ndarray) -> float:
    """Return the manuscript complex-velocity normalized RMS error."""
    denominator = float(np.sum(np.abs(reference) ** 2))
    if denominator <= 0.0:
        raise FloatingPointError("The HBE reference energy is not positive.")
    return float(np.sqrt(np.sum(np.abs(model - reference) ** 2) / denominator))


def _hbe_rhs(state: np.ndarray, grid: Grid, c: float) -> np.ndarray:
    spectra = fft2(state)
    dx = ifft2(1j * grid.kx * spectra).real
    dy = ifft2(1j * grid.ky * spectra).real
    u, v = state[..., 0, :, :], state[..., 1, :, :]
    ux, vx, px = dx[..., 0, :, :], dx[..., 1, :, :], dx[..., 2, :, :]
    uy, vy, py = dy[..., 0, :, :], dy[..., 1, :, :], dy[..., 2, :, :]
    rhs = np.empty_like(state)
    rhs[..., 0, :, :] = (
        -(grid.u * ux + grid.v * uy) - (u * grid.ux + v * grid.uy) + v - px
    )
    rhs[..., 1, :, :] = (
        -(grid.u * vx + grid.v * vy) - (u * grid.vx + v * grid.vy) - u - py
    )
    rhs[..., 2, :, :] = -(grid.u * px + grid.v * py) - c**2 * (ux + vy)
    return spectral_filter(rhs, grid).real


def _rk4_hbe(state: np.ndarray, dt: float, grid: Grid, c: float) -> np.ndarray:
    k1 = _hbe_rhs(state, grid, c)
    k2 = _hbe_rhs(state + 0.5 * dt * k1, grid, c)
    k3 = _hbe_rhs(state + 0.5 * dt * k2, grid, c)
    k4 = _hbe_rhs(state + dt * k3, grid, c)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _scalar_rhs(envelopes: np.ndarray, grid: Grid) -> np.ndarray:
    spectra = fft2(envelopes)
    dx = ifft2(1j * grid.kx * spectra)
    dy = ifft2(1j * grid.ky * spectra)
    rhs = -(grid.u * dx + grid.v * dy) - 0.5j * grid.xi1 * envelopes
    return spectral_filter(rhs, grid)


def _rk4_scalar(
    envelopes: np.ndarray,
    dt: float,
    grid: Grid,
) -> np.ndarray:
    k1 = _scalar_rhs(envelopes, grid)
    k2 = _scalar_rhs(envelopes + 0.5 * dt * k1, grid)
    k3 = _scalar_rhs(envelopes + 0.5 * dt * k2, grid)
    k4 = _scalar_rhs(envelopes + dt * k3, grid)
    return envelopes + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _scalar_strang_step(
    envelopes: np.ndarray,
    factors: np.ndarray,
    dt: float,
    grid: Grid,
) -> np.ndarray:
    envelopes = _rk4_scalar(envelopes, 0.5 * dt, grid)
    envelopes = ifft2(factors * fft2(envelopes))
    return _rk4_scalar(envelopes, 0.5 * dt, grid)


def _scalar_factors(c: float, dt: float, grid: Grid) -> np.ndarray:
    ybj = np.exp(-0.5j * dt * c**2 * grid.k2)
    tsb = np.exp(-0.5j * dt * c**2 * grid.k2 + 0.125j * dt * c**4 * grid.k2**2)
    lplus = 1.0 + 0.25 * c**2 * grid.k2
    ybj_plus = np.exp(-0.5j * dt * c**2 * grid.k2 / lplus)
    return np.stack((ybj, tsb, ybj_plus), axis=0)


def _matrix_from_scalar_function(
    z11: np.ndarray,
    z12: np.ndarray,
    z21: np.ndarray,
    z22: np.ndarray,
    function,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centre = 0.5 * (z11 + z22)
    radius = np.sqrt((0.5 * (z11 - z22)) ** 2 + z12 * z21)
    first = centre + radius
    second = centre - radius
    f_first = function(first)
    f_second = function(second)
    difference = first - second
    if np.any(np.abs(difference) < 1.0e-13):
        raise FloatingPointError("A repeated PSE linear eigenvalue was encountered.")
    slope = (f_first - f_second) / difference
    intercept = (first * f_second - second * f_first) / difference
    return (
        slope * z11 + intercept,
        slope * z12,
        slope * z21,
        slope * z22 + intercept,
    )


def _etdrk4_coefficients(
    c: float,
    dt: float,
    grid: Grid,
    contour_points: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build matrix ETDRK4 coefficients using vectorised contour averages."""
    d0 = -grid.k2
    d2 = -(grid.kx**2 - grid.ky**2)
    d3 = -2.0 * grid.kx * grid.ky
    gamma = -(c**2)

    # Z=dt*L for L=i*sigma0-sigma1-(gamma/2)*(D0*sigma1+
    # D2*sigma2*sigma1+D3*sigma3*sigma1).
    z11 = dt * (-0.5 * gamma * (1j * d0))
    z22 = dt * (2j + 0.5 * gamma * (1j * d0))
    z12 = dt * (-0.5 * gamma * (-1j * d2 + d3))
    z21 = dt * (-0.5 * gamma * (1j * d2 + d3))

    roots = np.exp(2j * np.pi * (np.arange(contour_points) + 0.5) / contour_points)

    def contour(kind: str):
        def evaluate(z: np.ndarray) -> np.ndarray:
            w = z[..., None] + roots
            exponential = np.exp(w)
            if kind == "Q":
                values = dt * (np.exp(0.5 * w) - 1.0) / w
            elif kind == "f1":
                values = dt * (-4.0 - w + exponential * (4.0 - 3.0 * w + w**2)) / w**3
            elif kind == "f2":
                values = dt * (2.0 + w + exponential * (-2.0 + w)) / w**3
            elif kind == "f3":
                values = dt * (-4.0 - 3.0 * w - w**2 + exponential * (4.0 - w)) / w**3
            else:
                raise ValueError(kind)
            return np.mean(values, axis=-1)

        return evaluate

    return {
        "E": _matrix_from_scalar_function(z11, z12, z21, z22, np.exp),
        "E2": _matrix_from_scalar_function(
            z11, z12, z21, z22, lambda value: np.exp(0.5 * value)
        ),
        "Q": _matrix_from_scalar_function(z11, z12, z21, z22, contour("Q")),
        "f1": _matrix_from_scalar_function(z11, z12, z21, z22, contour("f1")),
        "f2": _matrix_from_scalar_function(z11, z12, z21, z22, contour("f2")),
        "f3": _matrix_from_scalar_function(z11, z12, z21, z22, contour("f3")),
    }


def _apply_matrix(
    coefficient: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c11, c12, c21, c22 = coefficient
    return c11 * first + c12 * second, c21 * first + c22 * second


def _pse_rhs(
    up: np.ndarray,
    down_conjugate: np.ndarray,
    up_hat: np.ndarray,
    down_hat: np.ndarray,
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray]:
    up_x = ifft2(1j * grid.kx * up_hat)
    up_y = ifft2(1j * grid.ky * up_hat)
    down_x = ifft2(1j * grid.kx * down_hat)
    down_y = ifft2(1j * grid.ky * down_hat)
    rhs_up = (
        -(grid.u * up_x + grid.v * up_y)
        - 0.5 * (grid.xi0 + 1j * grid.xi1) * up
        - 0.5 * (grid.xi2 + 1j * grid.xi3) * down_conjugate
    )
    rhs_down = (
        -(grid.u * down_x + grid.v * down_y)
        - 0.5 * (grid.xi2 - 1j * grid.xi3) * up
        + (-0.5 * grid.xi0 + 0.5j * grid.xi1) * down_conjugate
    )
    return fft2(rhs_up) * grid.mask, fft2(rhs_down) * grid.mask


def _pse_step(
    up_hat: np.ndarray,
    down_hat: np.ndarray,
    coefficients: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray]:
    up, down = ifft2(up_hat), ifft2(down_hat)
    fu, fd = _pse_rhs(up, down, up_hat, down_hat, grid)

    e2u, e2d = _apply_matrix(coefficients["E2"], up_hat, down_hat)
    qfu, qfd = _apply_matrix(coefficients["Q"], fu, fd)
    au, ad = e2u + qfu, e2d + qfd
    fau, fad = _pse_rhs(ifft2(au), ifft2(ad), au, ad, grid)

    qfau, qfad = _apply_matrix(coefficients["Q"], fau, fad)
    bu, bd = e2u + qfau, e2d + qfad
    fbu, fbd = _pse_rhs(ifft2(bu), ifft2(bd), bu, bd, grid)

    e2au, e2ad = _apply_matrix(coefficients["E2"], au, ad)
    qstage_u, qstage_d = _apply_matrix(
        coefficients["Q"], 2.0 * fbu - fu, 2.0 * fbd - fd
    )
    cu, cd = e2au + qstage_u, e2ad + qstage_d
    fcu, fcd = _pse_rhs(ifft2(cu), ifft2(cd), cu, cd, grid)

    eu, ed = _apply_matrix(coefficients["E"], up_hat, down_hat)
    f1u, f1d = _apply_matrix(coefficients["f1"], fu, fd)
    f2u, f2d = _apply_matrix(coefficients["f2"], fau + fbu, fad + fbd)
    f3u, f3d = _apply_matrix(coefficients["f3"], fcu, fcd)
    return eu + f1u + 2.0 * f2u + f3u, ed + f1d + 2.0 * f2d + f3d


def _initial_pse(
    target: np.ndarray,
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Apply the paper's frozen-local, strain-only O(Ro) initialisation."""
    down = 0.25j * (grid.xi2 + 1j * grid.xi3) * np.conj(target)
    up = target - down
    up_hat = fft2(up) * grid.mask
    down_hat = fft2(np.conj(down)) * grid.mask
    scale = np.linalg.norm(target)
    reconstructed = ifft2(up_hat) + np.conj(ifft2(down_hat))
    residual = float(np.linalg.norm(reconstructed - target) / scale)
    return up_hat, down_hat, 0, residual


def _physical_fields(
    scalar_envelopes: np.ndarray,
    up_hat: np.ndarray,
    down_hat: np.ndarray,
    hbe_state: np.ndarray,
    time_dimensionless: float,
) -> np.ndarray:
    clockwise = np.exp(-1j * time_dimensionless)
    counterclockwise = np.conj(clockwise)
    pse = ifft2(up_hat) * clockwise + np.conj(ifft2(down_hat)) * counterclockwise
    hbe = hbe_state[0] + 1j * hbe_state[1]
    return np.concatenate(
        (scalar_envelopes * clockwise, pse[None, ...], hbe[None, ...]),
        axis=0,
    )


def simulate_mode(
    parameters: Parameters,
    mode: int,
    group: h5py.Group,
    saved_periods: Iterable[int],
    *,
    progress_interval_periods: int = 5,
) -> dict[str, Any]:
    """Integrate one vertical mode and write its diagnostics to an HDF5 group."""
    start = time.perf_counter()
    np.random.seed(parameters.seed)
    grid = Grid(parameters)
    c = parameters.dimensionless_phase_speed(mode)
    dt = parameters.dt
    scalar_factors = _scalar_factors(c, dt, grid)
    coefficients = _etdrk4_coefficients(c, dt, grid, parameters.contour_points)

    target = np.full((grid.n, grid.n), parameters.amplitude_m_s, dtype=complex)
    scalar = np.stack((target.copy(), target.copy(), target.copy()), axis=0)
    hbe = np.zeros((3, grid.n, grid.n), dtype=float)
    hbe[0] = parameters.amplitude_m_s
    up_hat, down_hat, iterations, initial_residual = _initial_pse(target, grid)

    saved_periods_array = np.asarray(sorted(set(saved_periods)), dtype=int)
    if np.any(saved_periods_array < 0) or np.any(
        saved_periods_array > parameters.total_periods
    ):
        raise ValueError(f"n={mode}: a requested saved time is outside the run.")
    saved_steps = saved_periods_array * parameters.steps_per_period
    field_shape = (saved_steps.size, len(MODEL_NAMES), grid.n, grid.n)
    if saved_steps.size:
        field_data = group.create_dataset(
            "complex_velocity",
            shape=field_shape,
            dtype=np.dtype(parameters.output_precision),
            chunks=(1, 1, grid.n, grid.n),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
    else:
        field_data = group.create_dataset(
            "complex_velocity",
            shape=field_shape,
            dtype=np.dtype(parameters.output_precision),
        )
    group.create_dataset("field_times_ip", data=saved_periods_array.astype(float))
    times = np.arange(parameters.step_count + 1) / parameters.steps_per_period
    nre = np.empty((len(NRE_MODEL_NAMES), times.size), dtype=float)
    field_cursor = 0

    def sample(step: int) -> None:
        nonlocal field_cursor
        fields = _physical_fields(scalar, up_hat, down_hat, hbe, step * dt)
        reference = fields[-1]
        for index in range(len(NRE_MODEL_NAMES)):
            nre[index, step] = relative_l2(reference, fields[index])
        if field_cursor < saved_steps.size and step == saved_steps[field_cursor]:
            field_data[field_cursor] = fields.astype(parameters.output_precision)
            field_cursor += 1

    sample(0)
    for step in range(1, parameters.step_count + 1):
        hbe = _rk4_hbe(hbe, dt, grid, c)
        scalar = _scalar_strang_step(scalar, scalar_factors, dt, grid)
        up_hat, down_hat = _pse_step(up_hat, down_hat, coefficients, grid)
        sample(step)
        if (
            progress_interval_periods > 0
            and step % (progress_interval_periods * parameters.steps_per_period) == 0
        ):
            print(
                f"n={mode}: {step / parameters.steps_per_period:g}/"
                f"{parameters.total_periods:g} inertial periods",
                flush=True,
            )

    if field_cursor != saved_steps.size:
        raise RuntimeError(f"n={mode}: not all requested fields were saved.")
    if not np.all(np.isfinite(nre)):
        raise FloatingPointError(f"n={mode}: non-finite NRE values were produced.")

    group.create_dataset("times_ip", data=times)
    group.create_dataset("nre", data=nre)
    group.create_dataset("coordinate_over_length", data=grid.coordinate)
    metadata = mode_metadata(mode, parameters.depth_m)
    metadata.update(
        {
            "phase_speed_m_s": parameters.phase_speed(mode),
            "dimensionless_phase_speed": c,
            "pse_initial_iterations": iterations,
            "pse_initialisation": "frozen-local strain-only O(Ro)",
            "pse_initial_reconstruction_relative_l2": initial_residual,
            "runtime_seconds": time.perf_counter() - start,
        }
    )
    for key, value in metadata.items():
        group.attrs[key] = value
    return metadata


def create_simulation_file(
    path: Path,
    config: dict[str, Any],
    modes: Iterable[int],
    save_times_by_mode: dict[int, Iterable[int]],
    *,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run all requested modes and atomically create the simulation archive."""
    parameters = parameters_from_config(config)
    tolerance = config["validation_tolerances"]
    mode_list = sorted(set(int(mode) for mode in modes))
    for mode in mode_list:
        validate_vertical_mode(
            mode,
            parameters.depth_m,
            boundary_tolerance=float(
                tolerance["vertical_mode_boundary_derivative_abs"]
            ),
            mean_tolerance=float(tolerance["vertical_mode_mean_abs"]),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    records: list[dict[str, Any]] = []
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["config_json"] = json.dumps(config, sort_keys=True)
            handle.attrs["simulation_config_json"] = json.dumps(
                simulation_configuration(config), sort_keys=True
            )
            handle.attrs["simulation_signature_sha256"] = simulation_signature(config)
            handle.attrs["model_names"] = json.dumps(MODEL_NAMES)
            handle.attrs["nre_model_names"] = json.dumps(NRE_MODEL_NAMES)
            handle.attrs["coordinate_convention"] = "x/L,y/L in [-pi,pi)"
            handle.attrs["time_convention"] = "dimensionless time ft"
            groups = handle.create_group("modes")
            if workers <= 1:
                for mode in mode_list:
                    print(f"starting vertical mode n={mode}", flush=True)
                    records.append(
                        simulate_mode(
                            parameters,
                            mode,
                            groups.create_group(f"n{mode:04d}"),
                            save_times_by_mode.get(mode, ()),
                        )
                    )
            else:
                with tempfile.TemporaryDirectory(
                    dir=path.parent, prefix=f".{path.stem}_workers_"
                ) as directory:
                    work_directory = Path(directory)
                    future_to_mode = {}
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        for mode in mode_list:
                            worker_path = work_directory / f"n{mode:04d}.h5"
                            future = executor.submit(
                                _run_mode_worker,
                                config,
                                mode,
                                list(save_times_by_mode.get(mode, ())),
                                worker_path,
                            )
                            future_to_mode[future] = (mode, worker_path)
                        completed: dict[int, tuple[dict[str, Any], Path]] = {}
                        for future in as_completed(future_to_mode):
                            mode, worker_path = future_to_mode[future]
                            completed[mode] = (future.result(), worker_path)
                            print(f"completed vertical mode n={mode}", flush=True)
                    for mode in mode_list:
                        record, worker_path = completed[mode]
                        with h5py.File(worker_path, "r") as source:
                            source.copy("mode", groups, name=f"n{mode:04d}")
                        records.append(record)
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return records


def _run_mode_worker(
    config: dict[str, Any],
    mode: int,
    saved_periods: list[int],
    output_path: Path,
) -> dict[str, Any]:
    """Run one independent mode in a worker-owned HDF5 file."""
    parameters = parameters_from_config(config)
    with h5py.File(output_path, "w") as handle:
        return simulate_mode(
            parameters,
            mode,
            handle.create_group("mode"),
            saved_periods,
        )
