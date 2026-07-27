"""Prepare the fields for supplementary movie 2.

The calculation stage reads MATLAB v7.3 files, reconstructs the physical PSE
velocity at true saved integer-inertial-period times, verifies those fields
against the processed Figure 9--10 data, and writes a self-describing NPZ
archive.  It performs no plotting or video encoding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "sinusoidal_dipole_movie_fields.npz"
)

MODEL_NAMES = ("YBJ", "TSB", "YBJ+", "PSE", "HBEs")
NRE_MODEL_NAMES = MODEL_NAMES[:4]
VERTICAL_MODES = ((4, 1000.0), (16, 250.0), (32, 125.0))
BACKGROUND_SPEED = 0.25
DOMAIN_DEPTH_M = 4000.0
RAW_FILENAME = "four_model_raw_timeseries.mat"
PROCESSED_MODEL_DATASETS = (
    "ybj_uiv_full",
    "tsb_uiv_full",
    "ybjplus_uiv_full",
    "spin_uiv_full",
    "OUT_phi/phi",
)
NRE_DATASETS = (
    "err_complex_phi_ybj_step",
    "err_complex_phi_tsb_step",
    "err_complex_phi_ybjplus_step",
    "err_complex_phi_spin_step",
)

# The n=4 and n=32 ranges reproduce the published Figure 10 scales.  The
# intermediate n=16 range is centred on the unperturbed value one and covers
# all but the most intense upper tail. Clipping is counted over every model
# and time.
FIXED_ABSOLUTE_LIMITS = {
    4: (0.01, 10.0),
    16: (0.50, 1.50),
    32: (0.88, 1.12),
}


def complex_slice(dataset: h5py.Dataset, index: int) -> np.ndarray:
    """Read one MATLAB complex field with the Figure 9--10 orientation."""
    raw = dataset[index, :, :]
    return (raw["real"] + 1j * raw["imag"]).T


def scalar(dataset: h5py.Dataset) -> float:
    """Return a scalar value from a one-element MATLAB dataset."""
    values = np.asarray(dataset).squeeze()
    if values.ndim != 0:
        raise ValueError(f"Expected scalar dataset {dataset.name}; got {values.shape}.")
    return float(values)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a source file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def outward_significant(value: float, digits: int = 3) -> float:
    """Round a positive value upward to a fixed number of significant digits."""
    if value <= 0.0:
        return 0.0
    scale = 10.0 ** (digits - 1 - np.floor(np.log10(value)))
    return float(np.ceil(value * scale) / scale)


def resolve_index(index_path: Path | None, data_root: Path | None) -> Path:
    """Resolve either an explicit CSV index or the unique index in a data root."""
    if (index_path is None) == (data_root is None):
        raise ValueError("Provide exactly one of --index or --data-root.")
    if index_path is not None:
        resolved = index_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Simulation index does not exist: {resolved}")
        return resolved

    if data_root is None:
        raise AssertionError("data_root unexpectedly missing")
    root = data_root.resolve()
    candidates = sorted(root.glob("*_index.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one *_index.csv in {root}; found {len(candidates)}."
        )
    return candidates[0]


def find_processed_cases(index_path: Path) -> list[dict[str, Any]]:
    """Find the unique processed Figure 9--10 case for each movie mode."""
    with index_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    selected: list[dict[str, Any]] = []
    for mode, wavelength in VERTICAL_MODES:
        matches = [
            row
            for row in rows
            if np.isclose(
                float(row["background_velocity_mps"]),
                BACKGROUND_SPEED,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.isclose(
                float(row["Lv_m"]),
                wavelength,
                rtol=0.0,
                atol=1.0e-9,
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one n={mode} case (Lv={wavelength:g} m); "
                f"found {len(matches)}."
            )
        data_path = Path(matches[0]["data_mat"])
        if not data_path.is_absolute():
            data_path = index_path.parent / data_path
        data_path = data_path.resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"Processed data file is missing: {data_path}")
        selected.append(
            {
                "mode": mode,
                "wavelength_m": wavelength,
                "row": matches[0],
                "processed_path": data_path,
            }
        )
    return selected


def raw_case_matches(path: Path, wavelength_m: float) -> bool:
    """Return whether a raw file has the controlled movie parameters."""
    try:
        with h5py.File(path, "r") as handle:
            speed = scalar(handle["background_velocity_mps"])
            kz = scalar(handle["P_phi/kz"])
            wavelength = 2.0 * np.pi / kz
            return bool(
                np.isclose(speed, BACKGROUND_SPEED, rtol=0.0, atol=1.0e-12)
                and np.isclose(
                    wavelength,
                    wavelength_m,
                    rtol=0.0,
                    atol=1.0e-8,
                )
            )
    except (KeyError, OSError, ValueError):
        return False


def find_raw_case(
    raw_data_root: Path,
    processed_path: Path,
    wavelength_m: float,
) -> Path:
    """Find the parameter-matched raw file that retains PSE components."""
    processed_name = processed_path.parent.name
    base_name = (
        processed_name[:-3] if processed_name.endswith("_01") else processed_name
    )
    direct = raw_data_root / base_name / RAW_FILENAME
    if direct.is_file() and raw_case_matches(direct, wavelength_m):
        return direct.resolve()

    candidates = [
        path
        for path in raw_data_root.glob(f"*/{RAW_FILENAME}")
        if raw_case_matches(path, wavelength_m)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one raw case for Lv={wavelength_m:g} m in "
            f"{raw_data_root}; found {len(candidates)}."
        )
    return candidates[0].resolve()


def require_time_alignment(handle: h5py.File) -> np.ndarray:
    """Validate that every raw model uses the same true saved times."""
    names = (
        "hbe_time_periods",
        "pse_time_periods",
        "ybjplus_time_periods",
        "ybj_time_periods",
        "tsb_time_periods",
    )
    arrays = [np.asarray(handle[name]).squeeze().astype(float) for name in names]
    reference = arrays[0]
    if reference.ndim != 1 or reference.size < 2:
        raise ValueError("Raw movie times must be a non-trivial one-dimensional array.")
    if not np.all(np.diff(reference) > 0.0):
        raise ValueError("Raw movie times are not strictly increasing.")
    for name, values in zip(names[1:], arrays[1:], strict=True):
        if values.shape != reference.shape or not np.allclose(
            values,
            reference,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"{name} is not aligned with hbe_time_periods.")
    return reference


def select_true_times(
    available: np.ndarray,
    *,
    start: float,
    stop: float,
    interval: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exact integer-period states without interpolating physical fields."""
    if interval <= 0.0:
        raise ValueError("--sample-interval-ip must be positive.")
    expected = np.arange(start, stop + 0.5 * interval, interval, dtype=float)
    if expected.size == 0 or not np.isclose(expected[-1], stop, atol=1.0e-12):
        raise ValueError("The requested time interval must land exactly on --stop-ip.")
    if not np.allclose(expected, np.rint(expected), rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "Movie field times must be integer inertial periods so the requested "
            "PSE reconstruction A_up + conj(stored_conj_A_dn) is exact."
        )

    indices = np.empty(expected.size, dtype=int)
    for position, target in enumerate(expected):
        index = int(np.argmin(np.abs(available - target)))
        if not np.isclose(available[index], target, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"True saved time {target:g} IP is unavailable.")
        indices[position] = index
    if not np.all(np.diff(indices) > 0):
        raise ValueError("Selected source-time indices are not strictly increasing.")
    return expected, indices


def normalized_root_mean_square_error(
    reference: np.ndarray,
    model: np.ndarray,
) -> float:
    """Return the manuscript's complex-velocity relative L2 error."""
    denominator = np.sum(np.abs(reference) ** 2)
    if denominator <= 0.0:
        raise ValueError("HBEs reference energy is not positive.")
    return float(np.sqrt(np.sum(np.abs(model - reference) ** 2) / denominator))


def processed_time_index(handle: h5py.File, target: float) -> int:
    """Return the exact processed-field index for one inertial-period time."""
    available = np.asarray(handle["phi_saved_periods"]).squeeze().astype(float)
    index = int(np.argmin(np.abs(available - target)))
    if not np.isclose(available[index], target, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Processed field time {target:g} IP is unavailable.")
    return index


def clipping_record(values: np.ndarray, lower: float, upper: float) -> dict[str, Any]:
    """Return explicit clipping counts for a fixed color scale."""
    below = int(np.count_nonzero(values < lower))
    above = int(np.count_nonzero(values > upper))
    total = int(values.size)
    return {
        "lower": float(lower),
        "upper": float(upper),
        "actual_minimum": float(np.min(values)),
        "actual_maximum": float(np.max(values)),
        "below_count": below,
        "above_count": above,
        "total_count": total,
        "clipped_fraction": float((below + above) / total),
    }


def compute_archive(
    index_path: Path,
    raw_data_root: Path,
    *,
    start_ip: float,
    stop_ip: float,
    sample_interval_ip: float,
    difference_quantile: float,
) -> dict[str, np.ndarray]:
    """Extract, reconstruct, compare, and package all movie data."""
    if not 0.5 < difference_quantile < 1.0:
        raise ValueError("--difference-quantile must lie between 0.5 and 1.")

    cases = find_processed_cases(index_path)
    mode_fields: list[np.ndarray] = []
    mode_nre: list[np.ndarray] = []
    mode_nre_times: list[np.ndarray] = []
    selected_source_indices: list[np.ndarray] = []
    normalization_amplitudes: list[float] = []
    raw_sources: list[str] = []
    processed_sources: list[str] = []
    raw_hashes: list[str] = []
    processed_hashes: list[str] = []
    field_reference_differences: list[np.ndarray] = []
    nre_reference_differences: list[np.ndarray] = []
    mode_metadata: list[dict[str, Any]] = []
    selected_times: np.ndarray | None = None

    for case in cases:
        mode = int(case["mode"])
        wavelength_m = float(case["wavelength_m"])
        processed_path = Path(case["processed_path"])
        raw_path = find_raw_case(raw_data_root, processed_path, wavelength_m)
        print(f"n={mode}: raw={raw_path}")
        print(f"n={mode}: processed reference={processed_path}")

        with (
            h5py.File(raw_path, "r") as raw,
            h5py.File(processed_path, "r") as processed,
        ):
            available_times = require_time_alignment(raw)
            times, source_indices = select_true_times(
                available_times,
                start=start_ip,
                stop=stop_ip,
                interval=sample_interval_ip,
            )
            if selected_times is None:
                selected_times = times
            elif not np.array_equal(times, selected_times):
                raise ValueError("Movie modes selected different field times.")

            amplitude = scalar(raw["P_phi/initial_condition/amplitude"])
            if amplitude <= 0.0:
                raise ValueError("Initial modal amplitude must be positive.")

            processed_error_times = (
                np.asarray(processed["error_time_periods"]).squeeze().astype(float)
            )
            if processed_error_times.shape != available_times.shape or not np.allclose(
                processed_error_times,
                available_times,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"Raw and processed NRE times differ for vertical mode n={mode}."
                )
            nre_curves = np.stack(
                [
                    np.asarray(processed[name]).squeeze().astype(float)
                    for name in NRE_DATASETS
                ],
                axis=0,
            )
            if nre_curves.shape != (len(NRE_MODEL_NAMES), available_times.size):
                raise ValueError(f"Unexpected NRE array shape for n={mode}.")
            if not np.all(np.isfinite(nre_curves)) or np.any(nre_curves < 0.0):
                raise ValueError(f"Invalid NRE values for n={mode}.")

            fields_at_times: list[np.ndarray] = []
            max_field_difference = np.zeros(len(MODEL_NAMES), dtype=float)
            max_nre_difference = np.zeros(len(NRE_MODEL_NAMES), dtype=float)

            for target_time, raw_index in zip(
                times,
                source_indices,
                strict=True,
            ):
                component_up = complex_slice(raw["pse_A_up"], int(raw_index))
                stored_conjugate_down = complex_slice(
                    raw["pse_conj_A_dn"],
                    int(raw_index),
                )
                # Selected times are integer inertial periods, so the carrier
                # phase factors are exactly unity.
                pse_velocity = component_up + np.conj(stored_conjugate_down)
                raw_velocities = (
                    complex_slice(raw["ybj_LA"], int(raw_index)),
                    complex_slice(raw["tsb_LA"], int(raw_index)),
                    complex_slice(raw["ybjplus_LplusA"], int(raw_index)),
                    pse_velocity,
                    complex_slice(raw["hbe_uiv"], int(raw_index)),
                )

                processed_index = processed_time_index(processed, float(target_time))
                processed_velocities = tuple(
                    complex_slice(processed[name], processed_index)
                    for name in PROCESSED_MODEL_DATASETS
                )
                for model_index, (raw_value, reference_value) in enumerate(
                    zip(raw_velocities, processed_velocities, strict=True)
                ):
                    max_field_difference[model_index] = max(
                        max_field_difference[model_index],
                        float(np.max(np.abs(raw_value - reference_value))),
                    )

                hbes = raw_velocities[-1]
                for model_index, model_velocity in enumerate(raw_velocities[:-1]):
                    recomputed_nre = normalized_root_mean_square_error(
                        hbes,
                        model_velocity,
                    )
                    stored_nre = float(nre_curves[model_index, raw_index])
                    max_nre_difference[model_index] = max(
                        max_nre_difference[model_index],
                        abs(recomputed_nre - stored_nre),
                    )

                fields_at_times.append(
                    np.stack(
                        [
                            np.abs(velocity) ** 2 / amplitude**2
                            for velocity in raw_velocities
                        ],
                        axis=0,
                    )
                )

            fields = np.stack(fields_at_times, axis=0)
            if not np.all(np.isfinite(fields)) or np.any(fields < 0.0):
                raise ValueError(f"Invalid normalized field values for n={mode}.")
            if np.max(max_field_difference) > 1.0e-10:
                raise ValueError(
                    f"Raw/processed field mismatch for n={mode}: "
                    f"{np.max(max_field_difference):.3e}."
                )
            if np.max(max_nre_difference) > 1.0e-12:
                raise ValueError(
                    f"Raw/processed NRE mismatch for n={mode}: "
                    f"{np.max(max_nre_difference):.3e}."
                )

            mode_fields.append(fields)
            mode_nre.append(nre_curves)
            mode_nre_times.append(processed_error_times)
            selected_source_indices.append(source_indices)
            normalization_amplitudes.append(amplitude)
            field_reference_differences.append(max_field_difference)
            nre_reference_differences.append(max_nre_difference)
            raw_sources.append(str(raw_path))
            processed_sources.append(str(processed_path))
            raw_hashes.append(sha256_file(raw_path))
            processed_hashes.append(sha256_file(processed_path))
            mode_metadata.append(
                {
                    "vertical_mode": mode,
                    "vertical_wavelength_m": wavelength_m,
                    "raw_time_count": int(available_times.size),
                    "raw_time_step_ip": float(
                        np.median(np.diff(available_times))
                    ),
                    "selected_time_count": int(times.size),
                    "normalization_amplitude": amplitude,
                }
            )

    if selected_times is None:
        raise RuntimeError("No movie fields were extracted.")

    field_array = np.stack(mode_fields, axis=0)
    nre_array = np.stack(mode_nre, axis=0)
    nre_time_array = np.stack(mode_nre_times, axis=0)
    source_index_array = np.stack(selected_source_indices, axis=0)
    differences = field_array[:, :, :4] - field_array[:, :, 4:5]

    absolute_limits = np.asarray(
        [FIXED_ABSOLUTE_LIMITS[int(mode)] for mode, _ in VERTICAL_MODES],
        dtype=float,
    )
    difference_limits = []
    clipping: dict[str, Any] = {}
    for mode_index, (mode, _) in enumerate(VERTICAL_MODES):
        absolute_lower, absolute_upper = absolute_limits[mode_index]
        difference_limit = outward_significant(
            float(
                np.quantile(
                    np.abs(differences[mode_index]),
                    difference_quantile,
                )
            )
        )
        if difference_limit <= 0.0:
            raise ValueError(f"Difference color limit is zero for n={mode}.")
        difference_limits.append((-difference_limit, difference_limit))
        clipping[str(mode)] = {
            "absolute_field": clipping_record(
                field_array[mode_index],
                float(absolute_lower),
                float(absolute_upper),
            ),
            "difference_field": clipping_record(
                differences[mode_index],
                -difference_limit,
                difference_limit,
            ),
        }
    difference_limit_array = np.asarray(difference_limits, dtype=float)

    grid_size = field_array.shape[-1]
    coordinate = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    metadata = {
        "schema_version": 1,
        "product": "supplementary movie 2",
        "background_flow": "sinusoidal dipole",
        "background_velocity_m_s": BACKGROUND_SPEED,
        "domain_depth_m": DOMAIN_DEPTH_M,
        "model_order": list(MODEL_NAMES),
        "nre_model_order": list(NRE_MODEL_NAMES),
        "vertical_modes": [mode for mode, _ in VERTICAL_MODES],
        "time_units": "inertial periods",
        "field_quantity_tex": r"|\\phi|^2/|\\phi_{\\mathrm{amp}}|^2",
        "difference_definition": (
            "normalized squared-velocity field of the named model minus HBEs"
        ),
        "nre_definition_tex": (
            r"\\left[\\int|\\phi_M-\\phi_{\\mathrm{HBEs}}|^2/"
            r"\\int|\\phi_{\\mathrm{HBEs}}|^2\\right]^{1/2}"
        ),
        "pse_reconstruction": (
            "pse_velocity = A_up + conj(stored_conj_A_dn)"
        ),
        "pse_reconstruction_note": (
            "The selected field times are integer inertial periods, at which "
            "the carrier phase factors are unity."
        ),
        "orientation": {
            "hdf5_slice_transform": "transpose",
            "origin": "lower",
            "extent": [-np.pi, np.pi, -np.pi, np.pi],
        },
        "sampling": {
            "start_ip": float(start_ip),
            "stop_ip": float(stop_ip),
            "sample_interval_ip": float(sample_interval_ip),
            "physical_field_interpolation": False,
            "description": (
                "Exact saved integer-period states were selected in strictly "
                "increasing source-index order."
            ),
        },
        "color_limits": {
            "absolute_strategy": (
                "Fixed mode-specific limits: n=4 and n=32 reproduce Figure 10; "
                "n=16 uses [0.50, 1.50], centred on the unperturbed value one "
                "and covering all but the most intense upper tail."
            ),
            "difference_strategy": (
                f"Symmetric full-movie {difference_quantile:g} quantile of "
                "absolute model-minus-HBEs differences, rounded outward."
            ),
            "difference_quantile": float(difference_quantile),
            "clipping": clipping,
        },
        "validation": {
            "raw_processed_field_tolerance": 1.0e-10,
            "raw_processed_nre_tolerance": 1.0e-12,
            "all_selected_times_compared": True,
            "includes_10_ip": bool(np.any(np.isclose(selected_times, 10.0))),
            "includes_50_ip": bool(np.any(np.isclose(selected_times, 50.0))),
        },
        "modes": mode_metadata,
        "source_sha256": {
            "raw": raw_hashes,
            "processed_reference": processed_hashes,
        },
    }

    return {
        "times_in_inertial_periods": selected_times.astype(float),
        "source_time_indices": source_index_array,
        "vertical_modes": np.asarray(
            [mode for mode, _ in VERTICAL_MODES],
            dtype=int,
        ),
        "vertical_wavelengths_m": np.asarray(
            [wavelength for _, wavelength in VERTICAL_MODES],
            dtype=float,
        ),
        "model_names": np.asarray(MODEL_NAMES),
        "nre_model_names": np.asarray(NRE_MODEL_NAMES),
        "normalized_squared_velocity": field_array,
        "nre_times_in_inertial_periods": nre_time_array,
        "nre_complex_relative_l2": nre_array,
        "x_over_L": coordinate,
        "y_over_L": coordinate,
        "normalization_amplitude": np.asarray(
            normalization_amplitudes,
            dtype=float,
        ),
        "absolute_color_limits": absolute_limits,
        "difference_color_limits": difference_limit_array,
        "reference_max_abs_complex_difference": np.stack(
            field_reference_differences,
            axis=0,
        ),
        "reference_max_abs_nre_difference": np.stack(
            nre_reference_differences,
            axis=0,
        ),
        "raw_source_files": np.asarray(raw_sources),
        "processed_reference_files": np.asarray(processed_sources),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare sinusoidal-dipole fields for supplementary movie 2."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--index",
        type=Path,
        help="CSV index of the processed Figure 9--10 simulation files.",
    )
    source.add_argument(
        "--data-root",
        type=Path,
        help="Directory containing exactly one *_index.csv file.",
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        required=True,
        help="Directory containing raw case folders with PSE components.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-ip", type=float, default=0.0)
    parser.add_argument("--stop-ip", type=float, default=50.0)
    parser.add_argument("--sample-interval-ip", type=float, default=1.0)
    parser.add_argument("--difference-quantile", type=float, default=0.995)
    return parser.parse_args()


def main() -> None:
    """Compute and atomically save the intermediate movie archive."""
    args = parse_args()
    index_path = resolve_index(args.index, args.data_root)
    raw_data_root = args.raw_data_root.resolve()
    if not raw_data_root.is_dir():
        raise FileNotFoundError(f"Raw data root does not exist: {raw_data_root}")

    archive = compute_archive(
        index_path,
        raw_data_root,
        start_ip=args.start_ip,
        stop_ip=args.stop_ip,
        sample_interval_ip=args.sample_interval_ip,
        difference_quantile=args.difference_quantile,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    np.savez_compressed(temporary, **archive)
    os.replace(temporary, output)
    print(f"wrote {output}")
    print(
        "shape="
        f"{archive['normalized_squared_velocity'].shape}; "
        f"times={archive['times_in_inertial_periods'][0]:g}--"
        f"{archive['times_in_inertial_periods'][-1]:g} IP"
    )


if __name__ == "__main__":
    main()
