"""Extract the sinusoidal-dipole fields used in Figures 9 and 10.

The script reads the MATLAB v7.3 simulation files, reconstructs the physical
PSE velocity, and stores the normalized squared velocity magnitude for the
five models and five vertical modes shown in the paper.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "sinusoidal_dipole_wave_velocity_fields.npz"
)

MODEL_NAMES = ("YBJ", "TSB", "YBJ+", "PSE", "HBEs")
VERTICAL_MODES = (
    (1, 4000.0),
    (4, 1000.0),
    (8, 500.0),
    (16, 250.0),
    (32, 125.0),
)
TARGET_TIMES = (10.0, 50.0)
BACKGROUND_SPEED = 0.25


def complex_slice(dataset: h5py.Dataset, index: int) -> np.ndarray:
    """Read one complex two-dimensional slice from a MATLAB v7.3 dataset."""
    raw = dataset[index, :, :]
    return (raw["real"] + 1j * raw["imag"]).T


def find_cases(index_path: Path) -> list[tuple[int, float, Path]]:
    """Find the unique simulation file for each displayed vertical mode."""
    with index_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    selected = []
    for vertical_mode, vertical_wavelength in VERTICAL_MODES:
        matches = [
            row
            for row in rows
            if np.isclose(
                float(row["background_velocity_mps"]),
                BACKGROUND_SPEED,
            )
            and np.isclose(
                float(row["Lv_m"]),
                vertical_wavelength,
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Expected one case for "
                f"U={BACKGROUND_SPEED:g} m s^-1 and "
                f"Lv={vertical_wavelength:g} m; found {len(matches)}."
            )
        data_path = Path(matches[0]["data_mat"])
        if not data_path.is_absolute():
            data_path = index_path.parent / data_path
        selected.append(
            (
                vertical_mode,
                vertical_wavelength,
                data_path,
            )
        )
    return selected


def load_squared_velocity(
    data_path: Path,
    target_times: tuple[float, ...],
) -> np.ndarray:
    """Load normalized squared velocity magnitude at selected times."""
    with h5py.File(data_path, "r") as file:
        available_times = np.asarray(file["hbe_time_periods"]).squeeze()
        fields = []
        for target_time in target_times:
            time_index = int(np.argmin(np.abs(available_times - target_time)))
            if not np.isclose(
                available_times[time_index],
                target_time,
                atol=1.0e-10,
            ):
                raise ValueError(f"{target_time:g} IP is unavailable in {data_path}.")

            component_up = complex_slice(
                file["pse_A_up"],
                time_index,
            )
            stored_conjugate_down = complex_slice(
                file["pse_conj_A_dn"],
                time_index,
            )
            pse_velocity = component_up + np.conj(stored_conjugate_down)
            model_velocities = (
                complex_slice(file["ybj_LA"], time_index),
                complex_slice(file["tsb_LA"], time_index),
                complex_slice(
                    file["ybjplus_LplusA"],
                    time_index,
                ),
                pse_velocity,
                complex_slice(file["hbe_uiv"], time_index),
            )
            fields.append(
                np.stack(
                    [np.abs(velocity) ** 2 for velocity in model_velocities],
                    axis=0,
                )
            )
    return np.stack(fields, axis=0)


def compute_fields(
    index_path: Path,
    target_times: tuple[float, ...],
) -> dict[str, np.ndarray]:
    """Extract all model fields for the requested modes and times."""
    cases = find_cases(index_path)
    fields = []
    source_files = []
    for vertical_mode, _, data_path in cases:
        print(f"n={vertical_mode}: {data_path}")
        fields.append(load_squared_velocity(data_path, target_times))
        source_files.append(str(data_path))

    # Stored order: time, vertical mode, model, y, x.
    field_array = np.stack(fields, axis=1)
    return {
        "times_in_inertial_periods": np.asarray(target_times, dtype=float),
        "vertical_modes": np.asarray(
            [mode for mode, _ in VERTICAL_MODES],
            dtype=int,
        ),
        "vertical_wavelengths_m": np.asarray(
            [wavelength for _, wavelength in VERTICAL_MODES],
            dtype=float,
        ),
        "model_names": np.asarray(MODEL_NAMES),
        "squared_velocity": field_array,
        "source_files": np.asarray(source_files),
    }


def parse_float_list(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of target inertial periods."""
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one time is required.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract sinusoidal-dipole wave-velocity fields."
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="CSV index of the simulation files.",
    )
    parser.add_argument(
        "--times",
        type=parse_float_list,
        default=TARGET_TIMES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Extract the fields and write the compressed NumPy file."""
    args = parse_args()
    results = compute_fields(args.index, args.times)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **results)
    print(args.output)


if __name__ == "__main__":
    main()
