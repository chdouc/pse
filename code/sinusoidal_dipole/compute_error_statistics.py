"""Compute the sinusoidal-dipole error statistics used in Figure 8.

The script reads the model-error time series from MATLAB v7.3 files and writes
the time-averaged normalized root-mean-square errors to a CSV file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "sinusoidal_dipole_error_statistics.csv"
)

MODEL_DATASETS = {
    "PSE": "err_complex_phi_spin_step",
    "TSB": "err_complex_phi_tsb_step",
    "YBJ": "err_complex_phi_ybj_step",
    "YBJ+": "err_complex_phi_ybjplus_step",
}
AVERAGING_WINDOWS = (
    ("0-10IP", 0.0, 10.0),
    ("0-50IP", 0.0, 50.0),
)
VERTICAL_MODES = tuple(range(1, 13)) + tuple(range(16, 33, 4))
DOMAIN_DEPTH = 4000.0


def mean_over_window(
    error: np.ndarray,
    time_in_periods: np.ndarray,
    start: float,
    stop: float,
) -> float:
    """Return the trapezoidal time mean on a complete inertial-period window."""
    mask = (time_in_periods >= start) & (time_in_periods <= stop)
    selected_time = time_in_periods[mask]
    selected_error = error[mask]
    if (
        selected_time.size < 2
        or not np.isclose(selected_time[0], start)
        or not np.isclose(selected_time[-1], stop)
    ):
        raise ValueError(f"Incomplete {start:g}-{stop:g} IP window.")
    return float(np.trapezoid(selected_error, selected_time) / (stop - start))


def compute_statistics(index_path: Path) -> pd.DataFrame:
    """Compute all model and averaging-window statistics."""
    index = pd.read_csv(index_path)
    rows: list[dict[str, object]] = []

    for case in index.itertuples(index=False):
        vertical_mode = int(round(DOMAIN_DEPTH / case.Lv_m))
        if vertical_mode not in VERTICAL_MODES:
            continue

        data_path = Path(case.data_mat)
        if not data_path.is_absolute():
            data_path = index_path.parent / data_path
        with h5py.File(data_path, "r") as file:
            time_in_periods = np.asarray(file["error_time_periods"])[0]
            errors = {
                model: np.asarray(file[dataset])[0]
                for model, dataset in MODEL_DATASETS.items()
            }

        for window_name, start, stop in AVERAGING_WINDOWS:
            for model, error in errors.items():
                mean_error = mean_over_window(
                    error,
                    time_in_periods,
                    start,
                    stop,
                )
                rows.append(
                    {
                        "Ro": case.Ro,
                        "background_velocity_m_s": (case.background_velocity_mps),
                        "vertical_wavelength_m": case.Lv_m,
                        "vertical_mode": vertical_mode,
                        "model": model,
                        "window": window_name,
                        "mean_error": mean_error,
                        "mean_error_percent": 100.0 * mean_error,
                        "source_file": str(data_path),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["window", "vertical_mode", "model"],
        kind="stable",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute sinusoidal-dipole model-error statistics."
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="CSV index of the simulation files.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Compute the statistics and write the CSV file."""
    args = parse_args()
    statistics = compute_statistics(args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
