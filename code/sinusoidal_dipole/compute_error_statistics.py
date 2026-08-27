"""Compute the Figure 8 error statistics from the repository solver output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "sinusoidal_dipole_error_statistics.csv"
)
AVERAGING_WINDOWS = (("0-10IP", 0.0, 10.0), ("0-50IP", 0.0, 50.0))


def mean_over_window(
    error: np.ndarray,
    time_in_periods: np.ndarray,
    start: float,
    stop: float,
) -> float:
    """Return the trapezoidal time mean on one complete time window."""
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


def compute_statistics(simulation_path: Path) -> pd.DataFrame:
    """Compute every model/window statistic directly from a solver archive."""
    rows: list[dict[str, object]] = []
    with h5py.File(simulation_path, "r") as handle:
        config = json.loads(handle.attrs["config_json"])
        model_names = json.loads(handle.attrs["nre_model_names"])
        physical = config["physical_parameters"]
        rossby = physical["background_velocity_m_s"] / (
            physical["coriolis_frequency_s-1"] * physical["background_length_scale_m"]
        )
        for name in sorted(handle["modes"]):
            group = handle["modes"][name]
            mode = int(group.attrs["vertical_mode"])
            times = np.asarray(group["times_ip"])
            errors = np.asarray(group["nre"])
            for window_name, start, stop in AVERAGING_WINDOWS:
                for model_index, model in enumerate(model_names):
                    mean_error = mean_over_window(
                        errors[model_index], times, start, stop
                    )
                    rows.append(
                        {
                            "Ro": rossby,
                            "background_velocity_m_s": physical[
                                "background_velocity_m_s"
                            ],
                            "vertical_wavelength_m": group.attrs[
                                "vertical_wavelength_m"
                            ],
                            "vertical_mode": mode,
                            "model": model,
                            "window": window_name,
                            "mean_error": mean_error,
                            "mean_error_percent": 100.0 * mean_error,
                            "source_file": simulation_path.name,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["window", "vertical_mode", "model"], kind="stable"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = compute_statistics(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
