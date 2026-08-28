"""Numerical and structural validation for the self-contained reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from specification import MODEL_NAMES, reference_metrics_from_config, validate_config
from vertical_modes import validate_vertical_mode, vertical_wavelength


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_smoke(
    simulation_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate a deliberately small end-to-end numerical run."""
    depth = float(config["physical_parameters"]["domain_depth_m"])
    mode_checks = {}
    with h5py.File(simulation_path, "r") as handle:
        for name in sorted(handle["modes"]):
            group = handle["modes"][name]
            mode = int(group.attrs["vertical_mode"])
            mode_checks[str(mode)] = validate_vertical_mode(mode, depth)
            require(
                np.isclose(
                    group.attrs["vertical_wavelength_m"],
                    vertical_wavelength(mode, depth),
                ),
                f"n={mode}: vertical wavelength metadata is inconsistent.",
            )
            require(
                group.attrs["pse_initial_reconstruction_relative_l2"] < 1.0e-10,
                f"n={mode}: PSE initial reconstruction failed.",
            )
            nre = np.asarray(group["nre"])
            fields = np.asarray(group["complex_velocity"])
            require(np.all(np.isfinite(nre)), f"n={mode}: non-finite NRE values.")
            require(np.all(np.isfinite(fields)), f"n={mode}: non-finite fields.")
            require(np.all(nre >= 0.0), f"n={mode}: negative NRE values.")
    return {
        "status": "passed",
        "kind": "smoke-test",
        "vertical_mode_checks": mode_checks,
    }


def validate_all(
    simulation_path: Path,
    error_path: Path,
    wave_path: Path,
    movie_fields_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply the manuscript-resolution regression and consistency checks."""
    physical = config["physical_parameters"]
    tolerance = config["validation_tolerances"]
    validate_config(config, manuscript_resolution=True)
    references = reference_metrics_from_config(config)

    vertical_checks: dict[str, Any] = {}
    with h5py.File(simulation_path, "r") as handle:
        groups = handle["modes"]
        expected_modes = config["vertical_modes"]["error_statistics"]
        actual_modes = [
            int(groups[name].attrs["vertical_mode"]) for name in sorted(groups)
        ]
        require(
            actual_modes == expected_modes, "The simulated vertical-mode set changed."
        )
        for mode in actual_modes:
            group = groups[f"n{mode:04d}"]
            vertical_checks[str(mode)] = validate_vertical_mode(
                mode,
                float(physical["domain_depth_m"]),
                boundary_tolerance=float(
                    tolerance["vertical_mode_boundary_derivative_abs"]
                ),
                mean_tolerance=float(tolerance["vertical_mode_mean_abs"]),
            )
            require(
                group.attrs["normalisation"] == "unnormalised",
                f"n={mode}: an undocumented mode normalization was used.",
            )
            require(
                group.attrs["physical_reconstruction_factor"] == 1.0,
                f"n={mode}: physical reconstruction factor is not one.",
            )
            require(
                group.attrs["pse_initial_reconstruction_relative_l2"]
                <= tolerance["initial_reconstruction_relative_l2"],
                f"n={mode}: PSE initial field reconstruction failed.",
            )
            require(
                group.attrs["pse_initialisation"]
                == "frozen-local strain-only O(Ro)",
                f"n={mode}: the PSE initialisation does not match the appendix.",
            )
            require(
                group.attrs["pse_initial_iterations"] == 0,
                f"n={mode}: the explicit PSE initialisation was iterated.",
            )

    statistics = pd.read_csv(error_path)
    require(statistics.shape[0] == 136, "Figure 8 table must contain 136 rows.")

    def selected(model: str, window: str, minimum_mode: int) -> np.ndarray:
        mask = (
            (statistics["model"] == model)
            & (statistics["window"] == window)
            & (statistics["vertical_mode"] >= minimum_mode)
        )
        return statistics.loc[mask, "mean_error_percent"].to_numpy()

    pse_10 = selected("PSE", "0-10IP", 8)
    scalar_10 = np.concatenate(
        [selected(model, "0-10IP", 8) for model in ("YBJ", "TSB", "YBJ+")]
    )
    pse_50 = selected("PSE", "0-50IP", 12)
    scalar_50 = np.concatenate(
        [selected(model, "0-50IP", 12) for model in ("YBJ", "TSB", "YBJ+")]
    )
    require(
        np.max(pse_10) <= tolerance["pse_mean_nre_0_10ip_n_ge_8_percent_max"],
        f"PSE 0--10 IP mean NRE exceeds tolerance: {np.max(pse_10):.4g}%.",
    )
    low, high = tolerance["scalar_mean_nre_0_10ip_n_ge_8_percent_range"]
    require(
        low <= np.mean(scalar_10) <= high,
        f"Scalar-model 0--10 IP mean NRE is outside [{low}, {high}]%.",
    )
    low, high = tolerance["pse_mean_nre_0_50ip_n_ge_12_percent_range"]
    require(
        low <= np.mean(pse_50) <= high,
        f"PSE n>=12 ensemble-mean 0--50 IP NRE is outside [{low}, {high}]%.",
    )
    require(
        np.max(pse_50)
        <= tolerance["pse_individual_nre_0_50ip_n_ge_12_percent_max"],
        f"A PSE n>=12 0--50 IP mean NRE exceeds the individual-mode bound: "
        f"{np.max(pse_50):.4g}%.",
    )
    low, high = tolerance["scalar_mean_nre_0_50ip_n_ge_12_percent_range"]
    require(
        low <= np.mean(scalar_50) <= high,
        f"Scalar-model 0--50 IP mean NRE is outside [{low}, {high}]%.",
    )

    with np.load(wave_path) as wave:
        times = wave["times_in_inertial_periods"]
        modes = wave["vertical_modes"]
        fields = wave["squared_velocity"]
        time_50 = int(np.flatnonzero(times == 50.0)[0])
        mode_4 = int(np.flatnonzero(modes == 4)[0])
        maxima = fields[time_50, mode_4].max(axis=(-2, -1))
        reference = references["n4_50ip_squared_velocity_maxima"]
        expected = np.asarray([reference[name] for name in MODEL_NAMES])
        maximum_error = float(np.max(np.abs(maxima - expected)))
        require(
            maximum_error <= tolerance["n4_50ip_squared_velocity_maxima_abs"],
            f"n=4, 50-IP field maxima differ by {maximum_error:.4g}.",
        )
        time_10 = int(np.flatnonzero(times == 10.0)[0])
        mode_32 = int(np.flatnonzero(modes == 32)[0])
        pointwise = float(
            np.max(np.abs(fields[time_10, mode_32, 3] - fields[time_10, mode_32, 4]))
        )
        require(
            pointwise <= tolerance["n32_10ip_pse_hbe_squared_velocity_pointwise_max"],
            f"n=32, 10-IP PSE--HBE pointwise difference is {pointwise:.4g}.",
        )

    with np.load(movie_fields_path) as movie:
        nre_consistency = float(np.max(movie["recomputed_nre_max_abs_difference"]))
        require(
            nre_consistency <= tolerance["saved_field_nre_consistency_abs"],
            f"Saved-field NRE consistency error is {nre_consistency:.4g}.",
        )

    return {
        "status": "passed",
        "kind": "full-reproduction",
        "vertical_mode_checks": vertical_checks,
        "metrics": {
            "pse_0_10ip_n_ge_8_max_percent": float(np.max(pse_10)),
            "scalar_0_10ip_n_ge_8_mean_percent": float(np.mean(scalar_10)),
            "pse_0_50ip_n_ge_12_ensemble_mean_percent": float(np.mean(pse_50)),
            "pse_0_50ip_n_ge_12_individual_max_percent": float(np.max(pse_50)),
            "scalar_0_50ip_n_ge_12_mean_percent": float(np.mean(scalar_50)),
            "n4_50ip_squared_velocity_maxima": dict(
                zip(MODEL_NAMES, maxima.tolist(), strict=True)
            ),
            "n4_50ip_maxima_max_abs_error": maximum_error,
            "n32_10ip_pse_hbe_squared_velocity_pointwise_max": pointwise,
            "saved_field_nre_consistency_max_abs": nre_consistency,
        },
    }
