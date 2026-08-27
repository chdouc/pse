"""Small deterministic integration tests for the five-model solver."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from solver import create_simulation_file


def smoke_config() -> dict[str, object]:
    return {
        "random_seed": 20260826,
        "physical_parameters": {
            "coriolis_frequency_s-1": 1.0e-4,
            "buoyancy_frequency_ratio": 20.0,
            "domain_depth_m": 2000.0,
            "background_length_scale_m": 50000.0,
            "background_velocity_m_s": 0.25,
            "initial_velocity_amplitude_m_s": 1.0,
        },
        "numerical_parameters": {
            "horizontal_grid": 16,
            "time_steps_per_inertial_period": 8,
            "total_inertial_periods": 1,
            "pse_etdrk4_contour_points": 16,
        },
        "validation_tolerances": {
            "vertical_mode_boundary_derivative_abs": 1.0e-12,
            "vertical_mode_mean_abs": 1.0e-12,
        },
    }


def test_solver_is_deterministic_and_self_consistent(tmp_path: Path) -> None:
    outputs = []
    for run in (1, 2):
        path = tmp_path / f"run{run}.h5"
        create_simulation_file(path, smoke_config(), [4], {4: [0, 1]})
        with h5py.File(path, "r") as handle:
            group = handle["modes/n0004"]
            outputs.append(
                (
                    np.asarray(group["nre"]),
                    np.asarray(group["complex_velocity"]),
                )
            )
            assert group.attrs["pse_initial_reconstruction_relative_l2"] < 1e-10
            assert json.loads(handle.attrs["model_names"]) == [
                "YBJ",
                "TSB",
                "YBJ+",
                "PSE",
                "HBEs",
            ]
    assert np.array_equal(outputs[0][0], outputs[1][0])
    assert np.array_equal(outputs[0][1], outputs[1][1])


def test_error_only_mode_allows_no_field_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "error_only.h5"
    create_simulation_file(path, smoke_config(), [2], {2: []})
    with h5py.File(path, "r") as handle:
        group = handle["modes/n0002"]
        assert group["complex_velocity"].shape == (0, 5, 16, 16)
        assert group["field_times_ip"].shape == (0,)
        assert np.all(np.isfinite(group["nre"]))
