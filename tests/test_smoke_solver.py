"""Small deterministic integration tests for the five-model solver."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from solver import Grid, _initial_pse, create_simulation_file, ifft2, parameters_from_config
from check_convergence import check_refinement, periodic_resample
from reproduce import source_inventory, validate_reusable_simulation


def smoke_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "random_seed": 20260826,
        "reference_metrics_file": "config/reference_metrics.json",
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
            "dealiasing": "two-thirds",
            "diffusion": "none",
            "pse_etdrk4_contour_points": 16,
            "output_precision": "complex64",
        },
        "vertical_modes": {
            "error_statistics": [1, 4, 8, 16, 32],
            "figures_9_10": [1, 4, 8, 16, 32],
            "movie_2": [4, 16, 32],
        },
        "saved_times_in_inertial_periods": {
            "figures_9_10": [0, 1],
            "movie_2_start": 0,
            "movie_2_stop": 1,
            "movie_2_interval": 1,
        },
        "validation_tolerances": {
            "vertical_mode_boundary_derivative_abs": 1.0e-12,
            "vertical_mode_mean_abs": 1.0e-12,
        },
    }


def test_pse_initialisation_matches_appendix_formula() -> None:
    parameters = parameters_from_config(smoke_config())
    grid = Grid(parameters)
    target = np.full((grid.n, grid.n), 1.0 + 0.2j)
    up_hat, down_hat, iterations, residual = _initial_pse(target, grid)

    expected_down = 0.25j * (grid.xi2 + 1j * grid.xi3) * np.conj(target)
    expected_up = target - expected_down
    assert iterations == 0
    assert residual < 1.0e-14
    assert np.allclose(ifft2(up_hat), expected_up, rtol=0.0, atol=1.0e-14)
    assert np.allclose(
        np.conj(ifft2(down_hat)), expected_down, rtol=0.0, atol=1.0e-14
    )


def test_periodic_resampling_preserves_resolved_fourier_modes() -> None:
    coordinate = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    x, y = np.meshgrid(coordinate, coordinate)
    values = np.exp(2j * x - 3j * y)
    smaller = periodic_resample(values, 16)
    small_coordinate = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    small_x, small_y = np.meshgrid(small_coordinate, small_coordinate)
    expected = np.exp(2j * small_x - 3j * small_y)
    assert np.allclose(smaller, expected, rtol=0.0, atol=2.0e-14)


def test_convergence_check_reports_decreasing_errors() -> None:
    reference = np.ones((5, 16, 16), dtype=complex)
    fields = [reference * 1.01, reference * 1.001, reference]
    report = check_refinement(
        fields,
        [16, 32, 64],
        maximum_finest_pair_relative_l2=0.01,
        require_monotone=True,
        monotonicity_floor_relative_l2=1.0e-6,
    )
    assert report["relative_l2_by_model"][1]["PSE"] < 0.01


def test_convergence_monotonicity_ignores_the_storage_precision_floor() -> None:
    reference = np.ones((5, 16, 16), dtype=np.complex64)
    fields = [reference * (1.0 + 1.0e-7), reference * (1.0 + 2.0e-7), reference]
    report = check_refinement(
        fields,
        [16, 32, 64],
        maximum_finest_pair_relative_l2=1.0e-5,
        require_monotone=True,
        monotonicity_floor_relative_l2=1.0e-6,
    )
    assert report["monotonicity_floor_relative_l2"] == 1.0e-6


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("dealiasing", "none", "two-thirds"),
        ("diffusion", "laplacian", "no numerical diffusion"),
        ("output_precision", "float32", "output_precision"),
    ],
)
def test_unsupported_numerical_choices_fail(
    key: str,
    value: str,
    message: str,
) -> None:
    config = smoke_config()
    config["numerical_parameters"][key] = value
    with pytest.raises(ValueError, match=message):
        parameters_from_config(config)


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
            assert group.attrs["pse_initialisation"] == "frozen-local strain-only O(Ro)"
            assert group.attrs["pse_initial_iterations"] == 0
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


def test_reuse_rejects_a_different_numerical_configuration(tmp_path: Path) -> None:
    path = tmp_path / "simulation.h5"
    config = smoke_config()
    create_simulation_file(path, config, [4], {4: [0, 1]})

    changed = copy.deepcopy(config)
    changed["numerical_parameters"]["horizontal_grid"] = 32
    with pytest.raises(ValueError, match="different configuration"):
        validate_reusable_simulation(path, changed)


def test_source_inventory_works_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("reproduce.subprocess.run", missing_git)
    records = source_inventory()
    paths = {record["path"] for record in records}
    assert "README.md" in paths
    assert "code/sinusoidal_dipole/solver.py" in paths
    assert not any("__pycache__" in path for path in paths)
