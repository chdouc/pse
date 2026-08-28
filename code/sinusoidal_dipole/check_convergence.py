"""Run the representative spatial and temporal refinement checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.signal import resample

from solver import create_simulation_file
from specification import MODEL_NAMES


def periodic_resample(values: np.ndarray, size: int) -> np.ndarray:
    """Spectrally resample periodic fields on their last two axes."""
    return resample(resample(values, size, axis=-1), size, axis=-2)


def relative_l2(reference: np.ndarray, approximation: np.ndarray) -> float:
    denominator = float(np.sum(np.abs(reference) ** 2))
    if denominator <= 0.0:
        raise FloatingPointError("The convergence reference has zero norm.")
    return float(np.sqrt(np.sum(np.abs(approximation - reference) ** 2) / denominator))


def read_final_fields(path: Path, mode: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        fields = np.asarray(handle[f"modes/n{mode:04d}/complex_velocity"])
    if fields.shape[0] != 1:
        raise ValueError(f"Expected one final field in {path}; found {fields.shape[0]}.")
    return fields[0]


def pair_errors(
    coarse: np.ndarray,
    fine: np.ndarray,
) -> dict[str, float]:
    """Compare each fine field after projection onto the coarser grid."""
    if coarse.shape[-1] != fine.shape[-1]:
        fine = periodic_resample(fine, coarse.shape[-1])
    return {
        name: relative_l2(fine[index], coarse[index])
        for index, name in enumerate(MODEL_NAMES)
    }


def check_refinement(
    fields: list[np.ndarray],
    labels: list[int],
    *,
    maximum_finest_pair_relative_l2: float,
    require_monotone: bool,
    monotonicity_floor_relative_l2: float,
) -> dict[str, Any]:
    errors = [pair_errors(fields[index], fields[index + 1]) for index in range(2)]
    failures = []
    for name in MODEL_NAMES:
        coarse_error = errors[0][name]
        fine_error = errors[1][name]
        if fine_error > maximum_finest_pair_relative_l2:
            failures.append(
                f"{name}: finest-pair relative L2 {fine_error:.6g} exceeds "
                f"{maximum_finest_pair_relative_l2:.6g}"
            )
        below_precision_floor = (
            max(coarse_error, fine_error) <= monotonicity_floor_relative_l2
        )
        if require_monotone and not below_precision_floor and fine_error >= coarse_error:
            failures.append(
                f"{name}: refinement error did not decrease "
                f"({coarse_error:.6g} to {fine_error:.6g})"
            )
    if failures:
        raise AssertionError("; ".join(failures))
    return {
        "levels": labels,
        "pair_labels": [f"{labels[0]}-{labels[1]}", f"{labels[1]}-{labels[2]}"],
        "relative_l2_by_model": errors,
        "maximum_finest_pair_relative_l2": maximum_finest_pair_relative_l2,
        "require_monotone_refinement": require_monotone,
        "monotonicity_floor_relative_l2": monotonicity_floor_relative_l2,
    }


def run_case(
    base_config: dict[str, Any],
    output_path: Path,
    *,
    mode: int,
    periods: int,
    grid: int,
    steps_per_period: int,
) -> np.ndarray:
    config = copy.deepcopy(base_config)
    config["numerical_parameters"].update(
        {
            "horizontal_grid": grid,
            "time_steps_per_inertial_period": steps_per_period,
            "total_inertial_periods": periods,
        }
    )
    create_simulation_file(
        output_path,
        config,
        [mode],
        {mode: [periods]},
        workers=1,
    )
    return read_final_fields(output_path, mode)


def run_convergence(
    base_config: dict[str, Any],
    convergence_config: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    """Run and validate the configured three-level refinement study."""
    if convergence_config.get("schema_version") != 1:
        raise ValueError("Only convergence configuration schema 1 is supported.")
    mode = int(convergence_config["vertical_mode"])
    periods = int(convergence_config["final_inertial_period"])
    require_monotone = bool(convergence_config["require_monotone_refinement"])
    monotonicity_floor = float(
        convergence_config["monotonicity_floor_relative_l2"]
    )
    if monotonicity_floor < 0.0:
        raise ValueError("monotonicity_floor_relative_l2 must be non-negative.")
    output_directory.mkdir(parents=True, exist_ok=True)

    spatial = convergence_config["spatial_refinement"]
    spatial_levels = [int(value) for value in spatial["horizontal_grids"]]
    if len(spatial_levels) != 3 or spatial_levels != sorted(spatial_levels):
        raise ValueError("spatial_refinement must define three increasing grids.")
    spatial_fields = []
    for grid in spatial_levels:
        path = output_directory / f"spatial_N{grid}.h5"
        spatial_fields.append(
            run_case(
                base_config,
                path,
                mode=mode,
                periods=periods,
                grid=grid,
                steps_per_period=int(spatial["time_steps_per_inertial_period"]),
            )
        )
    spatial_report = check_refinement(
        spatial_fields,
        spatial_levels,
        maximum_finest_pair_relative_l2=float(
            spatial["maximum_finest_pair_relative_l2"]
        ),
        require_monotone=require_monotone,
        monotonicity_floor_relative_l2=monotonicity_floor,
    )

    temporal = convergence_config["temporal_refinement"]
    temporal_levels = [
        int(value) for value in temporal["time_steps_per_inertial_period"]
    ]
    if len(temporal_levels) != 3 or temporal_levels != sorted(temporal_levels):
        raise ValueError("temporal_refinement must define three increasing step counts.")
    temporal_fields = []
    for steps in temporal_levels:
        path = output_directory / f"temporal_fc{steps}.h5"
        temporal_fields.append(
            run_case(
                base_config,
                path,
                mode=mode,
                periods=periods,
                grid=int(temporal["horizontal_grid"]),
                steps_per_period=steps,
            )
        )
    temporal_report = check_refinement(
        temporal_fields,
        temporal_levels,
        maximum_finest_pair_relative_l2=float(
            temporal["maximum_finest_pair_relative_l2"]
        ),
        require_monotone=require_monotone,
        monotonicity_floor_relative_l2=monotonicity_floor,
    )

    report = {
        "status": "passed",
        "kind": "convergence-test",
        "vertical_mode": mode,
        "final_inertial_period": periods,
        "spatial_refinement": spatial_report,
        "temporal_refinement": temporal_report,
    }
    (output_directory / "convergence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
