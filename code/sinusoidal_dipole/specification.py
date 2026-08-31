"""Shared manuscript conventions for the sinusoidal-dipole calculations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.paper_parameters import (  # noqa: E402
    BUOYANCY_FREQUENCY_RATIO,
    CORIOLIS_FREQUENCY,
    DOMAIN_DEPTH,
    FLOW_SPEED,
)

REFERENCE_METRICS_PATH = ROOT / "config" / "reference_metrics.json"
SIMULATION_SOURCE_PATHS = (
    ROOT / "code" / "common" / "paper_parameters.py",
    ROOT / "code" / "sinusoidal_dipole" / "solver.py",
    ROOT / "code" / "sinusoidal_dipole" / "specification.py",
    ROOT / "code" / "sinusoidal_dipole" / "vertical_modes.py",
)

MODEL_NAMES = ("YBJ", "TSB", "YBJ+", "PSE", "HBEs")
NRE_MODEL_NAMES = MODEL_NAMES[:-1]

SUPPORTED_DEALIASING = "two-thirds"
SUPPORTED_DIFFUSION = "none"
SUPPORTED_OUTPUT_PRECISIONS = ("complex64", "complex128")

PAPER_PHYSICAL_PARAMETERS = {
    "coriolis_frequency_s-1": CORIOLIS_FREQUENCY,
    "buoyancy_frequency_ratio": BUOYANCY_FREQUENCY_RATIO,
    "domain_depth_m": DOMAIN_DEPTH,
    "background_length_scale_m": 50000.0,
    "background_velocity_m_s": FLOW_SPEED,
    "initial_velocity_amplitude_m_s": 1.0,
}
PAPER_NUMERICAL_PARAMETERS = {
    "horizontal_grid": 128,
    "time_steps_per_inertial_period": 64,
    "total_inertial_periods": 50,
    "dealiasing": SUPPORTED_DEALIASING,
    "diffusion": SUPPORTED_DIFFUSION,
    "pse_etdrk4_contour_points": 16,
    "output_precision": "complex64",
}
PAPER_VERTICAL_MODES = {
    "error_statistics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32],
    "figures_9_10": [1, 4, 8, 16, 32],
    "movie_2": [4, 16, 32],
}
PAPER_SAVED_TIMES = {
    "figures_9_10": [10, 50],
    "movie_2_start": 0,
    "movie_2_stop": 50,
    "movie_2_interval": 1,
}


def load_reference_metrics(path: Path = REFERENCE_METRICS_PATH) -> dict[str, Any]:
    """Load the version-controlled numerical regression targets."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported reference-metrics schema in {path}.")
    if tuple(data["model_names"]) != MODEL_NAMES:
        raise ValueError(f"The model order in {path} is inconsistent.")
    return data


def reference_metrics_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the repository-owned reference file named by a configuration."""
    relative = Path(config["reference_metrics_file"])
    if relative.is_absolute():
        raise ValueError("reference_metrics_file must be repository-relative.")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            "reference_metrics_file must remain inside the repository."
        ) from error
    return load_reference_metrics(path)


def validate_config(
    config: dict[str, Any],
    *,
    manuscript_resolution: bool = False,
) -> None:
    """Validate supported numerical choices and, if requested, paper settings."""
    required = {
        "schema_version",
        "random_seed",
        "reference_metrics_file",
        "physical_parameters",
        "numerical_parameters",
        "vertical_modes",
        "saved_times_in_inertial_periods",
        "validation_tolerances",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("Configuration is missing: " + ", ".join(missing))
    if config["schema_version"] != 1:
        raise ValueError("Only reproduction configuration schema 1 is supported.")

    physical = config["physical_parameters"]
    numerical = config["numerical_parameters"]
    grid = int(numerical["horizontal_grid"])
    steps = int(numerical["time_steps_per_inertial_period"])
    periods = int(numerical["total_inertial_periods"])
    contour_points = int(numerical["pse_etdrk4_contour_points"])
    if grid < 8 or grid % 2:
        raise ValueError("The horizontal grid must be an even integer of at least 8.")
    if steps < 4:
        raise ValueError("At least four time steps per inertial period are required.")
    if periods < 1:
        raise ValueError("total_inertial_periods must be positive.")
    if contour_points < 8:
        raise ValueError("At least eight ETDRK4 contour points are required.")
    if numerical["dealiasing"] != SUPPORTED_DEALIASING:
        raise ValueError("Only two-thirds dealiasing is implemented.")
    if numerical["diffusion"] != SUPPORTED_DIFFUSION:
        raise ValueError("The published calculation uses no numerical diffusion.")
    if numerical["output_precision"] not in SUPPORTED_OUTPUT_PRECISIONS:
        allowed = ", ".join(SUPPORTED_OUTPUT_PRECISIONS)
        raise ValueError(f"output_precision must be one of: {allowed}.")
    for key, value in physical.items():
        if float(value) <= 0.0:
            raise ValueError(f"physical_parameters.{key} must be positive.")

    reference_metrics_from_config(config)
    if manuscript_resolution:
        if int(config["random_seed"]) != 20260826:
            raise ValueError("Full reproduction requires the published random seed.")
        if physical != PAPER_PHYSICAL_PARAMETERS:
            raise ValueError(
                "Full reproduction requires the published physical parameters."
            )
        if numerical != PAPER_NUMERICAL_PARAMETERS:
            raise ValueError(
                "Full reproduction requires the published numerical parameters."
            )
        if config["vertical_modes"] != PAPER_VERTICAL_MODES:
            raise ValueError(
                "Full reproduction requires the published vertical-mode sets."
            )
        if config["saved_times_in_inertial_periods"] != PAPER_SAVED_TIMES:
            raise ValueError("Full reproduction requires the published saved times.")


def simulation_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Return the inputs that determine the simulation archive."""
    return {
        "schema_version": config["schema_version"],
        "random_seed": config["random_seed"],
        "physical_parameters": config["physical_parameters"],
        "numerical_parameters": config["numerical_parameters"],
        "vertical_modes": config["vertical_modes"],
        "saved_times_in_inertial_periods": config["saved_times_in_inertial_periods"],
    }


def simulation_signature(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 identity for all simulation-determining inputs."""
    payload = json.dumps(
        simulation_configuration(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def simulation_source_inventory() -> list[dict[str, str]]:
    """Hash the source files that determine the simulation archive."""
    return [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in SIMULATION_SOURCE_PATHS
    ]


def simulation_source_signature() -> str:
    """Return a stable identity for the simulation implementation."""
    payload = json.dumps(
        {
            "schema_version": 1,
            "files": simulation_source_inventory(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
