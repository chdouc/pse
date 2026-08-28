"""Prepare supplementary Movie 2 from the repository solver output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from specification import MODEL_NAMES, NRE_MODEL_NAMES, reference_metrics_from_config

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "sinusoidal_dipole_movie_fields.npz"
)


def outward_significant(value: float, digits: int = 3) -> float:
    if value <= 0.0:
        return 0.0
    scale = 10.0 ** (digits - 1 - np.floor(np.log10(value)))
    return float(np.ceil(value * scale) / scale)


def clipping_record(values: np.ndarray, lower: float, upper: float) -> dict[str, Any]:
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


def relative_l2(reference: np.ndarray, model: np.ndarray) -> float:
    return float(
        np.sqrt(np.sum(np.abs(model - reference) ** 2) / np.sum(np.abs(reference) ** 2))
    )


def compute_archive(
    simulation_path: Path,
    *,
    difference_quantile: float | None = None,
) -> dict[str, np.ndarray]:
    """Build the movie archive without interpolation or external inputs."""
    fields_by_mode: list[np.ndarray] = []
    nre_by_mode: list[np.ndarray] = []
    nre_times_by_mode: list[np.ndarray] = []
    wavelengths: list[float] = []
    nre_differences: list[np.ndarray] = []

    with h5py.File(simulation_path, "r") as handle:
        config = json.loads(handle.attrs["config_json"])
        references = reference_metrics_from_config(config)
        movie_reference = references["movie2"]
        vertical_modes = tuple(int(mode) for mode in movie_reference["vertical_modes"])
        absolute_limit_map = {
            int(mode): tuple(limits)
            for mode, limits in movie_reference["absolute_color_limits"].items()
        }
        if difference_quantile is None:
            difference_quantile = float(movie_reference["difference_quantile"])
        if not 0.5 < difference_quantile < 1.0:
            raise ValueError("The difference quantile must lie between 0.5 and 1.")
        physical = config["physical_parameters"]
        numerical = config["numerical_parameters"]
        amplitude = float(physical["initial_velocity_amplitude_m_s"])
        if tuple(json.loads(handle.attrs["model_names"])) != MODEL_NAMES:
            raise ValueError("The simulation archive model order changed.")
        for mode in vertical_modes:
            group = handle[f"modes/n{mode:04d}"]
            field_times = np.asarray(group["field_times_ip"])
            expected_times = np.arange(51.0)
            if not np.array_equal(field_times, expected_times):
                raise ValueError(f"n={mode}: Movie 2 requires every integer 0--50 IP.")
            velocity = np.asarray(group["complex_velocity"])
            fields_by_mode.append(np.abs(velocity) ** 2 / amplitude**2)
            times = np.asarray(group["times_ip"])
            nre = np.asarray(group["nre"])
            nre_by_mode.append(nre)
            nre_times_by_mode.append(times)
            wavelengths.append(float(group.attrs["vertical_wavelength_m"]))

            consistency = np.zeros(len(NRE_MODEL_NAMES))
            for field_index, target_time in enumerate(field_times):
                error_index = int(
                    round(target_time * numerical["time_steps_per_inertial_period"])
                )
                reference = velocity[field_index, -1]
                for model_index in range(len(NRE_MODEL_NAMES)):
                    consistency[model_index] = max(
                        consistency[model_index],
                        abs(
                            relative_l2(reference, velocity[field_index, model_index])
                            - nre[model_index, error_index]
                        ),
                    )
            nre_differences.append(consistency)

    field_array = np.stack(fields_by_mode)
    nre_array = np.stack(nre_by_mode)
    nre_times = np.stack(nre_times_by_mode)
    differences = field_array[:, :, :4] - field_array[:, :, 4:5]
    absolute_limits = np.asarray(
        [absolute_limit_map[mode] for mode in vertical_modes]
    )
    difference_limits: list[tuple[float, float]] = []
    clipping: dict[str, Any] = {}
    for index, mode in enumerate(vertical_modes):
        limit = outward_significant(
            float(np.quantile(np.abs(differences[index]), difference_quantile))
        )
        difference_limits.append((-limit, limit))
        clipping[str(mode)] = {
            "absolute_field": clipping_record(
                field_array[index], *absolute_limits[index]
            ),
            "difference_field": clipping_record(differences[index], -limit, limit),
        }

    coordinate = np.linspace(-np.pi, np.pi, field_array.shape[-1], endpoint=False)
    metadata = {
        "schema_version": 3,
        "product": "supplementary movie 2",
        "background_flow": "sinusoidal dipole",
        "background_velocity_m_s": physical["background_velocity_m_s"],
        "domain_depth_m": physical["domain_depth_m"],
        "vertical_wavelength_relation": "h=2H/n",
        "source_kind": "fields generated by the in-repository equation solver",
        "spatial_discretisation": {
            "grid_points": [
                numerical["horizontal_grid"],
                numerical["horizontal_grid"],
            ]
        },
        "time_discretisation": {
            "parameter": "fc",
            "steps_per_inertial_period": numerical["time_steps_per_inertial_period"],
        },
        "model_order": list(MODEL_NAMES),
        "nre_model_order": list(NRE_MODEL_NAMES),
        "vertical_modes": list(vertical_modes),
        "time_units": "inertial periods",
        "field_quantity_tex": r"|\phi|^2/|\phi_{\mathrm{amp}}|^2",
        "difference_definition": "named model minus HBEs normalized squared velocity",
        "nre_definition_tex": (
            r"[\int|\phi_M-\phi_{\mathrm{HBEs}}|^2/"
            r"\int|\phi_{\mathrm{HBEs}}|^2]^{1/2}"
        ),
        "pse_field_source": "A_up exp(-ift)+A_down exp(ift) from the PSE solver",
        "orientation": {
            "array_axes": "time, model, y, x",
            "origin": "lower",
            "extent": [-np.pi, np.pi, -np.pi, np.pi],
        },
        "sampling": {
            "start_ip": 0.0,
            "stop_ip": 50.0,
            "sample_interval_ip": 1.0,
            "physical_field_interpolation": False,
        },
        "color_limits": {
            "absolute_strategy": "fixed manuscript-aligned mode-specific limits",
            "difference_strategy": (
                f"symmetric {difference_quantile:g} quantile, rounded outward"
            ),
            "difference_quantile": difference_quantile,
            "clipping": clipping,
        },
        "validation": {
            "source": "solver output",
            "all_selected_times_recomputed": True,
            "includes_10_ip": True,
            "includes_50_ip": True,
        },
    }
    return {
        "times_in_inertial_periods": np.arange(51.0),
        "source_time_indices": np.tile(np.arange(51), (3, 1)),
        "vertical_modes": np.asarray(vertical_modes),
        "vertical_wavelengths_m": np.asarray(wavelengths),
        "model_names": np.asarray(MODEL_NAMES),
        "nre_model_names": np.asarray(NRE_MODEL_NAMES),
        "normalized_squared_velocity": field_array,
        "nre_times_in_inertial_periods": nre_times,
        "nre_complex_relative_l2": nre_array,
        "x_over_L": coordinate,
        "y_over_L": coordinate,
        "normalization_amplitude": np.full(3, amplitude),
        "absolute_color_limits": absolute_limits,
        "difference_color_limits": np.asarray(difference_limits),
        "recomputed_nre_max_abs_difference": np.stack(nre_differences),
        "processed_source_files": np.asarray([simulation_path.name] * 3),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--difference-quantile", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = compute_archive(
        args.input.resolve(), difference_quantile=args.difference_quantile
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".partial.npz")
    np.savez_compressed(temporary, **archive)
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
