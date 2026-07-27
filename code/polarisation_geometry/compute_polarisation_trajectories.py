"""Compute the data used in supplementary movie 1.

The calculation follows the manuscript convention
``|A> = (A_up, conj(A_down))^T``.  Rendering is deliberately kept in a
separate script so every sphere vector and hodograph in the movie can be
traced to the same saved spinor state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm


SIGMA = np.stack(
    [
        np.eye(2, dtype=complex),
        np.array([[1j, 0.0], [0.0, -1j]], dtype=complex),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, 1j], [-1j, 0.0]], dtype=complex),
    ]
)
GENERATOR_NAMES = np.asarray(["sigma_0", "sigma_1", "sigma_2", "sigma_3"])
SPINOR_CONVENTION = "|A> = (A_up, conj(A_down))^T"
DATA_FILENAME = "movie1_data.npz"
METADATA_FILENAME = "movie1_metadata.json"


def complex_pair(value: complex) -> list[float]:
    """Return a JSON-safe real-imaginary pair."""
    return [float(np.real(value)), float(np.imag(value))]


def spinor_from_angles(
    stokes_magnitude: float,
    varphi: np.ndarray | float,
    longitude: np.ndarray | float,
    gamma: np.ndarray | float,
) -> np.ndarray:
    """Construct the manuscript spinor from latitude, longitude and phase."""
    varphi, longitude, gamma = np.broadcast_arrays(varphi, longitude, gamma)
    amplitude = np.sqrt(stokes_magnitude)
    component_up = (
        amplitude
        * np.cos(np.pi / 4.0 - varphi / 2.0)
        * np.exp(1j * (gamma + longitude / 2.0))
    )
    stored_conjugate_down = (
        amplitude
        * np.sin(np.pi / 4.0 - varphi / 2.0)
        * np.exp(1j * (gamma - longitude / 2.0))
    )
    return np.stack([component_up, stored_conjugate_down], axis=-1)


def stokes_from_spinor(spinor: np.ndarray) -> np.ndarray:
    """Return the unnormalised Stokes vector using the manuscript definition."""
    spinor = np.asarray(spinor)
    component_up = spinor[..., 0]
    component_down = np.conj(spinor[..., 1])
    product = component_up * component_down
    return np.stack(
        [
            2.0 * np.real(product),
            2.0 * np.imag(product),
            np.abs(component_up) ** 2 - np.abs(component_down) ** 2,
        ],
        axis=-1,
    )


def stokes_magnitude(spinor: np.ndarray) -> np.ndarray:
    """Return |A_up|^2 + |A_down|^2."""
    spinor = np.asarray(spinor)
    return np.sum(np.abs(spinor) ** 2, axis=-1)


def hodograph_from_spinor(
    spinor: np.ndarray,
    fast_phase: np.ndarray,
) -> np.ndarray:
    """Return phi(theta)=A_up exp(-i theta)+A_down exp(i theta)."""
    spinor = np.asarray(spinor)
    fast_phase = np.asarray(fast_phase)
    component_up = spinor[..., 0]
    component_down = np.conj(spinor[..., 1])
    return component_up[..., None] * np.exp(-1j * fast_phase) + component_down[
        ..., None
    ] * np.exp(1j * fast_phase)


def analytic_exponential(generator_index: int, parameter: float) -> np.ndarray:
    """Return exp(parameter*sigma_j) from the exact closed form."""
    identity = SIGMA[0]
    generator = SIGMA[generator_index]
    if generator_index == 0:
        return np.exp(parameter) * identity
    if generator_index == 1:
        return np.cos(parameter) * identity + np.sin(parameter) * generator
    return np.cosh(parameter) * identity + np.sinh(parameter) * generator


def evolve_generator_branches(
    initial_spinor: np.ndarray,
    generator_parameter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive and negative exact matrix-exponential trajectories."""
    positive = np.empty(
        (SIGMA.shape[0], generator_parameter.size, 2),
        dtype=complex,
    )
    negative = np.empty_like(positive)
    for generator_index in range(SIGMA.shape[0]):
        for index, scaled_time in enumerate(generator_parameter / 50.0):
            positive[generator_index, index] = (
                analytic_exponential(generator_index, scaled_time) @ initial_spinor
            )
            negative[generator_index, index] = (
                analytic_exponential(generator_index, -scaled_time) @ initial_spinor
            )
    return positive, negative


def signed_polygon_area(values: np.ndarray) -> float:
    """Return the signed area enclosed by a sampled complex curve."""
    x = np.real(values)
    y = np.imag(values)
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def wrapped_axis_error(angle: float, target: float) -> float:
    """Return an orientation error modulo pi."""
    difference = angle - target
    return float(0.5 * np.arctan2(np.sin(2.0 * difference), np.cos(2.0 * difference)))


def ellipse_axes_and_orientation(values: np.ndarray) -> tuple[float, float, float]:
    """Estimate semi-axis lengths and major-axis orientation from samples."""
    points = np.column_stack([np.real(values), np.imag(values)])
    covariance = np.cov(points[:-1], rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    semiaxes = np.sqrt(2.0 * np.maximum(eigenvalues, 0.0))
    major_vector = eigenvectors[:, 0]
    orientation = float(np.arctan2(major_vector[1], major_vector[0]))
    return float(semiaxes[0]), float(semiaxes[1]), orientation


def matrix_relation_metrics() -> dict[str, float]:
    """Check the manuscript Clifford and commutator relations."""
    identity = SIGMA[0]
    metric = np.diag([-1.0, 1.0, 1.0])
    square_errors = {
        "sigma_1_squared": float(np.max(np.abs(SIGMA[1] @ SIGMA[1] + identity))),
        "sigma_2_squared": float(np.max(np.abs(SIGMA[2] @ SIGMA[2] - identity))),
        "sigma_3_squared": float(np.max(np.abs(SIGMA[3] @ SIGMA[3] - identity))),
    }
    anticommutator_error = 0.0
    for a in range(3):
        for b in range(3):
            left = SIGMA[a + 1] @ SIGMA[b + 1] + SIGMA[b + 1] @ SIGMA[a + 1]
            right = 2.0 * metric[a, b] * identity
            anticommutator_error = max(
                anticommutator_error,
                float(np.max(np.abs(left - right))),
            )
    commutators = (
        (SIGMA[1] @ SIGMA[2] - SIGMA[2] @ SIGMA[1], 2.0 * SIGMA[3]),
        (SIGMA[2] @ SIGMA[3] - SIGMA[3] @ SIGMA[2], -2.0 * SIGMA[1]),
        (SIGMA[3] @ SIGMA[1] - SIGMA[1] @ SIGMA[3], 2.0 * SIGMA[2]),
    )
    commutator_error = max(
        float(np.max(np.abs(left - right))) for left, right in commutators
    )
    return {
        **square_errors,
        "anticommutator_error": anticommutator_error,
        "commutator_error": commutator_error,
    }


def run_mathematical_validation(
    arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    """Run the mathematical checks requested for supplementary movie 1."""
    tolerance = 2.0e-11
    metrics = matrix_relation_metrics()

    expected_sigma = np.stack(
        [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
            np.array([[1j, 0.0], [0.0, -1j]], dtype=complex),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
            np.array([[0.0, 1j], [-1j, 0.0]], dtype=complex),
        ]
    )
    metrics["matrix_definition_error"] = float(np.max(np.abs(SIGMA - expected_sigma)))

    rng = np.random.default_rng(20260727)
    random_spinors = rng.normal(size=(32, 2)) + 1j * rng.normal(size=(32, 2))
    random_stokes = stokes_from_spinor(random_spinors)
    norm_error = np.max(
        np.abs(
            np.linalg.norm(random_stokes, axis=-1) - stokes_magnitude(random_spinors)
        )
    )
    metrics["stokes_norm_identity_error"] = float(norm_error)

    gamma_spinors = spinor_from_angles(
        1.0,
        arrays["initial_varphi"],
        arrays["initial_lambda"],
        np.linspace(-np.pi, np.pi, 41),
    )
    gamma_stokes = stokes_from_spinor(gamma_spinors)
    metrics["common_phase_stokes_invariance_error"] = float(
        np.max(np.abs(gamma_stokes - gamma_stokes[0]))
    )
    chapter_one_stokes = np.concatenate(
        [
            arrays[f"{stage}_stokes"]
            for stage in ("landmark", "ellipticity", "orientation", "phase")
        ],
        axis=0,
    )
    metrics["chapter1_unit_stokes_error"] = float(
        np.max(np.abs(np.linalg.norm(chapter_one_stokes, axis=-1) - 1.0))
    )

    fast_phase = arrays["fast_phase"]
    north = hodograph_from_spinor(
        spinor_from_angles(1.0, np.pi / 2.0, 0.0, 0.0),
        fast_phase,
    )
    south = hodograph_from_spinor(
        spinor_from_angles(1.0, -np.pi / 2.0, 0.0, 0.0),
        fast_phase,
    )
    equator = hodograph_from_spinor(
        spinor_from_angles(1.0, 0.0, 0.0, 0.0),
        fast_phase,
    )
    north_area = signed_polygon_area(north)
    south_area = signed_polygon_area(south)
    _, equator_minor, _ = ellipse_axes_and_orientation(equator)
    if not north_area < 0.0:
        raise ValueError("The north-pole hodograph is not clockwise.")
    if not south_area > 0.0:
        raise ValueError("The south-pole hodograph is not counter-clockwise.")
    metrics["north_clockwise_signed_area"] = north_area
    metrics["south_counterclockwise_signed_area"] = south_area
    metrics["equator_minor_semiaxis"] = equator_minor

    orientation_errors = []
    for longitude in np.linspace(-0.9 * np.pi, 0.9 * np.pi, 13):
        values = hodograph_from_spinor(
            spinor_from_angles(
                1.0,
                arrays["initial_varphi"],
                longitude,
                arrays["initial_gamma"],
            ),
            fast_phase,
        )
        _, _, orientation = ellipse_axes_and_orientation(values)
        orientation_errors.append(abs(wrapped_axis_error(orientation, longitude / 2.0)))
    metrics["longitude_half_orientation_error"] = float(max(orientation_errors))

    ellipticity_errors = []
    for latitude in np.linspace(-1.3, 1.3, 15):
        values = hodograph_from_spinor(
            spinor_from_angles(
                1.0,
                latitude,
                arrays["initial_lambda"],
                arrays["initial_gamma"],
            ),
            fast_phase,
        )
        major, minor, _ = ellipse_axes_and_orientation(values)
        ellipticity_errors.append(abs(np.arctan2(minor, major) - abs(latitude) / 2.0))
    metrics["latitude_half_ellipticity_error"] = float(max(ellipticity_errors))

    initial_spinor = arrays["initial_spinor"]
    positive = arrays["generator_spinor_positive"]
    negative = arrays["generator_spinor_negative"]
    positive_stokes = arrays["generator_stokes_positive"]
    negative_stokes = arrays["generator_stokes_negative"]
    initial_direction = stokes_from_spinor(initial_spinor)
    initial_direction = initial_direction / np.linalg.norm(initial_direction)
    sigma0_directions = np.concatenate(
        [
            positive_stokes[0] / np.linalg.norm(positive_stokes[0], axis=-1)[:, None],
            negative_stokes[0] / np.linalg.norm(negative_stokes[0], axis=-1)[:, None],
        ],
        axis=0,
    )
    metrics["sigma0_direction_invariance_error"] = float(
        np.max(np.abs(sigma0_directions - initial_direction))
    )

    component_magnitudes = np.abs(positive[1])
    metrics["sigma1_component_magnitude_error"] = float(
        np.max(np.abs(component_magnitudes - np.abs(initial_spinor)))
    )
    sigma1_longitude = np.unwrap(
        np.arctan2(positive_stokes[1, :, 1], positive_stokes[1, :, 0])
    )
    expected_longitude = (
        arrays["initial_lambda"] + 2.0 * arrays["generator_parameter"] / 50.0
    )
    metrics["sigma1_longitude_error"] = float(
        np.max(np.abs(sigma1_longitude - expected_longitude))
    )

    metrics["positive_zero_initial_error"] = float(
        np.max(np.abs(positive[:, 0] - initial_spinor))
    )
    metrics["negative_zero_initial_error"] = float(
        np.max(np.abs(negative[:, 0] - initial_spinor))
    )

    exponential_error = 0.0
    for generator_index in range(4):
        for parameter in (-0.47, -0.11, 0.0, 0.23, 0.61):
            exponential_error = max(
                exponential_error,
                float(
                    np.max(
                        np.abs(
                            analytic_exponential(generator_index, parameter)
                            - expm(parameter * SIGMA[generator_index])
                        )
                    )
                ),
            )
    metrics["analytic_vs_numeric_exponential_error"] = exponential_error

    for generator_index in (2, 3):
        parameter = arrays["generator_parameter"][-1] / 50.0
        expected_positive = (
            np.cosh(parameter) * initial_spinor
            + np.sinh(parameter) * SIGMA[generator_index] @ initial_spinor
        )
        metrics[f"sigma{generator_index}_hyperbolic_mixing_error"] = float(
            np.max(np.abs(positive[generator_index, -1] - expected_positive))
        )
        if np.allclose(
            np.abs(positive[generator_index, -1]),
            np.abs(initial_spinor),
            atol=1.0e-6,
        ):
            raise ValueError(
                f"sigma_{generator_index} did not produce the expected mixing."
            )

    recomputed_positive_stokes = stokes_from_spinor(positive)
    recomputed_negative_stokes = stokes_from_spinor(negative)
    metrics["saved_stokes_same_spinor_error"] = float(
        max(
            np.max(np.abs(recomputed_positive_stokes - positive_stokes)),
            np.max(np.abs(recomputed_negative_stokes - negative_stokes)),
        )
    )
    recomputed_positive_hodographs = hodograph_from_spinor(positive, fast_phase)
    recomputed_negative_hodographs = hodograph_from_spinor(negative, fast_phase)
    metrics["saved_hodograph_same_spinor_error"] = float(
        max(
            np.max(
                np.abs(
                    recomputed_positive_hodographs
                    - arrays["generator_hodograph_positive"]
                )
            ),
            np.max(
                np.abs(
                    recomputed_negative_hodographs
                    - arrays["generator_hodograph_negative"]
                )
            ),
        )
    )

    strict_metrics = [
        key
        for key in metrics
        if key.endswith("_error") and key != "equator_minor_semiaxis"
    ]
    failures = {key: metrics[key] for key in strict_metrics if metrics[key] > tolerance}
    if equator_minor > 2.0e-8:
        failures["equator_minor_semiaxis"] = equator_minor
    if failures:
        details = ", ".join(f"{key}={value:.3e}" for key, value in failures.items())
        raise ValueError(f"Mathematical validation failed: {details}")
    return metrics


def build_dataset(sample_count: int, fast_phase_count: int) -> dict[str, np.ndarray]:
    """Construct all spinor, Stokes and hodograph arrays for both chapters."""
    if sample_count < 101:
        raise ValueError("sample_count must be at least 101.")
    if fast_phase_count < 181:
        raise ValueError("fast_phase_count must be at least 181.")

    fast_phase = np.linspace(0.0, 2.0 * np.pi, fast_phase_count)
    reference_theta = np.arctan(1.0 / 5.0)
    initial_stokes_magnitude = 1.1**2
    initial_varphi = np.pi / 2.0 - 2.0 * reference_theta
    initial_lambda = np.pi / 3.0
    initial_gamma = -np.pi / 4.0
    initial_spinor = spinor_from_angles(
        initial_stokes_magnitude,
        initial_varphi,
        initial_lambda,
        initial_gamma,
    )

    unit_progress = np.linspace(0.0, 1.0, sample_count)
    landmark_varphi = np.linspace(np.pi / 2.0, -np.pi / 2.0, sample_count)
    landmark_spinor = spinor_from_angles(
        1.0,
        landmark_varphi,
        initial_lambda,
        initial_gamma,
    )

    ellipticity_varphi = initial_varphi * np.cos(2.0 * np.pi * unit_progress)
    ellipticity_spinor = spinor_from_angles(
        1.0,
        ellipticity_varphi,
        initial_lambda,
        initial_gamma,
    )

    orientation_lambda = initial_lambda + 2.0 * np.pi * unit_progress
    orientation_spinor = spinor_from_angles(
        1.0,
        initial_varphi,
        orientation_lambda,
        initial_gamma,
    )

    phase_gamma = initial_gamma + 2.0 * np.pi * unit_progress
    phase_spinor = spinor_from_angles(
        1.0,
        initial_varphi,
        initial_lambda,
        phase_gamma,
    )

    generator_parameter = np.linspace(0.0, 5.0 * np.pi, sample_count)
    generator_positive, generator_negative = evolve_generator_branches(
        initial_spinor,
        generator_parameter,
    )

    arrays: dict[str, np.ndarray] = {
        "sigma": SIGMA,
        "generator_names": GENERATOR_NAMES,
        "fast_phase": fast_phase,
        "unit_progress": unit_progress,
        "initial_spinor": initial_spinor,
        "initial_stokes": stokes_from_spinor(initial_spinor),
        "initial_stokes_magnitude": np.asarray(initial_stokes_magnitude),
        "initial_varphi": np.asarray(initial_varphi),
        "initial_lambda": np.asarray(initial_lambda),
        "initial_gamma": np.asarray(initial_gamma),
        "landmark_varphi": landmark_varphi,
        "landmark_lambda": np.full(sample_count, initial_lambda),
        "landmark_gamma": np.full(sample_count, initial_gamma),
        "landmark_spinor": landmark_spinor,
        "landmark_stokes": stokes_from_spinor(landmark_spinor),
        "landmark_hodograph": hodograph_from_spinor(landmark_spinor, fast_phase),
        "ellipticity_varphi": ellipticity_varphi,
        "ellipticity_lambda": np.full(sample_count, initial_lambda),
        "ellipticity_gamma": np.full(sample_count, initial_gamma),
        "ellipticity_spinor": ellipticity_spinor,
        "ellipticity_stokes": stokes_from_spinor(ellipticity_spinor),
        "ellipticity_hodograph": hodograph_from_spinor(
            ellipticity_spinor,
            fast_phase,
        ),
        "orientation_varphi": np.full(sample_count, initial_varphi),
        "orientation_lambda": orientation_lambda,
        "orientation_gamma": np.full(sample_count, initial_gamma),
        "orientation_spinor": orientation_spinor,
        "orientation_stokes": stokes_from_spinor(orientation_spinor),
        "orientation_hodograph": hodograph_from_spinor(
            orientation_spinor,
            fast_phase,
        ),
        "phase_varphi": np.full(sample_count, initial_varphi),
        "phase_lambda": np.full(sample_count, initial_lambda),
        "phase_gamma": phase_gamma,
        "phase_spinor": phase_spinor,
        "phase_stokes": stokes_from_spinor(phase_spinor),
        "phase_hodograph": hodograph_from_spinor(phase_spinor, fast_phase),
        "generator_parameter": generator_parameter,
        "generator_spinor_positive": generator_positive,
        "generator_spinor_negative": generator_negative,
        "generator_stokes_positive": stokes_from_spinor(generator_positive),
        "generator_stokes_negative": stokes_from_spinor(generator_negative),
        "generator_hodograph_positive": hodograph_from_spinor(
            generator_positive,
            fast_phase,
        ),
        "generator_hodograph_negative": hodograph_from_spinor(
            generator_negative,
            fast_phase,
        ),
    }
    return arrays


def metadata_from_dataset(
    arrays: dict[str, np.ndarray],
    validation: dict[str, float],
    sample_count: int,
    fast_phase_count: int,
) -> dict[str, Any]:
    """Create a portable metadata record without machine-specific paths."""
    initial_spinor = arrays["initial_spinor"]
    initial_down = np.conj(initial_spinor[1])
    initial_stokes = arrays["initial_stokes"]
    matrices = {
        name: [[complex_pair(value) for value in row] for row in SIGMA[index]]
        for index, name in enumerate(GENERATOR_NAMES)
    }
    return {
        "schema_version": 1,
        "artifact": "supplementary movie 1 calculation data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spinor_convention": SPINOR_CONVENTION,
        "physical_hodograph": ("phi(theta) = A_up exp(-i theta) + A_down exp(i theta)"),
        "stokes_definition": {
            "S_x": "2 Re(A_up A_down)",
            "S_y": "2 Im(A_up A_down)",
            "S_z": "|A_up|^2 - |A_down|^2",
            "|S|": "|A_up|^2 + |A_down|^2",
        },
        "matrices": matrices,
        "initial_state": {
            "source": (
                "Parameters reconstructed from the Figure 2 reference script; "
                "converted from its conjugated storage representation to the "
                "manuscript spinor convention."
            ),
            "A_up": complex_pair(initial_spinor[0]),
            "A_down": complex_pair(initial_down),
            "stored_conjugate_A_down": complex_pair(initial_spinor[1]),
            "spinor_radius": 1.1,
            "stokes_vector": [float(value) for value in initial_stokes],
            "stokes_magnitude": float(arrays["initial_stokes_magnitude"]),
            "varphi_radians": float(arrays["initial_varphi"]),
            "varphi_degrees": float(np.degrees(arrays["initial_varphi"])),
            "lambda_radians": float(arrays["initial_lambda"]),
            "lambda_degrees": float(np.degrees(arrays["initial_lambda"])),
            "gamma_radians": float(arrays["initial_gamma"]),
            "gamma_degrees": float(np.degrees(arrays["initial_gamma"])),
        },
        "chapter_1": {
            "stokes_magnitude": 1.0,
            "landmark_latitudes_radians": [
                float(np.pi / 2.0),
                float(arrays["initial_varphi"]),
                0.0,
                float(-arrays["initial_varphi"]),
                float(-np.pi / 2.0),
            ],
            "scans": {
                "ellipticity": "varphi varies; lambda and gamma are fixed",
                "orientation": "lambda advances by 2 pi; varphi and gamma are fixed",
                "common_phase": "gamma advances by 2 pi; Stokes vector is fixed",
            },
        },
        "chapter_2": {
            "evolution": "d|A>/dt = +/- f tau |A>/50",
            "solution": "|A(t)> = exp(+/- f t tau/50)|A(0)>",
            "generator_parameter_ft_range": [
                float(arrays["generator_parameter"][0]),
                float(arrays["generator_parameter"][-1]),
            ],
            "stokes_vectors_are_normalised": False,
            "unit_sphere_role": "scale reference only",
            "positive_branch": "blue solid",
            "negative_branch": "red dashed",
        },
        "sampling": {
            "state_samples_per_scan": sample_count,
            "fast_phase_samples_per_hodograph": fast_phase_count,
        },
        "display": {
            "reference_camera_azimuth_degrees": 110.75,
            "reference_camera_elevation_degrees": 30.85,
            "chapter_1_stokes_limits": [-1.55, 1.55],
            "chapter_1_hodograph_limits": [-1.55, 1.55],
            "chapter_2_stokes_limits": [-2.55, 2.55],
            "chapter_2_hodograph_limits": [-1.7, 1.7],
            "axes_are_fixed_within_each_chapter": True,
        },
        "validation": {
            "status": "passed",
            "tolerance": 2.0e-11,
            "metrics": validation,
        },
        "video_target": {
            "container": "MP4",
            "codec": "H.264",
            "pixel_format": "yuv420p",
            "resolution": [1920, 1080],
            "frame_rate_fps": 24,
            "audio": False,
            "maximum_file_size_mb": 50,
        },
    }


def validate_saved_outputs(
    data_path: Path,
    metadata_path: Path,
) -> dict[str, float]:
    """Re-run all mathematical checks against a saved archive."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing calculation archive: {data_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing calculation metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("spinor_convention") != SPINOR_CONVENTION:
        raise ValueError("The saved spinor convention changed.")
    with np.load(data_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "sigma",
        "fast_phase",
        "initial_spinor",
        "initial_stokes",
        "initial_varphi",
        "initial_lambda",
        "initial_gamma",
        "landmark_spinor",
        "landmark_stokes",
        "landmark_hodograph",
        "ellipticity_spinor",
        "ellipticity_stokes",
        "ellipticity_hodograph",
        "orientation_spinor",
        "orientation_stokes",
        "orientation_hodograph",
        "phase_spinor",
        "phase_stokes",
        "phase_hodograph",
        "generator_parameter",
        "generator_spinor_positive",
        "generator_spinor_negative",
        "generator_stokes_positive",
        "generator_stokes_negative",
        "generator_hodograph_positive",
        "generator_hodograph_negative",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"Calculation archive is missing: {', '.join(missing)}")
    for name, values in arrays.items():
        if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
    if not np.array_equal(arrays["sigma"], SIGMA):
        raise ValueError("The saved matrices differ from the manuscript definitions.")

    metrics = run_mathematical_validation(arrays)
    saved_metrics = metadata["validation"]["metrics"]
    for name, value in metrics.items():
        if name not in saved_metrics or not np.isclose(
            value,
            saved_metrics[name],
            rtol=1.0e-10,
            atol=2.0e-12,
        ):
            raise ValueError(f"Saved validation metric changed: {name}")
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute the polarisation trajectories for supplementary movie 1."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for the movie calculation archive and metadata.",
    )
    parser.add_argument("--sample-count", type=int, default=481)
    parser.add_argument("--fast-phase-count", type=int, default=361)
    return parser.parse_args()


def main() -> None:
    """Compute, validate and save the complete movie dataset."""
    args = parse_args()
    arrays = build_dataset(args.sample_count, args.fast_phase_count)
    validation = run_mathematical_validation(arrays)
    metadata = metadata_from_dataset(
        arrays,
        validation,
        args.sample_count,
        args.fast_phase_count,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    data_path = args.output_directory / DATA_FILENAME
    metadata_path = args.output_directory / METADATA_FILENAME
    np.savez_compressed(data_path, **arrays)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(data_path)
    print(metadata_path)
    print("mathematical validation: passed")


if __name__ == "__main__":
    main()
