"""Validate the configured numerical outputs and publication figures."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from run_workflow import ROOT, load_workflow, workflow_names

CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.files import validate_executable_provenance  # noqa: E402


REFERENCE_METRICS = json.loads(
    (ROOT / "config" / "reference_metrics.json").read_text(encoding="utf-8")
)
REPRODUCTION_CONFIG = json.loads(
    (ROOT / "config" / "reproduction.json").read_text(encoding="utf-8")
)
VALIDATION_TOLERANCES = REPRODUCTION_CONFIG["validation_tolerances"]


def require_keys(container: Any, keys: set[str], label: str) -> None:
    """Require a set of keys or columns in a data container."""
    available = set(container)
    missing = sorted(keys - available)
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(missing)}")


def require_finite(name: str, values: np.ndarray) -> None:
    """Require a non-empty array containing only finite values."""
    if values.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")


def validate_metadata(path: Path) -> dict[str, Any]:
    """Read and minimally validate a calculation metadata file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not metadata.get("background_flow"):
        raise ValueError(f"Missing background-flow metadata in {path}.")
    return metadata


def validate_vertical_mode_metadata(
    metadata: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    """Require the manuscript vertical-mode definition in saved metadata."""
    expected = {
        "domain_depth_m": reference["domain_depth_m"],
        "vertical_wavelength_m": reference["vertical_wavelength_m"],
        "normalisation": "unnormalised",
        "physical_reconstruction_factor": 1.0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Vertical-mode metadata changed: {key}.")
    if metadata.get("vertical_mode_definition") != "Z_n(z) = cos(n*pi*z/H)":
        raise ValueError("The vertical-mode definition is missing or inconsistent.")


def validate_parallel_shear(specification: dict[str, Any]) -> None:
    """Validate the parallel-shear eigenanalysis archive."""
    path = ROOT / specification["data"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing calculation output: {path}")

    required = {
        "along_stream_spectrum",
        "vertical_mode_spectrum",
        "overlay_coordinate",
        "overlay_wavenumbers",
        "overlay_component_up",
        "overlay_component_down",
        "overlay_frequencies",
        "overlay_ratios",
        "overlay_branch_indices",
    }
    reference = REFERENCE_METRICS["parallel_shear_figure5"]
    with np.load(path) as data:
        require_keys(data.files, required, path.name)
        along_stream = data["along_stream_spectrum"]
        vertical_modes = data["vertical_mode_spectrum"]
        coordinate = data["overlay_coordinate"]
        wavenumbers = data["overlay_wavenumbers"]
        component_up = data["overlay_component_up"]
        component_down = data["overlay_component_down"]
        frequencies = data["overlay_frequencies"]
        ratios = data["overlay_ratios"]
        branches = data["overlay_branch_indices"]

        for name, spectrum in (
            ("along_stream_spectrum", along_stream),
            ("vertical_mode_spectrum", vertical_modes),
        ):
            if spectrum.ndim != 2 or spectrum.shape[1] != 6:
                raise ValueError(f"{name} must have six columns.")
            require_finite(name, spectrum)

        require_finite("overlay_coordinate", coordinate)
        if coordinate.ndim != 1 or not np.all(np.diff(coordinate) > 0.0):
            raise ValueError("overlay_coordinate must be strictly increasing.")
        if component_up.shape != component_down.shape:
            raise ValueError("The two overlay components have different shapes.")
        expected_field_shape = (
            wavenumbers.size,
            frequencies.shape[1],
            coordinate.size,
        )
        if component_up.shape != expected_field_shape:
            raise ValueError("The overlay field dimensions are inconsistent.")
        if frequencies.shape != ratios.shape or frequencies.shape != branches.shape:
            raise ValueError("The overlay summary dimensions are inconsistent.")
        if np.any(ratios < 0.0):
            raise ValueError("overlay_ratios contains negative values.")
        for name, values in (
            ("overlay_component_up", component_up),
            ("overlay_component_down", component_down),
            ("overlay_frequencies", frequencies),
            ("overlay_ratios", ratios),
        ):
            require_finite(name, values)

        expected_wavenumbers = np.asarray(reference["wavenumbers_k_y_L"])
        if not np.array_equal(wavenumbers, expected_wavenumbers):
            raise ValueError("The Figure 5 along-stream wavenumbers changed.")
        normalized_frequencies = frequencies.real / 1.0e-4
        expected_frequencies = np.asarray(reference["frequencies_over_f"])
        if not np.allclose(
            normalized_frequencies,
            expected_frequencies,
            rtol=0.0,
            atol=VALIDATION_TOLERANCES["parallel_shear_frequency_over_f_abs"],
        ):
            raise ValueError("The Figure 5 branch frequencies changed.")
        if not np.all(np.diff(normalized_frequencies, axis=1) > 0.0):
            raise ValueError(
                "The Figure 5 branches are not in increasing-frequency order."
            )

    metadata = validate_metadata(ROOT / specification["metadata"])
    if metadata.get("vertical_mode") != reference["vertical_mode"]:
        raise ValueError("The parallel-shear vertical mode changed.")
    validate_vertical_mode_metadata(metadata, reference)


def validate_gaussian_vortex(specification: dict[str, Any]) -> None:
    """Validate the Gaussian-vortex eigenanalysis archive."""
    path = ROOT / specification["data"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing calculation output: {path}")

    required = {
        "vertical_mode_spectrum",
        "azimuthal_spectrum",
        "radial_domain",
        "mode_radius",
        "mode_azimuthal_wavenumbers",
        "mode_frequencies",
        "mode_branch_indices",
        "mode_component_up",
        "mode_component_down",
    }
    reference = REFERENCE_METRICS["gaussian_vortex_figure7"]
    with np.load(path) as data:
        require_keys(data.files, required, path.name)
        for name in ("vertical_mode_spectrum", "azimuthal_spectrum"):
            spectrum = data[name]
            if spectrum.ndim != 2 or spectrum.shape[1] != 5:
                raise ValueError(f"{name} must have five columns.")
            require_finite(name, spectrum)

        radius = data["mode_radius"]
        wavenumbers = data["mode_azimuthal_wavenumbers"]
        frequencies = data["mode_frequencies"]
        branches = data["mode_branch_indices"]
        component_up = data["mode_component_up"]
        component_down = data["mode_component_down"]
        require_finite("mode_radius", radius)
        if radius.ndim != 1 or not np.all(np.diff(radius) > 0.0):
            raise ValueError("mode_radius must be strictly increasing.")
        expected_shape = (wavenumbers.size, radius.size)
        if (
            component_up.shape != expected_shape
            or component_down.shape != expected_shape
        ):
            raise ValueError("The selected radial-mode dimensions are inconsistent.")
        if frequencies.shape != branches.shape or frequencies.size != wavenumbers.size:
            raise ValueError("The selected-mode summary dimensions are inconsistent.")
        for name, values in (
            ("mode_frequencies", frequencies),
            ("mode_component_up", component_up),
            ("mode_component_down", component_down),
        ):
            require_finite(name, values)

        expected_wavenumbers = np.asarray(reference["azimuthal_wavenumbers"])
        if np.array_equal(wavenumbers, expected_wavenumbers):
            expected = np.asarray(reference["frequencies_over_f"])
            normalized = frequencies.real / 1.0e-4
            if not np.allclose(
                normalized,
                expected,
                rtol=0.0,
                atol=VALIDATION_TOLERANCES[
                    "gaussian_vortex_frequency_over_f_abs"
                ],
            ):
                raise ValueError("The selected Gaussian-vortex frequencies changed.")
        else:
            raise ValueError("The selected Gaussian-vortex azimuthal modes changed.")

    metadata = validate_metadata(ROOT / specification["metadata"])
    if metadata.get("selected_mode_vertical_mode") != reference["vertical_mode"]:
        raise ValueError("The Gaussian-vortex vertical mode changed.")
    validate_vertical_mode_metadata(metadata, reference)


def validate_sinusoidal_dipole_error_path(path: Path) -> None:
    """Validate a sinusoidal-dipole error table at an explicit path."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing calculation output: {path}")

    table = pd.read_csv(path)
    required = {
        "Ro",
        "background_velocity_m_s",
        "vertical_wavelength_m",
        "vertical_mode",
        "model",
        "window",
        "mean_error",
        "mean_error_percent",
        "source_file",
    }
    require_keys(table.columns, required, path.name)
    expected_modes = list(range(1, 13)) + list(range(16, 33, 4))
    if sorted(table["vertical_mode"].unique()) != expected_modes:
        raise ValueError("The vertical-mode set in the error table changed.")
    if set(table["model"]) != {"YBJ", "TSB", "YBJ+", "PSE"}:
        raise ValueError("The model set in the error table changed.")
    if set(table["window"]) != {"0-10IP", "0-50IP"}:
        raise ValueError("The averaging-window set in the error table changed.")
    if table.shape[0] != 136:
        raise ValueError(f"Expected 136 error rows; found {table.shape[0]}.")
    if table.duplicated(["vertical_mode", "model", "window"]).any():
        raise ValueError("The error table contains duplicate model comparisons.")
    physical = REPRODUCTION_CONFIG["physical_parameters"]
    expected_velocity = float(physical["background_velocity_m_s"])
    expected_rossby = expected_velocity / (
        float(physical["coriolis_frequency_s-1"])
        * float(physical["background_length_scale_m"])
    )
    expected_wavelengths = (
        2.0
        * float(physical["domain_depth_m"])
        / table["vertical_mode"].to_numpy()
    )
    if not np.allclose(
        table["Ro"], expected_rossby, rtol=0.0, atol=np.finfo(float).eps
    ):
        raise ValueError("The Rossby number in the error table is inconsistent.")
    if not np.allclose(
        table["background_velocity_m_s"],
        expected_velocity,
        rtol=0.0,
        atol=np.finfo(float).eps,
    ):
        raise ValueError("The background velocity in the error table changed.")
    if not np.allclose(
        table["vertical_wavelength_m"],
        expected_wavelengths,
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise ValueError("The vertical wavelengths do not satisfy h=2H/n.")
    if not np.allclose(
        table["mean_error_percent"],
        100.0 * table["mean_error"],
        rtol=2.0e-15,
        atol=1.0e-14,
    ):
        raise ValueError("mean_error_percent is inconsistent with mean_error.")
    if set(table["source_file"]) != {"simulation.h5"}:
        raise ValueError("The error-table source file changed.")
    for column in ("mean_error", "mean_error_percent"):
        values = table[column].to_numpy()
        require_finite(column, values)
        if np.any(values < 0.0):
            raise ValueError(f"{column} contains negative values.")


def validate_sinusoidal_dipole_error(specification: dict[str, Any]) -> None:
    """Validate the configured sinusoidal-dipole error table."""
    validate_sinusoidal_dipole_error_path(ROOT / specification["data"])


def validate_sinusoidal_dipole_wave_path(
    path: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Validate a sinusoidal-dipole wave-field archive at an explicit path."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing calculation output: {path}")

    active_config = REPRODUCTION_CONFIG if config is None else config
    maxima_tolerance = active_config["validation_tolerances"][
        "n4_50ip_squared_velocity_maxima_abs"
    ]

    required = {
        "times_in_inertial_periods",
        "vertical_modes",
        "vertical_wavelengths_m",
        "model_names",
        "squared_velocity",
        "source_files",
    }
    with np.load(path) as data:
        require_keys(data.files, required, path.name)
        times = data["times_in_inertial_periods"]
        modes = data["vertical_modes"]
        wavelengths = data["vertical_wavelengths_m"]
        names = data["model_names"]
        fields = data["squared_velocity"]
        source_files = data["source_files"]

        if not np.array_equal(times, np.array([10.0, 50.0])):
            raise ValueError("The saved inertial-period times changed.")
        if not np.array_equal(modes, np.array([1, 4, 8, 16, 32])):
            raise ValueError("The saved vertical modes changed.")
        expected_names = REFERENCE_METRICS["model_names"]
        if list(names) != expected_names:
            raise ValueError("The saved model order changed.")
        expected_wavelengths = (
            2.0
            * float(active_config["physical_parameters"]["domain_depth_m"])
            / modes
        )
        if not np.allclose(
            wavelengths, expected_wavelengths, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("The wave archive does not satisfy h=2H/n.")
        grid_points = int(active_config["numerical_parameters"]["horizontal_grid"])
        expected_field_shape = (
            times.size,
            modes.size,
            names.size,
            grid_points,
            grid_points,
        )
        if fields.shape != expected_field_shape:
            raise ValueError("The wave-field dimensions are inconsistent.")
        if list(source_files) != ["simulation.h5"] * modes.size:
            raise ValueError("The wave archive source-file inventory changed.")
        require_finite("squared_velocity", fields)
        if np.any(fields < 0.0):
            raise ValueError("squared_velocity contains negative values.")

        time_index = int(np.flatnonzero(np.isclose(times, 50.0))[0])
        mode_index = int(np.flatnonzero(modes == 4)[0])
        maxima = fields[time_index, mode_index].max(axis=(-2, -1))
        expected_by_name = REFERENCE_METRICS["n4_50ip_squared_velocity_maxima"]
        expected = np.array([expected_by_name[name] for name in expected_names])
        if not np.allclose(maxima, expected, rtol=0.0, atol=maxima_tolerance):
            raise ValueError("The n=4 wave-field maxima at 50 IP changed.")


def validate_sinusoidal_dipole_wave(specification: dict[str, Any]) -> None:
    """Validate the configured sinusoidal-dipole wave-field archive."""
    validate_sinusoidal_dipole_wave_path(ROOT / specification["data"])


def validate_background_flows(specification: dict[str, Any]) -> None:
    """Validate the analytic fields used to plot Figure 3."""
    path = ROOT / specification["data"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing calculation output: {path}")
    required = {
        "flow_names",
        "coordinate_over_length",
        "velocity_u_over_reference",
        "velocity_v_over_reference",
        "speed_over_reference",
        "xi1_over_f",
        "xi2_over_f",
        "xi3_over_f",
        "sampled_v_over_reference_at_y0",
        "rossby_number",
    }
    grid_points = int(specification["grid_points"])
    rossby_number = float(specification["rossby_number"])
    module = load_script_module(
        "background_flows_compute_validation",
        ROOT / "code" / "background_flows" / "compute_background_flows.py",
    )
    expected = module.compute_fields(grid_points, rossby_number)
    with np.load(path) as data:
        require_keys(data.files, required, path.name)
        coordinate = data["coordinate_over_length"]
        u = data["velocity_u_over_reference"]
        v = data["velocity_v_over_reference"]
        speed = data["speed_over_reference"]
        if coordinate.shape != (grid_points,) or not np.all(np.diff(coordinate) > 0.0):
            raise ValueError("The Figure 3 coordinate must be strictly increasing.")
        expected_shape = (3, coordinate.size, coordinate.size)
        if u.shape != expected_shape or v.shape != expected_shape:
            raise ValueError("The Figure 3 velocity dimensions are inconsistent.")
        if not np.allclose(speed, np.sqrt(u**2 + v**2), rtol=0.0, atol=1e-14):
            raise ValueError("The saved Figure 3 speed is inconsistent with velocity.")
        for name in ("xi1_over_f", "xi2_over_f", "xi3_over_f"):
            values = data[name]
            if values.shape != expected_shape:
                raise ValueError(f"{name} has inconsistent dimensions.")
            require_finite(name, values)
        if not np.allclose(data["xi2_over_f"][[0, 2]], 0.0, atol=1e-15):
            raise ValueError("The shear and dipole normal strains must vanish.")
        for name, expected_values in expected.items():
            if not np.array_equal(data[name], expected_values):
                raise ValueError(f"The saved Figure 3 field changed: {name}.")
    metadata = validate_metadata(ROOT / specification["metadata"])
    expected_metadata = {
        "schema_version": 1,
        "background_flow": "analytic examples used in Figure 3",
        "flow_order": expected["flow_names"].tolist(),
        "coordinate_convention": "x/L and y/L in [-pi, pi)",
        "rossby_number": rossby_number,
        "fields": ["|U|/U_ref", "xi1/f", "xi2/f", "xi3/f"],
        "sampling_line": "y/L=0",
        "external_data": False,
    }
    if metadata != expected_metadata:
        raise ValueError("The Figure 3 metadata changed.")


VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "parallel_shear": validate_parallel_shear,
    "gaussian_vortex": validate_gaussian_vortex,
    "sinusoidal_dipole_error": validate_sinusoidal_dipole_error,
    "sinusoidal_dipole_wave": validate_sinusoidal_dipole_wave,
    "background_flows": validate_background_flows,
}


def load_script_module(name: str, path: Path) -> Any:
    """Load a repository script without making ``code`` a Python package."""
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load script module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def validate_polarisation_geometry(specification: dict[str, Any]) -> None:
    """Validate the shared calculation data behind Figures 1 and 2."""
    module = load_script_module(
        "polarisation_geometry_compute",
        ROOT
        / "code"
        / "polarisation_geometry"
        / "compute_polarisation_trajectories.py",
    )
    metrics = module.validate_saved_outputs(
        ROOT / specification["data"], ROOT / specification["metadata"]
    )
    if not metrics:
        raise ValueError("No polarisation-geometry validation metrics were produced.")


def validate_polarisation_geometry_movie(
    specification: dict[str, Any],
    *,
    output_directory: Path | None,
    data_only: bool,
) -> None:
    """Validate the movie-1 calculation, sidecars and encoded video."""
    if output_directory is None:
        raise ValueError(
            "polarisation_geometry_movie validation requires --output-directory."
        )
    data_path = output_directory / specification["data_filename"]
    metadata_path = output_directory / specification["metadata_filename"]
    compute_module = load_script_module(
        "polarisation_movie_compute",
        ROOT
        / "code"
        / "polarisation_geometry"
        / "compute_polarisation_trajectories.py",
    )
    metrics = compute_module.validate_saved_outputs(data_path, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["validation"]["status"] != "passed":
        raise ValueError("The saved mathematical-validation status is not passed.")
    if data_only:
        print(
            "polarisation_geometry_movie: mathematical validation passed "
            f"({len(metrics)} metrics)"
        )
        return

    encoding_environment = metadata.get("encoding_environment")
    if not isinstance(encoding_environment, dict) or set(encoding_environment) != {
        "ffmpeg"
    }:
        raise ValueError("Movie 1 metadata lacks media-tool provenance.")
    validate_executable_provenance(
        encoding_environment["ffmpeg"],
        label="Movie 1 FFmpeg",
    )

    required = [
        specification["movie_filename"],
        *specification["required_sidecars"],
    ]
    for filename in required:
        path = output_directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty movie-1 artifact: {path}")

    preview_path = output_directory / "movie1_preview.png"
    expected_preview_size = tuple(
        metadata.get("video", {}).get(
            key,
            metadata["video_target"]["resolution"][index],
        )
        for index, key in enumerate(("width", "height"))
    )
    with Image.open(preview_path) as preview:
        if preview.size != expected_preview_size:
            raise ValueError(
                "movie1_preview.png has size "
                f"{preview.size}; expected {expected_preview_size}."
            )

    caption = (output_directory / "movie1_caption.txt").read_text(encoding="utf-8")
    if not caption.lower().startswith("movie 1."):
        raise ValueError("The caption is not explicitly titled movie 1.")
    if caption.count("$$") % 2:
        raise ValueError("The movie caption contains unpaired $$ TeX delimiters.")

    render_module = load_script_module(
        "polarisation_movie_render",
        ROOT / "code" / "polarisation_geometry" / "render_polarisation_movie.py",
    )
    ffmpeg = render_module.locate_ffmpeg(None)
    movie_path = output_directory / specification["movie_filename"]
    video = render_module.probe_video(ffmpeg, movie_path)
    saved_video = metadata.get("video")
    if not saved_video:
        raise ValueError("The movie metadata has no encoded-video record.")
    render_module.validate_video_target(
        video,
        width=int(saved_video["width"]),
        height=int(saved_video["height"]),
        fps=int(round(saved_video["frame_rate_fps"])),
        expected_frames=int(saved_video["expected_frame_count"]),
    )
    for key in (
        "codec",
        "profile",
        "pixel_format",
        "width",
        "height",
        "frame_count",
        "audio_stream_present",
        "file_size_bytes",
        "faststart_moov_before_mdat",
    ):
        if video[key] != saved_video[key]:
            raise ValueError(f"The probed video field differs from metadata: {key}")
    print(
        "polarisation_geometry_movie: validation passed "
        f"({video['width']}x{video['height']}, "
        f"{video['frame_rate_fps']:g} fps, "
        f"{video['file_size_mb']:.3f} MB)"
    )


def validate_sinusoidal_dipole_movie(
    specification: dict[str, Any],
    *,
    output_directory: Path | None,
    data_only: bool,
) -> None:
    """Validate the Movie 2 numerical archive and, when requested, the MP4."""
    input_path = ROOT / specification["data"]
    module = load_script_module(
        "sinusoidal_dipole_movie_validation",
        ROOT / "code" / "sinusoidal_dipole" / "validate_wave_velocity_movie.py",
    )
    numerical = module.check_archive(input_path)
    if data_only:
        print(
            "sinusoidal_dipole_movie: numerical validation passed "
            f"({len(numerical)} checks)"
        )
        return
    if output_directory is None:
        raise ValueError(
            "sinusoidal_dipole_movie validation requires --output-directory."
        )
    module.validate_product(input_path, output_directory)


def validate_figure_stems(stems: Iterable[Path]) -> list[Path]:
    """Require decodable PNG and identifiable PDF output for figure stems."""
    outputs: list[Path] = []
    for stem in stems:
        for suffix in (".png", ".pdf"):
            path = stem.with_suffix(suffix)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing figure output: {path}")
            if suffix == ".png":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if image.width < 1 or image.height < 1:
                            raise ValueError(
                                f"Figure PNG has invalid dimensions: {path}"
                            )
                        image.verify()
            elif path.read_bytes()[:5] != b"%PDF-":
                raise ValueError(f"Figure PDF has an invalid header: {path}")
            outputs.append(path)
    return outputs


def validate_figures(specification: dict[str, Any]) -> None:
    """Require PNG and PDF output for every configured figure stem."""
    validate_figure_stems(
        ROOT / relative_stem
        for relative_stem in specification.get("figure_stems", [])
    )
    data_path = ROOT / specification["data"]
    for relative_path in specification.get("figure_metadata", []):
        validate_wave_figure_metadata(ROOT / relative_path, data_path=data_path)


def expected_wave_figure_rows(
    data_path: Path,
    *,
    target_time: float,
) -> list[dict[str, object]]:
    """Recompute Figure 9 or 10 row metadata from the numerical archive."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing wave-field archive: {data_path}")
    module = load_script_module(
        "sinusoidal_dipole_wave_figure_metadata",
        ROOT / "code" / "sinusoidal_dipole" / "plot_wave_velocity_fields.py",
    )
    with np.load(data_path) as data:
        require_keys(
            data.files,
            {
                "times_in_inertial_periods",
                "vertical_modes",
                "model_names",
                "squared_velocity",
            },
            data_path.name,
        )
        times = data["times_in_inertial_periods"]
        time_indices = np.flatnonzero(np.isclose(times, target_time))
        if time_indices.size != 1:
            raise ValueError(
                f"Wave-field archive must contain one {target_time:g}-IP state."
            )
        vertical_modes = data["vertical_modes"]
        model_names = data["model_names"]
        fields = data["squared_velocity"][int(time_indices[0])]

    rows: list[dict[str, object]] = []
    for row_index, vertical_mode in enumerate(vertical_modes):
        minimum, maximum, centered_on_one = module.row_color_limits(
            fields[row_index],
            target_time=target_time,
            vertical_mode=int(vertical_mode),
        )
        rows.append(
            module.row_color_metadata(
                fields[row_index],
                model_names,
                target_time=target_time,
                vertical_mode=int(vertical_mode),
                minimum=minimum,
                maximum=maximum,
                centered_on_one=centered_on_one,
            )
        )
    return rows


def validate_wave_figure_metadata(
    path: Path,
    *,
    data_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a Figure 9 or 10 sidecar and optionally its source data."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing figure metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported figure-metadata schema in {path}.")
    if metadata.get("model_names") != REFERENCE_METRICS["model_names"]:
        raise ValueError(f"The model order changed in {path}.")
    target_time = metadata.get("target_time_in_inertial_periods")
    if target_time not in (10.0, 50.0) or not path.stem.endswith(f"{target_time:g}IP"):
        raise ValueError(f"The figure time is inconsistent in {path}.")
    rows = metadata.get("rows")
    if not isinstance(rows, list) or [row.get("vertical_mode") for row in rows] != [
        1,
        4,
        8,
        16,
        32,
    ]:
        raise ValueError(f"The vertical-mode rows changed in {path}.")
    for row in rows:
        limits = np.asarray(row.get("color_limits"), dtype=float)
        extrema = np.asarray(
            [row.get("field_minimum"), row.get("field_maximum")], dtype=float
        )
        fractions = np.asarray(
            [
                row.get("below_color_limit_fraction"),
                row.get("above_color_limit_fraction"),
            ],
            dtype=float,
        )
        if limits.shape != (2,) or not np.all(np.isfinite(limits)):
            raise ValueError(f"Invalid color limits in {path}.")
        if not limits[0] < limits[1]:
            raise ValueError(f"Non-increasing color limits in {path}.")
        if not np.all(np.isfinite(extrema)) or extrema[0] > extrema[1]:
            raise ValueError(f"Invalid field extrema in {path}.")
        if not np.all(np.isfinite(fractions)) or np.any(
            (fractions < 0) | (fractions > 1)
        ):
            raise ValueError(f"Invalid clipped fractions in {path}.")
        sample_count = row.get("sample_count")
        below_count = row.get("below_color_limit_count")
        above_count = row.get("above_color_limit_count")
        if not all(
            isinstance(value, int) for value in (sample_count, below_count, above_count)
        ):
            raise ValueError(f"Invalid sample counts in {path}.")
        if sample_count <= 0 or below_count < 0 or above_count < 0:
            raise ValueError(f"Invalid sample counts in {path}.")
        if below_count + above_count > sample_count:
            raise ValueError(f"Clipped counts exceed the sample count in {path}.")
        model_extrema = row.get("model_extrema")
        if not isinstance(model_extrema, dict) or set(model_extrema) != set(
            metadata["model_names"]
        ):
            raise ValueError(f"Invalid model extrema in {path}.")
        for extrema_by_model in model_extrema.values():
            values = np.asarray(
                [extrema_by_model.get("minimum"), extrema_by_model.get("maximum")],
                dtype=float,
            )
            if not np.all(np.isfinite(values)) or values[0] > values[1]:
                raise ValueError(f"Invalid model extrema in {path}.")
        expected_fractions = np.asarray(
            [below_count / sample_count, above_count / sample_count]
        )
        if not np.allclose(fractions, expected_fractions, rtol=0.0, atol=1.0e-15):
            raise ValueError(f"Inconsistent clipped fractions in {path}.")
    if data_path is not None:
        expected_rows = expected_wave_figure_rows(
            data_path,
            target_time=float(target_time),
        )
        if rows != expected_rows:
            raise ValueError(
                f"Figure metadata does not match its wave-field archive: {path}."
            )
    return metadata


def validate_workflow(
    name: str,
    *,
    data_only: bool,
    output_directory: Path | None = None,
    artifact_root: Path | None = None,
) -> None:
    """Validate one configured workflow."""
    workflow = load_workflow(name, artifact_root=artifact_root)
    specification = workflow["validation"]
    if specification.get("uses_output_directory") and output_directory is None:
        configured_directory = workflow.get("default_output_directory")
        if configured_directory is None:
            raise ValueError(f"Workflow {name!r} has no default output directory.")
        output_directory = Path(configured_directory)
        if not output_directory.is_absolute():
            output_directory = ROOT / output_directory
    if specification["type"] == "polarisation_geometry":
        validate_polarisation_geometry(specification)
        if not data_only:
            validate_figures(specification)
        print(f"{name}: validation passed")
        return
    if specification["type"] == "polarisation_geometry_movie":
        validate_polarisation_geometry_movie(
            specification,
            output_directory=output_directory,
            data_only=data_only,
        )
        return
    if specification["type"] == "sinusoidal_dipole_movie":
        validate_sinusoidal_dipole_movie(
            specification,
            output_directory=output_directory,
            data_only=data_only,
        )
        return
    validator = VALIDATORS[specification["type"]]
    validator(specification)
    if not data_only:
        validate_figures(specification)
    print(f"{name}: validation passed")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    names = workflow_names()
    parser = argparse.ArgumentParser(
        description="Validate configured numerical outputs and figures."
    )
    parser.add_argument(
        "workflow",
        choices=[*names, "all"],
        help="Workflow to validate, or 'all'.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Validate numerical data without requiring figure files.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Artifact directory override for one movie workflow.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Root used to remap configured artifacts/... paths.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate one workflow or all configured workflows."""
    args = parse_args()
    if args.workflow == "all" and args.output_directory is not None:
        raise ValueError(
            "--output-directory cannot be combined with 'all'; "
            "each movie workflow uses its configured default directory."
        )
    names = workflow_names() if args.workflow == "all" else [args.workflow]
    for name in names:
        validate_workflow(
            name,
            data_only=args.data_only,
            output_directory=args.output_directory,
            artifact_root=(
                args.artifact_root.resolve() if args.artifact_root is not None else None
            ),
        )


if __name__ == "__main__":
    main()
