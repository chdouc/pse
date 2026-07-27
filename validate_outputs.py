"""Validate the configured numerical outputs and publication figures."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image

from run_workflow import ROOT, load_workflow, workflow_names


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

    validate_metadata(ROOT / specification["metadata"])


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

        if np.array_equal(wavenumbers, np.array([0, 1, 2, 3])):
            expected = np.array([-0.10078016, -0.17202856, -0.22812337, -0.27251197])
            normalized = frequencies.real / 1.0e-4
            if not np.allclose(normalized, expected, rtol=0.0, atol=5.0e-6):
                raise ValueError("The selected Gaussian-vortex frequencies changed.")

    validate_metadata(ROOT / specification["metadata"])


def validate_sinusoidal_dipole_error(specification: dict[str, Any]) -> None:
    """Validate the sinusoidal-dipole error table."""
    path = ROOT / specification["data"]
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
    if not np.allclose(table["background_velocity_m_s"], 0.25):
        raise ValueError("The background velocity in the error table changed.")
    for column in ("mean_error", "mean_error_percent"):
        values = table[column].to_numpy()
        require_finite(column, values)
        if np.any(values < 0.0):
            raise ValueError(f"{column} contains negative values.")


def validate_sinusoidal_dipole_wave(specification: dict[str, Any]) -> None:
    """Validate the sinusoidal-dipole wave-field archive."""
    path = ROOT / specification["data"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing calculation output: {path}")

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
        names = data["model_names"]
        fields = data["squared_velocity"]

        if not np.array_equal(times, np.array([10.0, 50.0])):
            raise ValueError("The saved inertial-period times changed.")
        if not np.array_equal(modes, np.array([1, 4, 8, 16, 32])):
            raise ValueError("The saved vertical modes changed.")
        if list(names) != ["YBJ", "TSB", "YBJ+", "PSE", "HBEs"]:
            raise ValueError("The saved model order changed.")
        if fields.ndim != 5 or fields.shape[:3] != (
            times.size,
            modes.size,
            names.size,
        ):
            raise ValueError("The wave-field dimensions are inconsistent.")
        require_finite("squared_velocity", fields)
        if np.any(fields < 0.0):
            raise ValueError("squared_velocity contains negative values.")

        time_index = int(np.flatnonzero(np.isclose(times, 50.0))[0])
        mode_index = int(np.flatnonzero(modes == 4)[0])
        maxima = fields[time_index, mode_index].max(axis=(-2, -1))
        expected = np.array(
            [
                31.786428361858587,
                31.786428864178912,
                37.239008041643494,
                37.54282297563831,
                27.56445542781792,
            ]
        )
        if not np.allclose(maxima, expected, rtol=0.0, atol=1.0e-6):
            raise ValueError("The n=4 wave-field maxima at 50 IP changed.")


VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "parallel_shear": validate_parallel_shear,
    "gaussian_vortex": validate_gaussian_vortex,
    "sinusoidal_dipole_error": validate_sinusoidal_dipole_error,
    "sinusoidal_dipole_wave": validate_sinusoidal_dipole_wave,
}


def load_script_module(name: str, path: Path) -> Any:
    """Load a repository script without making ``code`` a Python package."""
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load script module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


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

    caption = (output_directory / "movie1_caption.txt").read_text(
        encoding="utf-8"
    )
    if not caption.lower().startswith("movie 1."):
        raise ValueError("The caption is not explicitly titled movie 1.")
    if caption.count("$$") % 2:
        raise ValueError("The movie caption contains unpaired $$ TeX delimiters.")

    render_module = load_script_module(
        "polarisation_movie_render",
        ROOT
        / "code"
        / "polarisation_geometry"
        / "render_polarisation_movie.py",
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


def validate_figures(specification: dict[str, Any]) -> None:
    """Require PNG and PDF output for every configured figure stem."""
    for relative_stem in specification.get("figure_stems", []):
        stem = ROOT / relative_stem
        for suffix in (".png", ".pdf"):
            path = stem.with_suffix(suffix)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing figure output: {path}")


def validate_workflow(
    name: str,
    *,
    data_only: bool,
    output_directory: Path | None = None,
) -> None:
    """Validate one configured workflow."""
    workflow = load_workflow(name)
    specification = workflow["validation"]
    if specification["type"] == "polarisation_geometry_movie":
        validate_polarisation_geometry_movie(
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
        help="External artifact directory used by movie workflows.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate one workflow or all configured workflows."""
    args = parse_args()
    names = workflow_names() if args.workflow == "all" else [args.workflow]
    for name in names:
        validate_workflow(
            name,
            data_only=args.data_only,
            output_directory=args.output_directory,
        )


if __name__ == "__main__":
    main()
