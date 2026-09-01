from __future__ import annotations

import copy
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from reproduce import STATIC_WORKFLOWS
from common.spectrum_plotting import load_spectrum_colormap
from plot_wave_velocity_fields import row_color_limits, row_color_metadata
from render_wave_velocity_movie import resolve_executable
from run_workflow import (
    build_validation_command,
    load_workflow,
    remap_artifact_paths,
    resolve_workflow_output_directory,
)
from validate_outputs import (
    REFERENCE_METRICS,
    REPRODUCTION_CONFIG,
    expected_wave_figure_rows,
    validate_sinusoidal_dipole_wave_path,
    validate_wave_figure_metadata,
)


def test_remap_artifact_paths_handles_nested_configuration(tmp_path: Path) -> None:
    configuration = {
        "output": "artifacts/case/result.npz",
        "nested": ["unchanged.txt", "artifacts/case/figure"],
        "number": 3,
    }

    remapped = remap_artifact_paths(configuration, tmp_path)

    assert remapped["output"] == str(tmp_path / "case" / "result.npz")
    assert remapped["nested"] == [
        "unchanged.txt",
        str(tmp_path / "case" / "figure"),
    ]
    assert remapped["number"] == 3


def test_load_workflow_remaps_steps_and_validation(tmp_path: Path) -> None:
    workflow = load_workflow("background_flows", artifact_root=tmp_path)

    output = workflow["steps"][0]["arguments"]["output"]
    validation_data = workflow["validation"]["data"]
    assert output == str(
        tmp_path / "background_flows" / "data" / "background_flows.npz"
    )
    assert validation_data == output


def test_movie_default_output_directories_follow_artifact_root(tmp_path: Path) -> None:
    movie1 = load_workflow("polarisation_geometry_movie", artifact_root=tmp_path)
    movie2 = load_workflow("sinusoidal_dipole_movie", artifact_root=tmp_path)

    assert movie1["default_output_directory"] == str(tmp_path / "movie1")
    assert movie2["default_output_directory"] == str(
        tmp_path / "reproduction" / "full" / "movies"
    )


def test_movie_default_output_directory_is_used_for_execution(tmp_path: Path) -> None:
    workflow = load_workflow("polarisation_geometry_movie", artifact_root=tmp_path)

    assert resolve_workflow_output_directory(workflow, None) == tmp_path / "movie1"


def test_explicit_movie_output_directory_overrides_default(tmp_path: Path) -> None:
    workflow = load_workflow("polarisation_geometry_movie")
    requested = tmp_path / "custom-movie"

    assert resolve_workflow_output_directory(workflow, requested) == requested.resolve()


def test_validation_command_includes_every_path_override(tmp_path: Path) -> None:
    output_directory = tmp_path / "movie"
    artifact_root = tmp_path / "run"

    command = build_validation_command(
        "polarisation_geometry_movie",
        output_directory=output_directory,
        artifact_root=artifact_root,
        data_only=True,
    )

    assert command[-5:] == [
        "--output-directory",
        str(output_directory),
        "--artifact-root",
        str(artifact_root),
        "--data-only",
    ]


def test_unified_reproduction_covers_static_figure_workflows() -> None:
    assert STATIC_WORKFLOWS == (
        "polarisation_geometry",
        "background_flows",
        "parallel_shear",
        "gaussian_vortex",
    )


def test_repository_spectrum_colormap_is_finite_rgb_table() -> None:
    path = (
        Path(__file__).parents[1] / "code" / "common" / "custom_gradient_32_to_256.mat"
    )
    colors = np.asarray(load_spectrum_colormap(path).colors)

    assert colors.shape == (256, 3)
    assert np.all(np.isfinite(colors))
    assert np.all((0.0 <= colors) & (colors <= 1.0))


def test_figure10_n4_color_limits_match_movie2() -> None:
    fields = np.linspace(0.0, 40.0, 25).reshape(1, 5, 5)

    minimum, maximum, centered = row_color_limits(
        fields,
        target_time=50.0,
        vertical_mode=4,
    )

    assert (minimum, maximum, centered) == (0.01, 37.5, False)


def test_wave_figure_metadata_records_clipped_extrema() -> None:
    fields = np.asarray(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.5, 1.0], [1.5, 4.0]],
        ]
    )
    metadata = row_color_metadata(
        fields,
        np.asarray(["model-a", "model-b"]),
        target_time=10.0,
        vertical_mode=1,
        minimum=0.5,
        maximum=3.0,
        centered_on_one=False,
    )

    assert metadata["sample_count"] == 8
    assert metadata["field_minimum"] == 0.0
    assert metadata["field_maximum"] == 4.0
    assert metadata["below_color_limit_count"] == 1
    assert metadata["above_color_limit_count"] == 1


def test_wave_figure_metadata_is_cross_checked_against_archive(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "fields.npz"
    sidecar_path = tmp_path / "sinusoidal_dipole_wave_velocity_10IP.json"
    modes = np.asarray([1, 4, 8, 16, 32])
    names = np.asarray(REFERENCE_METRICS["model_names"])
    fields = np.linspace(0.2, 2.0, 2 * 5 * 5 * 3 * 3).reshape(2, 5, 5, 3, 3)
    np.savez_compressed(
        data_path,
        times_in_inertial_periods=np.asarray([10.0, 50.0]),
        vertical_modes=modes,
        model_names=names,
        squared_velocity=fields,
    )
    rows = expected_wave_figure_rows(data_path, target_time=10.0)
    metadata = {
        "schema_version": 1,
        "target_time_in_inertial_periods": 10.0,
        "model_names": names.tolist(),
        "rows": rows,
    }
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
    validate_wave_figure_metadata(sidecar_path, data_path=data_path)

    metadata["rows"][0]["field_minimum"] = 0.21
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_wave_figure_metadata(sidecar_path, data_path=data_path)


def test_wave_archive_maximum_tolerance_comes_from_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wave_fields.npz"
    times = np.asarray([10.0, 50.0])
    modes = np.asarray([1, 4, 8, 16, 32])
    names = np.asarray(REFERENCE_METRICS["model_names"])
    fields = np.zeros((2, 5, 5, 1, 1))
    expected_by_name = REFERENCE_METRICS["n4_50ip_squared_velocity_maxima"]
    fields[1, 1, :, 0, 0] = [expected_by_name[name] + 0.5 for name in names]
    np.savez_compressed(
        path,
        times_in_inertial_periods=times,
        vertical_modes=modes,
        vertical_wavelengths_m=np.asarray([4000, 1000, 500, 250, 125]),
        model_names=names,
        squared_velocity=fields,
        source_files=np.asarray(["simulation.h5"]),
    )

    validate_sinusoidal_dipole_wave_path(path)
    strict_config = copy.deepcopy(REPRODUCTION_CONFIG)
    strict_config["validation_tolerances"][
        "n4_50ip_squared_velocity_maxima_abs"
    ] = 0.1
    with pytest.raises(ValueError, match="maxima"):
        validate_sinusoidal_dipole_wave_path(path, strict_config)


def test_movie2_uses_bundled_ffmpeg_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = tmp_path / "bundled-ffmpeg.exe"
    bundled.write_bytes(b"bundled")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(bundled))
    monkeypatch.setattr(
        "render_wave_velocity_movie.shutil.which",
        lambda _: str(tmp_path / "system-ffmpeg.exe"),
    )

    assert resolve_executable(None, "ffmpeg") == bundled.resolve()
