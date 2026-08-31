from __future__ import annotations

from pathlib import Path

import numpy as np

from reproduce import STATIC_WORKFLOWS
from common.spectrum_plotting import load_spectrum_colormap
from plot_wave_velocity_fields import row_color_limits, row_color_metadata
from run_workflow import (
    build_validation_command,
    load_workflow,
    remap_artifact_paths,
    resolve_workflow_output_directory,
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
