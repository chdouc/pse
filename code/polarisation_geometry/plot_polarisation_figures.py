"""Generate the Poincare-sphere and hodograph diagrams in Figures 1 and 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Wedge
import numpy as np

from compute_polarisation_trajectories import hodograph_from_spinor
from render_polarisation_movie import (
    BLACK,
    BLUE,
    RED,
    TRIANGULAR_ARROW_STYLE,
    draw_hodograph_axes,
    draw_poincare_sphere,
    project_stokes,
    publication_style,
)


def arrow(
    axis: mpl.axes.Axes,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    linewidth: float = 1.8,
    scale: float = 12.0,
    zorder: float = 8.0,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle=TRIANGULAR_ARROW_STYLE,
            mutation_scale=scale,
            linewidth=linewidth,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
    )


def sphere_axis_labels(axis: mpl.axes.Axes, limit: float) -> None:
    for vector, label, alignment in (
        (np.array([1.0, 0.0, 0.0]), r"$x$", "right"),
        (np.array([0.0, 1.0, 0.0]), r"$y$", "left"),
        (np.array([0.0, 0.0, 1.0]), r"$z$", "center"),
    ):
        endpoints, _ = project_stokes(np.vstack([np.zeros(3), 0.82 * limit * vector]))
        position = 1.05 * endpoints[1]
        axis.text(
            position[0],
            position[1],
            label,
            ha=alignment,
            va="center",
            fontsize=10,
        )


def angle_wedge(
    axis: mpl.axes.Axes,
    centre: tuple[float, float],
    radius: float,
    start: float,
    stop: float,
    *,
    edge: str,
    face: str,
    label: str,
    label_radius: float = 0.68,
) -> None:
    start_degrees = np.degrees(start)
    stop_degrees = np.degrees(stop)
    while stop_degrees < start_degrees:
        stop_degrees += 360.0
    axis.add_patch(
        Wedge(
            centre,
            radius,
            start_degrees,
            stop_degrees,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
            alpha=0.8,
            zorder=5,
        )
    )
    middle = 0.5 * (start + stop)
    axis.text(
        centre[0] + label_radius * radius * np.cos(middle),
        centre[1] + label_radius * radius * np.sin(middle),
        label,
        ha="center",
        va="center",
        fontsize=9,
        zorder=9,
    )


def plot_figure_1(data: dict[str, np.ndarray], output_stem: Path) -> None:
    stokes = data["figure1_stokes"]
    direction = stokes / np.linalg.norm(stokes)
    fast_phase = data["fast_phase"]
    hodograph = data["figure1_hodograph"]
    varphi = float(data["figure1_varphi"])
    longitude = float(data["figure1_lambda"])

    with mpl.rc_context(publication_style(use_tex=False)):
        figure, (sphere_axis, hodograph_axis) = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.25),
            gridspec_kw={
                "left": 0.035,
                "right": 0.985,
                "bottom": 0.08,
                "top": 0.95,
                "wspace": 0.16,
            },
        )
        draw_poincare_sphere(
            sphere_axis,
            limit=1.45,
            longitude=longitude,
            label_axes=False,
        )
        sphere_axis_labels(sphere_axis, 1.45)
        projected_direction, _ = project_stokes(direction[None, :])
        projected_equator, _ = project_stokes(
            np.array([[np.cos(longitude), np.sin(longitude), 0.0]])
        )
        tip = projected_direction[0]
        equator_tip = projected_equator[0]
        arrow(
            sphere_axis,
            np.zeros(2),
            tip,
            color="#c84d49",
            linewidth=3.0,
            scale=15,
        )
        sphere_axis.plot(
            tip[0],
            tip[1],
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=BLACK,
            zorder=10,
        )
        sphere_axis.plot(
            [0.0, equator_tip[0], tip[0]],
            [0.0, equator_tip[1], tip[1]],
            color="0.35",
            linewidth=0.8,
            zorder=4,
        )
        sphere_axis.text(
            0.53 * tip[0] - 0.08,
            0.53 * tip[1],
            r"$\mathbf{S}$",
            fontsize=11,
        )
        sphere_axis.text(-1.33, 1.25, r"$(a)$", fontsize=11)

        x_projection, _ = project_stokes(np.array([[1.0, 0.0, 0.0]]))
        x_angle = float(np.arctan2(x_projection[0, 1], x_projection[0, 0]))
        equator_angle = float(np.arctan2(equator_tip[1], equator_tip[0]))
        tip_angle = float(np.arctan2(tip[1], tip[0]))
        angle_wedge(
            sphere_axis,
            (0.0, 0.0),
            0.34,
            x_angle,
            equator_angle,
            edge="#498b61",
            face="#b9d8bf",
            label=r"$\lambda$",
        )
        angle_wedge(
            sphere_axis,
            (0.0, 0.0),
            0.46,
            equator_angle,
            tip_angle,
            edge="#bd7a34",
            face="#ecd7ad",
            label=r"$\varphi$",
        )
        sphere_axis.add_patch(
            Arc(
                tuple(tip),
                0.30,
                0.14,
                angle=20.0,
                theta1=15.0,
                theta2=330.0,
                color="#2672ae",
                linewidth=1.5,
                zorder=9,
            )
        )
        sphere_axis.text(tip[0] + 0.18, tip[1] + 0.06, r"$\gamma$", fontsize=10)

        draw_hodograph_axes(hodograph_axis, limit=1.55, ticks=False)
        for spine in hodograph_axis.spines.values():
            spine.set_visible(False)
        hodograph_axis.set_xlabel("")
        hodograph_axis.set_ylabel("")
        axis_limit = 1.48
        arrow(
            hodograph_axis,
            np.array([-axis_limit, 0.0]),
            np.array([axis_limit, 0.0]),
            color=BLACK,
            linewidth=1.2,
        )
        arrow(
            hodograph_axis,
            np.array([0.0, -axis_limit]),
            np.array([0.0, axis_limit]),
            color=BLACK,
            linewidth=1.2,
        )
        hodograph_axis.text(
            axis_limit - 0.02,
            0.10,
            r"$\operatorname{Re}\phi$",
            ha="right",
            fontsize=10,
        )
        hodograph_axis.text(
            0.08,
            axis_limit - 0.02,
            r"$\operatorname{Im}\phi$",
            va="top",
            fontsize=10,
        )
        hodograph_axis.plot(hodograph.real, hodograph.imag, color=BLACK, linewidth=2.2)
        marker = hodograph[0]
        hodograph_axis.plot(
            marker.real,
            marker.imag,
            marker="o",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=BLACK,
            zorder=10,
        )
        hodograph_axis.text(
            marker.real + 0.03,
            marker.imag + 0.13,
            r"$\phi(\boldsymbol{x},0)$",
            fontsize=10,
        )

        major_angle = 0.5 * longitude
        major = 1.45 * np.array([np.cos(major_angle), np.sin(major_angle)])
        hodograph_axis.plot(
            [-major[0], major[0]],
            [-major[1], major[1]],
            color="0.55",
            linewidth=0.8,
            linestyle=(0, (3.0, 2.0)),
        )
        marker_angle = float(np.angle(marker))
        angle_wedge(
            hodograph_axis,
            (0.0, 0.0),
            0.48,
            0.0,
            major_angle,
            edge="#498b61",
            face="#b9d8bf",
            label=r"$\lambda/2$",
        )
        angle_wedge(
            hodograph_axis,
            (0.0, 0.0),
            0.68,
            major_angle,
            marker_angle,
            edge="#2672ae",
            face="#b9d7ea",
            label=r"$\gamma$",
        )
        chord_indices = (
            int(0.56 * (fast_phase.size - 1)),
            int(0.82 * (fast_phase.size - 1)),
        )
        chord = hodograph[list(chord_indices)]
        hodograph_axis.plot(chord.real, chord.imag, color="#b2182b", linewidth=2.0)
        chord_midpoint = 0.5 * (chord[0] + chord[1])
        hodograph_axis.text(
            chord_midpoint.real,
            chord_midpoint.imag - 0.16,
            r"$|\mathbf{S}|$",
            ha="center",
            fontsize=9,
        )
        angle_wedge(
            hodograph_axis,
            (float(chord[0].real), float(chord[0].imag)),
            0.24,
            0.0,
            abs(varphi) / 2.0,
            edge="#bd7a34",
            face="#ecd7ad",
            label=r"$\varphi/2$",
            label_radius=0.95,
        )
        hodograph_axis.text(-1.47, 1.25, r"$(b)$", fontsize=11)

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02
        )
        figure.savefig(
            output_stem.with_suffix(".png"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(figure)


def blend(color: str, fraction: float) -> tuple[float, float, float]:
    rgb = np.asarray(mpl.colors.to_rgb(color))
    return tuple((fraction * rgb).tolist())


def plot_figure_2(data: dict[str, np.ndarray], output_stem: Path) -> None:
    positive_stokes = data["generator_stokes_positive"]
    negative_stokes = data["generator_stokes_negative"]
    positive_hodographs = data["generator_hodograph_positive"]
    negative_hodographs = data["generator_hodograph_negative"]
    initial_stokes = data["initial_stokes"]
    initial_hodograph = hodograph_from_spinor(
        data["initial_spinor"], data["fast_phase"]
    )
    selected = np.linspace(0, positive_stokes.shape[1] - 1, 7, dtype=int)

    with mpl.rc_context(publication_style(use_tex=False)):
        figure, axes = plt.subplots(
            2,
            4,
            figsize=(7.35, 4.45),
            gridspec_kw={
                "left": 0.065,
                "right": 0.985,
                "bottom": 0.12,
                "top": 0.94,
                "wspace": 0.26,
                "hspace": 0.25,
            },
        )
        titles = (
            r"$\sigma_0$: $r$-change",
            r"$\sigma_1$: $z$-rotation",
            r"$\sigma_2$: $x$-translation",
            r"$\sigma_3$: $y$-translation",
        )
        panel_letters = "abcdefgh"
        initial_projected, _ = project_stokes(initial_stokes[None, :])
        for column in range(4):
            sphere_axis = axes[0, column]
            hodograph_axis = axes[1, column]
            draw_poincare_sphere(
                sphere_axis,
                limit=1.65,
                longitude=np.pi / 3.0,
                label_axes=False,
                dashed_hidden_guides=False,
            )
            sphere_axis_labels(sphere_axis, 1.65)
            sphere_axis.set_title(titles[column], pad=5)
            arrow(
                sphere_axis,
                np.zeros(2),
                initial_projected[0],
                color=BLACK,
                linewidth=1.2,
                scale=10,
            )
            sphere_axis.plot(
                initial_projected[0, 0],
                initial_projected[0, 1],
                marker="o",
                markersize=5.5,
                markerfacecolor="white",
                markeredgecolor=BLACK,
                zorder=11,
            )
            for trajectory, color in (
                (positive_stokes[column], BLUE),
                (negative_stokes[column], RED),
            ):
                projected, _ = project_stokes(trajectory)
                vector = projected[9] - projected[0]
                norm = np.linalg.norm(vector)
                if norm > 0.0:
                    vector = 0.28 * vector / norm
                    arrow(
                        sphere_axis,
                        projected[0],
                        projected[0] + vector,
                        color=color,
                        linewidth=2.1,
                        scale=14,
                    )
            sphere_axis.text(
                -1.52,
                1.37,
                rf"$({panel_letters[column]})$",
                fontsize=10,
            )

            draw_hodograph_axes(hodograph_axis, limit=2.02, ticks=True)
            hodograph_axis.set_title(rf"$\phi$ track for $\sigma_{column}$", pad=5)
            if column > 0:
                hodograph_axis.set_ylabel("")
                hodograph_axis.tick_params(labelleft=False)
            for branch, color in (
                (negative_hodographs[column], RED),
                (positive_hodographs[column], BLUE),
            ):
                for order, state_index in enumerate(selected[1:], start=1):
                    curve = branch[state_index]
                    fraction = order / (selected.size - 1)
                    hodograph_axis.plot(
                        curve.real,
                        curve.imag,
                        color=blend(color, fraction),
                        linewidth=1.3,
                    )
            hodograph_axis.plot(
                initial_hodograph.real,
                initial_hodograph.imag,
                color=BLACK,
                linewidth=1.4,
            )
            marker = initial_hodograph[0]
            hodograph_axis.plot(
                marker.real,
                marker.imag,
                marker="o",
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor=BLACK,
                zorder=9,
            )
            hodograph_axis.text(
                -1.96,
                1.79,
                rf"$({panel_letters[column + 4]})$",
                fontsize=10,
            )

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02
        )
        figure.savefig(
            output_stem.with_suffix(".png"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input.resolve()) as archive:
        data = {name: archive[name] for name in archive.files}
    output = args.output_directory.resolve()
    plot_figure_1(data, output / "polarisation_geometry")
    plot_figure_2(data, output / "matrix_basis_actions")
    print(output)


if __name__ == "__main__":
    main()
