"""Render supplementary movie 1 from precomputed polarisation data.

This script performs no spinor evolution, Stokes conversion or hodograph
calculation.  It reads those quantities from ``movie1_data.npz``, draws fixed
publication-style layouts, encodes H.264 video and writes submission sidecars.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import imageio_ffmpeg
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
import numpy as np
from PIL import Image


DATA_FILENAME = "movie1_data.npz"
METADATA_FILENAME = "movie1_metadata.json"
MOVIE_FILENAME = "movie1.mp4"
PREVIEW_FILENAME = "movie1_preview.png"
BLUE = "#1f77b4"
RED = "#d62728"
GREY = "0.62"
BLACK = "0.08"
ANGLE_GREEN = "#30664e"
ANGLE_GREEN_FILL = "#cae1b3"
ANGLE_BLUE = "#1c588c"
ANGLE_BLUE_FILL = "#bbc9db"
ANGLE_ORANGE = "#b85900"
ANGLE_ORANGE_FILL = "#fcd7b3"
CAMERA_AZIMUTH = 110.75
CAMERA_ELEVATION = 30.85
CHAPTER_ONE_PHASE_PERIOD_SECONDS = 7.0


def publication_style(*, use_tex: bool) -> dict[str, object]:
    """Return a movie-safe version of the manuscript figure style."""
    style: dict[str, object] = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.size": 15.0,
        "axes.labelsize": 16.0,
        "axes.titlesize": 17.0,
        "xtick.labelsize": 13.0,
        "ytick.labelsize": 13.0,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
    if use_tex:
        style.update(
            {
                "text.usetex": True,
                "text.latex.preamble": (
                    r"\usepackage{amsmath}"
                    r"\usepackage{newtxtext}"
                    r"\usepackage{newtxmath}"
                ),
            }
        )
    else:
        style["text.usetex"] = False
    return style


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return horizontal, vertical and depth axes for the fixed camera."""
    azimuth = np.deg2rad(CAMERA_AZIMUTH)
    elevation = np.deg2rad(CAMERA_ELEVATION)
    depth = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    vertical = np.cross(depth, horizontal)
    return horizontal, vertical, depth


def project_stokes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthographically project Stokes coordinates at the fixed view."""
    points = np.asarray(points, dtype=float)
    displayed = np.stack(
        [-points[..., 1], points[..., 0], points[..., 2]],
        axis=-1,
    )
    horizontal, vertical, depth_axis = camera_basis()
    projected = np.stack(
        [displayed @ horizontal, displayed @ vertical],
        axis=-1,
    )
    depth = displayed @ depth_axis
    return projected, depth


def split_visibility(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a unit-sphere guide into visible and hidden projected pieces."""
    projected, depth = project_stokes(points)
    front = projected.copy()
    back = projected.copy()
    front[depth < 0.0] = np.nan
    back[depth >= 0.0] = np.nan
    return front, back


def sphere_shading(size: int = 420) -> np.ndarray:
    """Create restrained grayscale shading for a unit sphere."""
    coordinate = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(coordinate, coordinate)
    rr2 = xx**2 + yy**2
    zz = np.sqrt(np.maximum(1.0 - rr2, 0.0))
    light = np.array([-0.45, 0.36, 0.82])
    light /= np.linalg.norm(light)
    intensity = xx * light[0] + yy * light[1] + zz * light[2]
    intensity = 0.78 + 0.18 * (intensity + 1.0) / 2.0
    intensity[rr2 > 1.0] = 1.0
    return intensity


def draw_poincare_sphere(
    axis: mpl.axes.Axes,
    *,
    limit: float,
    longitude: float,
    label_axes: bool,
) -> None:
    """Draw the fixed unit-sphere reference and projected coordinate axes."""
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()

    sphere_circle = Circle(
        (0.0, 0.0),
        1.0,
        facecolor="none",
        edgecolor=BLACK,
        linewidth=1.7,
        zorder=2,
    )
    axis.add_patch(sphere_circle)
    shading = axis.imshow(
        sphere_shading(),
        extent=(-1.0, 1.0, -1.0, 1.0),
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        zorder=0,
    )
    shading.set_clip_path(sphere_circle)

    angle = np.linspace(0.0, 2.0 * np.pi, 721)
    equator = np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])
    meridian = np.column_stack(
        [
            np.cos(angle) * np.cos(longitude),
            np.cos(angle) * np.sin(longitude),
            np.sin(angle),
        ]
    )
    for guide in (equator, meridian):
        front, back = split_visibility(guide)
        axis.plot(
            back[:, 0],
            back[:, 1],
            color="0.67",
            linestyle=(0, (3.0, 2.5)),
            linewidth=0.9,
            zorder=1,
        )
        axis.plot(
            front[:, 0],
            front[:, 1],
            color="0.22",
            linewidth=1.0,
            zorder=3,
        )

    axis_length = min(1.48, 0.78 * limit)
    labels = (r"$S_x$", r"$S_y$", r"$S_z$")
    basis = np.eye(3)
    for vector, label in zip(basis, labels):
        endpoints, _ = project_stokes(
            np.vstack([-axis_length * vector, axis_length * vector])
        )
        axis.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            color="0.23",
            linewidth=1.25,
            zorder=4,
        )
        axis.add_patch(
            FancyArrowPatch(
                tuple(0.84 * endpoints[1]),
                tuple(endpoints[1]),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color="0.23",
                zorder=5,
            )
        )
        if label_axes:
            label_position = 1.10 * endpoints[1]
            axis.text(
                label_position[0],
                label_position[1],
                label,
                ha="center",
                va="center",
                fontsize=15,
                zorder=6,
            )


def draw_hodograph_axes(
    axis: mpl.axes.Axes,
    *,
    limit: float,
    ticks: bool,
) -> None:
    """Draw a fixed, equal-aspect complex hodograph frame."""
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.axhline(0.0, color="0.74", linewidth=0.8, zorder=0)
    axis.axvline(0.0, color="0.74", linewidth=0.8, zorder=0)
    axis.set_xlabel(r"$u=\operatorname{Re}\phi$")
    axis.set_ylabel(r"$v=\operatorname{Im}\phi$")
    if ticks:
        axis.set_xticks([-1.0, 0.0, 1.0])
        axis.set_yticks([-1.0, 0.0, 1.0])
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)


def canvas_rgb(figure: mpl.figure.Figure) -> np.ndarray:
    """Draw a figure and return a contiguous RGB frame."""
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[..., :3])


def plain_dynamic_text(text: mpl.text.Text) -> mpl.text.Text:
    """Keep changing numerical text out of the external TeX pipeline."""
    text.set_usetex(False)
    text.set_fontfamily("serif")
    return text


@dataclass
class AngleArcArtists:
    """Mutable filled arc and mathematical label."""

    fill: Polygon
    arc: Line2D
    label: mpl.text.Text


@dataclass
class ChapterOneArtists:
    """Mutable artists for the two-panel mapping chapter."""

    figure: mpl.figure.Figure
    sphere_axis: mpl.axes.Axes
    hodograph_axis: mpl.axes.Axes
    stokes_trail: Line2D
    stokes_arrow: FancyArrowPatch
    stokes_marker: Line2D
    hodograph: Line2D
    hodograph_marker: Line2D
    phase_arrow: FancyArrowPatch
    major_axis: Line2D
    sphere_lambda: AngleArcArtists
    sphere_varphi: AngleArcArtists
    sphere_gamma: AngleArcArtists
    sphere_gamma_ring: Line2D
    hodograph_lambda_half: AngleArcArtists
    hodograph_varphi_half: AngleArcArtists
    hodograph_gamma: AngleArcArtists
    ellipticity_chord: Line2D
    spinor_top: mpl.text.Text
    spinor_bottom: mpl.text.Text
    subtitle: mpl.text.Text
    explanation: mpl.text.Text
    parameter_text: mpl.text.Text


def make_angle_arc(
    axis: mpl.axes.Axes,
    *,
    edge_color: str,
    fill_color: str,
    label: str,
    zorder: float,
    fill_alpha: float = 0.70,
) -> AngleArcArtists:
    """Create one initially empty filled angle arc."""
    fill = Polygon(
        np.zeros((3, 2)),
        closed=True,
        facecolor=fill_color,
        edgecolor="none",
        alpha=fill_alpha,
        zorder=zorder,
    )
    axis.add_patch(fill)
    (arc,) = axis.plot(
        [],
        [],
        color=edge_color,
        linewidth=2.0,
        zorder=zorder + 0.2,
    )
    text = axis.text(
        0.0,
        0.0,
        label,
        ha="center",
        va="center",
        fontsize=18,
        color=BLACK,
        zorder=zorder + 0.4,
    )
    return AngleArcArtists(fill=fill, arc=arc, label=text)


def set_angle_arc(
    artists: AngleArcArtists,
    center: np.ndarray,
    arc_points: np.ndarray,
    label_position: np.ndarray,
) -> None:
    """Update one filled arc from display-coordinate points."""
    center = np.asarray(center, dtype=float)
    arc_points = np.asarray(arc_points, dtype=float)
    artists.fill.set_xy(np.vstack([center, arc_points, center]))
    artists.arc.set_data(arc_points[:, 0], arc_points[:, 1])
    artists.label.set_position(tuple(label_position))


def make_chapter_one(
    width: int,
    height: int,
    *,
    use_tex: bool,
) -> ChapterOneArtists:
    """Create the fixed Figure 1-style two-panel layout."""
    dpi = 120
    with mpl.rc_context(publication_style(use_tex=use_tex)):
        figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(
            1,
            3,
            width_ratios=(1.0, 0.48, 1.0),
            left=0.035,
            right=0.975,
            bottom=0.14,
            top=0.80,
            wspace=0.035,
        )
        sphere_axis = figure.add_subplot(grid[0, 0])
        spinor_axis = figure.add_subplot(grid[0, 1])
        hodograph_axis = figure.add_subplot(grid[0, 2])
        spinor_axis.set_axis_off()
        spinor_axis.set_xlim(0.0, 1.0)
        spinor_axis.set_ylim(0.0, 1.0)
        draw_poincare_sphere(
            sphere_axis,
            limit=1.55,
            longitude=np.pi / 3.0,
            label_axes=True,
        )
        draw_hodograph_axes(hodograph_axis, limit=1.55, ticks=False)
        sphere_axis.set_title("Stokes-Poincare representation", pad=15)
        hodograph_axis.set_title("Horizontal velocity hodograph", pad=15)
        sphere_axis.text(
            -1.43,
            -1.34,
            r"$\left|\widehat{\mathbf{S}}\right|=1$",
            ha="left",
            va="center",
            fontsize=17,
            color="0.20",
            zorder=10,
        )

        (stokes_trail,) = sphere_axis.plot(
            [],
            [],
            color=RED,
            linewidth=2.2,
            alpha=0.72,
            zorder=7,
        )
        stokes_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=3.0,
            color=RED,
            zorder=8,
        )
        sphere_axis.add_patch(stokes_arrow)
        (stokes_marker,) = sphere_axis.plot(
            [],
            [],
            marker="o",
            markersize=8.0,
            markerfacecolor="white",
            markeredgecolor=BLACK,
            markeredgewidth=1.4,
            linestyle="None",
            zorder=9,
        )
        (hodograph,) = hodograph_axis.plot(
            [],
            [],
            color=BLACK,
            linewidth=3.0,
            zorder=4,
        )
        (hodograph_marker,) = hodograph_axis.plot(
            [],
            [],
            marker="o",
            markersize=8.0,
            markerfacecolor="white",
            markeredgecolor=BLACK,
            markeredgewidth=1.4,
            linestyle="None",
            zorder=6,
        )
        phase_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.3,
            color=ANGLE_BLUE,
            shrinkA=0.0,
            shrinkB=5.5,
            zorder=5,
        )
        hodograph_axis.add_patch(phase_arrow)
        (major_axis,) = hodograph_axis.plot(
            [],
            [],
            color="0.52",
            linestyle=(0, (4.0, 3.0)),
            linewidth=1.2,
            zorder=2,
        )
        sphere_lambda = make_angle_arc(
            sphere_axis,
            edge_color=ANGLE_GREEN,
            fill_color=ANGLE_GREEN_FILL,
            label=r"$\lambda$",
            zorder=5.1,
        )
        sphere_varphi = make_angle_arc(
            sphere_axis,
            edge_color=ANGLE_ORANGE,
            fill_color=ANGLE_ORANGE_FILL,
            label=r"$\varphi$",
            zorder=5.3,
        )
        sphere_gamma = make_angle_arc(
            sphere_axis,
            edge_color=ANGLE_BLUE,
            fill_color=ANGLE_BLUE_FILL,
            label=r"$\gamma$",
            zorder=9.2,
        )
        (sphere_gamma_ring,) = sphere_axis.plot(
            [],
            [],
            color="0.20",
            linewidth=1.1,
            zorder=9.0,
        )
        hodograph_lambda_half = make_angle_arc(
            hodograph_axis,
            edge_color=ANGLE_GREEN,
            fill_color=ANGLE_GREEN_FILL,
            label=r"$\lambda/2$",
            zorder=3.2,
        )
        hodograph_varphi_half = make_angle_arc(
            hodograph_axis,
            edge_color=ANGLE_ORANGE,
            fill_color=ANGLE_ORANGE_FILL,
            label=r"$\varphi/2$",
            zorder=4.4,
        )
        hodograph_gamma = make_angle_arc(
            hodograph_axis,
            edge_color=ANGLE_BLUE,
            fill_color=ANGLE_BLUE_FILL,
            label=r"$\gamma$",
            zorder=2.5,
            fill_alpha=0.18,
        )
        (ellipticity_chord,) = hodograph_axis.plot(
            [],
            [],
            color=RED,
            linewidth=2.5,
            zorder=4.2,
        )

        spinor_axis.text(
            0.5,
            0.86,
            r"$\left|\mathscr{A}\right\rangle=$",
            ha="center",
            va="center",
            fontsize=20,
        )
        spinor_axis.text(
            0.12,
            0.625,
            "(",
            ha="center",
            va="center",
            fontsize=78,
            color=BLACK,
        )
        spinor_axis.text(
            0.88,
            0.625,
            ")",
            ha="center",
            va="center",
            fontsize=78,
            color=BLACK,
        )
        spinor_top = plain_dynamic_text(
            spinor_axis.text(
                0.5,
                0.685,
                "",
                ha="center",
                va="center",
                fontsize=14,
                color=BLACK,
            )
        )
        spinor_bottom = plain_dynamic_text(
            spinor_axis.text(
                0.5,
                0.565,
                "",
                ha="center",
                va="center",
                fontsize=14,
                color=BLACK,
            )
        )
        spinor_axis.text(
            0.5,
            0.355,
            r"$\phi=u+\mathrm{i}v$",
            ha="center",
            va="center",
            fontsize=17,
        )
        spinor_axis.text(
            0.5,
            0.265,
            r"$=\mathscr{A}_\uparrow\mathrm{e}^{-\mathrm{i}ft}$",
            ha="center",
            va="center",
            fontsize=15,
        )
        spinor_axis.text(
            0.5,
            0.195,
            r"$+\mathscr{A}_\downarrow\mathrm{e}^{\mathrm{i}ft}$",
            ha="center",
            va="center",
            fontsize=15,
        )
        spinor_axis.text(
            0.5,
            0.075,
            r"$\left|\mathscr{A}\right\rangle"
            r":=(\mathscr{A}_\uparrow,\mathscr{A}_\downarrow^\ast)^{\mathsf{T}}$",
            ha="center",
            va="center",
            fontsize=12.5,
            color="0.25",
        )

        figure.text(
            0.5,
            0.945,
            "Chapter 1: one spinor, two geometric representations",
            ha="center",
            va="top",
            fontsize=24,
        )
        subtitle = figure.text(
            0.5,
            0.855,
            "",
            ha="center",
            va="top",
            fontsize=20,
        )
        explanation = figure.text(
            0.5,
            0.075,
            "",
            ha="center",
            va="center",
            fontsize=16,
        )
        parameter_text = plain_dynamic_text(
            figure.text(
                0.5,
                0.025,
                "",
                ha="center",
                va="center",
                fontsize=14,
                color="0.25",
            )
        )
    return ChapterOneArtists(
        figure=figure,
        sphere_axis=sphere_axis,
        hodograph_axis=hodograph_axis,
        stokes_trail=stokes_trail,
        stokes_arrow=stokes_arrow,
        stokes_marker=stokes_marker,
        hodograph=hodograph,
        hodograph_marker=hodograph_marker,
        phase_arrow=phase_arrow,
        major_axis=major_axis,
        sphere_lambda=sphere_lambda,
        sphere_varphi=sphere_varphi,
        sphere_gamma=sphere_gamma,
        sphere_gamma_ring=sphere_gamma_ring,
        hodograph_lambda_half=hodograph_lambda_half,
        hodograph_varphi_half=hodograph_varphi_half,
        hodograph_gamma=hodograph_gamma,
        ellipticity_chord=ellipticity_chord,
        spinor_top=spinor_top,
        spinor_bottom=spinor_bottom,
        subtitle=subtitle,
        explanation=explanation,
        parameter_text=parameter_text,
    )


@dataclass
class GeneratorPanelArtists:
    """Mutable artists for one generator column."""

    positive_trail: Line2D
    negative_trail: Line2D
    positive_arrow: FancyArrowPatch
    negative_arrow: FancyArrowPatch
    positive_marker: Line2D
    negative_marker: Line2D
    initial_stokes_marker: Line2D
    positive_hodograph: Line2D
    negative_hodograph: Line2D
    initial_hodograph: Line2D
    positive_phi_trail: Line2D
    negative_phi_trail: Line2D
    initial_phi_marker: Line2D


@dataclass
class ChapterTwoArtists:
    """Mutable artists for the Figure 2-style eight-panel chapter."""

    figure: mpl.figure.Figure
    panels: list[GeneratorPanelArtists]
    parameter_text: mpl.text.Text


def make_chapter_two(
    width: int,
    height: int,
    *,
    use_tex: bool,
) -> ChapterTwoArtists:
    """Create the fixed four-column, two-row generator layout."""
    dpi = 120
    with mpl.rc_context(publication_style(use_tex=use_tex)):
        figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(
            2,
            4,
            left=0.035,
            right=0.985,
            bottom=0.09,
            top=0.82,
            wspace=0.20,
            hspace=0.20,
            height_ratios=[1.03, 1.0],
        )
        figure.text(
            0.5,
            0.955,
            "Chapter 2: local actions of the four matrix basis directions",
            ha="center",
            va="top",
            fontsize=23,
        )
        figure.legend(
            handles=[
                Line2D([0], [0], color=BLUE, linewidth=2.8, label="+ direction"),
                Line2D(
                    [0],
                    [0],
                    color=RED,
                    linewidth=2.8,
                    linestyle=(0, (5.0, 3.0)),
                    label="- direction",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor=BLACK,
                    linestyle="None",
                    markersize=8,
                    label="common initial state",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905),
            ncol=3,
            frameon=False,
            fontsize=14,
        )
        top_titles = (
            r"$\sigma_0$: amplitude",
            r"$\sigma_1$: rotation",
            r"$\sigma_2$: stretching",
            r"$\sigma_3$: stretching",
        )
        panels: list[GeneratorPanelArtists] = []
        for column in range(4):
            sphere_axis = figure.add_subplot(grid[0, column])
            hodograph_axis = figure.add_subplot(grid[1, column])
            draw_poincare_sphere(
                sphere_axis,
                limit=2.55,
                longitude=np.pi / 3.0,
                label_axes=True,
            )
            draw_hodograph_axes(hodograph_axis, limit=1.7, ticks=True)
            sphere_axis.set_title(top_titles[column], pad=4)
            hodograph_axis.set_title(rf"hodograph for $\sigma_{column}$", pad=8)
            if column > 0:
                hodograph_axis.set_ylabel("")
                hodograph_axis.tick_params(labelleft=False)

            (positive_trail,) = sphere_axis.plot(
                [],
                [],
                color=BLUE,
                linewidth=2.2,
                zorder=7,
            )
            (negative_trail,) = sphere_axis.plot(
                [],
                [],
                color=RED,
                linestyle=(0, (5.0, 3.0)),
                linewidth=2.2,
                zorder=7,
            )
            positive_arrow = FancyArrowPatch(
                (0.0, 0.0),
                (0.0, 0.0),
                arrowstyle="-|>",
                mutation_scale=14,
                color=BLUE,
                linewidth=2.3,
                zorder=8,
            )
            negative_arrow = FancyArrowPatch(
                (0.0, 0.0),
                (0.0, 0.0),
                arrowstyle="-|>",
                mutation_scale=14,
                color=RED,
                linewidth=2.3,
                linestyle=(0, (5.0, 3.0)),
                zorder=8,
            )
            sphere_axis.add_patch(positive_arrow)
            sphere_axis.add_patch(negative_arrow)
            (positive_marker,) = sphere_axis.plot(
                [],
                [],
                marker="o",
                markersize=5.5,
                markerfacecolor=BLUE,
                markeredgecolor=BLUE,
                linestyle="None",
                zorder=9,
            )
            (negative_marker,) = sphere_axis.plot(
                [],
                [],
                marker="s",
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor=RED,
                markeredgewidth=1.4,
                linestyle="None",
                zorder=9,
            )
            (initial_stokes_marker,) = sphere_axis.plot(
                [],
                [],
                marker="o",
                markersize=7.5,
                markerfacecolor="white",
                markeredgecolor=BLACK,
                markeredgewidth=1.2,
                linestyle="None",
                zorder=10,
            )

            (initial_hodograph,) = hodograph_axis.plot(
                [],
                [],
                color="0.67",
                linewidth=1.4,
                linestyle=(0, (2.0, 2.0)),
                zorder=1,
            )
            (negative_hodograph,) = hodograph_axis.plot(
                [],
                [],
                color=RED,
                linewidth=2.1,
                linestyle=(0, (5.0, 3.0)),
                zorder=3,
            )
            (positive_hodograph,) = hodograph_axis.plot(
                [],
                [],
                color=BLUE,
                linewidth=2.1,
                zorder=4,
            )
            (negative_phi_trail,) = hodograph_axis.plot(
                [],
                [],
                color=RED,
                linewidth=1.0,
                linestyle=(0, (3.0, 2.0)),
                alpha=0.75,
                zorder=5,
            )
            (positive_phi_trail,) = hodograph_axis.plot(
                [],
                [],
                color=BLUE,
                linewidth=1.0,
                alpha=0.75,
                zorder=5,
            )
            (initial_phi_marker,) = hodograph_axis.plot(
                [],
                [],
                marker="o",
                markersize=6.0,
                markerfacecolor="white",
                markeredgecolor=BLACK,
                markeredgewidth=1.1,
                linestyle="None",
                zorder=7,
            )
            panels.append(
                GeneratorPanelArtists(
                    positive_trail=positive_trail,
                    negative_trail=negative_trail,
                    positive_arrow=positive_arrow,
                    negative_arrow=negative_arrow,
                    positive_marker=positive_marker,
                    negative_marker=negative_marker,
                    initial_stokes_marker=initial_stokes_marker,
                    positive_hodograph=positive_hodograph,
                    negative_hodograph=negative_hodograph,
                    initial_hodograph=initial_hodograph,
                    positive_phi_trail=positive_phi_trail,
                    negative_phi_trail=negative_phi_trail,
                    initial_phi_marker=initial_phi_marker,
                )
            )

        parameter_text = plain_dynamic_text(
            figure.text(
                0.5,
                0.025,
                "",
                ha="center",
                va="center",
                fontsize=14,
                color="0.23",
            )
        )
    return ChapterTwoArtists(
        figure=figure,
        panels=panels,
        parameter_text=parameter_text,
    )


def set_chapter_one_frame(
    artists: ChapterOneArtists,
    arrays: dict[str, np.ndarray],
    stage: str,
    index: int,
    *,
    phase_angle: float,
    landmark_label: str | None = None,
) -> None:
    """Update the two-panel mapping layout from saved arrays."""
    stokes = arrays[f"{stage}_stokes"]
    hodographs = arrays[f"{stage}_hodograph"]
    spinors = arrays[f"{stage}_spinor"]
    varphi = arrays[f"{stage}_varphi"]
    longitude = arrays[f"{stage}_lambda"]
    gamma = arrays[f"{stage}_gamma"]
    trail_norms = np.linalg.norm(stokes[: index + 1], axis=-1, keepdims=True)
    unit_trail = stokes[: index + 1] / trail_norms
    current_stokes = unit_trail[-1]
    projected, _ = project_stokes(unit_trail)
    endpoint, _ = project_stokes(current_stokes[None, :])
    endpoint = endpoint[0]
    artists.stokes_trail.set_data(projected[:, 0], projected[:, 1])
    artists.stokes_arrow.set_positions((0.0, 0.0), tuple(endpoint))
    artists.stokes_marker.set_data([endpoint[0]], [endpoint[1]])

    values = hodographs[index]
    fast_phase = arrays["fast_phase"]
    target_phase = float((gamma[index] + phase_angle) % (2.0 * np.pi))
    phase_difference = np.angle(np.exp(1j * (fast_phase - target_phase)))
    phase_index = int(np.argmin(np.abs(phase_difference)))
    marker = np.array(
        [np.real(values[phase_index]), np.imag(values[phase_index])],
        dtype=float,
    )
    artists.hodograph.set_data(np.real(values), np.imag(values))
    artists.hodograph_marker.set_data([marker[0]], [marker[1]])
    artists.phase_arrow.set_positions((0.0, 0.0), tuple(marker))
    orientation = longitude[index] / 2.0
    guide = 1.45 * np.array([np.cos(orientation), np.sin(orientation)])
    if abs(abs(varphi[index]) - np.pi / 2.0) < 0.02:
        # A circle has no distinguished major-axis orientation.
        artists.major_axis.set_data([], [])
    else:
        artists.major_axis.set_data(
            [-guide[0], guide[0]],
            [-guide[1], guide[1]],
        )

    # Left panel: longitude and latitude are drawn as projected spherical
    # sectors, while the phase is a local tangent-plane sector at S-hat.
    lambda_display = float(longitude[index] % (2.0 * np.pi))
    lambda_theta = np.linspace(0.0, lambda_display, 121)
    lambda_radius = 0.34
    lambda_points_3d = lambda_radius * np.column_stack(
        [
            np.cos(lambda_theta),
            np.sin(lambda_theta),
            np.zeros_like(lambda_theta),
        ]
    )
    lambda_points, _ = project_stokes(lambda_points_3d)
    lambda_label_3d = 0.46 * np.array(
        [
            np.cos(0.5 * lambda_display),
            np.sin(0.5 * lambda_display),
            0.0,
        ]
    )
    lambda_label, _ = project_stokes(lambda_label_3d[None, :])
    set_angle_arc(
        artists.sphere_lambda,
        np.zeros(2),
        lambda_points,
        lambda_label[0],
    )

    latitude_theta = np.linspace(0.0, float(varphi[index]), 121)
    varphi_radius = 0.43
    varphi_points_3d = varphi_radius * np.column_stack(
        [
            np.cos(latitude_theta) * np.cos(longitude[index]),
            np.cos(latitude_theta) * np.sin(longitude[index]),
            np.sin(latitude_theta),
        ]
    )
    varphi_points, _ = project_stokes(varphi_points_3d)
    varphi_label_3d = 0.57 * np.array(
        [
            np.cos(0.5 * varphi[index]) * np.cos(longitude[index]),
            np.cos(0.5 * varphi[index]) * np.sin(longitude[index]),
            np.sin(0.5 * varphi[index]),
        ]
    )
    varphi_label, _ = project_stokes(varphi_label_3d[None, :])
    set_angle_arc(
        artists.sphere_varphi,
        np.zeros(2),
        varphi_points,
        varphi_label[0],
    )

    reference_axis = np.array([0.0, 0.0, 1.0])
    tangent_a = np.cross(reference_axis, current_stokes)
    if np.linalg.norm(tangent_a) < 1.0e-8:
        tangent_a = np.array([1.0, 0.0, 0.0])
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(current_stokes, tangent_a)
    tangent_b /= np.linalg.norm(tangent_b)
    phase_direction = -1.0 if current_stokes[2] >= 0.0 else 1.0
    directed_phase = phase_direction * phase_angle
    gamma_radius = 0.15
    gamma_theta = np.linspace(0.0, directed_phase, 121)
    gamma_points_3d = current_stokes + gamma_radius * (
        np.cos(gamma_theta)[:, None] * tangent_a
        + np.sin(gamma_theta)[:, None] * tangent_b
    )
    gamma_points, _ = project_stokes(gamma_points_3d)
    gamma_center = endpoint
    gamma_label_angle = 0.5 * directed_phase
    if abs(directed_phase) < 0.16:
        gamma_label_angle = 0.18 * phase_direction
    gamma_label_3d = current_stokes + 1.55 * gamma_radius * (
        np.cos(gamma_label_angle) * tangent_a
        + np.sin(gamma_label_angle) * tangent_b
    )
    gamma_label, _ = project_stokes(gamma_label_3d[None, :])
    set_angle_arc(
        artists.sphere_gamma,
        gamma_center,
        gamma_points,
        gamma_label[0],
    )
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 181)
    ring_points_3d = current_stokes + gamma_radius * (
        np.cos(ring_theta)[:, None] * tangent_a
        + np.sin(ring_theta)[:, None] * tangent_b
    )
    ring_points, _ = project_stokes(ring_points_3d)
    artists.sphere_gamma_ring.set_data(ring_points[:, 0], ring_points[:, 1])

    # Right panel: use the same three-angle construction as the manuscript
    # reference.  The phase ray ends at the white instantaneous marker.
    lambda_half_theta = np.linspace(0.0, orientation, 121)
    lambda_half_radius = 0.40
    lambda_half_points = lambda_half_radius * np.column_stack(
        [np.cos(lambda_half_theta), np.sin(lambda_half_theta)]
    )
    lambda_half_label_angle = 0.5 * orientation
    lambda_half_label = 1.38 * lambda_half_radius * np.array(
        [np.cos(lambda_half_label_angle), np.sin(lambda_half_label_angle)]
    )
    set_angle_arc(
        artists.hodograph_lambda_half,
        np.zeros(2),
        lambda_half_points,
        lambda_half_label,
    )

    cosine_latitude = max(float(np.cos(varphi[index])), 0.0)
    semi_major = np.sqrt(1.0 + cosine_latitude)
    semi_minor = np.sqrt(max(1.0 - cosine_latitude, 0.0))
    major_direction = np.array([np.cos(orientation), np.sin(orientation)])
    minor_direction = np.array([-np.sin(orientation), np.cos(orientation)])
    latitude_sign = 1.0 if varphi[index] >= 0.0 else -1.0
    major_negative = -semi_major * major_direction
    minor_endpoint = -latitude_sign * semi_minor * minor_direction
    chord_direction = minor_endpoint - major_negative
    chord_angle = float(
        np.arctan2(
            major_direction[0] * chord_direction[1]
            - major_direction[1] * chord_direction[0],
            np.dot(major_direction, chord_direction),
        )
    )
    varphi_half_radius = 0.25 * semi_major
    varphi_half_theta = np.linspace(
        orientation,
        orientation + chord_angle,
        101,
    )
    varphi_half_points = major_negative + varphi_half_radius * np.column_stack(
        [np.cos(varphi_half_theta), np.sin(varphi_half_theta)]
    )
    varphi_half_label_angle = orientation + 0.5 * chord_angle
    varphi_half_label = major_negative + 1.42 * varphi_half_radius * np.array(
        [np.cos(varphi_half_label_angle), np.sin(varphi_half_label_angle)]
    )
    if abs(chord_angle) < 0.08:
        varphi_half_label += 0.16 * minor_direction
    set_angle_arc(
        artists.hodograph_varphi_half,
        major_negative,
        varphi_half_points,
        varphi_half_label,
    )
    artists.ellipticity_chord.set_data(
        [major_negative[0], minor_endpoint[0]],
        [major_negative[1], minor_endpoint[1]],
    )

    marker_angle = float(np.arctan2(marker[1], marker[0]))
    raw_gamma_delta = float(
        np.arctan2(
            np.sin(marker_angle - orientation),
            np.cos(marker_angle - orientation),
        )
    )
    if phase_angle < 1.0e-10:
        gamma_delta = 0.0
    elif phase_direction < 0.0:
        gamma_delta = -float((-raw_gamma_delta) % (2.0 * np.pi))
    else:
        gamma_delta = float(raw_gamma_delta % (2.0 * np.pi))
    hodograph_gamma_radius = 0.54
    hodograph_gamma_theta = np.linspace(
        orientation,
        orientation + gamma_delta,
        181,
    )
    hodograph_gamma_points = hodograph_gamma_radius * np.column_stack(
        [np.cos(hodograph_gamma_theta), np.sin(hodograph_gamma_theta)]
    )
    hodograph_gamma_label_angle = orientation + 0.35 * gamma_delta
    if abs(gamma_delta) < 0.16:
        hodograph_gamma_label_angle = orientation + 0.20 * phase_direction
    hodograph_gamma_label = 1.62 * hodograph_gamma_radius * np.array(
        [
            np.cos(hodograph_gamma_label_angle),
            np.sin(hodograph_gamma_label_angle),
        ]
    )
    set_angle_arc(
        artists.hodograph_gamma,
        np.zeros(2),
        hodograph_gamma_points,
        hodograph_gamma_label,
    )

    def formatted_complex(value: complex) -> str:
        sign = "+" if np.imag(value) >= 0.0 else "-"
        return (
            rf"${np.real(value):.3f}{sign}{abs(np.imag(value)):.3f}"
            r"\,\mathrm{i}$"
        )

    artists.spinor_top.set_text(formatted_complex(spinors[index, 0]))
    artists.spinor_bottom.set_text(formatted_complex(spinors[index, 1]))

    if stage == "landmark":
        artists.subtitle.set_text("Polarisation landmarks")
        artists.explanation.set_text(landmark_label or "")
    elif stage == "ellipticity":
        artists.subtitle.set_text("Ellipticity")
        artists.explanation.set_text(
            r"Changing $\varphi$ changes handedness and ellipticity; "
            r"the ellipticity angle is $\varphi/2$."
        )
    elif stage == "orientation":
        artists.subtitle.set_text("Orientation")
        artists.explanation.set_text(
            r"Changing $\lambda$ rotates the ellipse through $\lambda/2$."
        )
    elif stage == "phase":
        artists.subtitle.set_text("Common phase")
        artists.explanation.set_text(
            r"The phase runs uniformly; the unit Stokes vector and "
            r"ellipse geometry remain fixed."
        )
    else:
        raise ValueError(f"Unknown chapter-one stage: {stage}")
    displayed_gamma = phase_direction * phase_angle
    artists.parameter_text.set_text(
        rf"$\varphi={np.degrees(varphi[index]):.1f}^\circ,\quad"
        rf"\lambda={np.degrees(longitude[index]):.1f}^\circ,\quad"
        rf"\gamma={np.degrees(displayed_gamma):.1f}^\circ$"
    )


def set_chapter_two_frame(
    artists: ChapterTwoArtists,
    arrays: dict[str, np.ndarray],
    index: int,
) -> None:
    """Update all eight panels from the saved generator trajectories."""
    positive_stokes = arrays["generator_stokes_positive"]
    negative_stokes = arrays["generator_stokes_negative"]
    positive_hodographs = arrays["generator_hodograph_positive"]
    negative_hodographs = arrays["generator_hodograph_negative"]
    for column, panel in enumerate(artists.panels):
        positive_projected, _ = project_stokes(positive_stokes[column, : index + 1])
        negative_projected, _ = project_stokes(negative_stokes[column, : index + 1])
        positive_endpoint = positive_projected[-1]
        negative_endpoint = negative_projected[-1]
        initial_endpoint = positive_projected[0]
        panel.positive_trail.set_data(
            positive_projected[:, 0],
            positive_projected[:, 1],
        )
        panel.negative_trail.set_data(
            negative_projected[:, 0],
            negative_projected[:, 1],
        )
        panel.positive_arrow.set_positions((0.0, 0.0), tuple(positive_endpoint))
        panel.negative_arrow.set_positions((0.0, 0.0), tuple(negative_endpoint))
        panel.positive_marker.set_data(
            [positive_endpoint[0]],
            [positive_endpoint[1]],
        )
        panel.negative_marker.set_data(
            [negative_endpoint[0]],
            [negative_endpoint[1]],
        )
        panel.initial_stokes_marker.set_data(
            [initial_endpoint[0]],
            [initial_endpoint[1]],
        )

        positive_values = positive_hodographs[column, index]
        negative_values = negative_hodographs[column, index]
        initial_values = positive_hodographs[column, 0]
        panel.positive_hodograph.set_data(
            np.real(positive_values),
            np.imag(positive_values),
        )
        panel.negative_hodograph.set_data(
            np.real(negative_values),
            np.imag(negative_values),
        )
        panel.initial_hodograph.set_data(
            np.real(initial_values),
            np.imag(initial_values),
        )
        positive_phi_track = positive_hodographs[column, : index + 1, 0]
        negative_phi_track = negative_hodographs[column, : index + 1, 0]
        panel.positive_phi_trail.set_data(
            np.real(positive_phi_track),
            np.imag(positive_phi_track),
        )
        panel.negative_phi_trail.set_data(
            np.real(negative_phi_track),
            np.imag(negative_phi_track),
        )
        panel.initial_phi_marker.set_data(
            [np.real(initial_values[0])],
            [np.imag(initial_values[0])],
        )

    parameter = arrays["generator_parameter"][index]
    artists.parameter_text.set_text(
        f"generator parameter f t = {parameter:.3f}; "
        "top row uses unnormalised Stokes vectors, and every unit sphere is a scale reference"
    )


def title_frame(
    width: int,
    height: int,
    title: str,
    subtitle: str,
    *,
    use_tex: bool,
) -> np.ndarray:
    """Render one clean, static title card."""
    dpi = 120
    with mpl.rc_context(publication_style(use_tex=use_tex)):
        figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        FigureCanvasAgg(figure)
        figure.text(
            0.5,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=31,
        )
        figure.text(
            0.5,
            0.43,
            subtitle,
            ha="center",
            va="center",
            fontsize=20,
            color="0.28",
        )
        frame = canvas_rgb(figure)
        plt.close(figure)
    return frame


def landmark_schedule(progress: float) -> tuple[float, str]:
    """Return a smooth latitude fraction with readable holds at five states."""
    positions = np.linspace(0.0, 1.0, 5)
    labels = (
        "North pole: pure clockwise circular polarisation",
        "Positive latitude: clockwise elliptical polarisation",
        "Equator: linear polarisation",
        "Negative latitude: counter-clockwise elliptical polarisation",
        "South pole: pure counter-clockwise circular polarisation",
    )
    hold = 0.08
    move = 0.15
    cursor = 0.0
    for state_index in range(5):
        if progress <= cursor + hold or state_index == 4:
            return float(positions[state_index]), labels[state_index]
        cursor += hold
        if state_index < 4:
            if progress <= cursor + move:
                local = (progress - cursor) / move
                smooth = local * local * (3.0 - 2.0 * local)
                fraction = (1.0 - smooth) * positions[state_index] + smooth * positions[
                    state_index + 1
                ]
                return float(fraction), (
                    f"{labels[state_index]}  ->  {labels[state_index + 1]}"
                )
            cursor += move
    return 1.0, labels[-1]


def locate_ffmpeg(explicit_path: Path | None) -> str:
    """Locate the bundled imageio-ffmpeg executable or use an explicit path."""
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"ffmpeg executable not found: {explicit_path}")
        return str(explicit_path)
    return imageio_ffmpeg.get_ffmpeg_exe()


def start_encoder(
    ffmpeg: str,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    crf: int,
) -> subprocess.Popen[bytes]:
    """Start a constant-frame-rate H.264 encoder accepting raw RGB frames."""
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
        "-g",
        str(2 * fps),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def atom_offsets(path: Path) -> dict[str, int]:
    """Locate top-level MP4 atoms sufficiently to verify fast-start ordering."""
    data = path.read_bytes()
    return {
        "moov": data.find(b"moov"),
        "mdat": data.find(b"mdat"),
    }


def probe_video(ffmpeg: str, path: Path) -> dict[str, Any]:
    """Probe and decode-check the movie without requiring ffprobe."""
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = probe.stderr
    video_lines = [line.strip() for line in text.splitlines() if "Video:" in line]
    if len(video_lines) != 1:
        raise ValueError(f"Expected one video stream; found {len(video_lines)}.")
    video_line = video_lines[0]
    profile_match = re.search(r"Video:\s+h264\s+\(([^)]+)\)", video_line)
    pixel_match = re.search(r",\s*(yuv[0-9a-z]+)(?:\([^)]*\))?,", video_line)
    size_match = re.search(r",\s*(\d{2,5})x(\d{2,5})[\s,]", video_line)
    fps_match = re.search(r",\s*([0-9.]+)\s+fps", video_line)
    duration_match = re.search(
        r"Duration:\s*(\d+):(\d+):([0-9.]+)",
        text,
    )
    if not all((profile_match, pixel_match, size_match, fps_match, duration_match)):
        raise ValueError(f"Could not parse ffmpeg probe output:\n{text}")
    hours, minutes, seconds = duration_match.groups()
    duration = 3600.0 * int(hours) + 60.0 * int(minutes) + float(seconds)

    decode = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if decode.returncode != 0:
        raise ValueError(f"Video decode check failed:\n{decode.stderr}")

    frame_count, counted_duration = imageio_ffmpeg.count_frames_and_secs(str(path))
    offsets = atom_offsets(path)
    return {
        "container": "MP4",
        "codec": "h264",
        "profile": profile_match.group(1),
        "pixel_format": pixel_match.group(1),
        "width": int(size_match.group(1)),
        "height": int(size_match.group(2)),
        "frame_rate_fps": float(fps_match.group(1)),
        "duration_seconds": float(duration),
        "counted_duration_seconds": float(counted_duration),
        "frame_count": int(frame_count),
        "audio_stream_present": "Audio:" in text,
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": path.stat().st_size / 1_000_000.0,
        "faststart_moov_before_mdat": (
            offsets["moov"] >= 0
            and offsets["mdat"] >= 0
            and offsets["moov"] < offsets["mdat"]
        ),
        "decode_check": "passed",
    }


def validate_video_target(
    video: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: int,
    expected_frames: int,
) -> None:
    """Require the final movie to meet the requested technical target."""
    failures = []
    if video["codec"] != "h264":
        failures.append("codec is not H.264")
    if video["pixel_format"] != "yuv420p":
        failures.append("pixel format is not yuv420p")
    if (video["width"], video["height"]) != (width, height):
        failures.append("resolution changed")
    if not np.isclose(video["frame_rate_fps"], fps, atol=1.0e-6):
        failures.append("frame rate changed")
    if video["frame_count"] != expected_frames:
        failures.append(
            f"frame count is {video['frame_count']}, expected {expected_frames}"
        )
    if video["audio_stream_present"]:
        failures.append("an audio stream is present")
    if video["file_size_bytes"] >= 50_000_000:
        failures.append("file is not smaller than 50 MB")
    if not video["faststart_moov_before_mdat"]:
        failures.append("fast-start atom ordering was not detected")
    if failures:
        raise ValueError("Video validation failed: " + "; ".join(failures))


def write_text(path: Path, content: str) -> None:
    """Write one UTF-8 sidecar with a trailing newline."""
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def update_movies_readme(path: Path, section: str) -> None:
    """Add or replace only the movie 1 section, preserving all other content."""
    begin = "<!-- BEGIN MOVIE 1 -->"
    end = "<!-- END MOVIE 1 -->"
    block = f"{begin}\n{section.rstrip()}\n{end}"
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        pattern = re.compile(
            re.escape(begin) + r".*?" + re.escape(end),
            flags=re.DOTALL,
        )
        if pattern.search(existing):
            updated = pattern.sub(block, existing)
        else:
            updated = existing.rstrip() + "\n\n" + block + "\n"
    else:
        updated = "# Supplementary movies\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def format_complex(pair: list[float]) -> str:
    """Format a metadata real-imaginary pair for human-readable notes."""
    real, imaginary = pair
    sign = "+" if imaginary >= 0.0 else "-"
    return f"{real:.12g} {sign} {abs(imaginary):.12g} i"


def write_auxiliary_files(
    output_directory: Path,
    metadata: dict[str, Any],
    video: dict[str, Any],
) -> None:
    """Write the caption, accessibility text and submission notes."""
    initial = metadata["initial_state"]
    caption = r"""movie 1. Dynamic Stokes-Poincare and hodograph geometry of a local near-inertial-wave polarisation state. The NIW polarisation spinor is $$|\mathscr A\rangle=(\mathscr A_\uparrow,\mathscr A_\downarrow^\ast)^T$$, with $$\mathrm S_x=2\operatorname{Re}(\mathscr A_\uparrow\mathscr A_\downarrow)$$, $$\mathrm S_y=2\operatorname{Im}(\mathscr A_\uparrow\mathscr A_\downarrow)$$ and $$\mathrm S_z=|\mathscr A_\uparrow|^2-|\mathscr A_\downarrow|^2$$. Chapter 1 uses the unit Bloch/Stokes vector $$\widehat{\mathbf S}$$ and displays the corresponding numerical spinor between the two panels. It maps this state to $$\phi=u+\mathrm i v=\mathscr A_\uparrow\exp(-\mathrm i f t)+\mathscr A_\downarrow\exp(\mathrm i f t)$$. Green, orange and blue arcs show $$\lambda$$, $$\varphi$$ and the phase coordinate $$\gamma$$ on the sphere, and $$\lambda/2$$, $$\varphi/2$$ and $$\gamma$$ on the hodograph. The phase advances uniformly at $$2\pi/7$$ radians per second of movie time and resets after each turn. Northern-hemisphere states run clockwise; southern-hemisphere states run counter-clockwise. Chapter 2 shows the exact local actions $$|\mathscr A(t)\rangle=\exp(\pm f t\tau/50)|\mathscr A(0)\rangle$$ for $$\tau\in\{\sigma_0,\sigma_1,\sigma_2,\sigma_3\}$$. Blue solid and red dashed curves denote the positive and negative directions, respectively, and white circles denote the common initial state. The Chapter 2 Stokes vectors are unnormalised; each unit sphere is only a scale reference. The displayed Chapter 2 parameter $$f t$$ is the matrix-generator action parameter, not the time of a background-flow simulation."""
    accessibility = """Accessibility description for movie 1

The movie has a white background, dark serif labels and fixed axes. It has no audio. Blue solid curves and circular markers are always labelled as the positive generator direction. Red dashed curves and square markers are always labelled as the negative generator direction, so the two directions can be distinguished without colour.

Chapter 1 uses a left-centre-right layout. The left panel is a pale grey unit Stokes-Poincare sphere with fixed S_x, S_y and S_z axes, a red unit Stokes arrow and a white outlined endpoint. Green lambda, orange varphi and blue gamma sectors are drawn on the sphere. The centre shows the current two numerical entries of the spinor (A_up, conjugate(A_down)) and the equation phi=u+i v=A_up exp(-i f t)+A_down exp(i f t). The right panel is an equal-aspect horizontal-velocity plane with u=Re(phi) horizontally and v=Im(phi) vertically. A black hodograph, a white outlined instantaneous-position marker and a blue arrow from the origin to that marker are shown. Green lambda/2, orange varphi/2 and blue gamma sectors reproduce the angle construction of the manuscript reference. The phase marker advances at a constant rate throughout Chapter 1; the blue sectors reset after each seven-second turn. At the north pole the hodograph is a clockwise circle; at positive latitude it is a clockwise ellipse; at the equator it collapses to a line; at negative latitude it is a counter-clockwise ellipse; and at the south pole it is a counter-clockwise circle. During the ellipticity segment the ellipse changes shape and handedness. During the orientation segment the ellipse rotates while its shape is fixed. During the common-phase segment the sphere arrow and full ellipse stay fixed while the phase motion continues at the same rate.

Chapter 2 uses four columns and two rows. Columns are labelled sigma_0, sigma_1, sigma_2 and sigma_3. The top row contains pale grey unit spheres and unnormalised Stokes vectors; the bottom row contains equal-aspect hodographs. Every column begins at the same white outlined state. In the sigma_0 column the Stokes vector changes radially and the hodograph changes scale. In the sigma_1 column the Stokes vector moves around a constant-latitude circle and the hodograph rotates without changing ellipticity. In the sigma_2 and sigma_3 columns the vector follows non-compact stretching paths and the two rotary components mix, visibly changing the hodograph shape and orientation. Axis limits and camera views remain fixed throughout each chapter."""
    manuscript_reference = (
        "The Stokes-Poincare representation and the local actions of the four "
        "matrix basis directions are illustrated dynamically in supplementary movie 1."
    )
    submission_notes = f"""Submission notes for movie 1

- Upload movie1.mp4 in ScholarOne with file designation `Movie`.
- Title and number: movie 1.
- Container: MP4.
- Video codec: H.264 ({video["profile"]} profile).
- Pixel format: {video["pixel_format"]}.
- Resolution: {video["width"]} x {video["height"]} pixels.
- Constant frame rate: {video["frame_rate_fps"]:.6g} fps.
- Duration: {video["duration_seconds"]:.2f} s.
- Frame count: {video["frame_count"]}.
- File size: {video["file_size_mb"]:.3f} MB ({video["file_size_bytes"]} bytes).
- Audio stream: none.
- Decode check: {video["decode_check"]}.
- Fast-start check (moov before mdat): {video["faststart_moov_before_mdat"]}.
- JFM checks: MP4/H.264 passed; separate English caption supplied; TeX maths is bounded by $$ in the caption; numbered and titled movie 1; below 50 MB; intended ScholarOne file designation is `Movie`.

Official requirements checked on 2026-07-27:
- JFM preparing your materials: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/information/author-instructions/preparing-your-materials
- JFM submitting your materials: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/information/author-instructions/submitting-your-materials
- Cambridge supplementary-material guidance: https://www.cambridge.org/core/services/authors/publishing-supplementary-material
"""
    validation = metadata["validation"]["metrics"]
    validation_summary = f"""Mathematical and technical validation summary for movie 1

Status: passed.

Matrix and Clifford checks
- Matrix-definition maximum error: {validation["matrix_definition_error"]:.3e}
- sigma_1^2=-sigma_0 error: {validation["sigma_1_squared"]:.3e}
- sigma_2^2=sigma_0 error: {validation["sigma_2_squared"]:.3e}
- sigma_3^2=sigma_0 error: {validation["sigma_3_squared"]:.3e}
- Anticommutator maximum error: {validation["anticommutator_error"]:.3e}
- Commutator maximum error: {validation["commutator_error"]:.3e}
- Analytic versus numerical matrix exponential error: {validation["analytic_vs_numeric_exponential_error"]:.3e}

Spinor, Stokes and hodograph checks
- Stokes norm identity error: {validation["stokes_norm_identity_error"]:.3e}
- Chapter 1 unit-Stokes error: {validation["chapter1_unit_stokes_error"]:.3e}
- Common-phase Stokes invariance error: {validation["common_phase_stokes_invariance_error"]:.3e}
- Equator minor semiaxis: {validation["equator_minor_semiaxis"]:.3e}
- Longitude/2 orientation error: {validation["longitude_half_orientation_error"]:.3e}
- Latitude/2 ellipticity error: {validation["latitude_half_ellipticity_error"]:.3e}
- sigma_0 normalised-direction error: {validation["sigma0_direction_invariance_error"]:.3e}
- sigma_1 component-magnitude error: {validation["sigma1_component_magnitude_error"]:.3e}
- sigma_1 longitude error: {validation["sigma1_longitude_error"]:.3e}
- Saved Stokes/same-spinor error: {validation["saved_stokes_same_spinor_error"]:.3e}
- Saved hodograph/same-spinor error: {validation["saved_hodograph_same_spinor_error"]:.3e}
- North-pole signed area: {validation["north_clockwise_signed_area"]:.6f} (negative means clockwise)
- South-pole signed area: {validation["south_counterclockwise_signed_area"]:.6f} (positive means counter-clockwise)
- Chapter 1 phase period/angular speed: {CHAPTER_ONE_PHASE_PERIOD_SECONDS:.1f} s / {2.0 * np.pi / CHAPTER_ONE_PHASE_PERIOD_SECONDS:.9f} rad s^-1

Video checks
- Codec/pixel format: {video["codec"]} / {video["pixel_format"]}
- Resolution/frame rate: {video["width"]} x {video["height"]} / {video["frame_rate_fps"]:.6g} fps
- Frame count/duration: {video["frame_count"]} / {video["duration_seconds"]:.2f} s
- Audio stream present: {video["audio_stream_present"]}
- Decode check: {video["decode_check"]}
- File size below 50 MB: {video["file_size_bytes"] < 50_000_000}
"""
    visual_inspection = """Visual inspection record for movie 1

Reference comparison
- Figure 1 was rendered before production. The movie uses the same left-right Stokes-sphere/hodograph logic, white background, serif mathematical typography, pale grey sphere, black hodograph and white outlined state marker.
- Chapter 1 also reproduces the reference green lambda, orange varphi and blue gamma angle sectors, with half-angle labels on the hodograph.
- Figure 2 was rendered before production. Chapter 2 uses the same four generator columns, Stokes panels above hodograph panels, blue positive direction, red negative direction and common white initial state. Dashed red versus solid blue line style and square versus circular markers add colour-independent identification.

Representative encoded frames inspected
- 4.3 s: north-pole clockwise circular polarisation.
- 6.3 s: positive-latitude elliptical polarisation.
- 8.5 s: equatorial linear polarisation.
- 12.6 s: south-pole counter-clockwise circular polarisation.
- 16.5 s: ellipticity scan.
- 23.5 s: longitude-driven ellipse rotation.
- 30.5 s: common-phase motion with fixed Stokes vector and ellipse.
- 34.7 s: transition between chapters.
- 36.1 s: common initial frame for all four generators.
- 43.0 s: intermediate generator actions.
- 49.8 s: positive and negative generator endpoints.
- 52.5 s: final held frame.

Checks and result
- Mathematical labels and spinor convention: passed.
- Clockwise/counter-clockwise handedness: passed.
- Sphere and hodograph synchronisation: passed.
- Unit-vector normalisation on the Chapter 1 Bloch sphere: passed.
- Numerical spinor entries and the displayed spinor convention: passed.
- Uniform seven-second phase turns and blue-arc reset: passed.
- Northern-hemisphere clockwise and southern-hemisphere counter-clockwise motion: passed.
- Blue hodograph arrow terminates at the white phase marker: passed.
- Fixed camera, projection and axis limits within each chapter: passed.
- Text legibility at 1920 x 1080: passed.
- Label, arrow and trajectory overlap: passed.
- Cropping, flicker, axis jumps and compression artefacts: no defects observed.
- sigma_0 radial change: clearly visible.
- sigma_1 compact rotation versus sigma_2/sigma_3 non-compact stretching: clearly distinguishable.
- Final result: passed."""

    write_text(output_directory / "movie1_caption.txt", caption)
    write_text(
        output_directory / "movie1_accessibility_description.txt",
        accessibility,
    )
    write_text(
        output_directory / "movie1_submission_notes.txt",
        submission_notes,
    )
    write_text(
        output_directory / "movie1_manuscript_reference_suggestion.txt",
        manuscript_reference,
    )
    write_text(
        output_directory / "movie1_validation_summary.txt",
        validation_summary,
    )
    write_text(
        output_directory / "movie1_visual_inspection.txt",
        visual_inspection,
    )

    readme_section = f"""## Movie 1 - polarisation geometry

`movie1.mp4` dynamically explains the Stokes-Poincare mapping in Figure 1 and the local matrix-basis actions in Figure 2. Chapter 1 shows the unit Bloch/Stokes vector, the corresponding numerical spinor, all three angle arcs, and a uniformly advancing seven-second phase cycle. It is silent, encoded as H.264/yuv420p at {video["width"]} x {video["height"]} and {video["frame_rate_fps"]:.6g} fps, and is accompanied by a separate caption and accessibility description.

Files:

- `movie1.mp4`: submission movie.
- `movie1_preview.png`: representative still image.
- `movie1_caption.txt`: separate JFM caption with TeX mathematics.
- `movie1_accessibility_description.txt`: visual description independent of audio and colour.
- `movie1_submission_notes.txt`: ScholarOne and technical checks.
- `movie1_manuscript_reference_suggestion.txt`: suggested sentence only; it has not been inserted into the manuscript.
- `movie1_validation_summary.txt`: numerical and encoding validation results.
- `movie1_visual_inspection.txt`: representative-frame visual inspection record.
- `movie1_data.npz`: spinor, Stokes and hodograph arrays used by the renderer.
- `movie1_metadata.json`: definitions, initial state, display limits and validation metrics.

The manuscript spinor convention is `{metadata["spinor_convention"]}`. The common Figure 2 initial state is:

- `A_up = {format_complex(initial["A_up"])}`;
- `A_down = {format_complex(initial["A_down"])}`;
- `conj(A_down) = {format_complex(initial["stored_conjugate_A_down"])}`;
- `S = ({initial["stokes_vector"][0]:.12g}, {initial["stokes_vector"][1]:.12g}, {initial["stokes_vector"][2]:.12g})`;
- `|S| = {initial["stokes_magnitude"]:.12g}`;
- `varphi = {initial["varphi_radians"]:.12g} rad ({initial["varphi_degrees"]:.6f} deg)`;
- `lambda = {initial["lambda_radians"]:.12g} rad ({initial["lambda_degrees"]:.6f} deg)`;
- `gamma = {initial["gamma_radians"]:.12g} rad ({initial["gamma_degrees"]:.6f} deg)`.

The displayed Chapter 2 parameter is the matrix-generator action parameter `f t`, not a background-flow simulation time. The unit spheres in Chapter 2 are scale references; the Stokes vectors are not normalised.

Recreate the deliverables from the repository root with an output directory supplied at run time:

```bash
python run_workflow.py polarisation_geometry_movie --output-directory path/to/movies --validate
```
"""
    update_movies_readme(output_directory / "README.md", readme_section)


def render_movie(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_directory: Path,
    *,
    ffmpeg: str,
    width: int,
    height: int,
    fps: int,
    crf: int,
    use_tex: bool,
    preview_only: bool,
) -> tuple[dict[str, Any] | None, int]:
    """Render the complete timeline or only the representative preview."""
    chapter_one = make_chapter_one(width, height, use_tex=use_tex)
    chapter_two = make_chapter_two(width, height, use_tex=use_tex)
    sample_count = arrays["unit_progress"].size

    preview_index = int(round(0.55 * (sample_count - 1)))
    set_chapter_two_frame(chapter_two, arrays, preview_index)
    preview_frame = canvas_rgb(chapter_two.figure)
    Image.fromarray(preview_frame).save(output_directory / PREVIEW_FILENAME)
    if preview_only:
        plt.close(chapter_one.figure)
        plt.close(chapter_two.figure)
        return None, 0

    segments = (
        ("opening", 2.0),
        ("chapter1_title", 2.0),
        ("landmark", 9.0),
        ("ellipticity", 7.0),
        ("orientation", 7.0),
        ("phase", 7.0),
        ("chapter2_title", 2.0),
        ("generator", 14.0),
        ("final_hold", 3.0),
    )
    frame_counts = {name: int(round(duration * fps)) for name, duration in segments}
    total_frames = sum(frame_counts.values())
    opening = title_frame(
        width,
        height,
        "Supplementary movie 1",
        "Polarisation geometry of a near-inertial wave",
        use_tex=use_tex,
    )
    chapter1_title = title_frame(
        width,
        height,
        "Chapter 1",
        "Stokes-Poincare sphere and physical hodograph",
        use_tex=use_tex,
    )
    chapter2_title = title_frame(
        width,
        height,
        "Chapter 2",
        "Exact positive and negative actions of the four matrix basis directions",
        use_tex=use_tex,
    )
    output_path = output_directory / MOVIE_FILENAME
    encoder = start_encoder(
        ffmpeg,
        output_path,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
    )
    if encoder.stdin is None:
        raise RuntimeError("Could not open the ffmpeg input pipe.")

    written = 0

    def send(frame: np.ndarray) -> None:
        nonlocal written
        encoder.stdin.write(frame.tobytes())
        written += 1

    for _ in range(frame_counts["opening"]):
        send(opening)
    for _ in range(frame_counts["chapter1_title"]):
        send(chapter1_title)

    chapter_one_frame = 0

    def current_phase_angle() -> float:
        elapsed = chapter_one_frame / fps
        return float(
            2.0
            * np.pi
            * ((elapsed / CHAPTER_ONE_PHASE_PERIOD_SECONDS) % 1.0)
        )

    count = frame_counts["landmark"]
    for frame_index in range(count):
        progress = frame_index / max(count - 1, 1)
        fraction, label = landmark_schedule(progress)
        data_index = int(round(fraction * (sample_count - 1)))
        set_chapter_one_frame(
            chapter_one,
            arrays,
            "landmark",
            data_index,
            phase_angle=current_phase_angle(),
            landmark_label=label,
        )
        send(canvas_rgb(chapter_one.figure))
        chapter_one_frame += 1

    for stage in ("ellipticity", "orientation", "phase"):
        count = frame_counts[stage]
        for frame_index in range(count):
            progress = frame_index / max(count - 1, 1)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            data_index = int(round(smooth * (sample_count - 1)))
            set_chapter_one_frame(
                chapter_one,
                arrays,
                stage,
                data_index,
                phase_angle=current_phase_angle(),
            )
            send(canvas_rgb(chapter_one.figure))
            chapter_one_frame += 1

    for _ in range(frame_counts["chapter2_title"]):
        send(chapter2_title)

    count = frame_counts["generator"]
    final_generator_frame: np.ndarray | None = None
    for frame_index in range(count):
        progress = frame_index / max(count - 1, 1)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        data_index = int(round(smooth * (sample_count - 1)))
        set_chapter_two_frame(chapter_two, arrays, data_index)
        final_generator_frame = canvas_rgb(chapter_two.figure)
        send(final_generator_frame)
    if final_generator_frame is None:
        raise RuntimeError("No Chapter 2 frames were rendered.")
    for _ in range(frame_counts["final_hold"]):
        send(final_generator_frame)

    encoder.stdin.close()
    return_code = encoder.wait()
    plt.close(chapter_one.figure)
    plt.close(chapter_two.figure)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}.")
    if written != total_frames:
        raise RuntimeError(f"Wrote {written} frames; expected {total_frames}.")

    video = probe_video(ffmpeg, output_path)
    validate_video_target(
        video,
        width=width,
        height=height,
        fps=fps,
        expected_frames=total_frames,
    )
    return video, total_frames


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render and encode supplementary movie 1."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory containing movie1_data.npz and movie1_metadata.json.",
    )
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--no-tex", action="store_true")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Render movie1_preview.png without encoding the movie.",
    )
    return parser.parse_args()


def main() -> None:
    """Load calculated states, render the movie and write sidecar files."""
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("Width, height and fps must be positive.")
    if args.width % 2 or args.height % 2:
        raise ValueError("yuv420p output requires even width and height.")
    # The figures are updated and drawn after their construction contexts have
    # exited, so retain the typography settings for the complete render.
    mpl.rcParams.update(publication_style(use_tex=not args.no_tex))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    data_path = args.output_directory / DATA_FILENAME
    metadata_path = args.output_directory / METADATA_FILENAME
    if not data_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Run compute_polarisation_trajectories.py before rendering."
        )
    with np.load(data_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ffmpeg = locate_ffmpeg(args.ffmpeg)
    video, total_frames = render_movie(
        arrays,
        metadata,
        args.output_directory,
        ffmpeg=ffmpeg,
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
        use_tex=not args.no_tex,
        preview_only=args.preview_only,
    )
    if args.preview_only:
        print(args.output_directory / PREVIEW_FILENAME)
        return

    if video is None:
        raise RuntimeError("Video metadata was not produced.")
    metadata["video"] = {
        **video,
        "expected_frame_count": total_frames,
        "ffmpeg_build": Path(ffmpeg).name,
    }
    metadata.setdefault("display", {}).update(
        {
            "chapter_1_phase_period_seconds": CHAPTER_ONE_PHASE_PERIOD_SECONDS,
            "chapter_1_phase_angular_speed_radians_per_second": (
                2.0 * np.pi / CHAPTER_ONE_PHASE_PERIOD_SECONDS
            ),
            "chapter_1_phase_resets_after_radians": 2.0 * np.pi,
            "chapter_1_northern_hemisphere_motion": "clockwise",
            "chapter_1_southern_hemisphere_motion": "counter-clockwise",
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_auxiliary_files(args.output_directory, metadata, video)
    print(args.output_directory / MOVIE_FILENAME)
    print(args.output_directory / PREVIEW_FILENAME)
    print("video validation: passed")


if __name__ == "__main__":
    main()
