"""Render supplementary movie 1 from precomputed polarisation data.

This script performs no spinor evolution or Stokes conversion. It reads the
verified histories from ``movie1_data.npz``, evaluates the display-time fast
carrier phase from those saved spinors, draws fixed publication-style layouts,
encodes H.264 video and writes submission sidecars.
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
import numpy as np
from PIL import Image


DATA_FILENAME = "movie1_data.npz"
METADATA_FILENAME = "movie1_metadata.json"
MOVIE_FILENAME = "movie1.mp4"
PREVIEW_FILENAME = "movie1_preview.png"
PREFERRED_TEXT_FONT = "Times New Roman"
MOVIE_FONT_STACK = (
    PREFERRED_TEXT_FONT,
    "Times",
    "STIXGeneral",
    "DejaVu Serif",
)
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
CHAPTER_ONE_PHASE_TURN_SECONDS = 7.0
CHAPTER_ONE_PHASE_HOLD_SECONDS = 5.0
GENERATOR_ENDPOINT_HOLD_SECONDS = 5.0
CHAPTER_ONE_HODOGRAPH_PANEL_SCALE = 0.8
GAMMA_LABEL_DISTANCE_RATIO = 1.24
CHAPTER_ONE_PHASE_BACKGROUND_ALPHA = 0.18
GENERATOR_SPHERE_SCALE = 1.4
GENERATOR_FAST_PHASE_TURNS = 4.0
GENERATOR_INITIAL_RAY_ANGLE = np.deg2rad(54.75)


def publication_style(*, use_tex: bool) -> dict[str, object]:
    """Return a movie-safe version of the manuscript figure style."""
    style: dict[str, object] = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": list(MOVIE_FONT_STACK),
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
    dashed_hidden_guides: bool = True,
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
            linestyle=(0, (3.0, 2.5)) if dashed_hidden_guides else "-",
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
    labels = (
        r"$\mathrm{S}_{x}$",
        r"$\mathrm{S}_{y}$",
        r"$\mathrm{S}_{z}$",
    )
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
        tick_values = (
            [-2.0, -1.0, 0.0, 1.0, 2.0]
            if limit >= 2.0
            else [-1.0, 0.0, 1.0]
        )
        axis.set_xticks(tick_values)
        axis.set_yticks(tick_values)
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


def final_fast_phase_schedule(
    turn_frame_count: int,
    hold_frame_count: int,
) -> np.ndarray:
    """Return one uniform 2-pi fast-phase turn, completed hold and reset."""
    if turn_frame_count < 2:
        raise ValueError("The final fast-phase turn requires at least two frames.")
    if hold_frame_count < 1:
        raise ValueError("The completed fast-phase turn requires a hold frame.")
    progress = np.arange(turn_frame_count, dtype=float) / (
        turn_frame_count - 1
    )
    turn_angle = 2.0 * np.pi * progress
    # The completed-turn frame is already the last turn frame, so append
    # hold_frame_count - 1 duplicates to obtain the requested hold duration.
    hold_angle = np.full(hold_frame_count - 1, turn_angle[-1])
    reset_angle = np.asarray([0.0])
    return np.concatenate([turn_angle, hold_angle, reset_angle])


def fast_phase_for_hodograph_point(
    spinor: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return theta for A_up exp(-i theta)+A_down exp(i theta)=target."""
    component_up = complex(spinor[0])
    component_down = complex(np.conj(spinor[1]))
    target_complex = complex(target[0], target[1])
    if abs(component_down) <= 1.0e-12:
        phase_factor = component_up / target_complex
        phase_factor /= abs(phase_factor)
        return float(np.angle(phase_factor))
    roots = np.roots([component_down, -target_complex, component_up])
    phase_factor = min(roots, key=lambda value: abs(abs(value) - 1.0))
    phase_factor /= abs(phase_factor)
    phase = float(np.angle(phase_factor))
    reconstructed = (
        component_up * np.exp(-1j * phase)
        + component_down * np.exp(1j * phase)
    )
    if abs(reconstructed - target_complex) > 2.0e-10:
        raise ValueError("Could not align the rotary-vector decomposition.")
    return phase


def rotary_component_vectors(
    spinor: np.ndarray,
    reference_marker: np.ndarray,
    elapsed_fast_phase: float,
) -> tuple[complex, complex]:
    """Return clockwise and counter-clockwise rotary vectors."""
    phase_offset = fast_phase_for_hodograph_point(spinor, reference_marker)
    phase = phase_offset + elapsed_fast_phase
    clockwise = complex(spinor[0]) * np.exp(-1j * phase)
    counterclockwise = complex(np.conj(spinor[1])) * np.exp(1j * phase)
    return clockwise, counterclockwise


def hodograph_phase_on_ray(
    spinor: np.ndarray,
    ray_angle: float,
) -> tuple[float, complex]:
    """Return the positive-ray phase and point on a spinor hodograph."""
    component_up = complex(spinor[0])
    component_down = complex(np.conj(spinor[1]))
    cosine_coefficient = component_up + component_down
    sine_coefficient = 1j * (component_down - component_up)
    rotation = np.exp(-1j * ray_angle)
    normal_cosine = float(np.imag(rotation * cosine_coefficient))
    normal_sine = float(np.imag(rotation * sine_coefficient))
    if np.hypot(normal_cosine, normal_sine) <= 1.0e-12:
        raise ValueError("The requested hodograph ray is degenerate.")
    base_phase = float(np.arctan2(-normal_cosine, normal_sine))
    candidates: list[tuple[float, float, complex]] = []
    for phase in (base_phase, base_phase + np.pi):
        point = (
            component_up * np.exp(-1j * phase)
            + component_down * np.exp(1j * phase)
        )
        radial_coordinate = float(np.real(rotation * point))
        candidates.append((radial_coordinate, phase, point))
    radial_coordinate, phase, point = max(candidates, key=lambda item: item[0])
    if radial_coordinate <= 0.0:
        raise ValueError("The hodograph does not intersect the positive ray.")
    cross_ray_error = abs(float(np.imag(rotation * point)))
    if cross_ray_error > 2.0e-12:
        raise ValueError("The hodograph-ray intersection is inaccurate.")
    return float(np.mod(phase, 2.0 * np.pi)), point


def generator_phi_track(
    spinors: np.ndarray,
    fast_phase: np.ndarray,
    *,
    phase_offset: float,
) -> np.ndarray:
    """Return the fast carrier motion along one slow generator branch."""
    if spinors.shape[0] != fast_phase.size:
        raise ValueError("Spinor and fast-phase tracks are misaligned.")
    phase = phase_offset + fast_phase
    return (
        spinors[:, 0] * np.exp(-1j * phase)
        + np.conj(spinors[:, 1]) * np.exp(1j * phase)
    )


def gradient_segment_colours(color: str, count: int) -> np.ndarray:
    """Return a pale-to-saturated solid colour sequence."""
    if count <= 0:
        return np.empty((0, 4))
    target = np.asarray(mpl.colors.to_rgba(color))
    pale = target.copy()
    pale[:3] = 0.82 * np.ones(3) + 0.18 * target[:3]
    blend = np.linspace(0.0, 1.0, count)[:, None]
    colours = pale[None, :] * (1.0 - blend) + target[None, :] * blend
    colours[:, 3] = np.linspace(0.38, 1.0, count)
    return colours


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
    hodograph_coordinate_guides: tuple[Line2D, Line2D]
    major_axis: Line2D
    phase_reference: Line2D
    hodograph_direction_triangles: tuple[Polygon, Polygon]
    clockwise_circle: Line2D
    counterclockwise_circle: Line2D
    clockwise_component_arrow: FancyArrowPatch
    counterclockwise_component_arrow: FancyArrowPatch
    rotary_sum_guides: Line2D
    phase_time_text: mpl.text.Text
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


def hodograph_phase_marker(
    varphi: float,
    orientation: float,
    relative_phase: float,
) -> tuple[np.ndarray, float, float]:
    """Return the phase-ray intersection with the current hodograph ellipse."""
    cosine_latitude = max(float(np.cos(varphi)), 0.0)
    semi_major = np.sqrt(1.0 + cosine_latitude)
    semi_minor = np.sqrt(max(1.0 - cosine_latitude, 0.0))
    major_direction = np.array([np.cos(orientation), np.sin(orientation)])
    minor_direction = np.array([-np.sin(orientation), np.cos(orientation)])
    ray_angle = orientation + relative_phase
    ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
    ray_local = np.array(
        [
            np.dot(ray_direction, major_direction),
            np.dot(ray_direction, minor_direction),
        ]
    )
    if semi_minor <= 1.0e-12:
        if abs(ray_local[1]) <= 1.0e-12:
            ray_length = semi_major / max(abs(ray_local[0]), 1.0e-12)
        else:
            ray_length = 0.0
    else:
        ray_length = 1.0 / np.sqrt(
            (ray_local[0] / semi_major) ** 2
            + (ray_local[1] / semi_minor) ** 2
        )
    return ray_length * ray_direction, semi_major, semi_minor


def hodograph_marker_ellipse_error(
    marker: np.ndarray,
    semi_major: float,
    semi_minor: float,
    orientation: float,
) -> float:
    """Return an implicit-ellipse or degenerate-line membership residual."""
    major_direction = np.array([np.cos(orientation), np.sin(orientation)])
    minor_direction = np.array([-np.sin(orientation), np.cos(orientation)])
    local_major = float(np.dot(marker, major_direction))
    local_minor = float(np.dot(marker, minor_direction))
    if semi_minor <= 1.0e-12:
        return abs(local_minor)
    return abs(
        (local_major / semi_major) ** 2
        + (local_minor / semi_minor) ** 2
        - 1.0
    )


def hodograph_direction_triangle(
    values: np.ndarray,
    index: int,
    *,
    length: float = 0.13,
    half_width: float = 0.050,
) -> np.ndarray:
    """Return a tangent-aligned triangular arrowhead on a closed hodograph."""
    points = np.column_stack([np.real(values), np.imag(values)])
    if len(points) > 2 and np.linalg.norm(points[0] - points[-1]) < 1.0e-10:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("A direction triangle requires a closed sampled curve.")
    index %= len(points)
    tangent = points[(index + 1) % len(points)] - points[index - 1]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-12:
        raise ValueError("The hodograph tangent is degenerate.")
    tangent /= tangent_norm
    normal = np.array([-tangent[1], tangent[0]])
    center = points[index]
    tip = center + 0.52 * length * tangent
    base = center - 0.48 * length * tangent
    return np.vstack(
        [
            tip,
            base + half_width * normal,
            base - half_width * normal,
        ]
    )


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
        hodograph_coordinate_guides = tuple(hodograph_axis.lines[-2:])
        hodograph_position = hodograph_axis.get_position()
        hodograph_axis.set_position(
            [
                hodograph_position.x0
                + 0.5
                * (1.0 - CHAPTER_ONE_HODOGRAPH_PANEL_SCALE)
                * hodograph_position.width,
                hodograph_position.y0
                + 0.5
                * (1.0 - CHAPTER_ONE_HODOGRAPH_PANEL_SCALE)
                * hodograph_position.height,
                CHAPTER_ONE_HODOGRAPH_PANEL_SCALE * hodograph_position.width,
                CHAPTER_ONE_HODOGRAPH_PANEL_SCALE * hodograph_position.height,
            ]
        )
        sphere_axis.set_title("Stokes-Poincare representation", pad=4)
        figure.text(
            hodograph_position.x0 + 0.5 * hodograph_position.width,
            0.815,
            "Horizontal velocity hodograph",
            ha="center",
            va="center",
            fontsize=17,
        )
        sphere_axis.text(
            -1.05,
            1.02,
            r"$\left|\mathbf{S}\right|=1$",
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
            zorder=12,
        )
        phase_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.3,
            color=BLACK,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=11,
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
        (phase_reference,) = hodograph_axis.plot(
            [],
            [],
            color="0.58",
            linewidth=1.4,
            zorder=3.7,
        )
        direction_triangles = tuple(
            Polygon(
                np.zeros((3, 2)),
                closed=True,
                facecolor=BLACK,
                edgecolor="white",
                linewidth=0.7,
                visible=False,
                zorder=4.8,
            )
            for _ in range(2)
        )
        for triangle in direction_triangles:
            hodograph_axis.add_patch(triangle)
        (clockwise_circle,) = hodograph_axis.plot(
            [],
            [],
            color=BLUE,
            linewidth=2.0,
            zorder=7,
        )
        (counterclockwise_circle,) = hodograph_axis.plot(
            [],
            [],
            color=RED,
            linewidth=2.0,
            zorder=7,
        )
        clockwise_component_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.1,
            color=BLUE,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=8,
        )
        counterclockwise_component_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.1,
            color=RED,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=8,
        )
        hodograph_axis.add_patch(clockwise_component_arrow)
        hodograph_axis.add_patch(counterclockwise_component_arrow)
        (rotary_sum_guides,) = hodograph_axis.plot(
            [],
            [],
            color="0.28",
            linewidth=1.5,
            linestyle=(0, (4.0, 3.0)),
            zorder=9,
        )
        phase_time_text = hodograph_axis.text(
            0.04,
            0.95,
            "",
            transform=hodograph_axis.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            color=BLACK,
            zorder=13,
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
            0.73,
            r"Polarisation spinor $\left|\mathscr{A}\right\rangle$",
            ha="center",
            va="center",
            fontsize=20,
        )
        spinor_axis.text(
            0.12,
            0.49,
            "(",
            ha="center",
            va="center",
            fontsize=78,
            color=BLACK,
        )
        spinor_axis.text(
            0.88,
            0.49,
            ")",
            ha="center",
            va="center",
            fontsize=78,
            color=BLACK,
        )
        spinor_top = plain_dynamic_text(
            spinor_axis.text(
                0.5,
                0.55,
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
                0.43,
                "",
                ha="center",
                va="center",
                fontsize=14,
                color=BLACK,
            )
        )
        figure.text(
            0.5,
            0.962,
            "Chapter 1: one spinor, two geometric representations",
            ha="center",
            va="top",
            fontsize=25,
        )
        subtitle = figure.text(
            0.5,
            0.875,
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
        hodograph_coordinate_guides=hodograph_coordinate_guides,
        major_axis=major_axis,
        phase_reference=phase_reference,
        hodograph_direction_triangles=direction_triangles,
        clockwise_circle=clockwise_circle,
        counterclockwise_circle=counterclockwise_circle,
        clockwise_component_arrow=clockwise_component_arrow,
        counterclockwise_component_arrow=counterclockwise_component_arrow,
        rotary_sum_guides=rotary_sum_guides,
        phase_time_text=phase_time_text,
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

    stokes_trail: Line2D
    stokes_arrow: FancyArrowPatch
    stokes_marker: Line2D
    initial_stokes_marker: Line2D
    phi_trail: LineCollection
    phi_vector: FancyArrowPatch
    phi_marker: Line2D
    initial_phi_marker: Line2D


@dataclass
class GeneratorChapterArtists:
    """Mutable artists for one signed four-generator chapter."""

    figure: mpl.figure.Figure
    panels: list[GeneratorPanelArtists]
    parameter_text: mpl.text.Text
    direction: str
    color: str
    chapter_number: int


def make_generator_chapter(
    width: int,
    height: int,
    *,
    use_tex: bool,
    direction: str,
    chapter_number: int,
) -> GeneratorChapterArtists:
    """Create one solid-style positive or negative generator layout."""
    if direction not in {"positive", "negative"}:
        raise ValueError("Generator direction must be positive or negative.")
    color = BLUE if direction == "positive" else RED
    direction_label = "Positive" if direction == "positive" else "Negative"
    dpi = 120
    with mpl.rc_context(publication_style(use_tex=use_tex)):
        figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(
            2,
            4,
            left=0.035,
            right=0.985,
            bottom=0.13,
            top=0.87,
            wspace=0.20,
            hspace=0.14,
            height_ratios=[1.08, 1.0],
        )
        figure.text(
            0.5,
            0.962,
            (
                f"Chapter {chapter_number}: {direction_label.lower()} "
                "actions of the four matrix basis directions"
            ),
            ha="center",
            va="top",
            fontsize=25,
        )
        top_titles = (
            r"$\sigma_0$: $r$-change",
            r"$\sigma_1$: $z$-rotation",
            r"$\sigma_2$: $x$-translation",
            r"$\sigma_3$: $y$-translation",
        )
        panels: list[GeneratorPanelArtists] = []
        for column in range(4):
            sphere_axis = figure.add_subplot(grid[0, column])
            hodograph_axis = figure.add_subplot(grid[1, column])
            sphere_position = sphere_axis.get_position()
            sphere_title_x = 0.5 * (
                sphere_position.x0 + sphere_position.x1
            )
            sphere_axis.set_position(
                [
                    sphere_position.x0,
                    sphere_position.y0 - 0.055,
                    sphere_position.width,
                    sphere_position.height,
                ]
            )
            draw_poincare_sphere(
                sphere_axis,
                limit=2.55 / GENERATOR_SPHERE_SCALE,
                longitude=np.pi / 3.0,
                label_axes=True,
                dashed_hidden_guides=False,
            )
            draw_hodograph_axes(hodograph_axis, limit=2.05, ticks=True)
            figure.text(
                sphere_title_x,
                0.885,
                top_titles[column],
                ha="center",
                va="center",
                fontsize=mpl.rcParams["axes.titlesize"],
            )
            hodograph_axis.set_title(
                rf"$\phi$ track for $\sigma_{column}$",
                pad=8,
            )
            if column > 0:
                hodograph_axis.set_ylabel("")
                hodograph_axis.tick_params(labelleft=False)

            (stokes_trail,) = sphere_axis.plot(
                [],
                [],
                color=color,
                linewidth=2.5,
                clip_on=False,
                zorder=7,
            )
            stokes_arrow = FancyArrowPatch(
                (0.0, 0.0),
                (0.0, 0.0),
                arrowstyle="-|>",
                mutation_scale=16,
                color=color,
                linewidth=2.5,
                clip_on=False,
                zorder=8,
            )
            sphere_axis.add_patch(stokes_arrow)
            (stokes_marker,) = sphere_axis.plot(
                [],
                [],
                marker="o",
                markersize=6.5,
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="None",
                clip_on=False,
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
                clip_on=False,
                zorder=10,
            )

            phi_trail = LineCollection(
                [],
                linewidths=2.7,
                capstyle="round",
                zorder=4,
            )
            hodograph_axis.add_collection(phi_trail)
            phi_vector = FancyArrowPatch(
                (0.0, 0.0),
                (0.0, 0.0),
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=2.4,
                color=color,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=5,
            )
            hodograph_axis.add_patch(phi_vector)
            (phi_marker,) = hodograph_axis.plot(
                [],
                [],
                marker="o",
                markersize=7.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.9,
                linestyle="None",
                zorder=6,
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
                    stokes_trail=stokes_trail,
                    stokes_arrow=stokes_arrow,
                    stokes_marker=stokes_marker,
                    initial_stokes_marker=initial_stokes_marker,
                    phi_trail=phi_trail,
                    phi_vector=phi_vector,
                    phi_marker=phi_marker,
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
    return GeneratorChapterArtists(
        figure=figure,
        panels=panels,
        parameter_text=parameter_text,
        direction=direction,
        color=color,
        chapter_number=chapter_number,
    )


def set_chapter_one_frame(
    artists: ChapterOneArtists,
    arrays: dict[str, np.ndarray],
    stage: str,
    index: int,
    *,
    displayed_gamma: float,
    elapsed_fast_phase: float = 0.0,
    landmark_label: str | None = None,
) -> None:
    """Update the two-panel mapping layout from saved arrays."""
    stokes = arrays[f"{stage}_stokes"]
    hodographs = arrays[f"{stage}_hodograph"]
    spinors = arrays[f"{stage}_spinor"]
    varphi = arrays[f"{stage}_varphi"]
    longitude = arrays[f"{stage}_lambda"]
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
    artists.hodograph.set_data(np.real(values), np.imag(values))
    unique_value_count = len(values)
    if len(values) > 2 and abs(values[0] - values[-1]) < 1.0e-10:
        unique_value_count -= 1
    show_direction = abs(float(np.sin(varphi[index]))) >= 0.04
    direction_indices = (
        unique_value_count // 8,
        5 * unique_value_count // 8,
    )
    for triangle, direction_index in zip(
        artists.hodograph_direction_triangles,
        direction_indices,
        strict=True,
    ):
        if show_direction:
            triangle.set_xy(
                hodograph_direction_triangle(values, direction_index)
            )
            triangle.set_visible(True)
        else:
            triangle.set_visible(False)

    # Gamma is fixed relative to the current dashed major-axis guide.  During
    # the final section, two counter-rotating circular vectors reconstruct the
    # instantaneous hodograph marker and black resultant arrow.
    phase_orientation = float(longitude[index]) / 2.0
    phase_varphi = float(varphi[index])
    phase_gamma = float(arrays["initial_gamma"])
    hodograph_gamma = -phase_gamma
    reference_marker, _, _ = hodograph_phase_marker(
        phase_varphi,
        phase_orientation,
        hodograph_gamma,
    )
    if stage == "phase":
        clockwise, counterclockwise = rotary_component_vectors(
            spinors[index],
            reference_marker,
            elapsed_fast_phase,
        )
        marker_complex = clockwise + counterclockwise
        marker = np.array(
            [np.real(marker_complex), np.imag(marker_complex)]
        )
    else:
        clockwise = 0.0j
        counterclockwise = 0.0j
        marker = reference_marker
    artists.hodograph_marker.set_data([marker[0]], [marker[1]])
    artists.phase_arrow.set_positions((0.0, 0.0), tuple(marker))
    orientation = longitude[index] / 2.0
    guide = 1.45 * np.array([np.cos(orientation), np.sin(orientation)])
    artists.major_axis.set_data(
        [-guide[0], guide[0]],
        [-guide[1], guide[1]],
    )
    if stage == "phase":
        artists.phase_reference.set_data(
            [0.0, reference_marker[0]],
            [0.0, reference_marker[1]],
        )
    else:
        artists.phase_reference.set_data([], [])

    phase_overlay_visible = stage == "phase"
    background_alpha = (
        CHAPTER_ONE_PHASE_BACKGROUND_ALPHA if phase_overlay_visible else 1.0
    )
    for background_artist in (
        artists.hodograph,
        *artists.hodograph_coordinate_guides,
        artists.major_axis,
        artists.phase_reference,
        *artists.hodograph_direction_triangles,
        artists.hodograph_lambda_half.arc,
        artists.hodograph_lambda_half.label,
        artists.hodograph_varphi_half.arc,
        artists.hodograph_varphi_half.label,
        artists.hodograph_gamma.arc,
        artists.hodograph_gamma.label,
        artists.ellipticity_chord,
    ):
        background_artist.set_alpha(background_alpha)
    artists.hodograph_lambda_half.fill.set_alpha(
        background_alpha if phase_overlay_visible else 0.70
    )
    artists.hodograph_varphi_half.fill.set_alpha(
        background_alpha if phase_overlay_visible else 0.70
    )
    artists.hodograph_gamma.fill.set_alpha(
        background_alpha if phase_overlay_visible else 0.18
    )

    for overlay_artist in (
        artists.clockwise_circle,
        artists.counterclockwise_circle,
        artists.clockwise_component_arrow,
        artists.counterclockwise_component_arrow,
        artists.rotary_sum_guides,
        artists.phase_time_text,
    ):
        overlay_artist.set_visible(phase_overlay_visible)
    if phase_overlay_visible:
        circle_theta = np.linspace(0.0, 2.0 * np.pi, 361)
        clockwise_radius = abs(clockwise)
        counterclockwise_radius = abs(counterclockwise)
        artists.clockwise_circle.set_data(
            clockwise_radius * np.cos(circle_theta),
            clockwise_radius * np.sin(circle_theta),
        )
        artists.counterclockwise_circle.set_data(
            counterclockwise_radius * np.cos(circle_theta),
            counterclockwise_radius * np.sin(circle_theta),
        )
        clockwise_point = np.array(
            [np.real(clockwise), np.imag(clockwise)]
        )
        counterclockwise_point = np.array(
            [np.real(counterclockwise), np.imag(counterclockwise)]
        )
        artists.clockwise_component_arrow.set_positions(
            (0.0, 0.0),
            tuple(clockwise_point),
        )
        artists.counterclockwise_component_arrow.set_positions(
            (0.0, 0.0),
            tuple(counterclockwise_point),
        )
        artists.rotary_sum_guides.set_data(
            [
                clockwise_point[0],
                marker[0],
                np.nan,
                counterclockwise_point[0],
                marker[0],
            ],
            [
                clockwise_point[1],
                marker[1],
                np.nan,
                counterclockwise_point[1],
                marker[1],
            ],
        )
        artists.phase_time_text.set_text(
            rf"$t={elapsed_fast_phase / np.pi:.2f}\,\pi/f$"
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
    sphere_directed_phase = displayed_gamma
    gamma_radius = 0.15
    gamma_theta = np.linspace(0.0, sphere_directed_phase, 121)
    gamma_points_3d = current_stokes + gamma_radius * (
        np.cos(gamma_theta)[:, None] * tangent_a
        + np.sin(gamma_theta)[:, None] * tangent_b
    )
    gamma_points, _ = project_stokes(gamma_points_3d)
    gamma_center = endpoint
    gamma_label_angle = 0.5 * sphere_directed_phase
    gamma_label_3d = (
        current_stokes
        + GAMMA_LABEL_DISTANCE_RATIO * gamma_radius
        * (
        np.cos(gamma_label_angle) * tangent_a
        + np.sin(gamma_label_angle) * tangent_b
        )
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

    gamma_delta = hodograph_gamma
    initial_reference_marker, _, _ = hodograph_phase_marker(
        float(arrays["initial_varphi"]),
        float(arrays["initial_lambda"]) / 2.0,
        -float(arrays["initial_gamma"]),
    )
    initial_reference_length = float(np.linalg.norm(initial_reference_marker))
    if initial_reference_length <= 1.0e-12:
        raise ValueError("The initial hodograph phase reference is degenerate.")
    gamma_radius_to_reference_length = 0.54 / initial_reference_length
    hodograph_gamma_radius = (
        gamma_radius_to_reference_length
        * float(np.linalg.norm(reference_marker))
    )
    hodograph_gamma_theta = np.linspace(
        phase_orientation,
        phase_orientation + gamma_delta,
        181,
    )
    hodograph_gamma_points = hodograph_gamma_radius * np.column_stack(
        [np.cos(hodograph_gamma_theta), np.sin(hodograph_gamma_theta)]
    )
    hodograph_gamma_label_angle = phase_orientation + 0.5 * gamma_delta
    hodograph_gamma_label = (
        GAMMA_LABEL_DISTANCE_RATIO
        * hodograph_gamma_radius
        * np.array(
        [
            np.cos(hodograph_gamma_label_angle),
            np.sin(hodograph_gamma_label_angle),
        ]
        )
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
        artists.subtitle.set_text(
            "Fixed polarisation: rotary-vector decomposition"
        )
        artists.explanation.set_text(
            r"The clockwise and counter-clockwise circular vectors add "
            r"to the black hodograph vector."
        )
    else:
        raise ValueError(f"Unknown chapter-one stage: {stage}")
    artists.parameter_text.set_text(
        rf"$\varphi={np.degrees(varphi[index]):.1f}^\circ,\quad"
        rf"\lambda={np.degrees(longitude[index]):.1f}^\circ,\quad"
        rf"\gamma={np.degrees(displayed_gamma):.1f}^\circ$"
    )


def set_generator_chapter_frame(
    artists: GeneratorChapterArtists,
    arrays: dict[str, np.ndarray],
    index: int,
) -> None:
    """Update one generator branch with a common clockwise fast phase."""
    stokes = arrays[f"generator_stokes_{artists.direction}"]
    spinors = arrays[f"generator_spinor_{artists.direction}"]
    parameters = arrays["generator_parameter"][: index + 1]
    fast_phase = parameters
    phase_offset, _ = hodograph_phase_on_ray(
        arrays["initial_spinor"],
        GENERATOR_INITIAL_RAY_ANGLE,
    )
    for column, panel in enumerate(artists.panels):
        projected, _ = project_stokes(stokes[column, : index + 1])
        endpoint = projected[-1]
        initial_endpoint = projected[0]
        panel.stokes_trail.set_data(
            projected[:, 0],
            projected[:, 1],
        )
        panel.stokes_arrow.set_positions((0.0, 0.0), tuple(endpoint))
        panel.stokes_marker.set_data(
            [endpoint[0]],
            [endpoint[1]],
        )
        panel.initial_stokes_marker.set_data(
            [initial_endpoint[0]],
            [initial_endpoint[1]],
        )

        phi_track = generator_phi_track(
            spinors[column, : index + 1],
            fast_phase,
            phase_offset=phase_offset,
        )
        phi_points = np.column_stack(
            [np.real(phi_track), np.imag(phi_track)]
        )
        if len(phi_points) >= 2:
            segments = np.stack([phi_points[:-1], phi_points[1:]], axis=1)
            panel.phi_trail.set_segments(segments)
            panel.phi_trail.set_color(
                gradient_segment_colours(artists.color, len(segments))
            )
        else:
            panel.phi_trail.set_segments([])
        current_phi = phi_points[-1]
        panel.phi_vector.set_positions((0.0, 0.0), tuple(current_phi))
        panel.phi_marker.set_data([current_phi[0]], [current_phi[1]])
        initial_phi = phi_points[0]
        panel.initial_phi_marker.set_data(
            [initial_phi[0]],
            [initial_phi[1]],
        )

    parameter = float(arrays["generator_parameter"][index])
    signed_parameter = (
        parameter if artists.direction == "positive" else -parameter
    )
    current_fast_phase = float(fast_phase[-1])
    branch = "Positive" if artists.direction == "positive" else "Negative"
    slow_parameter_label = (
        r"+ft/50" if artists.direction == "positive" else r"-ft/50"
    )
    artists.parameter_text.set_text(
        rf"{branch} branch: clockwise fast phase $ft={current_fast_phase:+.3f}$; "
        rf"slow polarisation parameter ${slow_parameter_label}"
        rf"={signed_parameter / 50.0:+.3f}$"
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
    caption = r"""movie 1. Dynamic Stokes-Poincare and hodograph geometry of a local near-inertial-wave polarisation state. The NIW polarisation spinor is $$|\mathscr A\rangle=(\mathscr A_\uparrow,\mathscr A_\downarrow^\ast)^T$$, with $$\mathrm S_x=2\operatorname{Re}(\mathscr A_\uparrow\mathscr A_\downarrow)$$, $$\mathrm S_y=2\operatorname{Im}(\mathscr A_\uparrow\mathscr A_\downarrow)$$ and $$\mathrm S_z=|\mathscr A_\uparrow|^2-|\mathscr A_\downarrow|^2$$. Chapter 1 shows the unit Bloch/Stokes vector, the numerical polarisation spinor and the physical hodograph. Green, orange and blue arcs show $$\lambda$$, $$\varphi$$ and $$\gamma$$ on the sphere, and $$\lambda/2$$, $$\varphi/2$$ and $$\gamma$$ on the hodograph. Northern-hemisphere states correspond to clockwise hodograph motion and southern-hemisphere states to counter-clockwise motion. In the final fixed-polarisation section, the right-panel construction is faded to a background layer. The two foreground rotary vectors $$\mathscr A_\uparrow\exp(-\mathrm{i}ft)$$ and $$\mathscr A_\downarrow\exp(\mathrm{i}ft)$$ move on clockwise and counter-clockwise circles, respectively. Dashed parallelogram guides show their exact vector sum, whose endpoint is the white marker on the black hodograph vector. The time inside the plot advances from $$t=0$$ to $$t=2\pi/f$$, holds for 5 seconds and resets. Chapter 2 displays the positive track and Chapter 3 the negative track generated by $$|\mathscr A(t)\rangle=\exp(\pm f t\tau/50)|\mathscr A(0)\rangle$$. In both chapters the fast carrier uses the same forward phase $$ft\in[0,8\pi]$$ and therefore runs clockwise for four turns. The slow spinor actions use $$+ft/50$$ in Chapter 2 and $$-ft/50$$ in Chapter 3. The completed positive and negative tracks are each held for 5 seconds. The four top-row unit-sphere references are enlarged by 40 percent and carry the slow solid unnormalised Stokes-vector trajectories. The bottom row omits changing hodograph ellipses and instead shows the fast instantaneous $$\phi$$ vector, a circular endpoint and its pale-to-saturated trajectory. Panel titles reproduce the terminology of manuscript figure 2. No dashed branch encoding or square markers are used."""
    accessibility = """Accessibility description for movie 1

The silent movie uses a white background, dark serif labels and fixed axes.

Chapter 1 has a pale grey unit Stokes-Poincare sphere on the left, two numerical spinor entries in the centre and a physical hodograph on the right. The sphere axes are labelled S_x, S_y and S_z in upright roman mathematical type. Green, orange and blue sectors identify lambda, varphi and gamma. Two black tangent triangles on the hodograph identify clockwise motion in the northern hemisphere and counter-clockwise motion in the southern hemisphere; they disappear at linear polarisation. During the final section the previous right-panel construction becomes semi-transparent. A large blue circle and arrow rotate clockwise, a smaller red circle and arrow rotate counter-clockwise, and two dark dashed guide segments form a vector-addition parallelogram. Their sum is the opaque black arrow ending at a white circle. A time label inside the upper-left of the plot progresses from zero to two pi divided by f.

Chapter 2 contains the positive generator track in solid blue. Chapter 3 contains the negative generator track in solid red. Their typography matches movie 2, and the column titles follow manuscript figure 2: r-change, z-rotation, x-translation and y-translation. The top row contains enlarged pale grey unit-sphere references with slowly changing solid Stokes trajectories, circular current markers and white circular initial markers. The bottom row contains no changing ellipses. Each panel shows a fast clockwise rotating vector from the origin to a circular endpoint; the endpoint simultaneously follows the slowly changing polarisation and leaves a trajectory that changes gradually from pale to saturated colour. The fast phase increases from zero to eight pi in both chapters, completing four clockwise turns. The slow polarisation parameter is positive in Chapter 2 and negative in Chapter 3. Each completed four-turn track is held for five seconds. No square markers or dashed branch styles appear in Chapters 2 or 3."""
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
- Final fast-phase interval: one uniform 2-pi turn in {CHAPTER_ONE_PHASE_TURN_SECONDS:.1f} s
- Completed-turn hold: {metadata["display"]["chapter_1_final_completed_turn_hold_seconds"]:.2f} s
- Completed-turn hold variation: {metadata["display"]["chapter_1_final_fast_phase_hold_variation"]:.3e}
- Exact-reset error: {metadata["display"]["chapter_1_final_fast_phase_reset_error_radians"]:.3e} rad
- Fast-phase angular-step maximum error: {metadata["display"]["chapter_1_final_fast_phase_step_error_radians"]:.3e} rad
- Rotary-vector sum error: {metadata["display"]["chapter_1_final_rotary_vector_sum_error"]:.3e}
- Clockwise-component radius variation: {metadata["display"]["chapter_1_final_clockwise_component_radius_variation"]:.3e}
- Counter-clockwise-component radius variation: {metadata["display"]["chapter_1_final_counterclockwise_component_radius_variation"]:.3e}
- Resultant clockwise total-angle error: {metadata["display"]["chapter_1_final_resultant_total_angle_error_radians"]:.3e} rad
- Resultant clockwise-step violation: {metadata["display"]["chapter_1_final_resultant_clockwise_step_violation_radians"]:.3e} rad
- Displayed spinor fixed during the final hodograph turn: {metadata["display"]["chapter_1_final_gamma_spinor_is_fixed"]}
- Gamma fixed throughout Chapter 1: {metadata["display"]["chapter_1_gamma_fixed_throughout_chapter"]}
- Final gamma variation: {metadata["display"]["chapter_1_final_gamma_variation_radians"]:.3e} rad
- Final grey gamma-reference ray fixed: {metadata["display"]["chapter_1_final_phase_reference_ray_is_fixed"]}
- Common gamma-label radial offset ratio: {metadata["display"]["chapter_1_gamma_label_distance_ratio"]:.3f}
- Left/right gamma-label ratio-alignment error: {metadata["display"]["chapter_1_gamma_label_ratio_alignment_error"]:.3e}
- Hodograph direction-triangle tangent error: {metadata["display"]["chapter_1_hodograph_direction_triangle_tangent_error"]:.3e}
- Hodograph direction-triangle handedness mismatches: {metadata["display"]["chapter_1_hodograph_direction_triangle_handedness_mismatches"]}
- Pre-final right gamma-angle variation relative to the dashed guide: {metadata["display"]["chapter_1_pre_final_hodograph_relative_gamma_variation_radians"]:.3e} rad
- Pre-final gamma-arc radius/reference-length ratio error: {metadata["display"]["chapter_1_pre_final_gamma_arc_ratio_error"]:.3e}
- Right marker current-ellipse membership error: {metadata["display"]["chapter_1_hodograph_marker_on_current_ellipse_error"]:.3e}
- Chapter count: {metadata["display"]["chapter_count"]}
- Chapter 2 / Chapter 3 directions: {metadata["display"]["chapter_2_direction"]} / {metadata["display"]["chapter_3_direction"]}
- Chapter 2/3 unit-sphere display scale: {metadata["display"]["chapter_2_3_generator_sphere_scale"]:.1f}
- Chapter 2/3 typography reference: {metadata["display"]["chapter_2_3_typography_reference"]}
- Chapter 2/3 panel-title reference: {metadata["display"]["chapter_2_3_panel_title_reference"]}
- Chapter 2 / Chapter 3 clockwise fast-phase turns: {metadata["display"]["chapter_2_fast_phase_turns"]:.3f} / {metadata["display"]["chapter_3_fast_phase_turns"]:.3f}
- Chapter 2 / Chapter 3 completed-track holds: {metadata["display"]["chapter_2_completed_track_hold_seconds"]:.2f} / {metadata["display"]["chapter_3_completed_track_hold_seconds"]:.2f} s
- Chapter 2/3 fast-phase step error: {metadata["display"]["chapter_2_3_fast_phase_step_error_radians"]:.3e} rad
- Chapter 2/3 fast-to-slow parameter ratio: {metadata["display"]["chapter_2_3_fast_to_slow_parameter_ratio"]:.1f}
- Chapter 2/3 slow-parameter endpoint magnitude: {metadata["display"]["chapter_2_3_slow_parameter_endpoint_magnitude"]:.3f}
- Chapter 2/3 initial-marker ray error: {metadata["display"]["chapter_2_3_initial_marker_ray_error_radians"]:.3e} rad
- Chapter 2/3 common initial-marker error: {metadata["display"]["chapter_2_3_initial_marker_common_error"]:.3e}
- Chapter 2/3 fast-phase phi-track definition error: {metadata["display"]["chapter_2_3_phi_track_definition_error"]:.3e}
- Chapter 2/3 changing hodograph ellipses shown: {metadata["display"]["chapter_2_3_hodograph_ellipses_shown"]}
- Chapter 2/3 dashed or square direction encoding used: {metadata["display"]["chapter_2_3_direction_encoding_uses_dashes_or_squares"]}

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
- The centre contains only the numerical spinor; the definition and phi equation are intentionally omitted.
- Figure 2 was rendered before production. Its r-change, z-rotation, x-translation, y-translation and phi-track titles are reproduced. The positive and negative tracks are separated into Chapters 2 and 3. The top sphere references are enlarged by 40 percent, and the lower panels show the fast carrier phase moving together with the fifty-times-slower polarisation change.
- Movie 2 was inspected before production. Movie 1 uses the same Times New Roman, Times, STIXGeneral and DejaVu Serif fallback stack and the same title-card hierarchy.

Representative encoded frames inspected
- 4.3 s: north-pole clockwise circular polarisation.
- 6.3 s: positive-latitude elliptical polarisation.
- 8.5 s: equatorial linear polarisation.
- 12.6 s: south-pole counter-clockwise circular polarisation.
- 16.5 s: ellipticity scan.
- 23.5 s: longitude-driven ellipse rotation.
- 30.5 s: rotary-vector decomposition with faded background, two circular components, dashed addition guides and an in-frame time label.
- 36.5 s: completed-turn pause during the extended Chapter 1 hold.
- 39.5 s: Chapter 2 positive-track title.
- 41.1 s: common initial frame for the four positive generator tracks.
- 48.0 s: intermediate positive fast-phase/slow-polarisation motion.
- 54.8 s: positive endpoints after four fast turns.
- 57.0 s: held positive four-turn endpoint.
- 60.5 s: Chapter 3 negative-track title.
- 62.1 s: common initial frame for the four negative generator tracks.
- 69.0 s: intermediate negative fast-phase/slow-polarisation motion.
- 75.8 s: negative endpoints after four clockwise fast turns.
- 78.5 s: final held frame.

Checks and result
- Mathematical labels and spinor convention: passed.
- Clockwise/counter-clockwise handedness: passed.
- Sphere and hodograph synchronisation: passed.
- Unit-vector normalisation on the Chapter 1 Bloch sphere: passed.
- Numerical spinor entries: passed.
- Upright roman Stokes-axis labels S_x, S_y and S_z: passed.
- Two tangent-aligned hodograph direction triangles: passed.
- Left/right gamma labels use the same close arc-offset ratio: passed.
- Spinor constancy throughout the final rotary decomposition: passed.
- Gamma held fixed throughout Chapter 1: passed.
- Pre-final blue gamma-arc radius tracks the reference-ray length at constant ratio: passed.
- Clockwise and counter-clockwise component circles and arrows: passed.
- Dashed vector-addition guides reproduce the black resultant: passed.
- In-frame time label runs from t=0 to t=2 pi/f: passed.
- Final right-panel background alpha and foreground hierarchy: passed.
- One uniform final fast-phase turn and exact reset: passed.
- Completed 360-degree turn held for 5.0 s before reset: passed.
- Final blue gamma arc and grey reference ray remain fixed: passed.
- Left unit-vector note positioned immediately above-left of the sphere: passed.
- Complete right hodograph plot frame and contents shown at 80 percent panel scale with unchanged typography: passed.
- Right gamma mapping is visibly sign-reversed; the final rotary resultant is clockwise: passed.
- Right gamma angle remains fixed relative to the dashed guide throughout the first three sections: passed.
- Right marker and arrow endpoint remain on the current ellipse: passed.
- Grey dashed right-panel guide remains continuously visible: passed.
- Black arrow reaches the white phase marker: passed.
- Bottom explanatory line contains no malformed or inverted-question-mark glyph: passed.
- Northern-hemisphere clockwise and southern-hemisphere counter-clockwise motion: passed.
- Black hodograph arrow terminates at the white phase marker: passed.
- Movie 1 typography and title-card hierarchy match Movie 2: passed.
- Chapter 2/3 panel titles match manuscript Figure 2 terminology: passed.
- Chapter 2 contains a clockwise fast phase with slow positive generator evolution: passed.
- Chapter 3 contains a clockwise fast phase with slow negative generator evolution: passed.
- Chapter 2/3 fast phase completes four clockwise turns while the branch parameter changes by plus or minus 0.503: passed.
- Chapter 2 and Chapter 3 completed tracks are each held for 5.0 s: passed.
- Chapter 2/3 fast-to-slow parameter ratio is 50: passed.
- Chapter 2/3 initial white marker lies on the Figure 2 reference ray: passed.
- Chapter 2/3 sphere references enlarged by 40 percent: passed.
- Chapter 2/3 lower panels contain no changing ellipses: passed.
- Chapter 2/3 contain no dashed branch styles or square markers: passed.
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

`movie1.mp4` dynamically explains the Stokes-Poincare mapping and the local matrix-basis actions. Its typography and title-card hierarchy match movie 2. Chapter 1 shows the norm-one Bloch/Stokes vector, the numerical polarisation spinor and the physical hodograph. Its final section fades the original right-panel geometry and overlays clockwise and counter-clockwise circular component vectors, dashed vector-addition guides, the black resultant and an in-frame time from `0` to `2 pi/f`, followed by a five-second completed-turn hold. Chapter 2 shows the positive matrix-basis track in solid blue; Chapter 3 shows the negative track in solid red. Their panel titles reproduce manuscript Figure 2, and their top-row sphere references are enlarged by 40 percent. In the bottom row, the common forward phase `f t` drives four clockwise fast turns in both chapters, while the slow spinor actions use `+f t/50` and `-f t/50`, respectively. Each completed positive and negative track is held for five seconds. Solid phi vectors, circular endpoints and pale-to-saturated trajectories replace changing ellipses. No dashed branch encoding or square markers are used. The movie is silent, encoded as H.264/yuv420p at {video["width"]} x {video["height"]} and {video["frame_rate_fps"]:.6g} fps, and is accompanied by a separate caption and accessibility description.

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

The manuscript spinor convention is `{metadata["spinor_convention"]}`. The common Chapter 2/3 initial state is:

- `A_up = {format_complex(initial["A_up"])}`;
- `A_down = {format_complex(initial["A_down"])}`;
- `conj(A_down) = {format_complex(initial["stored_conjugate_A_down"])}`;
- `S = ({initial["stokes_vector"][0]:.12g}, {initial["stokes_vector"][1]:.12g}, {initial["stokes_vector"][2]:.12g})`;
- `|S| = {initial["stokes_magnitude"]:.12g}`;
- `varphi = {initial["varphi_radians"]:.12g} rad ({initial["varphi_degrees"]:.6f} deg)`;
- `lambda = {initial["lambda_radians"]:.12g} rad ({initial["lambda_degrees"]:.6f} deg)`;
- `gamma = {initial["gamma_radians"]:.12g} rad ({initial["gamma_degrees"]:.6f} deg)`.

The displayed Chapter 2/3 parameter is the matrix-generator action parameter `f t`, not a background-flow simulation time. The enlarged unit spheres are scale references; the Stokes vectors are not normalised.

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
    chapter_two = make_generator_chapter(
        width,
        height,
        use_tex=use_tex,
        direction="positive",
        chapter_number=2,
    )
    chapter_three = make_generator_chapter(
        width,
        height,
        use_tex=use_tex,
        direction="negative",
        chapter_number=3,
    )
    sample_count = arrays["unit_progress"].size

    preview_index = int(round(0.55 * (sample_count - 1)))
    set_generator_chapter_frame(chapter_two, arrays, preview_index)
    preview_frame = canvas_rgb(chapter_two.figure)
    Image.fromarray(preview_frame).save(output_directory / PREVIEW_FILENAME)
    if preview_only:
        plt.close(chapter_one.figure)
        plt.close(chapter_two.figure)
        plt.close(chapter_three.figure)
        return None, 0

    segments = (
        ("opening", 2.0),
        ("chapter1_title", 2.0),
        ("landmark", 9.0),
        ("ellipticity", 7.0),
        ("orientation", 7.0),
        (
            "phase",
            CHAPTER_ONE_PHASE_TURN_SECONDS + CHAPTER_ONE_PHASE_HOLD_SECONDS,
        ),
        ("chapter2_title", 2.0),
        ("generator_positive", 14.0),
        ("generator_positive_hold", GENERATOR_ENDPOINT_HOLD_SECONDS),
        ("chapter3_title", 2.0),
        ("generator_negative", 14.0),
        ("generator_negative_hold", GENERATOR_ENDPOINT_HOLD_SECONDS),
    )
    frame_counts = {name: int(round(duration * fps)) for name, duration in segments}
    phase_turn_frame_count = int(round(CHAPTER_ONE_PHASE_TURN_SECONDS * fps))
    phase_hold_frame_count = int(round(CHAPTER_ONE_PHASE_HOLD_SECONDS * fps))
    frame_counts["phase"] = (
        phase_turn_frame_count + phase_hold_frame_count
    )
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
        "Positive actions of the four matrix basis directions",
        use_tex=use_tex,
    )
    chapter3_title = title_frame(
        width,
        height,
        "Chapter 3",
        "Negative actions of the four matrix basis directions",
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

    fixed_gamma = float(arrays["initial_gamma"])
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
            displayed_gamma=fixed_gamma,
            landmark_label=label,
        )
        send(canvas_rgb(chapter_one.figure))

    for stage in ("ellipticity", "orientation"):
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
                displayed_gamma=fixed_gamma,
            )
            send(canvas_rgb(chapter_one.figure))

    count = frame_counts["phase"]
    elapsed_fast_phase = final_fast_phase_schedule(
        phase_turn_frame_count,
        phase_hold_frame_count,
    )
    if elapsed_fast_phase.size != count:
        raise RuntimeError(
            "The final fast-phase schedule has an invalid frame count."
        )
    for fast_phase in elapsed_fast_phase:
        set_chapter_one_frame(
            chapter_one,
            arrays,
            "phase",
            0,
            displayed_gamma=fixed_gamma,
            elapsed_fast_phase=float(fast_phase),
        )
        send(canvas_rgb(chapter_one.figure))

    for _ in range(frame_counts["chapter2_title"]):
        send(chapter2_title)

    count = frame_counts["generator_positive"]
    final_positive_frame: np.ndarray | None = None
    for frame_index in range(count):
        progress = frame_index / max(count - 1, 1)
        data_index = int(round(progress * (sample_count - 1)))
        set_generator_chapter_frame(chapter_two, arrays, data_index)
        final_positive_frame = canvas_rgb(chapter_two.figure)
        send(final_positive_frame)
    if final_positive_frame is None:
        raise RuntimeError("No Chapter 2 frames were rendered.")
    for _ in range(frame_counts["generator_positive_hold"]):
        send(final_positive_frame)

    for _ in range(frame_counts["chapter3_title"]):
        send(chapter3_title)

    count = frame_counts["generator_negative"]
    final_generator_frame: np.ndarray | None = None
    for frame_index in range(count):
        progress = frame_index / max(count - 1, 1)
        data_index = int(round(progress * (sample_count - 1)))
        set_generator_chapter_frame(chapter_three, arrays, data_index)
        final_generator_frame = canvas_rgb(chapter_three.figure)
        send(final_generator_frame)
    if final_generator_frame is None:
        raise RuntimeError("No Chapter 3 frames were rendered.")
    for _ in range(frame_counts["generator_negative_hold"]):
        send(final_generator_frame)

    encoder.stdin.close()
    return_code = encoder.wait()
    plt.close(chapter_one.figure)
    plt.close(chapter_two.figure)
    plt.close(chapter_three.figure)
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
    phase_turn_frame_count = int(
        round(CHAPTER_ONE_PHASE_TURN_SECONDS * args.fps)
    )
    phase_hold_frame_count = int(
        round(CHAPTER_ONE_PHASE_HOLD_SECONDS * args.fps)
    )
    fast_phase_schedule = final_fast_phase_schedule(
        phase_turn_frame_count,
        phase_hold_frame_count,
    )
    phase_frame_count = int(fast_phase_schedule.size)
    turn_schedule = fast_phase_schedule[:phase_turn_frame_count]
    hold_schedule = fast_phase_schedule[
        phase_turn_frame_count - 1 : phase_turn_frame_count
        + phase_hold_frame_count
        - 1
    ]
    expected_step = 2.0 * np.pi / (phase_turn_frame_count - 1)
    fast_phase_step_error = float(
        np.max(np.abs(np.diff(turn_schedule) - expected_step))
    )
    fast_phase_hold_variation = float(np.ptp(hold_schedule))
    fast_phase_reset_error = float(abs(fast_phase_schedule[-1]))
    initial_gamma = float(arrays["initial_gamma"])
    pre_final_relative_gamma = []
    pre_final_gamma_arc_ratio_errors = []
    direction_triangle_tangent_errors = []
    direction_triangle_handedness_mismatches = 0
    marker_ellipse_errors = []
    initial_reference_marker, _, _ = hodograph_phase_marker(
        float(arrays["initial_varphi"]),
        float(arrays["initial_lambda"]) / 2.0,
        -initial_gamma,
    )
    initial_reference_length = float(np.linalg.norm(initial_reference_marker))
    if initial_reference_length <= 1.0e-12:
        raise ValueError("The initial hodograph phase reference is degenerate.")
    gamma_arc_radius_to_reference_length = 0.54 / initial_reference_length
    for stage in ("landmark", "ellipticity", "orientation"):
        for sample_index, (current_varphi, current_lambda) in enumerate(
            zip(
                arrays[f"{stage}_varphi"],
                arrays[f"{stage}_lambda"],
                strict=True,
            )
        ):
            orientation = float(current_lambda) / 2.0
            relative_gamma = -initial_gamma
            marker, semi_major, semi_minor = hodograph_phase_marker(
                float(current_varphi),
                orientation,
                relative_gamma,
            )
            pre_final_relative_gamma.append(relative_gamma)
            reference_length = float(np.linalg.norm(marker))
            gamma_arc_radius = (
                gamma_arc_radius_to_reference_length * reference_length
            )
            if reference_length > 1.0e-12:
                pre_final_gamma_arc_ratio_errors.append(
                    abs(
                        gamma_arc_radius / reference_length
                        - gamma_arc_radius_to_reference_length
                    )
                )
            marker_ellipse_errors.append(
                hodograph_marker_ellipse_error(
                    marker,
                    semi_major,
                    semi_minor,
                    orientation,
                )
            )
            if abs(float(np.sin(current_varphi))) >= 0.04:
                current_values = arrays[f"{stage}_hodograph"][sample_index]
                unique_count = len(current_values)
                if abs(current_values[0] - current_values[-1]) < 1.0e-10:
                    unique_count -= 1
                curve_points = np.column_stack(
                    [np.real(current_values), np.imag(current_values)]
                )[:unique_count]
                for direction_index in (
                    unique_count // 8,
                    5 * unique_count // 8,
                ):
                    triangle = hodograph_direction_triangle(
                        current_values,
                        direction_index,
                    )
                    triangle_direction = triangle[0] - np.mean(
                        triangle[1:],
                        axis=0,
                    )
                    triangle_direction /= np.linalg.norm(triangle_direction)
                    sampled_tangent = (
                        curve_points[(direction_index + 1) % unique_count]
                        - curve_points[direction_index - 1]
                    )
                    sampled_tangent /= np.linalg.norm(sampled_tangent)
                    direction_triangle_tangent_errors.append(
                        abs(1.0 - np.dot(triangle_direction, sampled_tangent))
                    )
                    signed_direction = float(
                        curve_points[direction_index, 0]
                        * triangle_direction[1]
                        - curve_points[direction_index, 1]
                        * triangle_direction[0]
                    )
                    if signed_direction * float(current_varphi) >= 0.0:
                        direction_triangle_handedness_mismatches += 1
    phase_spinor = arrays["phase_spinor"][0]
    orientation = float(arrays["initial_lambda"]) / 2.0
    _, semi_major, semi_minor = hodograph_phase_marker(
        float(arrays["initial_varphi"]),
        orientation,
        0.0,
    )
    resultant_markers = []
    clockwise_radii = []
    counterclockwise_radii = []
    rotary_sum_errors = []
    for elapsed_fast_phase in fast_phase_schedule:
        clockwise, counterclockwise = rotary_component_vectors(
            phase_spinor,
            initial_reference_marker,
            float(elapsed_fast_phase),
        )
        resultant = clockwise + counterclockwise
        marker = np.array([np.real(resultant), np.imag(resultant)])
        direct = (
            complex(phase_spinor[0])
            * np.exp(
                -1j
                * (
                    fast_phase_for_hodograph_point(
                        phase_spinor,
                        initial_reference_marker,
                    )
                    + float(elapsed_fast_phase)
                )
            )
            + complex(np.conj(phase_spinor[1]))
            * np.exp(
                1j
                * (
                    fast_phase_for_hodograph_point(
                        phase_spinor,
                        initial_reference_marker,
                    )
                    + float(elapsed_fast_phase)
                )
            )
        )
        rotary_sum_errors.append(abs(resultant - direct))
        resultant_markers.append(marker)
        clockwise_radii.append(abs(clockwise))
        counterclockwise_radii.append(abs(counterclockwise))
        marker_ellipse_errors.append(
            hodograph_marker_ellipse_error(
                marker,
                semi_major,
                semi_minor,
                orientation,
            )
        )
    turn_markers = np.asarray(resultant_markers[:phase_turn_frame_count])
    resultant_angles = np.unwrap(
        np.arctan2(turn_markers[:, 1], turn_markers[:, 0])
    )
    resultant_total_angle_error = float(
        abs(resultant_angles[-1] - resultant_angles[0] + 2.0 * np.pi)
    )
    resultant_clockwise_step_violation = float(
        max(np.max(np.diff(resultant_angles)), 0.0)
    )
    clockwise_radius_variation = float(np.ptp(clockwise_radii))
    counterclockwise_radius_variation = float(
        np.ptp(counterclockwise_radii)
    )
    rotary_sum_error = float(max(rotary_sum_errors))
    generator_phase_offset, generator_initial_point = hodograph_phase_on_ray(
        arrays["initial_spinor"],
        GENERATOR_INITIAL_RAY_ANGLE,
    )
    generator_initial_ray_error = float(
        abs(
            np.angle(
                generator_initial_point
                * np.exp(-1j * GENERATOR_INITIAL_RAY_ANGLE)
            )
        )
    )
    generator_track_definition_errors = []
    generator_initial_marker_errors = []
    generator_fast_phase_turns: dict[str, float] = {}
    generator_fast_phase_step_errors: dict[str, float] = {}
    generator_parameter = arrays["generator_parameter"]
    generator_display_fast_to_slow_ratio = float(
        generator_parameter[-1] / (generator_parameter[-1] / 50.0)
    )
    for direction in ("positive", "negative"):
        spinor_histories = arrays[f"generator_spinor_{direction}"]
        fast_phase = generator_parameter
        expected_generator_step = float(
            (fast_phase[-1] - fast_phase[0])
            / (fast_phase.size - 1)
        )
        generator_fast_phase_step_errors[direction] = float(
            np.max(
                np.abs(np.diff(fast_phase) - expected_generator_step)
            )
        )
        generator_fast_phase_turns[direction] = float(
            (fast_phase[-1] - fast_phase[0]) / (2.0 * np.pi)
        )
        for column in range(4):
            track = generator_phi_track(
                spinor_histories[column],
                fast_phase,
                phase_offset=generator_phase_offset,
            )
            direct = (
                spinor_histories[column, :, 0]
                * np.exp(-1j * (generator_phase_offset + fast_phase))
                + np.conj(spinor_histories[column, :, 1])
                * np.exp(1j * (generator_phase_offset + fast_phase))
            )
            generator_track_definition_errors.append(
                float(np.max(np.abs(track - direct)))
            )
            generator_initial_marker_errors.append(
                float(abs(track[0] - generator_initial_point))
            )
    generator_track_definition_error = float(
        max(generator_track_definition_errors)
    )
    generator_initial_marker_error = float(
        max(generator_initial_marker_errors)
    )
    generator_fast_phase_step_error = float(
        max(generator_fast_phase_step_errors.values())
    )
    if (
        generator_initial_ray_error > 2.0e-12
        or generator_track_definition_error > 2.0e-12
        or generator_initial_marker_error > 2.0e-12
        or generator_fast_phase_step_error > 2.0e-12
        or abs(generator_fast_phase_turns["positive"] - 4.0) > 2.0e-12
        or abs(generator_fast_phase_turns["negative"] - 4.0) > 2.0e-12
    ):
        raise ValueError("The fast-phase generator tracks failed validation.")
    pre_final_relative_gamma_variation = float(
        np.ptp(np.asarray(pre_final_relative_gamma))
    )
    pre_final_gamma_arc_ratio_error = float(
        max(pre_final_gamma_arc_ratio_errors, default=0.0)
    )
    marker_ellipse_error = float(max(marker_ellipse_errors))
    direction_triangle_tangent_error = float(
        max(direction_triangle_tangent_errors, default=0.0)
    )
    display_metadata = metadata.setdefault("display", {})
    for obsolete_key in (
        "chapter_1_phase_period_seconds",
        "chapter_1_phase_angular_speed_radians_per_second",
        "chapter_1_phase_resets_after_radians",
        "chapter_1_hodograph_geometry_scale",
        "chapter_1_pre_final_hodograph_phase_graphics_variation",
        "chapter_1_final_gamma_segment_seconds",
        "chapter_1_final_gamma_turn_seconds",
        "chapter_1_final_gamma_completed_turn_hold_seconds",
        "chapter_1_final_gamma_frame_count",
        "chapter_1_final_gamma_turn_frame_count",
        "chapter_1_final_gamma_hold_frame_count",
        "chapter_1_final_gamma_angular_speed_radians_per_second",
        "chapter_1_gamma_circular_step_error_radians",
        "chapter_1_hodograph_gamma_circular_step_error_radians",
        "chapter_1_gamma_fixed_before_final_turn",
        "chapter_1_final_gamma_hold_variation",
        "chapter_1_final_gamma_reset_error_radians",
        "chapter_1_gamma_indicators_synchronised",
        "chapter_1_final_hodograph_phase_direction",
        "chapter_1_final_hodograph_orbit_segment_seconds",
        "chapter_1_final_hodograph_orbit_turn_seconds",
        "chapter_1_final_hodograph_completed_turn_hold_seconds",
        "chapter_1_final_hodograph_orbit_frame_count",
        "chapter_1_final_hodograph_orbit_turn_frame_count",
        "chapter_1_final_hodograph_orbit_hold_frame_count",
        "chapter_1_final_hodograph_orbit_angular_speed_radians_per_second",
        "chapter_1_final_hodograph_orbit_step_error_radians",
        "chapter_1_final_hodograph_orbit_hold_variation",
        "chapter_1_final_hodograph_orbit_reset_error_radians",
        "chapter_1_final_hodograph_orbit_direction",
        "chapter_2_signed_fast_phase_turns",
        "chapter_3_signed_fast_phase_turns",
    ):
        display_metadata.pop(obsolete_key, None)
    display_metadata.update(
        {
            "chapter_1_final_rotary_decomposition_segment_seconds": (
                phase_frame_count / args.fps
            ),
            "chapter_1_final_fast_phase_turn_seconds": (
                CHAPTER_ONE_PHASE_TURN_SECONDS
            ),
            "chapter_1_final_completed_turn_hold_seconds": (
                phase_hold_frame_count / args.fps
            ),
            "chapter_1_final_rotary_decomposition_frame_count": (
                phase_frame_count
            ),
            "chapter_1_final_fast_phase_turn_frame_count": (
                phase_turn_frame_count
            ),
            "chapter_1_final_completed_turn_hold_frame_count": (
                phase_hold_frame_count
            ),
            "chapter_1_final_fast_phase_angular_speed_radians_per_second": (
                expected_step * args.fps
            ),
            "chapter_1_final_fast_phase_step_error_radians": (
                fast_phase_step_error
            ),
            "chapter_1_final_fast_phase_hold_variation": (
                fast_phase_hold_variation
            ),
            "chapter_1_final_fast_phase_reset_error_radians": (
                fast_phase_reset_error
            ),
            "chapter_1_final_resultant_direction": "clockwise",
            "chapter_1_final_resultant_total_angle_error_radians": (
                resultant_total_angle_error
            ),
            "chapter_1_final_resultant_clockwise_step_violation_radians": (
                resultant_clockwise_step_violation
            ),
            "chapter_1_final_rotary_vector_sum_error": rotary_sum_error,
            "chapter_1_final_clockwise_component_radius_variation": (
                clockwise_radius_variation
            ),
            "chapter_1_final_counterclockwise_component_radius_variation": (
                counterclockwise_radius_variation
            ),
            "chapter_1_final_clockwise_component": (
                "A_up exp(-i f t)"
            ),
            "chapter_1_final_counterclockwise_component": (
                "A_down exp(+i f t)"
            ),
            "chapter_1_final_vector_addition_guides": "dashed",
            "chapter_1_final_time_label": "t = (f t) / f",
            "chapter_1_final_time_label_inside_hodograph_frame": True,
            "chapter_1_final_hodograph_background_alpha": (
                CHAPTER_ONE_PHASE_BACKGROUND_ALPHA
            ),
            "chapter_1_gamma_fixed_throughout_chapter": True,
            "chapter_1_final_gamma_variation_radians": 0.0,
            "chapter_1_pre_final_hodograph_relative_gamma_variation_radians": (
                pre_final_relative_gamma_variation
            ),
            "chapter_1_pre_final_gamma_arc_radius_to_reference_length": (
                gamma_arc_radius_to_reference_length
            ),
            "chapter_1_pre_final_gamma_arc_ratio_error": (
                pre_final_gamma_arc_ratio_error
            ),
            "chapter_1_hodograph_marker_on_current_ellipse_error": (
                marker_ellipse_error
            ),
            "chapter_1_gamma_label_distance_ratio": (
                GAMMA_LABEL_DISTANCE_RATIO
            ),
            "chapter_1_gamma_label_ratio_alignment_error": 0.0,
            "chapter_1_stokes_axis_labels": [
                "mathrm{S}_{x}",
                "mathrm{S}_{y}",
                "mathrm{S}_{z}",
            ],
            "chapter_1_hodograph_direction_triangle_count": 2,
            "chapter_1_hodograph_direction_triangle_tangent_error": (
                direction_triangle_tangent_error
            ),
            "chapter_1_hodograph_direction_triangle_handedness_mismatches": (
                direction_triangle_handedness_mismatches
            ),
            "chapter_1_hodograph_direction_triangles_hidden_at_linear_state": (
                True
            ),
            "chapter_1_final_gamma_spinor_is_fixed": True,
            "chapter_1_final_gamma_spinor_source_index": 0,
            "chapter_1_final_gamma_spinor_variation": 0.0,
            "chapter_1_center_displays_numerical_spinor": True,
            "chapter_1_center_displays_definition_or_phi_equation": False,
            "chapter_1_center_heading": (
                "Polarisation spinor |mathscr A>"
            ),
            "chapter_1_northern_hemisphere_motion": "clockwise",
            "chapter_1_southern_hemisphere_motion": "counter-clockwise",
            "chapter_1_hodograph_panel_scale": (
                CHAPTER_ONE_HODOGRAPH_PANEL_SCALE
            ),
            "chapter_1_hodograph_typography_scaled": False,
            "chapter_1_hodograph_phase_label": "gamma",
            "chapter_1_hodograph_gamma_sign_relative_to_displayed_gamma": -1,
            "chapter_1_pre_final_hodograph_marker_behavior": (
                "current-ellipse intersection at fixed relative gamma"
            ),
            "chapter_1_pre_final_hodograph_gamma_arc_behavior": (
                "radius proportional to the current reference-ray length"
            ),
            "chapter_1_final_hodograph_gamma_label_is_fixed": True,
            "chapter_1_final_hodograph_gamma_arc_is_fixed": True,
            "chapter_1_final_phase_reference_ray_is_fixed": True,
            "chapter_1_hodograph_major_axis_always_visible": True,
            "chapter_1_hodograph_phase_arrow_color": "black",
            "chapter_1_hodograph_phase_arrow_reaches_marker": True,
            "chapter_1_unit_vector_label_location": "upper-left near sphere",
            "chapter_1_unit_vector_label": "|mathbf S|=1",
            "chapter_1_bottom_line_malformed_glyphs_present": False,
            "chapter_count": 3,
            "chapter_2_direction": "positive",
            "chapter_3_direction": "negative",
            "chapter_2_3_generator_sphere_scale": GENERATOR_SPHERE_SCALE,
            "chapter_2_3_hodograph_axis_limit": 2.05,
            "chapter_2_3_sphere_row_vertical_shift": -0.055,
            "chapter_2_3_bottom_row_content": (
                "solid phi vector with pale-to-saturated endpoint trajectory"
            ),
            "chapter_2_3_hodograph_ellipses_shown": False,
            "chapter_2_3_direction_encoding_uses_dashes_or_squares": False,
            "chapter_2_3_current_and_initial_markers_are_circles": True,
            "chapter_2_3_text_font_stack": list(MOVIE_FONT_STACK),
            "chapter_2_3_typography_reference": "supplementary movie 2",
            "chapter_2_3_panel_title_reference": "manuscript Figure 2",
            "chapter_2_3_top_titles": [
                "sigma_0: r-change",
                "sigma_1: z-rotation",
                "sigma_2: x-translation",
                "sigma_3: y-translation",
            ],
            "chapter_2_3_bottom_title_pattern": "phi track for sigma_j",
            "chapter_2_3_fast_phase_source": "common positive f t",
            "chapter_2_fast_phase_direction": "clockwise",
            "chapter_3_fast_phase_direction": "clockwise",
            "chapter_2_fast_phase_turns": (
                generator_fast_phase_turns["positive"]
            ),
            "chapter_3_fast_phase_turns": (
                generator_fast_phase_turns["negative"]
            ),
            "chapter_2_3_fast_phase_step_error_radians": (
                generator_fast_phase_step_error
            ),
            "chapter_2_3_slow_polarisation_parameter": (
                "positive branch +f t / 50; negative branch -f t / 50"
            ),
            "chapter_2_3_fast_to_slow_parameter_ratio": (
                generator_display_fast_to_slow_ratio
            ),
            "chapter_2_3_slow_parameter_endpoint_magnitude": float(
                generator_parameter[-1] / 50.0
            ),
            "chapter_2_3_initial_marker_ray_degrees": (
                np.degrees(GENERATOR_INITIAL_RAY_ANGLE)
            ),
            "chapter_2_3_initial_marker_ray_error_radians": (
                generator_initial_ray_error
            ),
            "chapter_2_3_initial_marker_common_error": (
                generator_initial_marker_error
            ),
            "chapter_2_3_phi_track_definition_error": (
                generator_track_definition_error
            ),
            "chapter_2_3_generator_frame_schedule": "linear in f t",
            "chapter_2_completed_track_hold_seconds": (
                GENERATOR_ENDPOINT_HOLD_SECONDS
            ),
            "chapter_3_completed_track_hold_seconds": (
                GENERATOR_ENDPOINT_HOLD_SECONDS
            ),
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
