"""Render and encode supplementary movie 2.

This stage reads only the verified NPZ archive produced by
``compute_movie_fields.py``.  It renders fixed-scale scientific frames,
encodes a browser-compatible H.264 MP4, compares representative frames before
and after encoding, and writes the submission-side explanatory files.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image

from plot_wave_velocity_fields import publication_style


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "sinusoidal_dipole_movie_fields.npz"
)
MAX_FILE_BYTES = 50_000_000
EXPECTED_VERTICAL_MODES = (4, 16, 32)
PREFERRED_TEXT_FONT = "Times New Roman"
MOVIE_FONT_STACK = (
    PREFERRED_TEXT_FONT,
    "Times",
    "STIXGeneral",
    "DejaVu Serif",
)
VERTICAL_WAVELENGTH_METRES = {
    4: 1000,
    16: 250,
    32: 125,
}
MODEL_STYLES = (
    ("YBJ", "#002BFF", "o", "-"),
    ("TSB", "#7A3E9D", "^", "--"),
    ("YBJ+", "#00C46A", "+", "-"),
    ("PSE", "#B2182B", "s", "-"),
)
NRE_LEGEND_HANDLE_ORDER = (0, 2, 1, 3)
NRE_LEGEND_VISUAL_ROWS = (
    ("YBJ", "TSB"),
    ("YBJ+", "PSE"),
)
TITLE_CARD_REFERENCE_RESOLUTION = (1920, 1080)
TITLE_CARD_REFERENCE_DPI = 120
TITLE_CARD_TITLE_POSITION = (0.5, 0.62)
TITLE_CARD_SUBTITLE_POSITION = (0.5, 0.43)
TITLE_CARD_REFERENCE_TITLE_FONTSIZE = 31.0
TITLE_CARD_REFERENCE_SUBTITLE_FONTSIZE = 20.0
SUBPLOT_TITLE_FONTSIZE = 20.0
NRE_TITLE_FONTSIZE = 17.0
JFM_PREPARING_URL = (
    "https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/"
    "information/author-instructions/preparing-your-materials"
)
JFM_SUBMITTING_URL = (
    "https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/"
    "information/author-instructions/submitting-your-materials"
)
CAMBRIDGE_SUPPLEMENT_URL = (
    "https://www.cambridge.org/core/services/authors/"
    "publishing-supplementary-material"
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse an even WIDTHxHEIGHT video resolution."""
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError("Resolution must be WIDTHxHEIGHT.") from error
    if width < 1280 or height < 720 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError(
            "Resolution must be at least 1280x720 with even dimensions."
        )
    return width, height


def resolve_executable(value: Path | None, name: str) -> Path:
    """Resolve an explicit executable or one available on PATH."""
    if value is not None:
        resolved = value.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{name} executable does not exist: {resolved}")
        return resolved
    discovered = shutil.which(name)
    if discovered is None:
        raise FileNotFoundError(
            f"{name} was not found. Pass --{name} with an executable path."
        )
    return Path(discovered).resolve()


def load_movie_data(path: Path) -> dict[str, Any]:
    """Load and structurally validate the computed movie archive."""
    required = {
        "times_in_inertial_periods",
        "vertical_modes",
        "model_names",
        "nre_model_names",
        "normalized_squared_velocity",
        "nre_times_in_inertial_periods",
        "nre_complex_relative_l2",
        "absolute_color_limits",
        "difference_color_limits",
        "metadata_json",
    }
    with np.load(path) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Movie archive is missing: {', '.join(missing)}")
        result: dict[str, Any] = {
            name: archive[name].copy()
            for name in archive.files
            if name != "metadata_json"
        }
        result["metadata"] = json.loads(str(archive["metadata_json"].item()))

    times = result["times_in_inertial_periods"]
    modes = result["vertical_modes"]
    model_names = list(result["model_names"])
    fields = result["normalized_squared_velocity"]
    if tuple(int(mode) for mode in modes) != EXPECTED_VERTICAL_MODES:
        raise ValueError(
            "Movie archive must contain vertical modes n=4, n=16 and n=32."
        )
    if model_names != ["YBJ", "TSB", "YBJ+", "PSE", "HBEs"]:
        raise ValueError("Movie model order changed.")
    if fields.shape[:3] != (modes.size, times.size, len(model_names)):
        raise ValueError("Movie field dimensions are inconsistent.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("Movie times are not strictly increasing.")
    return result


def heatmap_axis_style(axis: mpl.axes.Axes, *, show_x: bool, show_y: bool) -> None:
    """Apply the Figure 9--10 square-panel axes style."""
    axis.set_box_aspect(1)
    axis.set_xlim(-np.pi, np.pi)
    axis.set_ylim(-np.pi, np.pi)
    axis.set_xticks((-np.pi, 0.0, np.pi))
    axis.set_yticks((-np.pi, 0.0, np.pi))
    axis.tick_params(length=4.0, width=1.0, pad=2.5, direction="out")
    if show_x:
        axis.set_xticklabels((r"$-\pi$", "0", r"$\pi$"))
        axis.set_xlabel(r"$x/L$", labelpad=1.5)
    else:
        axis.set_xticklabels(())
    if show_y:
        axis.set_yticklabels((r"$-\pi$", "0", r"$\pi$"))
        axis.set_ylabel(r"$y/L$", labelpad=1.5)
    else:
        axis.set_yticklabels(())
    for spine in axis.spines.values():
        spine.set_linewidth(1.15)
        spine.set_zorder(20)


def nice_nre_upper(value_percent: float) -> float:
    """Return a stable, readable upper bound for one chapter's NRE panel."""
    if value_percent <= 1.0:
        step = 0.2
    elif value_percent <= 5.0:
        step = 1.0
    elif value_percent <= 20.0:
        step = 5.0
    else:
        step = 10.0
    return max(step, step * math.ceil(1.06 * value_percent / step))


def colorbar_number(value: float, _: int) -> str:
    """Format compact colorbar tick labels."""
    if abs(value) >= 10.0:
        return f"{value:.0f}"
    if abs(value) >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.2f}"


def movie_publication_style() -> dict[str, object]:
    """Return the movie typography aligned with manuscript figures 8--10."""
    style = publication_style(use_tex=False)
    style.update(
        {
            "font.family": "serif",
            "font.serif": list(MOVIE_FONT_STACK),
            "mathtext.fontset": "stix",
        }
    )
    return style


class ChapterRenderer:
    """Maintain one chapter figure while updating only time-dependent artists."""

    def __init__(
        self,
        data: dict[str, Any],
        mode_index: int,
        *,
        resolution: tuple[int, int],
        dpi: int,
    ) -> None:
        self.data = data
        self.mode_index = mode_index
        self.mode = int(data["vertical_modes"][mode_index])
        self.times = data["times_in_inertial_periods"]
        self.fields = data["normalized_squared_velocity"][mode_index]
        self.nre_times = data["nre_times_in_inertial_periods"][mode_index]
        self.nre = data["nre_complex_relative_l2"][mode_index]
        self.absolute_limits = data["absolute_color_limits"][mode_index]
        self.difference_limits = data["difference_color_limits"][mode_index]
        width, height = resolution

        style = movie_publication_style()
        style.update(
            {
                "font.size": 15.0,
                "axes.labelsize": 15.0,
                "axes.titlesize": SUBPLOT_TITLE_FONTSIZE,
                "xtick.labelsize": 13.0,
                "ytick.labelsize": 13.0,
                "legend.fontsize": 11.5,
                "axes.linewidth": 1.0,
            }
        )
        self.context = plt.rc_context(style)
        self.context.__enter__()
        self.figure = plt.figure(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            facecolor="white",
        )
        grid = self.figure.add_gridspec(
            2,
            5,
            left=0.040,
            right=0.918,
            bottom=0.095,
            top=0.850,
            wspace=0.10,
            hspace=0.24,
        )
        self.axes = np.asarray(
            [
                [self.figure.add_subplot(grid[row, column]) for column in range(5)]
                for row in range(2)
            ]
        )
        self.title = self.figure.suptitle("", x=0.50, y=0.962, fontsize=25.0)
        self.figure.text(
            0.014,
            0.635,
            "Normalised squared velocity",
            rotation=90,
            ha="center",
            va="center",
            fontsize=16.0,
        )
        self.figure.text(
            0.014,
            0.270,
            r"Model $-$ HBEs",
            rotation=90,
            ha="center",
            va="center",
            fontsize=16.0,
        )

        extent = (-np.pi, np.pi, -np.pi, np.pi)
        self.absolute_images = []
        for model_index, model_name in enumerate(data["model_names"]):
            axis = self.axes[0, model_index]
            image = axis.imshow(
                self.fields[0, model_index],
                origin="lower",
                extent=extent,
                interpolation="nearest",
                aspect="equal",
                cmap="twilight_shifted",
                vmin=float(self.absolute_limits[0]),
                vmax=float(self.absolute_limits[1]),
            )
            display_name = r"YBJ$^{+}$" if model_name == "YBJ+" else str(model_name)
            axis.set_title(display_name, pad=7.0)
            heatmap_axis_style(
                axis,
                show_x=False,
                show_y=model_index == 0,
            )
            self.absolute_images.append(image)

        self.difference_images = []
        for model_index, model_name in enumerate(data["model_names"][:4]):
            axis = self.axes[1, model_index]
            difference = self.fields[0, model_index] - self.fields[0, 4]
            image = axis.imshow(
                difference,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                aspect="equal",
                cmap="RdBu_r",
                vmin=float(self.difference_limits[0]),
                vmax=float(self.difference_limits[1]),
            )
            display_name = r"YBJ$^{+}$" if model_name == "YBJ+" else str(model_name)
            axis.set_title(rf"{display_name} $-$ HBEs", pad=7.0)
            heatmap_axis_style(
                axis,
                show_x=True,
                show_y=model_index == 0,
            )
            self.difference_images.append(image)

        nre_axis = self.axes[1, 4]
        nre_axis.set_box_aspect(1)
        nre_axis.set_title(
            "Instantaneous NRE relative to HBEs",
            fontsize=NRE_TITLE_FONTSIZE,
            pad=7.0,
        )
        nre_axis.set_xlabel(r"$t$ (IP)", labelpad=3.0)
        nre_axis.set_ylabel("NRE (%)", labelpad=3.0)
        nre_axis.set_xlim(0.0, 50.0)
        nre_axis.set_xticks((0, 10, 20, 30, 40, 50))
        upper = nice_nre_upper(float(np.max(self.nre)) * 100.0)
        nre_axis.set_ylim(0.0, upper)
        nre_axis.grid(axis="y", color="0.82", linewidth=0.8, zorder=0)
        for spine in nre_axis.spines.values():
            spine.set_linewidth(1.15)

        self.nre_lines = []
        self.nre_markers = []
        for model_index, (label, color, marker, linestyle) in enumerate(MODEL_STYLES):
            display_label = r"YBJ$^{+}$" if label == "YBJ+" else label
            (line,) = nre_axis.plot(
                self.nre_times,
                100.0 * self.nre[model_index],
                color=color,
                linestyle=linestyle,
                linewidth=1.65,
                marker=marker,
                markevery=320,
                markersize=4.8 if marker != "+" else 7.0,
                markeredgewidth=1.0 if marker != "+" else 1.5,
                label=display_label,
                zorder=3 + model_index,
            )
            (moving,) = nre_axis.plot(
                [self.times[0]],
                [100.0 * self.nre[model_index, 0]],
                color=color,
                marker=marker,
                linestyle="None",
                markersize=9.0 if marker != "+" else 12.0,
                markeredgewidth=1.4 if marker != "+" else 2.0,
                markeredgecolor="black" if marker != "+" else color,
                zorder=20 + model_index,
            )
            self.nre_lines.append(line)
            self.nre_markers.append(moving)
        self.time_line = nre_axis.axvline(
            float(self.times[0]),
            color="0.25",
            linewidth=1.2,
            linestyle=":",
            zorder=2,
        )
        legend_handles = [
            self.nre_lines[index] for index in NRE_LEGEND_HANDLE_ORDER
        ]
        legend_labels = [
            (
                r"YBJ$^{+}$"
                if MODEL_STYLES[index][0] == "YBJ+"
                else MODEL_STYLES[index][0]
            )
            for index in NRE_LEGEND_HANDLE_ORDER
        ]
        nre_axis.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=0.95,
            borderpad=0.35,
            columnspacing=0.8,
            handlelength=1.7,
        )

        self.figure.canvas.draw()
        top_position = self.axes[0, -1].get_position()
        bottom_position = self.axes[1, -2].get_position()
        absolute_cax = self.figure.add_axes(
            [0.938, top_position.y0, 0.010, top_position.height]
        )
        difference_cax = self.figure.add_axes(
            [0.938, bottom_position.y0, 0.010, bottom_position.height]
        )
        absolute_colorbar = self.figure.colorbar(
            self.absolute_images[-1],
            cax=absolute_cax,
            orientation="vertical",
            extend="both",
        )
        absolute_colorbar.set_ticks(
            (
                float(self.absolute_limits[0]),
                float(np.mean(self.absolute_limits)),
                float(self.absolute_limits[1]),
            )
        )
        absolute_colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(colorbar_number)
        )
        absolute_colorbar.set_label(
            r"$|\phi|^2/|\phi_{\mathrm{amp}}|^2$",
            rotation=90,
            labelpad=8.0,
        )
        absolute_colorbar.outline.set_linewidth(1.0)
        absolute_colorbar.ax.tick_params(width=0.8, length=3.0, pad=3.0)

        difference_colorbar = self.figure.colorbar(
            self.difference_images[-1],
            cax=difference_cax,
            orientation="vertical",
            extend="both",
        )
        difference_colorbar.set_ticks(
            (
                float(self.difference_limits[0]),
                0.0,
                float(self.difference_limits[1]),
            )
        )
        difference_colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(colorbar_number)
        )
        difference_colorbar.set_label(
            (
                r"$(|\phi_{\mathrm{model}}|^2-|\phi_{\mathrm{HBEs}}|^2)"
                r"/|\phi_{\mathrm{amp}}|^2$"
            ),
            rotation=90,
            labelpad=8.0,
        )
        difference_colorbar.outline.set_linewidth(1.0)
        difference_colorbar.ax.tick_params(width=0.8, length=3.0, pad=3.0)

    def update(self, time_index: int) -> None:
        """Update all field images and moving NRE indicators."""
        time_value = float(self.times[time_index])
        for model_index, image in enumerate(self.absolute_images):
            image.set_data(self.fields[time_index, model_index])
        for model_index, image in enumerate(self.difference_images):
            image.set_data(
                self.fields[time_index, model_index]
                - self.fields[time_index, 4]
            )
        nre_time_index = int(np.argmin(np.abs(self.nre_times - time_value)))
        if not np.isclose(
            self.nre_times[nre_time_index],
            time_value,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"NRE time {time_value:g} IP is unavailable.")
        for model_index, marker in enumerate(self.nre_markers):
            marker.set_data(
                [time_value],
                [100.0 * self.nre[model_index, nre_time_index]],
            )
        self.time_line.set_xdata([time_value, time_value])
        self.title.set_text(
            rf"Supplementary movie 2: $n={self.mode}$, "
            rf"$t={time_value:g}$ IP"
        )

    def save(self, path: Path, *, dpi: int) -> None:
        """Save the current frame at the exact configured canvas size."""
        self.figure.savefig(
            path,
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
            pil_kwargs={"compress_level": 3},
        )

    def close(self) -> None:
        """Close the figure and restore the previous matplotlib style."""
        plt.close(self.figure)
        self.context.__exit__(None, None, None)


def render_text_title_frame(
    path: Path,
    *,
    title: str,
    subtitle: str,
    resolution: tuple[int, int],
    dpi: int,
) -> None:
    """Render one static card with Movie 1's hierarchy and placement."""
    width, height = resolution
    reference_height = TITLE_CARD_REFERENCE_RESOLUTION[1]
    font_scale = (
        TITLE_CARD_REFERENCE_DPI
        / dpi
        * height
        / reference_height
    )
    title_fontsize = TITLE_CARD_REFERENCE_TITLE_FONTSIZE * font_scale
    subtitle_fontsize = TITLE_CARD_REFERENCE_SUBTITLE_FONTSIZE * font_scale
    with plt.rc_context(movie_publication_style()):
        figure = plt.figure(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            facecolor="white",
        )
        figure.text(
            *TITLE_CARD_TITLE_POSITION,
            title,
            ha="center",
            va="center",
            fontsize=title_fontsize,
        )
        figure.text(
            *TITLE_CARD_SUBTITLE_POSITION,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_fontsize,
            color="0.28",
            linespacing=1.65,
        )
        figure.savefig(
            path,
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
            pil_kwargs={"compress_level": 3},
        )
        plt.close(figure)


def render_opening_frame(
    path: Path,
    *,
    resolution: tuple[int, int],
    dpi: int,
) -> None:
    """Render the overall opening page before the first chapter page."""
    render_text_title_frame(
        path,
        title="Supplementary movie 2",
        subtitle=(
            "Wave-field evolution in a sinusoidal-dipole background flow"
        ),
        resolution=resolution,
        dpi=dpi,
    )


def render_title_frame(
    path: Path,
    *,
    mode: int,
    chapter: int,
    chapter_count: int,
    resolution: tuple[int, int],
    dpi: int,
) -> None:
    """Render one mode page after the overall opening page."""
    if not 1 <= chapter <= chapter_count:
        raise ValueError("Chapter index is outside the declared chapter count.")
    wavelength_metres = VERTICAL_WAVELENGTH_METRES[mode]
    render_text_title_frame(
        path,
        title=f"Chapter {chapter}",
        subtitle=(
            rf"Vertical mode $n={mode}$ (vertical wavelength "
            rf"$h={wavelength_metres}\,\mathrm{{m}}$); "
            "0-50 inertial periods"
        ),
        resolution=resolution,
        dpi=dpi,
    )


def build_sequence(
    unique_frames: dict[str, Path],
    times: np.ndarray,
    modes: np.ndarray,
    *,
    frame_stride: int,
    hold_frames: int,
    opening_frames: int,
    title_frames: int,
    chapter_end_frames: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Build the ordered CFR source list without duplicating frame files."""
    if (
        frame_stride <= 0
        or hold_frames <= 0
        or opening_frames <= 0
        or title_frames <= 0
        or chapter_end_frames <= 0
    ):
        raise ValueError("Frame stride and all hold counts must be positive.")
    selected = list(range(0, times.size, frame_stride))
    if selected[-1] != times.size - 1:
        selected.append(times.size - 1)

    segments: list[dict[str, Any]] = []
    sources: list[Path] = []
    opening_key = "opening_title"
    sources.extend([unique_frames[opening_key]] * opening_frames)
    segments.append(
        {
            "kind": "opening_title",
            "start_frame": 0,
            "frame_count": opening_frames,
            "source_key": opening_key,
        }
    )
    for mode in modes:
        title_key = f"n{int(mode)}_title"
        start = len(sources)
        sources.extend([unique_frames[title_key]] * title_frames)
        segments.append(
            {
                "kind": "chapter_title",
                "vertical_mode": int(mode),
                "start_frame": start,
                "frame_count": title_frames,
                "source_key": title_key,
            }
        )
        for time_index in selected:
            time_value = float(times[time_index])
            field_key = f"n{int(mode)}_t{time_index:03d}"
            start = len(sources)
            sources.extend([unique_frames[field_key]] * hold_frames)
            segments.append(
                {
                    "kind": "scientific_frame",
                    "vertical_mode": int(mode),
                    "time_ip": time_value,
                    "source_time_index": int(time_index),
                    "start_frame": start,
                    "frame_count": hold_frames,
                    "source_key": field_key,
                }
            )
        final_time_index = selected[-1]
        final_time_value = float(times[final_time_index])
        final_key = f"n{int(mode)}_t{final_time_index:03d}"
        start = len(sources)
        sources.extend([unique_frames[final_key]] * chapter_end_frames)
        segments.append(
            {
                "kind": "chapter_end_hold",
                "vertical_mode": int(mode),
                "time_ip": final_time_value,
                "source_time_index": int(final_time_index),
                "start_frame": start,
                "frame_count": chapter_end_frames,
                "source_key": final_key,
            }
        )

    return segments, sources


def encode_video(
    ffmpeg: Path,
    sequence_sources: list[Path],
    output_path: Path,
    *,
    resolution: tuple[int, int],
    fps: int,
    crf: int,
    maximum_bytes: int,
) -> tuple[int, list[str]]:
    """Stream lossless RGB frames to H.264, increasing CRF only if needed."""
    attempted: list[str] = []
    width, height = resolution
    for candidate_crf in range(crf, 31, 2):
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(candidate_crf),
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
            "-r",
            str(fps),
            "-vsync",
            "cfr",
            "-movflags",
            "+faststart",
            "-metadata",
            "title=Supplementary movie 2",
            str(output_path),
        ]
        attempted.append(
            "lossless RGB frame stream -> libx264 "
            f"(CRF {candidate_crf}, yuv420p, CFR, fast-start)"
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stderr is None:
            raise RuntimeError("Failed to open the FFmpeg frame stream.")
        cached_path: Path | None = None
        cached_bytes: bytes | None = None
        try:
            for source in sequence_sources:
                if source != cached_path:
                    with Image.open(source) as image:
                        rgb_image = image.convert("RGB")
                        if rgb_image.size != resolution:
                            raise ValueError(
                                f"Frame {source.name} has size {rgb_image.size}; "
                                f"expected {resolution}."
                            )
                        cached_bytes = rgb_image.tobytes()
                    cached_path = source
                if cached_bytes is None:
                    raise RuntimeError("Frame cache unexpectedly remained empty.")
                process.stdin.write(cached_bytes)
        except BrokenPipeError:
            pass
        finally:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed with exit code {return_code}:\n{stderr}"
            )
        if output_path.stat().st_size < maximum_bytes:
            return candidate_crf, attempted
    raise RuntimeError(
        f"Encoded movie remains above {maximum_bytes} bytes through CRF 30."
    )


def probe_video(ffprobe: Path, path: Path) -> dict[str, Any]:
    """Return JSON media metadata from ffprobe."""
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    probe = json.loads(result.stdout)
    if "format" in probe:
        probe["format"]["filename"] = path.name
    return probe


def video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    """Return the unique video stream from ffprobe output."""
    videos = [
        stream
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    if len(videos) != 1:
        raise ValueError(f"Expected one video stream; found {len(videos)}.")
    return videos[0]


def rate_as_float(value: str) -> float:
    """Convert an ffprobe rational frame-rate string to float."""
    return float(Fraction(value))


def representative_segments(
    segments: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Select the required pre/post-encoding visual-check frames."""

    def scientific(mode: int, time_ip: float) -> dict[str, Any]:
        return next(
            segment
            for segment in segments
            if segment["kind"] == "scientific_frame"
            and segment["vertical_mode"] == mode
            and np.isclose(segment["time_ip"], time_ip)
        )

    opening = next(
        segment for segment in segments if segment["kind"] == "opening_title"
    )
    transition_4 = next(
        segment
        for segment in segments
        if segment["kind"] == "chapter_title"
        and segment["vertical_mode"] == 4
    )
    transition_16 = next(
        segment
        for segment in segments
        if segment["kind"] == "chapter_title"
        and segment["vertical_mode"] == 16
    )
    transition_32 = next(
        segment
        for segment in segments
        if segment["kind"] == "chapter_title"
        and segment["vertical_mode"] == 32
    )
    return [
        ("Opening title", opening),
        ("Chapter 1 title", transition_4),
        ("n=4, t=0 IP", scientific(4, 0.0)),
        ("n=4, t=10 IP", scientific(4, 10.0)),
        ("n=4, t=25 IP", scientific(4, 25.0)),
        ("n=4, t=50 IP", scientific(4, 50.0)),
        ("Transition to n=16", transition_16),
        ("n=16, t=25 IP", scientific(16, 25.0)),
        ("n=16, t=50 IP", scientific(16, 50.0)),
        ("Transition to n=32", transition_32),
        ("n=32, t=50 IP", scientific(32, 50.0)),
    ]


def extract_encoded_frames(
    ffmpeg: Path,
    video_path: Path,
    output_directory: Path,
    frame_numbers: list[int],
) -> list[Path]:
    """Decode exact representative frame numbers from the encoded movie."""
    expression = "+".join(f"eq(n\\,{number})" for number in frame_numbers)
    pattern = output_directory / "encoded_%02d.bmp"
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select={expression}",
        "-vsync",
        "0",
        str(pattern),
    ]
    subprocess.run(command, check=True)
    bitmap_paths = sorted(output_directory.glob("encoded_*.bmp"))
    if len(bitmap_paths) != len(frame_numbers):
        raise RuntimeError(
            "Expected "
            f"{len(frame_numbers)} decoded QC frames; found {len(bitmap_paths)}."
        )
    paths: list[Path] = []
    for bitmap_path in bitmap_paths:
        png_path = bitmap_path.with_suffix(".png")
        with Image.open(bitmap_path) as image:
            image.convert("RGB").save(png_path, compress_level=3)
        bitmap_path.unlink()
        paths.append(png_path)
    return paths


def image_psnr(reference_path: Path, encoded_path: Path) -> float:
    """Return RGB peak signal-to-noise ratio for a representative frame."""
    with (
        Image.open(reference_path).convert("RGB") as reference_image,
        Image.open(encoded_path).convert("RGB") as encoded_image,
    ):
        reference = np.asarray(reference_image, dtype=np.float64)
        encoded = np.asarray(encoded_image, dtype=np.float64)
    if reference.shape != encoded.shape:
        raise ValueError("Pre/post-encoding QC frames have different dimensions.")
    mean_square_error = float(np.mean((reference - encoded) ** 2))
    if mean_square_error == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mean_square_error)))


def thumbnail(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Return a downsampled RGB thumbnail for the contact sheet."""
    with Image.open(path).convert("RGB") as image:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        left = (size[0] - image.width) // 2
        top = (size[1] - image.height) // 2
        canvas.paste(image, (left, top))
        return np.asarray(canvas)


def create_qc_contact_sheet(
    reference_paths: list[Path],
    encoded_paths: list[Path],
    labels: list[str],
    output_path: Path,
) -> None:
    """Create a compact visual comparison before and after H.264 encoding."""
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            2,
            len(labels),
            figsize=(4.0 * len(labels), 7.0),
            dpi=100,
        )
        for column, label in enumerate(labels):
            axes[0, column].imshow(thumbnail(reference_paths[column], (420, 236)))
            axes[1, column].imshow(thumbnail(encoded_paths[column], (420, 236)))
            axes[0, column].set_title(label, fontsize=11.0, pad=5.0)
            for row in range(2):
                axes[row, column].axis("off")
        axes[0, 0].text(
            -0.03,
            0.5,
            "Before encoding",
            transform=axes[0, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=12.0,
        )
        axes[1, 0].text(
            -0.03,
            0.5,
            "After H.264 encoding",
            transform=axes[1, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=12.0,
        )
        figure.subplots_adjust(
            left=0.035,
            right=0.995,
            top=0.93,
            bottom=0.02,
            wspace=0.025,
            hspace=0.08,
        )
        figure.savefig(output_path, dpi=100)
        plt.close(figure)


def clipping_text(metadata: dict[str, Any]) -> str:
    """Return a concise clipping disclosure for captions and notes."""
    records = metadata["color_limits"]["clipping"]
    pieces = []
    for mode in (str(value) for value in metadata["vertical_modes"]):
        absolute = records[mode]["absolute_field"]
        difference = records[mode]["difference_field"]
        pieces.append(
            f"n={mode}: absolute {100.0 * absolute['clipped_fraction']:.3f}% "
            f"and difference {100.0 * difference['clipped_fraction']:.3f}%"
        )
    return "; ".join(pieces)


def write_auxiliary_files(
    output_directory: Path,
    data: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Write the caption, accessibility text, notes, and local README."""
    metadata = data["metadata"]
    spatial = metadata["spatial_discretisation"]
    time_discretisation = metadata["time_discretisation"]
    grid_points = spatial["grid_points"]
    steps_per_ip = int(time_discretisation["steps_per_inertial_period"])
    grid_label = f"{int(grid_points[0])}x{int(grid_points[1])}"
    grid_tex = f"{int(grid_points[0])}\\times{int(grid_points[1])}"
    probe = manifest["ffprobe"]
    stream = video_stream(probe)
    format_info = probe["format"]
    size_bytes = int(format_info["size"])
    duration = float(format_info["duration"])
    frame_rate = rate_as_float(stream["avg_frame_rate"])
    clipping = clipping_text(metadata)
    sample_interval = metadata["sampling"]["sample_interval_ip"]
    hold_frames = manifest["hold_frames"]
    opening_seconds = manifest["opening_seconds"]
    title_seconds = manifest["title_seconds"]
    chapter_end_frames = manifest["chapter_end_frames"]
    chapter_end_seconds = manifest["chapter_end_seconds"]

    caption = (
        "Movie 2. Evolution of the horizontal wave-velocity field in the "
        "sinusoidal-dipole background flow for YBJ, TSB, YBJ+, PSE and the "
        "hydrostatic Boussinesq equations (HBEs). The opening follows movie 1's "
        f"two-page structure: an overall title page held for {opening_seconds:g} "
        f"s followed by the first chapter page held for {title_seconds:g} s. "
        "The three sequential chapters "
        "show vertical modes $$n=4$$, $$n=16$$ and $$n=32$$ from 0 to 50 "
        "inertial periods (IP). For the common depth "
        "$$H=4000\\,\\mathrm{m}$$, their respective vertical wavelengths are "
        "$$h=1000\\,\\mathrm{m}$$, $$h=250\\,\\mathrm{m}$$ and "
        "$$h=125\\,\\mathrm{m}$$. "
        f"All five models use the manuscript's $${grid_tex}$$ horizontal "
        f"grid and {steps_per_ip} time steps per inertial period "
        f"($$f_c={steps_per_ip}$$). The upper row is "
        "$$|\\phi|^2/|\\phi_{\\mathrm{amp}}|^2$$ in the model order YBJ, TSB, "
        "YBJ+, PSE and HBEs. The first four panels of the lower row are the "
        "named model minus HBEs, explicitly "
        "$$(|\\phi_{\\mathrm{model}}|^2-|\\phi_{\\mathrm{HBEs}}|^2)"
        "/|\\phi_{\\mathrm{amp}}|^2$$; the final panel shows the instantaneous "
        "complex-velocity normalised root-mean-square error relative to HBEs, "
        "with moving markers at the displayed time. "
        "The NRE vertical axis spans 0--40% for $$n=4$$ and 0--10% for "
        "$$n=16$$ and $$n=32$$; these mode-specific limits remain fixed "
        "within their chapters. "
        "Within each mode, absolute and difference colour limits are fixed for "
        "the complete movie; difference limits are symmetric about zero. The "
        f"movie uses every {sample_interval:g}-IP saved state in strictly "
        f"increasing time order, holds each state for {hold_frames} video "
        f"frames, and holds each chapter's final state for an additional "
        f"{chapter_end_frames} frames ({chapter_end_seconds:g} s) before the "
        "next transition or the end of the movie. It does not interpolate "
        "physical fields. Clipping is marked "
        f"by extended colourbar ends ({clipping})."
    )
    (output_directory / "movie2_caption.txt").write_text(
        caption + "\n",
        encoding="utf-8",
    )

    accessibility = (
        "Accessibility description for movie 2\n\n"
        "The movie begins with two successive title pages matching movie 1: "
        f"an overall movie page displayed for {opening_seconds:g} seconds and "
        f"then the Chapter 1 page displayed for {title_seconds:g} seconds. The "
        "movie has three chapters, in the order "
        "n=4, n=16 and n=32. Each "
        "chapter title also gives the corresponding vertical wavelength: "
        "1000 m, 250 m and 125 m, respectively, for the common 4000-m depth. "
        f"The displayed fields are the manuscript's {grid_label}, fc="
        f"{steps_per_ip} results ({steps_per_ip} time steps per inertial period). "
        "The NRE vertical axis spans 0 to 40% for n=4 and 0 to 10% for n=16 "
        "and n=32; each range stays fixed throughout its chapter. "
        "Each "
        "scientific frame is a two-row, five-column layout on a white "
        "background. The "
        "current mode and time in inertial periods are written in a large "
        "heading at the top. In the upper row, left to right, square panels are "
        "labelled YBJ, TSB, YBJ+, PSE and HBEs. Purple-to-white-to-brown colour "
        "variation represents low-to-intermediate-to-high normalised squared "
        "velocity, with the same scale for all five models throughout a "
        "chapter. In the lower row, the first four square panels are explicitly "
        "labelled YBJ minus HBEs, TSB minus HBEs, YBJ+ minus HBEs and PSE minus "
        "HBEs. The lower-row colour scale is the displayed model's normalised "
        "squared velocity minus the HBEs normalised squared velocity. Blue and "
        "red sides of its zero-centred diverging scale indicate negative and "
        "positive differences; the signs, panel positions and labels make the "
        "comparisons understandable without relying on colour alone. The "
        "lower-right panel plots instantaneous NRE against time. YBJ is a blue "
        "solid line with circles, TSB a purple dashed line with triangles, YBJ+ "
        "a green solid line with plus signs, and PSE a red solid line with "
        "squares. A vertical dotted line and enlarged symbols identify the "
        "current time.\n\n"
        "For n=4, compact high-amplitude regions develop and strengthen, and "
        "the models increasingly differ in peak magnitude at long times while "
        "remaining spatially aligned. The n=16 chapter shows the intermediate "
        "regime: its modulation and model-minus-HBEs structure lie between the "
        "strong localisation at n=4 and the weaker modulation at n=32. For "
        "n=32, the PSE and HBEs retain closely matched fine, diagonal and "
        "curved structures, while the scalar-model difference panels and NRE "
        "curves expose smaller polarisation-related discrepancies. Static "
        "chapter title frames separate the modes; no morphing or field "
        f"interpolation is used. After each case reaches 50 IP, that final "
        f"frame remains still for an additional {chapter_end_seconds:g} seconds "
        "before the next chapter or the end of the movie."
    )
    (output_directory / "movie2_accessibility_description.txt").write_text(
        accessibility + "\n",
        encoding="utf-8",
    )

    audio_streams = [
        item
        for item in probe.get("streams", [])
        if item.get("codec_type") == "audio"
    ]
    submission = (
        "Submission notes for movie 2\n\n"
        "ScholarOne file designation: Movie\n"
        "File: movie2.mp4\n"
        f"Container: MP4 ({format_info.get('format_name', 'unknown')})\n"
        f"Video codec: {stream.get('codec_long_name', stream.get('codec_name'))} "
        f"({stream.get('codec_name')})\n"
        f"Profile: {stream.get('profile', 'not reported')}\n"
        f"Resolution: {stream['width']}x{stream['height']}\n"
        f"Frame rate: {frame_rate:g} fps, constant frame rate\n"
        f"Duration: {duration:.3f} s\n"
        f"Frame count: {stream.get('nb_read_frames', stream.get('nb_frames'))}\n"
        f"Pixel format: {stream.get('pix_fmt')}\n"
        f"File size: {size_bytes} bytes ({size_bytes / 1_000_000:.3f} MB)\n"
        f"Audio streams: {len(audio_streams)} (the movie is silent)\n"
        f"Fast start: {'yes' if manifest['faststart'] else 'no'}\n\n"
        "Simulation data\n"
        f"- Horizontal grid: {grid_label}\n"
        f"- Time stepping: fc={steps_per_ip} ({steps_per_ip} steps per "
        "inertial period)\n"
        "- Source: processed full complex-velocity fields used for Figures "
        "9--10\n\n"
        "JFM/Cambridge checks\n"
        "- MP4 container: passed\n"
        "- H.264 video codec: passed\n"
        "- File designation Movie: use this in ScholarOne\n"
        "- Numbered and titled movie 2: passed\n"
        "- Separate caption supplied: passed\n"
        "- TeX maths in the caption is bounded by $$: passed\n"
        "- File smaller than 50 MB: passed\n"
        "- Browser-oriented H.264/yuv420p encoding and decode test: passed\n"
        "- No audio stream or background music: passed\n\n"
        f"The overall opening page is held for {opening_seconds:g} seconds; "
        f"each Chapter page is held for {title_seconds:g} seconds to ensure "
        "its mode information can be read comfortably.\n"
        f"Each chapter ends with an additional {chapter_end_seconds:g}-second "
        "still hold of its true 50-IP terminal frame.\n\n"
        f"JFM preparing-materials guidance: {JFM_PREPARING_URL}\n"
        f"JFM submitting-materials guidance: {JFM_SUBMITTING_URL}\n"
        f"Cambridge supplementary-material guidance: {CAMBRIDGE_SUPPLEMENT_URL}\n"
    )
    (output_directory / "movie2_submission_notes.txt").write_text(
        submission,
        encoding="utf-8",
    )

    reference = (
        "The evolution leading from figures 9 and 10 is shown in "
        "supplementary movie 2.\n"
    )
    (
        output_directory / "movie2_manuscript_reference_suggestion.txt"
    ).write_text(reference, encoding="utf-8")

    readme = (
        "# Supplementary movie 2\n\n"
        "This directory contains the submission-ready movie and its supporting "
        "description and quality-control files. The manuscript source is not "
        "modified by this workflow.\n\n"
        "## Submission files\n\n"
        "- `movie2.mp4`: silent MP4/H.264 movie for upload with the ScholarOne "
        "file designation `Movie`.\n"
        "- `movie2_caption.txt`: separate JFM caption.\n"
        "- `movie2_accessibility_description.txt`: non-visual description of "
        "layout, encodings and scientific evolution.\n"
        "- `movie2_submission_notes.txt`: technical metadata and JFM checklist.\n"
        "- `movie2_manuscript_reference_suggestion.txt`: suggested citation "
        "sentence only; it has not been inserted into the manuscript.\n\n"
        "## Review files\n\n"
        "- `movie2_preview.png`: representative frame decoded from the final "
        "H.264 movie.\n"
        "- `movie2_qc_contact_sheet.png`: representative frames before and "
        "after encoding.\n"
        "- `movie2_render_manifest.json`: frame timing, fixed scales and encoder "
        "metadata.\n"
        "- `movie2_quality_report.json`: numerical, media and visual-QC results.\n\n"
        "The three chapters show n=4, n=16 and n=32, with the manuscript "
        "vertical wavelengths 1000 m, 250 m and 125 m, "
        f"respectively. The movie opens with a {opening_seconds:g}-second "
        f"overall title page followed by a {title_seconds:g}-second Chapter 1 "
        "page; the later Chapter pages use the same longer duration. The "
        "curve panel shows instantaneous NRE, distinct from "
        "the time-averaged NRE in manuscript figure 8. Its y-axis is fixed at "
        "0--40% for n=4 and 0--10% for n=16 and n=32. The 51 physical states "
        f"in each chapter are true saved {grid_label}, fc={steps_per_ip} "
        "integer-period outputs from 0 to 50 IP. "
        "Video-frame "
        f"holding controls playback speed; each terminal state is held for an "
        f"additional {chapter_end_seconds:g} seconds, and no field "
        "interpolation is used."
    )
    readme_path = output_directory / "README.md"
    movie1_marker = "<!-- BEGIN MOVIE 1 -->"
    if readme_path.is_file():
        existing_readme = readme_path.read_text(encoding="utf-8")
        marker_position = existing_readme.find(movie1_marker)
        if marker_position >= 0:
            readme = (
                readme.rstrip()
                + "\n\n"
                + existing_readme[marker_position:].lstrip()
            )
    readme_path.write_text(readme + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render and encode sinusoidal-dipole supplementary movie 2."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=parse_resolution("2560x1440"),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--hold-frames", type=int, default=6)
    parser.add_argument("--opening-seconds", type=float, default=2.0)
    parser.add_argument("--title-seconds", type=float, default=4.0)
    parser.add_argument("--chapter-end-seconds", type=float, default=1.5)
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the managed rendered-frame directory for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    """Render unique frames, encode the movie, and write submission files."""
    args = parse_args()
    if not 20 <= args.fps <= 24:
        raise ValueError("Final movie frame rate must be between 20 and 24 fps.")
    if args.opening_seconds <= 0.0 or args.title_seconds <= 0.0:
        raise ValueError("--opening-seconds and --title-seconds must be positive.")
    if not 0 <= args.crf <= 30:
        raise ValueError("--crf must lie between 0 and 30.")
    opening_frames = int(round(args.opening_seconds * args.fps))
    title_frames = int(round(args.title_seconds * args.fps))
    if opening_frames <= 0 or title_frames <= 0:
        raise ValueError("A title-page duration produced no video frames.")
    chapter_end_frames = int(round(args.chapter_end_seconds * args.fps))
    if chapter_end_frames <= 0:
        raise ValueError("The chapter-end duration produced no video frames.")

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Movie archive does not exist: {input_path}")
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_executable(args.ffmpeg, "ffmpeg")
    ffprobe = resolve_executable(args.ffprobe, "ffprobe")
    data = load_movie_data(input_path)
    times = data["times_in_inertial_periods"]
    modes = data["vertical_modes"]
    video_path = output_directory / "movie2.mp4"

    managed_root: Path
    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_frames:
        managed_root = output_directory / "movie2_managed_frames"
        if managed_root.exists():
            raise FileExistsError(
                f"Managed frame directory already exists: {managed_root}"
            )
        managed_root.mkdir(parents=True)
    else:
        temporary_context = tempfile.TemporaryDirectory(
            prefix=".movie2_frames_",
            dir=output_directory,
        )
        managed_root = Path(temporary_context.name)

    try:
        unique_directory = managed_root / "unique"
        qc_directory = managed_root / "qc"
        unique_directory.mkdir(parents=True)
        qc_directory.mkdir(parents=True)
        unique_frames: dict[str, Path] = {}
        opening_path = unique_directory / "opening_title.png"
        render_opening_frame(
            opening_path,
            resolution=args.resolution,
            dpi=args.dpi,
        )
        unique_frames["opening_title"] = opening_path

        for chapter, (mode_index, mode) in enumerate(
            enumerate(modes),
            start=1,
        ):
            title_path = unique_directory / f"n{int(mode)}_title.png"
            render_title_frame(
                title_path,
                mode=int(mode),
                chapter=chapter,
                chapter_count=len(modes),
                resolution=args.resolution,
                dpi=args.dpi,
            )
            unique_frames[f"n{int(mode)}_title"] = title_path
            renderer = ChapterRenderer(
                data,
                mode_index,
                resolution=args.resolution,
                dpi=args.dpi,
            )
            try:
                for time_index, time_value in enumerate(times):
                    renderer.update(time_index)
                    frame_path = (
                        unique_directory
                        / f"n{int(mode)}_t{time_index:03d}_{time_value:g}IP.png"
                    )
                    renderer.save(frame_path, dpi=args.dpi)
                    unique_frames[f"n{int(mode)}_t{time_index:03d}"] = frame_path
                    if time_index % 10 == 0 or time_index == times.size - 1:
                        print(
                            f"rendered n={int(mode)}, t={float(time_value):g} IP"
                        )
            finally:
                renderer.close()

        segments, sequence_sources = build_sequence(
            unique_frames,
            times,
            modes,
            frame_stride=args.frame_stride,
            hold_frames=args.hold_frames,
            opening_frames=opening_frames,
            title_frames=title_frames,
            chapter_end_frames=chapter_end_frames,
        )
        selected_crf, attempted_commands = encode_video(
            ffmpeg,
            sequence_sources,
            video_path,
            resolution=args.resolution,
            fps=args.fps,
            crf=args.crf,
            maximum_bytes=MAX_FILE_BYTES,
        )
        probe = probe_video(ffprobe, video_path)
        selected_qc = representative_segments(segments)
        qc_frame_numbers = [
            int(segment["start_frame"] + segment["frame_count"] // 2)
            for _, segment in selected_qc
        ]
        reference_paths = [
            sequence_sources[frame_number] for frame_number in qc_frame_numbers
        ]
        encoded_paths = extract_encoded_frames(
            ffmpeg,
            video_path,
            qc_directory,
            qc_frame_numbers,
        )
        labels = [label for label, _ in selected_qc]
        psnr_values = [
            image_psnr(reference, encoded)
            for reference, encoded in zip(
                reference_paths,
                encoded_paths,
                strict=True,
            )
        ]
        create_qc_contact_sheet(
            reference_paths,
            encoded_paths,
            labels,
            output_directory / "movie2_qc_contact_sheet.png",
        )
        preview_index = labels.index("n=4, t=50 IP")
        shutil.copy2(
            encoded_paths[preview_index],
            output_directory / "movie2_preview.png",
        )

        video_bytes = video_path.read_bytes()
        moov_position = video_bytes.find(b"moov")
        mdat_position = video_bytes.find(b"mdat")
        faststart = (
            moov_position >= 0
            and mdat_position >= 0
            and moov_position < mdat_position
        )
        stream = video_stream(probe)
        frame_count = int(
            stream.get("nb_read_frames")
            or stream.get("nb_frames")
            or len(sequence_sources)
        )
        duration = float(probe["format"]["duration"])
        nre_y_limits_percent = {
            str(int(mode)): [
                0.0,
                nice_nre_upper(
                    float(np.max(data["nre_complex_relative_l2"][mode_index]))
                    * 100.0
                ),
            ]
            for mode_index, mode in enumerate(modes)
        }
        manifest = {
            "schema_version": 1,
            "product": "supplementary movie 2",
            "video_file": "movie2.mp4",
            "video_sha256": sha256_file(video_path),
            "input_archive_sha256": sha256_file(input_path),
            "resolution": list(args.resolution),
            "fps": args.fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "frame_stride": args.frame_stride,
            "hold_frames": args.hold_frames,
            "opening_frames": opening_frames,
            "opening_seconds": args.opening_seconds,
            "title_frames": title_frames,
            "title_seconds": args.title_seconds,
            "opening_page_count": 2,
            "chapter_end_frames": chapter_end_frames,
            "chapter_end_seconds": args.chapter_end_seconds,
            "vertical_modes": modes.tolist(),
            "source_data": {
                "kind": data["metadata"]["source_kind"],
                "spatial_discretisation": data["metadata"][
                    "spatial_discretisation"
                ],
                "time_discretisation": data["metadata"][
                    "time_discretisation"
                ],
                "pse_field_source": data["metadata"]["pse_field_source"],
            },
            "selected_crf": selected_crf,
            "encoder": "libx264",
            "pixel_format": stream.get("pix_fmt"),
            "faststart": faststart,
            "audio_stream_count": len(
                [
                    item
                    for item in probe.get("streams", [])
                    if item.get("codec_type") == "audio"
                ]
            ),
            "segments": segments,
            "representative_qc": [
                {
                    "label": label,
                    "frame_number": frame_number,
                    "psnr_db": psnr,
                }
                for (label, _), frame_number, psnr in zip(
                    selected_qc,
                    qc_frame_numbers,
                    psnr_values,
                    strict=True,
                )
            ],
            "fixed_absolute_color_limits": data[
                "absolute_color_limits"
            ].tolist(),
            "fixed_difference_color_limits": data[
                "difference_color_limits"
            ].tolist(),
            "style_alignment": {
                "manuscript_figures": [8, 9, 10],
                "preferred_text_font": PREFERRED_TEXT_FONT,
                "font_stack": list(MOVIE_FONT_STACK),
                "mathtext_fontset": "stix",
                "model_colors": {
                    label: color for label, color, _, _ in MODEL_STYLES
                },
                "nre_legend_visual_rows": [
                    list(row) for row in NRE_LEGEND_VISUAL_ROWS
                ],
                "vertical_wavelength_metres": {
                    str(mode): wavelength
                    for mode, wavelength in VERTICAL_WAVELENGTH_METRES.items()
                },
                "nre_y_limits_percent": nre_y_limits_percent,
                "subplot_title_fontsize": SUBPLOT_TITLE_FONTSIZE,
                "nre_title_fontsize": NRE_TITLE_FONTSIZE,
                "upper_row_quantity": "|phi|^2/|phi_amp|^2",
                "difference_quantity": (
                    "(|phi_model|^2-|phi_HBEs|^2)/|phi_amp|^2"
                ),
                "nre_quantity": (
                    "instantaneous complex-velocity NRE relative to HBEs"
                ),
                "title_card_alignment": {
                    "reference": "supplementary movie 1",
                    "reference_resolution": list(
                        TITLE_CARD_REFERENCE_RESOLUTION
                    ),
                    "reference_dpi": TITLE_CARD_REFERENCE_DPI,
                    "title_position": list(TITLE_CARD_TITLE_POSITION),
                    "subtitle_position": list(
                        TITLE_CARD_SUBTITLE_POSITION
                    ),
                    "reference_title_fontsize": (
                        TITLE_CARD_REFERENCE_TITLE_FONTSIZE
                    ),
                    "reference_subtitle_fontsize": (
                        TITLE_CARD_REFERENCE_SUBTITLE_FONTSIZE
                    ),
                    "opening_page_count": 2,
                    "opening_title": "Supplementary movie 2",
                    "opening_subtitle": (
                        "Wave-field evolution in a sinusoidal-dipole "
                        "background flow"
                    ),
                    "chapter_title_pattern": "Chapter {chapter}",
                    "chapter_subtitle_pattern": (
                        "Vertical mode n={mode} (vertical wavelength "
                        "h={wavelength} m); 0-50 inertial periods"
                    ),
                },
            },
            "physical_field_interpolation": False,
            "ffmpeg_attempts": [
                {
                    "crf": args.crf + 2 * index,
                    "command_summary": (
                        "lossless RGB frame stream -> libx264, "
                        "yuv420p, CFR, +faststart"
                    ),
                }
                for index, _ in enumerate(attempted_commands)
            ],
            "ffprobe": probe,
        }
        manifest_path = output_directory / "movie2_render_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_auxiliary_files(output_directory, data, manifest)
        print(f"wrote {video_path}")
        print(
            f"{frame_count} frames, {duration:.3f} s, "
            f"{video_path.stat().st_size / 1_000_000:.3f} MB, CRF {selected_crf}"
        )
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()


if __name__ == "__main__":
    main()
