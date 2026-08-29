"""Shared plotting definitions for the two eigenanalysis spectra."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def spectrum_publication_style(*, use_tex: bool) -> dict[str, object]:
    """Return the unchanged publication style for eigenanalysis spectra."""
    font_size = 10
    scale = 0.8
    style: dict[str, object] = {
        "font.family": "serif",
        "font.size": font_size * scale,
        "axes.labelsize": font_size * scale,
        "axes.titlesize": font_size * scale,
        "axes.linewidth": 1.2 * scale,
        "axes.unicode_minus": False,
        "xtick.labelsize": font_size * scale,
        "ytick.labelsize": font_size * scale,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.8 * scale,
        "ytick.major.size": 2.8 * scale,
        "xtick.major.width": 1.2 * scale,
        "ytick.major.width": 1.2 * scale,
        "legend.fontsize": font_size * scale,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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
        style.update({"text.usetex": False, "mathtext.fontset": "stix"})
    return style


def load_spectrum_colormap(path: Path) -> mpl.colors.ListedColormap:
    """Load the colour gradient used in the published spectra."""
    with h5py.File(path, "r") as file:
        colors = np.asarray(file["C"], dtype=float)
    if colors.ndim != 2:
        raise ValueError("The spectrum colour table must be a two-dimensional array.")
    if colors.shape[0] == 3 and colors.shape[1] != 3:
        colors = colors.T
    if colors.shape[1] != 3 or colors.shape[0] < 2:
        raise ValueError("The spectrum colour table must contain N x 3 RGB values.")
    if not np.all(np.isfinite(colors)):
        raise ValueError("The spectrum colour table contains non-finite values.")
    colors = np.clip(0.88 * colors[::-1], 0.0, 1.0)

    transition_size = 52
    gray_weight = 0.65 * (1.0 - np.linspace(0.0, 1.0, transition_size)) ** 1.4
    gray = colors[:transition_size].mean(axis=1, keepdims=True)
    colors[:transition_size] = (
        gray_weight[:, None] * gray
        + (1.0 - gray_weight[:, None]) * colors[:transition_size]
    )
    return mpl.colors.ListedColormap(colors, name="custom_gradient")


def add_ratio_colorbar(
    figure: mpl.figure.Figure,
    colorbar_axis: mpl.axes.Axes,
    *,
    normalization: mpl.colors.Normalize,
    colormap: mpl.colors.Colormap,
    ratio_min: float,
    ratio_max: float,
    scale: float,
) -> None:
    """Add the shared rotary-component ratio colorbar to a spectrum figure."""
    color_scale = mpl.cm.ScalarMappable(norm=normalization, cmap=colormap)
    color_scale.set_array([])
    colorbar = figure.colorbar(
        color_scale,
        cax=colorbar_axis,
        orientation="vertical",
    )
    ticks = np.linspace(ratio_min, ratio_max, 5)
    colorbar.set_ticks(ticks.tolist())
    colorbar.set_ticklabels([rf"${tick:.2g}$" for tick in ticks])
    colorbar.ax.tick_params(direction="out", pad=2 * scale)
    colorbar.ax.set_ylabel(
        r"Averaged ratio "
        r"$|\mathscr{A}_{\downarrow}|/|\mathscr{A}_{\uparrow}|$",
        rotation=90,
        labelpad=8 * scale,
        va="center",
    )


def save_spectrum_figure(figure: mpl.figure.Figure, output_stem: Path) -> None:
    """Save one spectrum in the repository's vector and raster formats."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(
            output_stem.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)
