"""Plot the Gaussian-vortex eigenfunctions shown in Figure 7.

The script reads selected radial eigenfunctions from the data file generated
by ``compute_eigenanalysis.py`` and performs no eigensystem calculation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np


CORIOLIS_FREQUENCY = 1.0e-4
FLOW_LENGTH = 25.0e3

DEFAULT_INPUT = (
    Path(__file__).resolve().parent / "data" / "gaussian_vortex_eigenanalysis.npz"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "figures" / "gaussian_vortex_eigenfunctions"
)


def publication_style(*, use_tex: bool) -> dict[str, object]:
    """Return the typography used for the published mode maps."""
    style: dict[str, object] = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.size": 20.0,
        "axes.labelsize": 22.0,
        "axes.titlesize": 22.0,
        "xtick.labelsize": 22.0,
        "ytick.labelsize": 22.0,
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if use_tex:
        style.update(
            {
                "text.usetex": True,
                "text.latex.preamble": (
                    r"\usepackage{newtxtext}"
                    r"\usepackage{newtxmath}"
                ),
            }
        )
    else:
        style.update({"text.usetex": False, "mathtext.fontset": "stix"})
    return style


def signed_colormap() -> mpl.colors.LinearSegmentedColormap:
    """Return the signed blue--white--red map used for both components."""
    return mpl.colors.LinearSegmentedColormap.from_list(
        "custom_blue_white_red",
        [
            (0.00, "#34509a"),
            (0.25, "#4f80d6"),
            (0.40, "#d0deef"),
            (0.50, "#f7f7f7"),
            (0.60, "#ecd4ce"),
            (0.75, "#d8674a"),
            (1.00, "#96324f"),
        ],
        N=256,
    )


def interpolate_complex(
    target_radius: np.ndarray,
    source_radius: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate a complex radial field by its real and imaginary parts."""
    return np.interp(
        target_radius,
        source_radius,
        values.real,
    ) + 1j * np.interp(
        target_radius,
        source_radius,
        values.imag,
    )


def build_spatial_fields(
    radius: np.ndarray,
    azimuthal_wavenumbers: np.ndarray,
    component_up: np.ndarray,
    component_down: np.ndarray,
    radial_domain: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, ...]],
    float,
]:
    """Reconstruct the two complex components on a Cartesian grid."""
    view_limit = 0.5 * np.pi * FLOW_LENGTH
    grid = np.linspace(-view_limit, view_limit, 501)
    coordinate_x, coordinate_y = np.meshgrid(grid, grid)
    map_radius = np.hypot(coordinate_x, coordinate_y)
    angle = np.arctan2(coordinate_y, coordinate_x)
    inside = map_radius <= radial_domain

    fields: list[tuple[np.ndarray, ...]] = []
    upper_limit = 0.0
    for index, azimuthal_wavenumber in enumerate(azimuthal_wavenumbers):
        upper_radial = interpolate_complex(
            map_radius.ravel(),
            radius,
            component_up[index],
        ).reshape(map_radius.shape)
        lower_radial = interpolate_complex(
            map_radius.ravel(),
            radius,
            component_down[index],
        ).reshape(map_radius.shape)

        upper_complex = upper_radial * np.exp(1j * azimuthal_wavenumber * angle)
        lower_complex = lower_radial * np.exp(1j * (azimuthal_wavenumber - 2) * angle)
        upper_map = upper_complex.real
        lower_map = lower_complex.real
        upper_map[~inside] = np.nan
        lower_map[~inside] = np.nan
        upper_complex[~inside] = np.nan + 1j * np.nan
        lower_complex[~inside] = np.nan + 1j * np.nan
        upper_limit = max(
            upper_limit,
            float(np.nanmax(np.abs(upper_map))),
        )
        fields.append(
            (
                upper_map,
                lower_map,
                upper_complex,
                lower_complex,
            )
        )
    return coordinate_x, coordinate_y, fields, upper_limit


def plot_eigenfunctions(
    radius: np.ndarray,
    azimuthal_wavenumbers: np.ndarray,
    frequencies: np.ndarray,
    component_up: np.ndarray,
    component_down: np.ndarray,
    radial_domain: float,
    output_stem: Path,
    *,
    use_tex: bool,
) -> None:
    """Create the two-row Cartesian eigenfunction montage."""
    (
        coordinate_x,
        coordinate_y,
        fields,
        upper_limit,
    ) = build_spatial_fields(
        radius,
        azimuthal_wavenumbers,
        component_up,
        component_down,
        radial_domain,
    )
    lower_limit = 0.20

    with plt.rc_context(publication_style(use_tex=use_tex)):
        figure, axes = plt.subplots(
            2,
            azimuthal_wavenumbers.size,
            figsize=(26.4, 12.1),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(
            2,
            azimuthal_wavenumbers.size,
        )
        colormap = signed_colormap()
        normalizations = (
            mpl.colors.TwoSlopeNorm(
                vmin=-upper_limit,
                vcenter=0.0,
                vmax=upper_limit,
            ),
            mpl.colors.TwoSlopeNorm(
                vmin=-lower_limit,
                vcenter=0.0,
                vmax=lower_limit,
            ),
        )
        images: list[mpl.collections.QuadMesh | None] = [None, None]

        for column, azimuthal_wavenumber in enumerate(azimuthal_wavenumbers):
            (
                upper_map,
                lower_map,
                _,
                _,
            ) = fields[column]
            for row, (
                field,
                radial_values,
                component_limit,
            ) in enumerate(
                (
                    (
                        upper_map,
                        component_up[column],
                        upper_limit,
                    ),
                    (
                        lower_map,
                        component_down[column],
                        lower_limit,
                    ),
                )
            ):
                axis = axes[row, column]
                images[row] = axis.pcolormesh(
                    coordinate_x / FLOW_LENGTH,
                    coordinate_y / FLOW_LENGTH,
                    field,
                    shading="auto",
                    cmap=colormap,
                    norm=normalizations[row],
                    rasterized=True,
                )

                quiver_line = np.linspace(
                    -0.5 * np.pi,
                    0.5 * np.pi,
                    13,
                )
                quiver_x, quiver_y = np.meshgrid(
                    quiver_line,
                    quiver_line,
                )
                quiver_radius = FLOW_LENGTH * np.hypot(
                    quiver_x,
                    quiver_y,
                )
                quiver_angle = np.arctan2(quiver_y, quiver_x)
                interpolated = interpolate_complex(
                    quiver_radius.ravel(),
                    radius,
                    radial_values,
                ).reshape(quiver_x.shape)
                angular_order = (
                    azimuthal_wavenumber if row == 0 else azimuthal_wavenumber - 2
                )
                vector_field = (
                    interpolated
                    * np.exp(1j * angular_order * quiver_angle)
                    / component_limit
                )
                vector_amplitude = np.abs(vector_field)
                vector_mask = np.isfinite(vector_amplitude) & (vector_amplitude > 0.12)
                axis.quiver(
                    quiver_x[vector_mask],
                    quiver_y[vector_mask],
                    vector_field.real[vector_mask],
                    vector_field.imag[vector_mask],
                    angles="xy",
                    scale_units="xy",
                    scale=3.2 if row == 0 else 4.2,
                    minlength=0.2,
                    pivot="middle",
                    rasterized=False,
                    width=0.0110,
                    headwidth=3.8,
                    headlength=4.8,
                    headaxislength=4.8,
                    color="black",
                    edgecolor="black",
                    linewidth=0.0,
                    antialiased=False,
                    zorder=8,
                )
                axis.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        radial_domain / FLOW_LENGTH,
                        fill=False,
                        color="0.25",
                        linewidth=0.45,
                    )
                )
                axis.set_aspect("equal")
                for spine in axis.spines.values():
                    spine.set_linewidth(4.2)
                    spine.set_zorder(10)
                axis.set(
                    xlim=(-0.5 * np.pi, 0.5 * np.pi),
                    ylim=(-0.5 * np.pi, 0.5 * np.pi),
                )
                axis.set_xticks([-0.5 * np.pi, 0.0, 0.5 * np.pi])
                axis.set_yticks([-0.5 * np.pi, 0.0, 0.5 * np.pi])
                axis.set_xticklabels([r"$-\pi/2$", r"$x/L$", r"$\pi/2$"])
                axis.set_yticklabels([r"$-\pi/2$", r"$y/L$", r"$\pi/2$"])
                axis.xaxis.set_ticks_position("bottom")
                axis.yaxis.set_ticks_position("right")
                axis.yaxis.set_label_position("right")
                axis.tick_params(
                    axis="x",
                    length=10.5,
                    width=4.0,
                    pad=4.8,
                    direction="out",
                )
                axis.tick_params(
                    axis="y",
                    length=10.5,
                    width=4.0,
                    pad=4.8,
                    direction="out",
                    labelleft=False,
                    labelright=(column == azimuthal_wavenumbers.size - 1),
                )

            axes[0, column].set_title(
                rf"$\ell={azimuthal_wavenumber}$",
                fontsize=37.0,
                pad=16.0,
                loc="left",
            )
            axes[0, column].set_title(
                rf"$\omega\simeq "
                rf"{frequencies[column].real / CORIOLIS_FREQUENCY:.2f}f$",
                fontsize=37.0,
                pad=16.0,
                loc="right",
            )

        for axis in axes.flat:
            axis.tick_params(labelsize=32)
        for axis in axes[0, :]:
            axis.tick_params(axis="x", labelbottom=False)
        for axis in axes[1, :]:
            labels = axis.get_xticklabels()
            if len(labels) == 3:
                labels[0].set_horizontalalignment("left")
                labels[1].set_horizontalalignment("center")
                labels[2].set_horizontalalignment("right")
                labels[0].set_transform(
                    labels[0].get_transform()
                    + ScaledTranslation(
                        -7.0 / 72.0,
                        0.0,
                        figure.dpi_scale_trans,
                    )
                )
                labels[2].set_transform(
                    labels[2].get_transform()
                    + ScaledTranslation(
                        7.0 / 72.0,
                        0.0,
                        figure.dpi_scale_trans,
                    )
                )

        for row in range(2):
            labels = axes[row, -1].get_yticklabels()
            if len(labels) == 3:
                labels[0].set_verticalalignment("bottom")
                labels[1].set_rotation(0)
                labels[1].set_horizontalalignment("left")
                labels[1].set_verticalalignment("center")
                labels[2].set_verticalalignment("top")
                labels[0].set_transform(
                    labels[0].get_transform()
                    + ScaledTranslation(
                        0.0,
                        -7.0 / 72.0,
                        figure.dpi_scale_trans,
                    )
                )
                labels[2].set_transform(
                    labels[2].get_transform()
                    + ScaledTranslation(
                        0.0,
                        7.0 / 72.0,
                        figure.dpi_scale_trans,
                    )
                )

        if images[0] is None or images[1] is None:
            raise RuntimeError("No image data were plotted.")
        top_colorbar = figure.colorbar(
            images[0],
            ax=axes[0, :],
            location="right",
            shrink=1.0,
            pad=0.006,
        )
        bottom_colorbar = figure.colorbar(
            images[1],
            ax=axes[1, :],
            location="right",
            shrink=1.0,
            pad=0.006,
        )
        for colorbar in (top_colorbar, bottom_colorbar):
            if colorbar.norm.vmin is None or colorbar.norm.vmax is None:
                raise ValueError("The eigenfunction color limits are undefined.")
            colorbar.set_ticks(
                np.linspace(
                    colorbar.norm.vmin,
                    colorbar.norm.vmax,
                    3,
                ).tolist()
            )
            colorbar.ax.tick_params(
                length=11.0,
                width=4.6,
                pad=4.8,
                direction="out",
                labelsize=32,
            )
            colorbar.outline.set_linewidth(4.8)  # type: ignore[operator]
        bottom_colorbar.ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(
                lambda value, _: ("0" if np.isclose(value, 0.0) else f"{value:.1f}")
            )
        )

        figure.canvas.draw()
        for colorbar_index, colorbar in enumerate((top_colorbar, bottom_colorbar)):
            labels = colorbar.ax.get_yticklabels()
            if len(labels) >= 2:
                labels[0].set_verticalalignment("bottom")
                labels[-1].set_verticalalignment("top")
                labels[0].set_transform(
                    labels[0].get_transform()
                    + ScaledTranslation(
                        0.0,
                        -7.0 / 72.0,
                        figure.dpi_scale_trans,
                    )
                )
                labels[-1].set_transform(
                    labels[-1].get_transform()
                    + ScaledTranslation(
                        (3.8 if colorbar_index == 0 else 0.0) / 72.0,
                        (9.6 if colorbar_index == 0 else 7.0) / 72.0,
                        figure.dpi_scale_trans,
                    )
                )

        figure.canvas.draw()
        figure.set_layout_engine(None)
        original_panel_width = axes[0, 0].get_position().width
        original_column_gap = 0.022
        original_group_width = (
            azimuthal_wavenumbers.size * original_panel_width
            + (azimuthal_wavenumbers.size - 1) * original_column_gap
        )
        column_gap = 0.038
        row_gap_extra = -0.040
        outer_y_margin_extra = 0.010
        top_whitespace_extra = -0.004
        panel_width = (
            0.86
            * (original_group_width - (azimuthal_wavenumbers.size - 1) * column_gap)
            / azimuthal_wavenumbers.size
        )
        group_width = (
            azimuthal_wavenumbers.size * panel_width
            + (azimuthal_wavenumbers.size - 1) * column_gap
        )
        group_left = 0.5 - 0.5 * group_width

        for row in range(2):
            for column in range(azimuthal_wavenumbers.size):
                position = axes[row, column].get_position()
                vertical_shift = (
                    0.5 * row_gap_extra if row == 0 else -0.5 * row_gap_extra
                )
                vertical_shift += (
                    -outer_y_margin_extra if row == 0 else outer_y_margin_extra
                )
                vertical_shift += top_whitespace_extra
                panel_height = 0.94 * position.height
                panel_center = position.y0 + 0.5 * position.height + vertical_shift
                axes[row, column].set_position(
                    [
                        group_left + column * (panel_width + column_gap),
                        panel_center - 0.5 * panel_height,
                        panel_width,
                        panel_height,
                    ]
                )

        for row, colorbar in enumerate((top_colorbar, bottom_colorbar)):
            panel_position = axes[row, -1].get_position()
            colorbar_position = colorbar.ax.get_position()
            colorbar.ax.set_position(
                (
                    panel_position.x1 + 0.052,
                    panel_position.y0,
                    1.35 * colorbar_position.width,
                    panel_position.height,
                )
            )

        figure.canvas.draw()
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_stem.with_suffix(".png"),
            dpi=600,
            bbox_inches=None,
        )
        figure.savefig(
            output_stem.with_suffix(".pdf"),
            bbox_inches=None,
        )
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot the Gaussian-vortex selected eigenfunctions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-tex", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load the computed modes and create the eigenfunction montage."""
    args = parse_args()
    with np.load(args.input) as data:
        radius = data["mode_radius"]
        azimuthal_wavenumbers = data["mode_azimuthal_wavenumbers"]
        frequencies = data["mode_frequencies"]
        component_up = data["mode_component_up"]
        component_down = data["mode_component_down"]
        radial_domain = float(data["radial_domain"])
    plot_eigenfunctions(
        radius,
        azimuthal_wavenumbers,
        frequencies,
        component_up,
        component_down,
        radial_domain,
        args.output,
        use_tex=not args.no_tex,
    )
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
