"""Plot the analytic parallel-shear, Gaussian-vortex and dipole backgrounds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def publication_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }


def signed_colormap() -> mpl.colors.LinearSegmentedColormap:
    """Return the purple-white-burgundy map used for signed gradients."""
    return mpl.colors.LinearSegmentedColormap.from_list(
        "pse_signed",
        [
            (0.00, "#3b0f70"),
            (0.28, "#6a66a8"),
            (0.50, "#f1eef2"),
            (0.72, "#c45b65"),
            (1.00, "#68113b"),
        ],
        N=256,
    )


def pi_labels(values: list[float]) -> list[str]:
    labels = []
    for value in values:
        if np.isclose(value, -np.pi):
            labels.append(r"$-\pi$")
        elif np.isclose(value, np.pi):
            labels.append(r"$\pi$")
        else:
            labels.append("")
    return labels


def plot_figure(input_path: Path, output_stem: Path) -> None:
    with np.load(input_path) as data:
        coordinate = data["coordinate_over_length"]
        u = data["velocity_u_over_reference"]
        v = data["velocity_v_over_reference"]
        speed = data["speed_over_reference"]
        invariants = [data["xi1_over_f"], data["xi2_over_f"], data["xi3_over_f"]]
        profiles = data["sampled_v_over_reference_at_y0"]

    extent = [-np.pi, np.pi, -np.pi, np.pi]
    ticks = [-np.pi, 0.0, np.pi]
    with mpl.rc_context(publication_style()):
        figure, axes = plt.subplots(
            3,
            4,
            figsize=(7.2, 5.65),
            constrained_layout=False,
            gridspec_kw={
                "left": 0.085,
                "right": 0.975,
                "bottom": 0.155,
                "top": 0.925,
                "wspace": 0.22,
                "hspace": 0.31,
            },
        )
        speed_image = None
        invariant_image = None
        for row in range(3):
            for column in range(4):
                axis = axes[row, column]
                axis.set_aspect("equal")
                axis.set_xlim(-np.pi, np.pi)
                axis.set_ylim(-np.pi, np.pi)
                axis.set_xticks(ticks)
                axis.set_yticks(ticks)
                axis.tick_params(length=3.0, width=0.8)
                if row == 2:
                    axis.set_xticklabels(pi_labels(ticks))
                    axis.set_xlabel(r"$x/L$", labelpad=1)
                else:
                    axis.set_xticklabels([])
                if column == 0:
                    axis.set_yticklabels(pi_labels(ticks))
                    axis.set_ylabel(r"$y/L$", labelpad=1)
                else:
                    axis.set_yticklabels([])

                if column == 0:
                    speed_image = axis.imshow(
                        speed[row],
                        extent=extent,
                        origin="lower",
                        cmap="viridis",
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="bilinear",
                    )
                    stride = max(1, coordinate.size // 15)
                    axis.quiver(
                        coordinate[::stride],
                        coordinate[::stride],
                        u[row, ::stride, ::stride],
                        v[row, ::stride, ::stride],
                        color="black",
                        angles="xy",
                        scale_units="xy",
                        scale=7.5,
                        width=0.004,
                        headwidth=3.3,
                        headlength=4.2,
                    )
                else:
                    invariant_image = axis.imshow(
                        invariants[column - 1][row],
                        extent=extent,
                        origin="lower",
                        cmap=signed_colormap(),
                        vmin=-0.15,
                        vmax=0.15,
                        interpolation="bilinear",
                    )
                    axis.axhline(
                        0.0,
                        color="black",
                        linewidth=0.8,
                        linestyle=(0, (3.0, 2.0)),
                    )
                axis.plot(
                    coordinate,
                    0.62 * np.pi * profiles[row],
                    color="#ff00d4",
                    linewidth=2.0,
                )

        titles = (
            r"$|\boldsymbol{U}|/U_{\mathrm{ref}}$",
            r"$\xi_1/f$",
            r"$\xi_2/f$",
            r"$\xi_3/f$",
        )
        for axis, title in zip(axes[0], titles, strict=True):
            axis.set_title(title, pad=7)
        for row, label in enumerate((r"$(a)$", r"$(b)$", r"$(c)$")):
            axes[row, 0].text(
                -0.31,
                1.11,
                label,
                transform=axes[row, 0].transAxes,
                fontsize=10,
                clip_on=False,
            )

        if speed_image is None or invariant_image is None:
            raise RuntimeError("The background-flow images were not created.")
        speed_bar_axis = figure.add_axes([0.085, 0.075, 0.168, 0.018])
        speed_bar = figure.colorbar(speed_image, cax=speed_bar_axis, orientation="horizontal")
        speed_bar.set_ticks([0.0, 0.5, 1.0])
        speed_bar.ax.tick_params(length=2.5, pad=2)
        invariant_bar_axis = figure.add_axes([0.347, 0.075, 0.605, 0.018])
        invariant_bar = figure.colorbar(
            invariant_image, cax=invariant_bar_axis, orientation="horizontal"
        )
        invariant_bar.set_ticks([-0.15, -0.075, 0.0, 0.075, 0.15])
        invariant_bar.ax.set_xticklabels(["-0.15", "-0.075", "0.0", "0.075", "0.15"])
        invariant_bar.ax.tick_params(length=2.5, pad=2)

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_figure(args.input.resolve(), args.output.resolve())
    print(args.output)


if __name__ == "__main__":
    main()
