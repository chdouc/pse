"""Compute the three analytic background flows shown in Figure 3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


FLOW_NAMES = np.asarray(
    ["sinusoidal parallel shear", "Gaussian vortex", "sinusoidal dipole"]
)


def compute_fields(grid_points: int, rossby_number: float) -> dict[str, np.ndarray]:
    """Evaluate velocity, vorticity and strain on the periodic plotting grid."""
    if grid_points < 32:
        raise ValueError("grid_points must be at least 32.")
    if rossby_number <= 0.0:
        raise ValueError("rossby_number must be positive.")
    coordinate = np.linspace(-np.pi, np.pi, grid_points, endpoint=False)
    x, y = np.meshgrid(coordinate, coordinate)
    root_two = np.sqrt(2.0)

    velocity_u = []
    velocity_v = []
    xi1 = []
    xi2 = []
    xi3 = []
    profiles = []

    u = np.zeros_like(x)
    v = -np.sin(x)
    velocity_u.append(u)
    velocity_v.append(v)
    xi1.append(-rossby_number * np.cos(x))
    xi2.append(np.zeros_like(x))
    xi3.append(-rossby_number * np.cos(x))
    profiles.append(-np.sin(coordinate))

    gaussian = np.sqrt(np.e) * np.exp(-0.5 * (x**2 + y**2))
    u = gaussian * y
    v = -gaussian * x
    velocity_u.append(u)
    velocity_v.append(v)
    xi1.append(rossby_number * gaussian * (x**2 + y**2 - 2.0))
    xi2.append(-2.0 * rossby_number * gaussian * x * y)
    xi3.append(rossby_number * gaussian * (x**2 - y**2))
    profiles.append(-np.sqrt(np.e) * coordinate * np.exp(-0.5 * coordinate**2))

    u = -np.cos(y) / root_two
    v = -np.cos(x) / root_two
    velocity_u.append(u)
    velocity_v.append(v)
    xi1.append(rossby_number * (np.sin(x) - np.sin(y)) / root_two)
    xi2.append(np.zeros_like(x))
    xi3.append(rossby_number * (np.sin(x) + np.sin(y)) / root_two)
    profiles.append(-np.cos(coordinate) / root_two)

    u_array = np.stack(velocity_u)
    v_array = np.stack(velocity_v)
    return {
        "flow_names": FLOW_NAMES,
        "coordinate_over_length": coordinate,
        "velocity_u_over_reference": u_array,
        "velocity_v_over_reference": v_array,
        "speed_over_reference": np.sqrt(u_array**2 + v_array**2),
        "xi1_over_f": np.stack(xi1),
        "xi2_over_f": np.stack(xi2),
        "xi3_over_f": np.stack(xi3),
        "sampled_v_over_reference_at_y0": np.stack(profiles),
        "rossby_number": np.asarray(rossby_number),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-points", type=int, default=256)
    parser.add_argument("--rossby-number", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = compute_fields(args.grid_points, args.rossby_number)
    temporary = output.with_name(output.stem + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    metadata = {
        "schema_version": 1,
        "background_flow": "analytic examples used in Figure 3",
        "flow_order": FLOW_NAMES.tolist(),
        "coordinate_convention": "x/L and y/L in [-pi, pi)",
        "rossby_number": args.rossby_number,
        "fields": ["|U|/U_ref", "xi1/f", "xi2/f", "xi3/f"],
        "sampling_line": "y/L=0",
        "external_data": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
