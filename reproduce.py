"""Unified entry point for the manuscript-resolution reproduction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any

import h5py
import psutil


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "code" / "sinusoidal_dipole"
sys.path.insert(0, str(SOURCE))

from compute_error_statistics import compute_statistics  # noqa: E402
from compute_movie_fields import compute_archive  # noqa: E402
from compute_wave_velocity_fields import compute_fields  # noqa: E402
from check_convergence import run_convergence  # noqa: E402
from solver import create_simulation_file, load_config  # noqa: E402
from specification import (  # noqa: E402
    simulation_signature,
    validate_config,
)
from validate_reproduction import validate_all, validate_smoke  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "reproduction.json"
DEFAULT_CONVERGENCE_CONFIG = ROOT / "config" / "convergence.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "reproduction"


class PeakMemoryMonitor:
    """Sample combined resident memory for the parent and solver workers."""

    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while True:
            processes = [process, *process.children(recursive=True)]
            total = 0
            for child in processes:
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak_bytes = max(self.peak_bytes, total)
            if self._stop.wait(0.25):
                break

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return self.peak_bytes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path: Path, data: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def run_command(arguments: list[str]) -> None:
    print("running: " + " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def dependency_versions() -> dict[str, str]:
    names = (
        "numpy",
        "scipy",
        "h5py",
        "pandas",
        "matplotlib",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
        "psutil",
        "pytest",
    )
    return {name: importlib.metadata.version(name) for name in names}


def repository_path(path: Path) -> str:
    """Return a portable path label without recording a local absolute path."""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return f"external/{path.name}"


def git_output(*arguments: str) -> str | None:
    """Return Git output when metadata and the executable are available."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.rstrip() if result.returncode == 0 else None


def source_inventory() -> list[dict[str, Any]]:
    """Hash the repository files that define the reproduction."""
    pathspecs = (
        "code/common",
        "code/gaussian_vortex",
        "code/parallel_shear",
        "code/polarisation_geometry",
        "code/sinusoidal_dipole",
        "code/background_flows",
        "config",
        "tests",
        "workflows",
        "reproduce.py",
        "run_workflow.py",
        "validate_outputs.py",
        "requirements.txt",
        "README.md",
        "AUDIT.md",
        "CITATION.cff",
        ".gitignore",
    )
    git_files = git_output(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        *pathspecs,
    )
    if git_files is not None:
        relative_paths = sorted(line for line in git_files.splitlines() if line)
    else:
        relative_paths = []
        excluded_parts = {"__pycache__", "data", "figures"}
        for pathspec in pathspecs:
            candidate = ROOT / pathspec
            candidates = [candidate] if candidate.is_file() else candidate.rglob("*")
            for path in candidates:
                if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                    continue
                relative = path.relative_to(ROOT)
                if excluded_parts.intersection(relative.parts):
                    continue
                relative_paths.append(str(relative))
    records = []
    for relative in sorted(set(relative_paths)):
        path = ROOT / relative
        if path.is_file():
            records.append(
                {
                    "path": relative.replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def tree_snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Record output sizes and modification times before a run."""
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }


def changed_files(
    directory: Path,
    before: dict[str, tuple[int, int]],
) -> list[Path]:
    """Return only files created or replaced by the current run."""
    paths = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = str(path.relative_to(directory))
        state = (path.stat().st_size, path.stat().st_mtime_ns)
        if before.get(relative) != state:
            paths.append(path)
    return sorted(paths)


def validate_reusable_simulation(path: Path, config: dict[str, Any]) -> None:
    """Reject a cached archive generated from different numerical inputs."""
    expected = simulation_signature(config)
    with h5py.File(path, "r") as handle:
        stored = handle.attrs.get("simulation_signature_sha256")
        if stored is None:
            archived_config = json.loads(handle.attrs["config_json"])
            stored = simulation_signature(archived_config)
    if stored != expected:
        raise ValueError(
            "The cached simulation was generated from a different configuration; "
            "rerun without --reuse-simulation."
        )


def write_manifest(
    output_directory: Path,
    *,
    command: str,
    started: float,
    config_path: Path,
    validation: dict[str, Any],
    peak_combined_memory_bytes: int,
    outputs: list[Path],
    inputs: list[Path],
    additional_configurations: list[dict[str, str]],
) -> Path:
    tracked_changes = git_output("status", "--porcelain", "--untracked-files=no")
    git_commit = git_output("rev-parse", "HEAD")
    manifest = {
        "schema_version": 2,
        "command": command,
        "runtime_seconds": time.perf_counter() - started,
        "peak_combined_resident_memory_bytes": peak_combined_memory_bytes,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": Path(sys.executable).name,
            "logical_cpu_count": os.cpu_count(),
            "dependencies": dependency_versions(),
        },
        "configuration": {
            "source": repository_path(config_path),
            "snapshot": "config_used.json",
            "sha256": sha256_file(config_path),
            "additional": additional_configurations,
        },
        "source_provenance": {
            "git_repository_available": git_commit is not None,
            "git_commit": git_commit,
            "tracked_worktree_clean": (
                None if tracked_changes is None else not bool(tracked_changes)
            ),
            "tracked_changes": (
                [] if tracked_changes is None else tracked_changes.splitlines()
            ),
            "files": source_inventory(),
        },
        "validation": validation,
        "inputs": [
            {
                "path": repository_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inputs
            if path.is_file()
        ],
        "outputs": [
            {
                "path": str(path.relative_to(output_directory)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    path = output_directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def saved_times(config: dict[str, Any]) -> dict[int, list[int]]:
    figure_modes = set(config["vertical_modes"]["figures_9_10"])
    movie_modes = set(config["vertical_modes"]["movie_2"])
    figure_times = list(config["saved_times_in_inertial_periods"]["figures_9_10"])
    movie = config["saved_times_in_inertial_periods"]
    movie_times = list(
        range(
            int(movie["movie_2_start"]),
            int(movie["movie_2_stop"]) + 1,
            int(movie["movie_2_interval"]),
        )
    )
    return {
        mode: (
            movie_times
            if mode in movie_modes
            else figure_times if mode in figure_modes else []
        )
        for mode in config["vertical_modes"]["error_statistics"]
    }


def reproduce_all(
    config: dict[str, Any],
    output_directory: Path,
    *,
    workers: int,
    reuse_simulation: bool,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    data_directory = output_directory / "data"
    figure_directory = output_directory / "figures"
    movie_directory = output_directory / "movies"
    for directory in (data_directory, figure_directory, movie_directory):
        directory.mkdir(parents=True, exist_ok=True)

    simulation_path = data_directory / "simulation.h5"
    if reuse_simulation:
        if not simulation_path.is_file():
            raise FileNotFoundError(
                "--reuse-simulation requires an existing data/simulation.h5."
            )
        validate_reusable_simulation(simulation_path, config)
    else:
        create_simulation_file(
            simulation_path,
            config,
            config["vertical_modes"]["error_statistics"],
            saved_times(config),
            workers=workers,
        )

    error_path = data_directory / "sinusoidal_dipole_error_statistics.csv"
    compute_statistics(simulation_path).to_csv(error_path, index=False)
    wave_path = data_directory / "sinusoidal_dipole_wave_velocity_fields.npz"
    save_npz(wave_path, compute_fields(simulation_path))
    movie_fields_path = data_directory / "sinusoidal_dipole_movie_fields.npz"
    save_npz(movie_fields_path, compute_archive(simulation_path))

    validation = validate_all(
        simulation_path,
        error_path,
        wave_path,
        movie_fields_path,
        config,
    )

    run_command(
        [
            sys.executable,
            str(SOURCE / "plot_error_statistics.py"),
            "--input",
            str(error_path),
            "--output",
            str(figure_directory / "figure8_error_statistics"),
        ]
    )
    run_command(
        [
            sys.executable,
            str(SOURCE / "plot_wave_velocity_fields.py"),
            "--input",
            str(wave_path),
            "--output-directory",
            str(figure_directory),
            "--no-tex",
        ]
    )
    run_command(
        [
            sys.executable,
            str(SOURCE / "render_wave_velocity_movie.py"),
            "--input",
            str(movie_fields_path),
            "--output-directory",
            str(movie_directory),
        ]
    )
    run_command(
        [
            sys.executable,
            str(SOURCE / "validate_wave_velocity_movie.py"),
            "--input",
            str(movie_fields_path),
            "--output-directory",
            str(movie_directory),
        ]
    )

    movie_path = movie_directory / "movie2.mp4"
    if not movie_path.is_file() or movie_path.stat().st_size == 0:
        raise AssertionError("Movie 2 was not generated.")
    validation["movie2_bytes"] = movie_path.stat().st_size
    quality_path = movie_directory / "movie2_quality_report.json"
    movie_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if movie_quality.get("status") != "passed":
        raise AssertionError("Movie 2 quality validation did not pass.")
    validation["movie2_quality"] = {
        "status": movie_quality["status"],
        "media_checks": movie_quality["media_checks"],
    }
    report_path = output_directory / "validation.json"
    report_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    return validation


def smoke_test(config: dict[str, Any], output_directory: Path) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    smoke["numerical_parameters"].update(
        {
            "horizontal_grid": 16,
            "time_steps_per_inertial_period": 8,
            "total_inertial_periods": 1,
        }
    )
    modes = [1, 4, 8, 16, 32]
    smoke_path = output_directory / "data" / "smoke_simulation.h5"
    create_simulation_file(
        smoke_path,
        smoke,
        modes,
        {mode: [0, 1] for mode in modes},
        workers=1,
    )
    validation = validate_smoke(smoke_path, smoke)
    (output_directory / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--all", action="store_true")
    action.add_argument("--smoke-test", action="store_true")
    action.add_argument("--convergence-test", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--convergence-config",
        type=Path,
        default=DEFAULT_CONVERGENCE_CONFIG,
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    parser.add_argument(
        "--reuse-simulation",
        action="store_true",
        help=(
            "Reuse an existing simulation.h5 and regenerate downstream products; "
            "the default --all run always recomputes the simulation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if not args.all and args.reuse_simulation:
        raise ValueError("--reuse-simulation applies only to --all.")
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config, manuscript_resolution=not args.smoke_test)
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory
        else DEFAULT_OUTPUT
        / (
            "smoke-test"
            if args.smoke_test
            else "convergence" if args.convergence_test else "full"
        )
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    before = tree_snapshot(output_directory)
    config_snapshot = output_directory / "config_used.json"
    config_snapshot.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    memory_monitor = PeakMemoryMonitor()
    memory_monitor.start()
    command = " ".join([Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]])
    validation: dict[str, Any] = {"status": "running"}
    additional_configurations: list[dict[str, str]] = []
    failure: BaseException | None = None
    try:
        if args.smoke_test:
            validation = smoke_test(config, output_directory)
        elif args.convergence_test:
            convergence_config_path = args.convergence_config.resolve()
            convergence_config = json.loads(
                convergence_config_path.read_text(encoding="utf-8")
            )
            convergence_snapshot = output_directory / "convergence_config_used.json"
            convergence_snapshot.write_text(
                json.dumps(convergence_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            additional_configurations.append(
                {
                    "source": repository_path(convergence_config_path),
                    "snapshot": convergence_snapshot.name,
                    "sha256": sha256_file(convergence_config_path),
                }
            )
            validation = run_convergence(
                config,
                convergence_config,
                output_directory / "data",
            )
        else:
            validation = reproduce_all(
                config,
                output_directory,
                workers=args.workers,
                reuse_simulation=args.reuse_simulation,
            )
    except BaseException as error:
        failure = error
        validation = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }
    finally:
        peak_memory = memory_monitor.stop()
    outputs = changed_files(output_directory, before)
    reused_inputs = (
        [output_directory / "data" / "simulation.h5"]
        if args.reuse_simulation
        else []
    )
    manifest = write_manifest(
        output_directory,
        command=command,
        started=started,
        config_path=config_path,
        validation=validation,
        peak_combined_memory_bytes=peak_memory,
        outputs=outputs,
        inputs=reused_inputs,
        additional_configurations=additional_configurations,
    )
    if failure is not None:
        print(f"validation: failed ({type(failure).__name__}: {failure})")
        print(f"manifest: {manifest}")
        raise failure.with_traceback(failure.__traceback__)
    print(f"validation: {validation['status']}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
