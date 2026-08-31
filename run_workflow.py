"""Run the manuscript workflows from version-controlled configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKFLOW_DIRECTORY = ROOT / "workflows"
PLACEHOLDER_PATTERN = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
OPTIONAL_PLACEHOLDER_PATTERN = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\?\}$")


def workflow_names() -> list[str]:
    """Return the available workflow names."""
    return sorted(path.stem for path in WORKFLOW_DIRECTORY.glob("*.json"))


def remap_artifact_paths(value: Any, artifact_root: Path | None) -> Any:
    """Redirect configured ``artifacts/...`` paths below one run directory."""
    if artifact_root is None:
        return value
    if isinstance(value, dict):
        return {
            key: remap_artifact_paths(item, artifact_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [remap_artifact_paths(item, artifact_root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.parts and path.parts[0] == "artifacts":
            return str(artifact_root.joinpath(*path.parts[1:]))
    return value


def load_workflow(
    name: str,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Load and minimally validate one workflow configuration."""
    if name not in workflow_names():
        available = ", ".join(workflow_names())
        raise ValueError(f"Unknown workflow {name!r}. Available workflows: {available}")

    path = WORKFLOW_DIRECTORY / f"{name}.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    if workflow.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema version in {path}.")
    if workflow.get("name") != name:
        raise ValueError(f"Workflow name does not match the filename: {path}.")
    if not workflow.get("steps"):
        raise ValueError(f"No steps are defined in {path}.")
    return remap_artifact_paths(workflow, artifact_root)


def substitute(value: Any, replacements: dict[str, Any]) -> Any:
    """Replace a complete ``${name}`` placeholder in a configuration value."""
    if not isinstance(value, str):
        return value
    optional_match = OPTIONAL_PLACEHOLDER_PATTERN.fullmatch(value)
    if optional_match is not None:
        return replacements.get(optional_match.group(1))

    match = PLACEHOLDER_PATTERN.fullmatch(value)
    if match is None:
        return value

    name = match.group(1)
    replacement = replacements.get(name)
    if replacement is None:
        raise ValueError(
            f"The workflow requires --{name.replace('_', '-')}. "
            "Provide it when running the calculation stage."
        )
    return replacement


def argument_tokens(
    arguments: dict[str, Any],
    replacements: dict[str, Any],
) -> list[str]:
    """Convert a JSON argument mapping to command-line tokens."""
    tokens: list[str] = []
    for name, raw_value in arguments.items():
        value = substitute(raw_value, replacements)
        flag = f"--{name.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                tokens.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        tokens.extend((flag, str(value)))
    return tokens


def selected_steps(
    workflow: dict[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    """Select one workflow-step kind or every configured step."""
    if stage == "all":
        return workflow["steps"]
    return [step for step in workflow["steps"] if step["kind"] == stage]


def build_command(
    step: dict[str, Any],
    *,
    replacements: dict[str, Any],
    no_tex: bool,
) -> list[str]:
    """Build one Python command from a configured workflow step."""
    script = ROOT / step["script"]
    if not script.is_file():
        raise FileNotFoundError(f"Workflow script does not exist: {script}")

    command = [
        sys.executable,
        str(script),
        *argument_tokens(step.get("arguments", {}), replacements),
    ]
    if no_tex and step.get("supports_no_tex", False):
        command.append("--no-tex")
    return command


def build_validation_command(
    workflow: str,
    *,
    output_directory: Path | None,
    artifact_root: Path | None,
    data_only: bool,
) -> list[str]:
    """Build the validator command for one workflow invocation."""
    command = [sys.executable, str(ROOT / "validate_outputs.py"), workflow]
    if output_directory is not None:
        command.extend(("--output-directory", str(output_directory)))
    if artifact_root is not None:
        command.extend(("--artifact-root", str(artifact_root)))
    if data_only:
        command.append("--data-only")
    return command


def resolve_workflow_output_directory(
    workflow: dict[str, Any],
    requested: Path | None,
) -> Path | None:
    """Resolve an explicit or configured workflow output directory."""
    if requested is not None:
        return requested.resolve()
    configured = workflow.get("default_output_directory")
    if configured is None:
        return None
    directory = Path(configured)
    return directory if directory.is_absolute() else (ROOT / directory).resolve()


def run_workflow(args: argparse.Namespace) -> None:
    """Execute the selected workflow steps."""
    artifact_root = (
        args.artifact_root.resolve() if args.artifact_root is not None else None
    )
    workflow = load_workflow(args.workflow, artifact_root=artifact_root)
    output_directory = resolve_workflow_output_directory(
        workflow,
        args.output_directory,
    )
    if args.stage == "validate":
        validation_command = build_validation_command(
            args.workflow,
            output_directory=output_directory,
            artifact_root=artifact_root,
            data_only=False,
        )
        if args.dry_run:
            print(subprocess.list2cmdline(validation_command))
            return
        subprocess.run(validation_command, cwd=ROOT, check=True)
        return
    steps = selected_steps(workflow, args.stage)
    if not steps:
        raise ValueError(f"Workflow {args.workflow!r} has no {args.stage!r} stage.")

    replacements = {
        "output_directory": (
            str(output_directory) if output_directory is not None else None
        ),
        "ffmpeg": str(args.ffmpeg) if args.ffmpeg is not None else None,
        "ffprobe": str(args.ffprobe) if args.ffprobe is not None else None,
        "fps": args.fps,
        "resolution": args.resolution,
        "frame_stride": args.frame_stride,
        "hold_frames": args.hold_frames,
        "opening_seconds": args.opening_seconds,
        "title_seconds": args.title_seconds,
        "chapter_end_seconds": args.chapter_end_seconds,
        "crf": args.crf,
    }
    for step in steps:
        command = build_command(
            step,
            replacements=replacements,
            no_tex=args.no_tex,
        )
        print(f"[{step['name']}] {subprocess.list2cmdline(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    if args.validate and not args.dry_run:
        validation_command = build_validation_command(
            args.workflow,
            output_directory=output_directory,
            artifact_root=artifact_root,
            data_only=args.stage == "compute",
        )
        subprocess.run(validation_command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a configured supplementary-code workflow."
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        help="Workflow name; use --list to show the available names.",
    )
    parser.add_argument(
        "--stage",
        choices=("compute", "plot", "render", "validate", "all"),
        default="all",
        help="Run one configured step kind, or the complete workflow.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Output directory required by workflows that create external "
            "submission artifacts."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=("Redirect every configured artifacts/... path below this directory."),
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="Optional FFmpeg executable path for movie workflows.",
    )
    parser.add_argument(
        "--ffprobe",
        type=Path,
        help="Optional ffprobe executable path for movie workflows.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Movie frame-rate override.",
    )
    parser.add_argument(
        "--resolution",
        help="Movie resolution override in WIDTHxHEIGHT form.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        help="Stride through computed physical movie states.",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        help="Number of CFR video frames used to hold each physical state.",
    )
    parser.add_argument(
        "--chapter-end-seconds",
        type=float,
        help="Additional still hold after each movie chapter.",
    )
    parser.add_argument(
        "--opening-seconds",
        type=float,
        help="Duration of each opening card in seconds.",
    )
    parser.add_argument(
        "--title-seconds",
        type=float,
        help="Duration of each chapter title card in seconds.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        help="Initial H.264 constant-rate-factor value.",
    )
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable LaTeX in plotting steps that support this option.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated data and figures after the workflow.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the configured workflows and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for name in workflow_names():
            workflow = load_workflow(name)
            print(f"{name}: {workflow['description']}")
        raise SystemExit(0)
    if args.workflow is None:
        parser.error("a workflow name is required unless --list is used")
    return args


def main() -> None:
    """Run the requested workflow."""
    run_workflow(parse_args())


if __name__ == "__main__":
    main()
