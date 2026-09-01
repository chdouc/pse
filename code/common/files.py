"""Portable file helpers shared by reproduction and rendering commands."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable_version(path: Path) -> str:
    """Return the first non-empty line from an executable's version output."""
    result = subprocess.run(
        [str(path), "-version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout or result.stderr
    first_line = next(
        (line.strip() for line in output.splitlines() if line.strip()),
        "",
    )
    if not first_line:
        raise ValueError(f"Could not read version information from {path.name}.")
    return first_line


def executable_provenance(path: Path) -> dict[str, str]:
    """Return portable version and checksum metadata for an executable."""
    return {
        "filename": path.name,
        "version": executable_version(path),
        "sha256": sha256_file(path),
    }


def validate_executable_provenance(record: Any, *, label: str) -> None:
    """Require a portable executable provenance record."""
    if not isinstance(record, dict) or set(record) != {
        "filename",
        "version",
        "sha256",
    }:
        raise ValueError(f"Invalid {label} provenance record.")
    filename = record.get("filename")
    version = record.get("version")
    checksum = record.get("sha256")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or not isinstance(version, str)
        or not version
        or not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError(f"Invalid {label} provenance record.")
