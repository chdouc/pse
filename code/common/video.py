"""Small container checks shared by supplementary-movie workflows."""

from __future__ import annotations

from pathlib import Path


def mp4_atom_offsets(path: Path) -> dict[str, int]:
    """Return byte offsets used to check MP4 fast-start ordering."""
    data = path.read_bytes()
    return {"moov": data.find(b"moov"), "mdat": data.find(b"mdat")}


def mp4_has_faststart(path: Path) -> bool:
    """Return whether the MP4 metadata atom precedes its media-data atom."""
    offsets = mp4_atom_offsets(path)
    return (
        offsets["moov"] >= 0
        and offsets["mdat"] >= 0
        and offsets["moov"] < offsets["mdat"]
    )
