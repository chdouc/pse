"""Validate supplementary movie 2 numerically, visually, and technically."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
from PIL import Image

from render_wave_velocity_movie import (
    JFM_PREPARING_URL,
    JFM_SUBMITTING_URL,
    MAX_FILE_BYTES,
    colorbar_number,
    probe_video,
    resolve_executable,
)


MODEL_NAMES = ["YBJ", "TSB", "YBJ+", "PSE", "HBEs"]
NRE_MODEL_NAMES = MODEL_NAMES[:4]
EXPECTED_VERTICAL_MODES = np.asarray([4, 16, 32])
EXPECTED_DOMAIN_DEPTH_METRES = 2000.0
EXPECTED_VERTICAL_WAVELENGTHS_METRES = np.asarray([1000.0, 250.0, 125.0])
EXPECTED_GRID_POINTS = 128
EXPECTED_STEPS_PER_INERTIAL_PERIOD = 64
EXPECTED_T50_MAXIMA = np.asarray(
    [
        [
            31.78305252674567,
            38.59693198569784,
            37.10110161741679,
            37.56090526692235,
            27.573750991669126,
        ],
        [
            1.6165713245440108,
            1.6141988368892222,
            1.6145232022840763,
            1.6164613339815284,
            1.6794963091594228,
        ],
        [
            1.1192561550932818,
            1.119246472578306,
            1.1192467248495304,
            1.1191323896066654,
            1.1288687266184072,
        ],
    ]
)
EXPECTED_ABSOLUTE_LIMITS = np.asarray(
    [[0.01, 37.5], [0.39, 1.61], [0.88, 1.12]]
)
EXPECTED_STYLE_ALIGNMENT = {
    "manuscript_figures": [8, 9, 10],
    "domain_depth_metres": 2000,
    "vertical_wavelength_relation": "h=2H/n",
    "preferred_text_font": "Times New Roman",
    "font_stack": [
        "Times New Roman",
        "Times",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext_fontset": "stix",
    "model_colors": {
        "YBJ": "#002BFF",
        "TSB": "#7A3E9D",
        "YBJ+": "#00C46A",
        "PSE": "#B2182B",
    },
    "nre_legend_visual_rows": [
        ["YBJ", "TSB"],
        ["YBJ+", "PSE"],
    ],
    "vertical_wavelength_metres": {
        "4": 1000,
        "16": 250,
        "32": 125,
    },
    "nre_y_limits_percent": {
        "4": [0.0, 40.0],
        "16": [0.0, 10.0],
        "32": [0.0, 10.0],
    },
    "subplot_title_fontsize": 20.0,
    "nre_title_fontsize": 17.0,
    "upper_row_quantity": "|phi|^2/|phi_amp|^2",
    "difference_quantity": "(|phi_model|^2-|phi_HBEs|^2)/|phi_amp|^2",
    "nre_quantity": "instantaneous complex-velocity NRE relative to HBEs",
    "title_card_alignment": {
        "reference": "supplementary movie 1",
        "reference_resolution": [1920, 1080],
        "reference_dpi": 120,
        "title_position": [0.5, 0.62],
        "subtitle_position": [0.5, 0.43],
        "reference_title_fontsize": 31.0,
        "reference_subtitle_fontsize": 20.0,
        "opening_page_count": 2,
        "opening_title": "Supplementary movie 2",
        "opening_subtitle": (
            "Wave-field evolution in a sinusoidal-dipole background flow"
        ),
        "chapter_title_pattern": "Chapter {chapter}",
        "chapter_subtitle_pattern": (
            "Vertical mode n={mode} (H=2 km; vertical wavelength "
            "h={wavelength} m); "
            "0-50 inertial periods"
        ),
    },
}
REQUIRED_OUTPUTS = (
    "movie2.mp4",
    "movie2_preview.png",
    "movie2_caption.txt",
    "movie2_accessibility_description.txt",
    "movie2_submission_notes.txt",
    "movie2_manuscript_reference_suggestion.txt",
    "movie2_qc_contact_sheet.png",
    "movie2_render_manifest.json",
    "README.md",
)


def ffprobe_json(ffprobe: Path, video_path: Path) -> dict[str, Any]:
    """Probe all streams while counting decoded video frames."""
    if "ffprobe" not in ffprobe.name.lower():
        return probe_video(ffprobe, video_path)
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(video_path),
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
        probe["format"]["filename"] = video_path.name
    return probe


def unique_stream(probe: dict[str, Any], stream_type: str) -> dict[str, Any]:
    """Return the unique stream of one type."""
    matches = [
        stream
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == stream_type
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {stream_type} stream; found {len(matches)}."
        )
    return matches[0]


def check_archive(path: Path) -> dict[str, Any]:
    """Validate the intermediate NPZ and its numerical reference checks."""
    required = {
        "times_in_inertial_periods",
        "source_time_indices",
        "vertical_modes",
        "vertical_wavelengths_m",
        "model_names",
        "nre_model_names",
        "normalized_squared_velocity",
        "nre_times_in_inertial_periods",
        "nre_complex_relative_l2",
        "x_over_L",
        "y_over_L",
        "normalization_amplitude",
        "absolute_color_limits",
        "difference_color_limits",
        "recomputed_nre_max_abs_difference",
        "processed_source_files",
        "metadata_json",
    }
    with np.load(path) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Intermediate archive is missing: {missing}")
        times = archive["times_in_inertial_periods"]
        source_indices = archive["source_time_indices"]
        modes = archive["vertical_modes"]
        wavelengths = archive["vertical_wavelengths_m"]
        model_names = list(archive["model_names"])
        nre_names = list(archive["nre_model_names"])
        fields = archive["normalized_squared_velocity"]
        nre_times = archive["nre_times_in_inertial_periods"]
        nre = archive["nre_complex_relative_l2"]
        x = archive["x_over_L"]
        y = archive["y_over_L"]
        amplitudes = archive["normalization_amplitude"]
        absolute_limits = archive["absolute_color_limits"]
        difference_limits = archive["difference_color_limits"]
        nre_differences = archive["recomputed_nre_max_abs_difference"]
        metadata = json.loads(str(archive["metadata_json"].item()))

        if not np.array_equal(times, np.arange(51.0)):
            raise ValueError("Movie fields must contain every integer time from 0--50 IP.")
        if not np.array_equal(modes, EXPECTED_VERTICAL_MODES):
            raise ValueError("Movie vertical-mode order changed.")
        if not np.array_equal(wavelengths, EXPECTED_VERTICAL_WAVELENGTHS_METRES):
            raise ValueError("Movie vertical wavelengths changed.")
        if model_names != MODEL_NAMES or nre_names != NRE_MODEL_NAMES:
            raise ValueError("Movie model order changed.")
        if fields.shape != (
            3,
            51,
            5,
            EXPECTED_GRID_POINTS,
            EXPECTED_GRID_POINTS,
        ):
            raise ValueError(f"Unexpected movie field shape: {fields.shape}.")
        if source_indices.shape != (3, 51):
            raise ValueError("Unexpected source-time index shape.")
        expected_indices = np.arange(51)
        if not all(
            np.array_equal(indices, expected_indices)
            for indices in source_indices
        ):
            raise ValueError("Source-time selection no longer uses exact 1-IP states.")
        if not np.all(np.isfinite(fields)) or np.any(fields < 0.0):
            raise ValueError("Movie fields contain invalid values.")
        if nre.shape != (3, 4, 3201) or nre_times.shape != (3, 3201):
            raise ValueError("NRE curve dimensions changed.")
        if not np.all(np.diff(nre_times, axis=1) > 0.0):
            raise ValueError("NRE time coordinates are not strictly increasing.")
        if not np.all(np.isfinite(nre)) or np.any(nre < 0.0):
            raise ValueError("NRE curves contain invalid values.")
        expected_coordinate = np.linspace(
            -np.pi,
            np.pi,
            EXPECTED_GRID_POINTS,
            endpoint=False,
        )
        if not np.array_equal(x, expected_coordinate) or not np.array_equal(
            y,
            expected_coordinate,
        ):
            raise ValueError("Movie coordinates or orientation changed.")
        if not np.allclose(amplitudes, 1.0, rtol=0.0, atol=1.0e-15):
            raise ValueError("Initial modal-amplitude normalization changed.")
        if not np.array_equal(
            absolute_limits,
            EXPECTED_ABSOLUTE_LIMITS,
        ):
            raise ValueError("Fixed absolute color limits changed.")
        if not np.allclose(
            difference_limits[:, 0],
            -difference_limits[:, 1],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Difference color limits are not exactly symmetric.")
        if np.any(difference_limits[:, 1] <= 0.0):
            raise ValueError("Difference color limits are not positive.")
        if np.max(nre_differences) > 2.0e-6:
            raise ValueError("Recomputed NRE values disagree with stored error data.")
        time_50 = int(np.flatnonzero(np.isclose(times, 50.0))[0])
        maxima = fields[:, time_50].max(axis=(-2, -1))
        if not np.allclose(
            maxima,
            EXPECTED_T50_MAXIMA,
            rtol=0.0,
            atol=0.6,
        ):
            raise ValueError(
                f"Mode-wise t=50 IP maxima changed: {maxima.tolist()}."
            )
        for target in (10.0, 50.0):
            if not np.any(np.isclose(times, target, rtol=0.0, atol=1.0e-12)):
                raise ValueError(f"Required {target:g}-IP validation frame is absent.")

    if metadata["source_kind"] != "fields generated by the in-repository equation solver":
        raise ValueError("Movie source-kind metadata changed.")
    if not np.isclose(
        float(metadata.get("domain_depth_m", np.nan)),
        EXPECTED_DOMAIN_DEPTH_METRES,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Movie domain depth is not H=2 km.")
    if metadata.get("vertical_wavelength_relation") != "h=2H/n":
        raise ValueError("Movie vertical-wavelength relation is not h=2H/n.")
    if metadata["spatial_discretisation"] != {
        "grid_points": [EXPECTED_GRID_POINTS, EXPECTED_GRID_POINTS],
    }:
        raise ValueError("Movie spatial-discretisation metadata changed.")
    if metadata["time_discretisation"] != {
        "parameter": "fc",
        "steps_per_inertial_period": EXPECTED_STEPS_PER_INERTIAL_PERIOD,
    }:
        raise ValueError("Movie time-discretisation metadata changed.")
    if metadata["pse_field_source"] != (
        "A_up exp(-ift)+A_down exp(ift) from the PSE solver"
    ):
        raise ValueError("PSE field-source metadata changed.")
    if metadata["orientation"] != {
        "array_axes": "time, model, y, x",
        "extent": [-np.pi, np.pi, -np.pi, np.pi],
        "origin": "lower",
    }:
        raise ValueError("Movie orientation metadata changed.")
    if metadata["sampling"]["physical_field_interpolation"]:
        raise ValueError("Movie metadata incorrectly reports field interpolation.")

    return {
        "field_shape": list(fields.shape),
        "domain_depth_m": EXPECTED_DOMAIN_DEPTH_METRES,
        "vertical_wavelength_relation": "h=2H/n",
        "vertical_wavelengths_m": wavelengths.tolist(),
        "times_ip": [float(times[0]), float(times[-1])],
        "source_time_step": 1,
        "spatial_discretisation": metadata["spatial_discretisation"],
        "time_discretisation": metadata["time_discretisation"],
        "model_order": model_names,
        "max_recomputed_stored_nre_difference": float(np.max(nre_differences)),
        "t50_maxima_by_mode": {
            str(int(mode)): values.tolist()
            for mode, values in zip(modes, maxima, strict=True)
        },
        "absolute_color_limits": absolute_limits.tolist(),
        "difference_color_limits": difference_limits.tolist(),
        "clipping": metadata["color_limits"]["clipping"],
        "pse_field_source": metadata["pse_field_source"],
    }


def image_dimensions(path: Path) -> tuple[int, int]:
    """Return image dimensions after forcing a complete decode."""
    with Image.open(path) as image:
        image.load()
        return image.size


def measure_encoded_static_range(
    ffmpeg: Path,
    video_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Measure decoded luma changes across one intended still-frame range."""
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=between(n\\,{start_frame}\\,{end_frame})",
        "-vsync",
        "0",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Failed to open the decoded-frame validation stream.")

    frame_size = width * height
    previous: np.ndarray | None = None
    adjacent_mae: list[float] = []
    maximum_delta = 0
    frame_count = 0
    while True:
        raw = process.stdout.read(frame_size)
        if not raw:
            break
        while len(raw) < frame_size:
            more = process.stdout.read(frame_size - len(raw))
            if not more:
                raise RuntimeError("Decoded still-frame stream ended mid-frame.")
            raw += more
        current = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        if previous is not None:
            difference = np.abs(current - previous)
            maximum_delta = max(maximum_delta, int(np.max(difference)))
            adjacent_mae.append(float(np.mean(difference)))
        previous = current
        frame_count += 1

    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"FFmpeg still-range decode failed with code {return_code}:\n{stderr}"
        )
    expected_count = end_frame - start_frame + 1
    if frame_count != expected_count or not adjacent_mae:
        raise ValueError(
            f"Decoded {frame_count} still frames; expected {expected_count}."
        )
    mean_adjacent_mae = float(np.mean(adjacent_mae))
    if mean_adjacent_mae > 0.02:
        raise ValueError(
            "Encoded chapter-end hold is not visually static: "
            f"mean adjacent luma MAE {mean_adjacent_mae:.6f}."
        )
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "decoded_frame_count": frame_count,
        "maximum_adjacent_luma_delta": maximum_delta,
        "mean_adjacent_luma_mae": mean_adjacent_mae,
    }


def check_text_outputs(output_directory: Path) -> dict[str, Any]:
    """Check labels, TeX delimiters, links, and absence of local paths."""
    caption = (output_directory / "movie2_caption.txt").read_text(encoding="utf-8")
    accessibility = (
        output_directory / "movie2_accessibility_description.txt"
    ).read_text(encoding="utf-8")
    submission = (output_directory / "movie2_submission_notes.txt").read_text(
        encoding="utf-8"
    )
    reference = (
        output_directory / "movie2_manuscript_reference_suggestion.txt"
    ).read_text(encoding="utf-8")
    readme = (output_directory / "README.md").read_text(encoding="utf-8")
    if not caption.startswith("Movie 2."):
        raise ValueError("Caption is not explicitly titled Movie 2.")
    if caption.count("$$") < 4 or caption.count("$$") % 2:
        raise ValueError("Caption TeX maths is not consistently bounded by $$.")
    for term in (
        "YBJ",
        "TSB",
        "YBJ+",
        "PSE",
        "HBEs",
        "n=4",
        "n=16",
        "n=32",
    ):
        if term not in caption:
            raise ValueError(f"Caption is missing required term {term!r}.")
    if "0 to 50" not in caption or "does not interpolate" not in caption:
        raise ValueError("Caption does not document time coverage and sampling.")
    if (
        "128\\times128" not in caption
        or "f_c=64" not in caption
        or "64 time steps per inertial period" not in caption
    ):
        raise ValueError("Caption does not document the 128x128, fc=64 data.")
    if "additional 36 frames (1.5 s)" not in caption:
        raise ValueError("Caption does not document the chapter-end still holds.")
    for wavelength in (1000, 250, 125):
        if f"h={wavelength}" not in caption:
            raise ValueError(
                f"Caption is missing the h={wavelength} m vertical wavelength."
            )
    if "H=2\\,\\mathrm{km}" not in caption or "h=2H/n" not in caption:
        raise ValueError("Caption does not document H=2 km and h=2H/n.")
    if (
        "NRE vertical axis spans 0--40%" not in caption
        or "0--10%" not in caption
    ):
        raise ValueError("Caption does not document the mode-specific NRE axes.")
    if (
        "instantaneous complex-velocity" not in caption
        or "\\phi_{\\mathrm{model}}" not in caption
        or "\\phi_{\\mathrm{HBEs}}" not in caption
    ):
        raise ValueError(
            "Caption does not distinguish instantaneous NRE or define the "
            "model-minus-HBEs field explicitly."
        )
    if (
        "two-page structure" not in caption
        or "overall title page held for 2 s" not in caption
        or "first chapter page held for 4 s" not in caption
        or "two successive title pages" not in accessibility
        or "Chapter 1 page displayed for 4 seconds" not in accessibility
    ):
        raise ValueError(
            "Caption or accessibility text does not document the two-page opening."
        )
    if "additional 1.5 seconds" not in accessibility:
        raise ValueError(
            "Accessibility description does not document the chapter-end holds."
        )
    if (
        "NRE vertical axis spans 0 to 40%" not in accessibility
        or "0 to 10%" not in accessibility
    ):
        raise ValueError(
            "Accessibility description omits the mode-specific NRE axes."
        )
    if "Clipping" not in caption:
        raise ValueError("Caption does not disclose clipping.")
    if "without relying on colour alone" not in accessibility:
        raise ValueError("Accessibility description lacks non-colour guidance.")
    if "ScholarOne file designation: Movie" not in submission:
        raise ValueError("Submission notes do not specify the Movie designation.")
    if "Horizontal grid: 128x128" not in submission or "fc=64" not in submission:
        raise ValueError("Submission notes omit the simulation discretisation.")
    if JFM_PREPARING_URL not in submission or JFM_SUBMITTING_URL not in submission:
        raise ValueError("Submission notes omit official JFM links.")
    if "supplementary movie 2" not in reference.lower():
        raise ValueError("Suggested manuscript reference does not name movie 2.")
    combined = "\n".join((caption, accessibility, submission, reference, readme))
    if "4000-m depth" in combined or "H=4000" in combined:
        raise ValueError("Obsolete H=4 km wording appears in public-facing text.")
    if "\\Delta" in combined or "Δ" in combined:
        raise ValueError(
            "An ambiguous uppercase delta symbol appears in public-facing text."
        )
    if re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", combined):
        raise ValueError("A local absolute path leaked into a public-facing text file.")
    return {
        "caption_title": "Movie 2",
        "caption_tex_delimiter_count": caption.count("$$"),
        "official_jfm_links_present": True,
        "local_absolute_paths_absent": True,
        "ambiguous_uppercase_delta_absent": True,
        "instantaneous_nre_distinguished": True,
        "simulation_discretisation_documented": True,
        "domain_depth_2_km_documented": True,
        "vertical_wavelengths_unchanged": True,
        "two_page_opening_documented": True,
    }


def check_media(
    output_directory: Path,
    ffmpeg: Path,
    ffprobe: Path,
) -> dict[str, Any]:
    """Validate container, stream properties, frame count, fast start, and decode."""
    video_path = output_directory / "movie2.mp4"
    probe = ffprobe_json(ffprobe, video_path)
    video = unique_stream(probe, "video")
    audio_streams = [
        stream
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    format_info = probe["format"]
    format_names = set(str(format_info.get("format_name", "")).split(","))
    if not {"mov", "mp4"}.intersection(format_names):
        raise ValueError(f"Unexpected video container: {format_names}.")
    if video.get("codec_name") != "h264":
        raise ValueError(f"Unexpected video codec: {video.get('codec_name')}.")
    if video.get("pix_fmt") != "yuv420p":
        raise ValueError(f"Unexpected pixel format: {video.get('pix_fmt')}.")
    width, height = int(video["width"]), int(video["height"])
    if (width, height) not in {(1920, 1080), (2560, 1440)}:
        raise ValueError(f"Unexpected movie resolution: {width}x{height}.")
    avg_rate = float(Fraction(video["avg_frame_rate"]))
    real_rate = float(Fraction(video["r_frame_rate"]))
    if not np.isclose(avg_rate, real_rate, rtol=0.0, atol=1.0e-12):
        raise ValueError("Average and real frame rates differ; output may not be CFR.")
    if not 20.0 <= avg_rate <= 24.0:
        raise ValueError(f"Frame rate is outside 20--24 fps: {avg_rate}.")
    frame_count = int(video.get("nb_read_frames") or video.get("nb_frames"))
    duration = float(format_info["duration"])
    if not np.isclose(
        duration,
        frame_count / avg_rate,
        rtol=0.0,
        atol=1.5 / avg_rate,
    ):
        raise ValueError("Duration and CFR frame count are inconsistent.")
    if audio_streams:
        raise ValueError("Movie unexpectedly contains an audio stream.")
    size_bytes = int(format_info["size"])
    if size_bytes >= MAX_FILE_BYTES:
        raise ValueError(
            f"Movie is not smaller than {MAX_FILE_BYTES / 1_000_000:g} MB: "
            f"{size_bytes} bytes."
        )

    video_bytes = video_path.read_bytes()
    moov_position = video_bytes.find(b"moov")
    mdat_position = video_bytes.find(b"mdat")
    faststart = (
        moov_position >= 0
        and mdat_position >= 0
        and moov_position < mdat_position
    )
    if not faststart:
        raise ValueError("MP4 moov atom is not before mdat; fast start is absent.")

    decode_command = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    subprocess.run(decode_command, check=True)
    return {
        "container": format_info.get("format_name"),
        "codec": video.get("codec_name"),
        "codec_long_name": video.get("codec_long_name"),
        "profile": video.get("profile"),
        "pixel_format": video.get("pix_fmt"),
        "resolution": [width, height],
        "frame_rate_fps": avg_rate,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "size_bytes": size_bytes,
        "size_mb_decimal": size_bytes / 1_000_000.0,
        "audio_stream_count": 0,
        "faststart": True,
        "full_decode": "passed",
        "ffprobe": probe,
    }


def check_render_outputs(
    output_directory: Path,
    media: dict[str, Any],
    ffmpeg: Path,
) -> dict[str, Any]:
    """Validate the render manifest, representative images, and PSNR results."""
    manifest = json.loads(
        (output_directory / "movie2_render_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest["product"] != "supplementary movie 2":
        raise ValueError("Render manifest product label changed.")
    if manifest["physical_field_interpolation"]:
        raise ValueError("Render manifest reports physical-field interpolation.")
    if not np.array_equal(
        np.asarray(manifest.get("fixed_absolute_color_limits")),
        EXPECTED_ABSOLUTE_LIMITS,
    ):
        raise ValueError("Render manifest does not use the required colour limits.")
    if colorbar_number(37.5, 0) != "37.5":
        raise ValueError("The n=4 colourbar upper label does not display 37.5.")
    if manifest.get("source_data") != {
        "kind": "fields generated by the in-repository equation solver",
        "domain_depth_m": EXPECTED_DOMAIN_DEPTH_METRES,
        "vertical_wavelength_relation": "h=2H/n",
        "spatial_discretisation": {
            "grid_points": [EXPECTED_GRID_POINTS, EXPECTED_GRID_POINTS],
        },
        "time_discretisation": {
            "parameter": "fc",
            "steps_per_inertial_period": EXPECTED_STEPS_PER_INERTIAL_PERIOD,
        },
        "pse_field_source": (
            "A_up exp(-ift)+A_down exp(ift) from the PSE solver"
        ),
    }:
        raise ValueError("Render manifest source-data metadata changed.")
    style_alignment = manifest.get("style_alignment", {})
    if style_alignment != EXPECTED_STYLE_ALIGNMENT:
        raise ValueError(
            "Movie style metadata no longer matches manuscript figures 8--10."
        )
    opening_frames = int(manifest.get("opening_frames", 0))
    opening_seconds = float(manifest.get("opening_seconds", 0.0))
    title_frames = int(manifest.get("title_frames", 0))
    title_seconds = float(manifest.get("title_seconds", 0.0))
    if opening_seconds != 2.0 or opening_frames != 48:
        raise ValueError("The overall opening page must last 2 seconds.")
    if title_seconds != 4.0 or title_frames != 96:
        raise ValueError("Each Chapter page must last 4 seconds for readability.")
    if int(manifest.get("opening_page_count", 0)) != 2:
        raise ValueError("Movie 2 must open with an overall and a Chapter 1 page.")
    chapter_end_frames = int(manifest.get("chapter_end_frames", 0))
    chapter_end_seconds = float(manifest.get("chapter_end_seconds", 0.0))
    expected_chapter_end_frames = int(
        round(chapter_end_seconds * media["frame_rate_fps"])
    )
    if chapter_end_seconds != 1.5 or chapter_end_frames != 36:
        raise ValueError("The required 1.5-second chapter-end hold changed.")
    if chapter_end_frames != expected_chapter_end_frames:
        raise ValueError("Chapter-end seconds and frame count are inconsistent.")
    if manifest["frame_count"] != media["frame_count"]:
        raise ValueError("Manifest and probed frame counts differ.")
    if not np.isclose(
        manifest["duration_seconds"],
        media["duration_seconds"],
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("Manifest and probed durations differ.")
    segments = manifest["segments"]
    if not segments or segments[0]["start_frame"] != 0:
        raise ValueError("Render segment map is empty or starts late.")
    expected_start = 0
    previous_scientific_time: dict[int, float] = {}
    chapter_end_holds: dict[int, int] = {}
    chapter_end_segments: dict[int, dict[str, Any]] = {}
    opening_titles = [
        segment for segment in segments if segment["kind"] == "opening_title"
    ]
    chapter_titles = [
        segment for segment in segments if segment["kind"] == "chapter_title"
    ]
    if len(opening_titles) != 1 or opening_titles[0] != segments[0]:
        raise ValueError("The overall opening page is missing or misplaced.")
    if opening_titles[0]["frame_count"] != opening_frames:
        raise ValueError("The overall opening-page duration changed.")
    if [int(item["vertical_mode"]) for item in chapter_titles] != [4, 16, 32]:
        raise ValueError("The three mode chapter pages are missing or reordered.")
    if any(item["frame_count"] != title_frames for item in chapter_titles):
        raise ValueError("A mode chapter-page duration changed.")
    if len(segments) < 2 or segments[1] != chapter_titles[0]:
        raise ValueError("Chapter 1 must follow the overall opening page.")
    for segment_index, segment in enumerate(segments):
        if segment["start_frame"] != expected_start:
            raise ValueError("Render segments contain a frame gap or overlap.")
        expected_start += segment["frame_count"]
        if segment["kind"] == "scientific_frame":
            mode = int(segment["vertical_mode"])
            time_ip = float(segment["time_ip"])
            previous = previous_scientific_time.get(mode, -np.inf)
            if time_ip <= previous:
                raise ValueError(f"Scientific times are not increasing for n={mode}.")
            previous_scientific_time[mode] = time_ip
        elif segment["kind"] == "chapter_end_hold":
            mode = int(segment["vertical_mode"])
            if mode in chapter_end_holds:
                raise ValueError(f"Multiple chapter-end holds found for n={mode}.")
            if segment["frame_count"] != chapter_end_frames:
                raise ValueError(f"Chapter-end hold length changed for n={mode}.")
            if not np.isclose(float(segment["time_ip"]), 50.0):
                raise ValueError(f"Chapter-end hold is not at 50 IP for n={mode}.")
            if segment_index == 0:
                raise ValueError("A chapter-end hold cannot be the first segment.")
            previous_segment = segments[segment_index - 1]
            if (
                previous_segment["kind"] != "scientific_frame"
                or int(previous_segment["vertical_mode"]) != mode
                or not np.isclose(float(previous_segment["time_ip"]), 50.0)
                or previous_segment["source_key"] != segment["source_key"]
            ):
                raise ValueError(
                    f"Chapter-end hold for n={mode} does not repeat its true "
                    "50-IP terminal frame."
                )
            chapter_end_holds[mode] = int(segment["frame_count"])
            chapter_end_segments[mode] = segment
    if expected_start != media["frame_count"]:
        raise ValueError("Render segments do not cover every video frame.")
    if previous_scientific_time != {4: 50.0, 16: 50.0, 32: 50.0}:
        raise ValueError("All three movie chapters must end at 50 IP.")
    if manifest.get("vertical_modes") != [4, 16, 32]:
        raise ValueError("Render manifest vertical-mode order changed.")
    if chapter_end_holds != {4: 36, 16: 36, 32: 36}:
        raise ValueError("A required chapter-end hold is missing.")

    static_checks = {
        str(mode): measure_encoded_static_range(
            ffmpeg,
            output_directory / "movie2.mp4",
            start_frame=int(segment["start_frame"]) - int(manifest["hold_frames"]),
            end_frame=(
                int(segment["start_frame"])
                + int(segment["frame_count"])
                - 1
            ),
            width=int(media["resolution"][0]),
            height=int(media["resolution"][1]),
        )
        for mode, segment in chapter_end_segments.items()
    }

    qc = manifest["representative_qc"]
    required_labels = {
        "Opening title",
        "Chapter 1 title",
        "n=4, t=0 IP",
        "n=4, t=10 IP",
        "n=4, t=25 IP",
        "n=4, t=50 IP",
        "Transition to n=16",
        "n=16, t=25 IP",
        "n=16, t=50 IP",
        "Transition to n=32",
        "n=32, t=50 IP",
    }
    if {item["label"] for item in qc} != required_labels:
        raise ValueError("Representative QC set changed.")
    minimum_psnr = min(float(item["psnr_db"]) for item in qc)
    if minimum_psnr < 30.0:
        raise ValueError(f"Representative-frame PSNR is too low: {minimum_psnr:.2f}.")

    preview_dimensions = image_dimensions(output_directory / "movie2_preview.png")
    if list(preview_dimensions) != media["resolution"]:
        raise ValueError(
            f"Preview dimensions {preview_dimensions} do not match the movie."
        )
    contact_dimensions = image_dimensions(
        output_directory / "movie2_qc_contact_sheet.png"
    )
    if contact_dimensions[0] < 1800 or contact_dimensions[1] < 500:
        raise ValueError("QC contact sheet is too small for visual inspection.")
    return {
        "segments_cover_all_frames": True,
        "scientific_time_order": "strictly increasing within each chapter",
        "fixed_color_limits": True,
        "n4_absolute_colorbar_upper_label": "37.5",
        "style_alignment": style_alignment,
        "chapter_end_hold_seconds": chapter_end_seconds,
        "chapter_end_hold_frames_by_mode": chapter_end_holds,
        "encoded_chapter_end_static_checks": static_checks,
        "representative_frame_count": len(qc),
        "minimum_representative_psnr_db": minimum_psnr,
        "preview_dimensions": list(preview_dimensions),
        "contact_sheet_dimensions": list(contact_dimensions),
        "representative_qc": qc,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate sinusoidal-dipole supplementary movie 2."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run all movie checks and write a machine-readable quality report."""
    args = parse_args()
    input_path = args.input.resolve()
    output_directory = args.output_directory.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Intermediate archive is missing: {input_path}")
    if not output_directory.is_dir():
        raise FileNotFoundError(f"Movie output directory is missing: {output_directory}")
    for name in REQUIRED_OUTPUTS:
        path = output_directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required movie output is missing: {path}")

    ffmpeg = resolve_executable(args.ffmpeg, "ffmpeg")
    ffprobe = resolve_executable(args.ffprobe, "ffprobe")
    numerical = check_archive(input_path)
    media = check_media(output_directory, ffmpeg, ffprobe)
    visual = check_render_outputs(output_directory, media, ffmpeg)
    text_outputs = check_text_outputs(output_directory)
    report = {
        "schema_version": 1,
        "product": "supplementary movie 2",
        "status": "passed",
        "numerical_checks": numerical,
        "media_checks": media,
        "representative_visual_checks": visual,
        "text_and_submission_checks": text_outputs,
        "jfm_checks": {
            "mp4": True,
            "h264": True,
            "movie_designation_documented": True,
            "numbered_movie_2": True,
            "separate_caption": True,
            "caption_tex_math": True,
            "under_10_mb": True,
            "silent": True,
            "official_preparing_materials_url": JFM_PREPARING_URL,
            "official_submitting_materials_url": JFM_SUBMITTING_URL,
        },
    }
    report_path = output_directory / "movie2_quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"movie2 validation passed: {report_path}")


if __name__ == "__main__":
    main()
