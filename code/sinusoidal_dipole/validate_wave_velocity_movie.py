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
    resolve_executable,
)


MODEL_NAMES = ["YBJ", "TSB", "YBJ+", "PSE", "HBEs"]
NRE_MODEL_NAMES = MODEL_NAMES[:4]
EXPECTED_VERTICAL_MODES = np.asarray([4, 16, 32])
EXPECTED_T50_MAXIMA = np.asarray(
    [
        [
            31.786428361858587,
            37.32062604847504,
            37.239008041643494,
            37.54282297563831,
            27.56445542781792,
        ],
        [
            1.6125659389777067,
            1.6123885329555694,
            1.6123861068367464,
            1.6135188620971153,
            1.6784137820082063,
        ],
        [
            1.1200476980737526,
            1.1200508886292768,
            1.1200508180212077,
            1.189362150464363,
            1.1880821730307622,
        ],
    ]
)
EXPECTED_ABSOLUTE_LIMITS = np.asarray(
    [[0.01, 10.0], [0.50, 1.50], [0.88, 1.12]]
)
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
        "reference_max_abs_complex_difference",
        "reference_max_abs_nre_difference",
        "raw_source_files",
        "processed_reference_files",
        "metadata_json",
    }
    with np.load(path) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Intermediate archive is missing: {missing}")
        times = archive["times_in_inertial_periods"]
        source_indices = archive["source_time_indices"]
        modes = archive["vertical_modes"]
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
        field_differences = archive["reference_max_abs_complex_difference"]
        nre_differences = archive["reference_max_abs_nre_difference"]
        metadata = json.loads(str(archive["metadata_json"].item()))

        if not np.array_equal(times, np.arange(51.0)):
            raise ValueError("Movie fields must contain every integer time from 0--50 IP.")
        if not np.array_equal(modes, EXPECTED_VERTICAL_MODES):
            raise ValueError("Movie vertical-mode order changed.")
        if model_names != MODEL_NAMES or nre_names != NRE_MODEL_NAMES:
            raise ValueError("Movie model order changed.")
        if fields.shape != (3, 51, 5, 64, 64):
            raise ValueError(f"Unexpected movie field shape: {fields.shape}.")
        if source_indices.shape != (3, 51):
            raise ValueError("Unexpected source-time index shape.")
        expected_indices = np.arange(0, 3201, 64)
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
        expected_coordinate = np.linspace(-np.pi, np.pi, 64, endpoint=False)
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
        if np.max(field_differences) > 1.0e-10:
            raise ValueError("Raw PSE/model fields disagree with processed references.")
        if np.max(nre_differences) > 1.0e-12:
            raise ValueError("Recomputed NRE values disagree with stored error data.")
        time_50 = int(np.flatnonzero(np.isclose(times, 50.0))[0])
        maxima = fields[:, time_50].max(axis=(-2, -1))
        if not np.allclose(
            maxima,
            EXPECTED_T50_MAXIMA,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(
                f"Mode-wise t=50 IP maxima changed: {maxima.tolist()}."
            )
        for target in (10.0, 50.0):
            if not np.any(np.isclose(times, target, rtol=0.0, atol=1.0e-12)):
                raise ValueError(f"Required {target:g}-IP validation frame is absent.")

    if metadata["pse_reconstruction"] != (
        "pse_velocity = A_up + conj(stored_conj_A_dn)"
    ):
        raise ValueError("PSE reconstruction metadata changed.")
    if metadata["orientation"] != {
        "extent": [-np.pi, np.pi, -np.pi, np.pi],
        "hdf5_slice_transform": "transpose",
        "origin": "lower",
    }:
        raise ValueError("Movie orientation metadata changed.")
    if metadata["sampling"]["physical_field_interpolation"]:
        raise ValueError("Movie metadata incorrectly reports field interpolation.")

    return {
        "field_shape": list(fields.shape),
        "times_ip": [float(times[0]), float(times[-1])],
        "source_time_step": 64,
        "model_order": model_names,
        "max_raw_processed_field_difference": float(np.max(field_differences)),
        "max_recomputed_stored_nre_difference": float(np.max(nre_differences)),
        "t50_maxima_by_mode": {
            str(int(mode)): values.tolist()
            for mode, values in zip(modes, maxima, strict=True)
        },
        "absolute_color_limits": absolute_limits.tolist(),
        "difference_color_limits": difference_limits.tolist(),
        "clipping": metadata["color_limits"]["clipping"],
        "pse_reconstruction": metadata["pse_reconstruction"],
    }


def image_dimensions(path: Path) -> tuple[int, int]:
    """Return image dimensions after forcing a complete decode."""
    with Image.open(path) as image:
        image.load()
        return image.size


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
    if "Clipping" not in caption:
        raise ValueError("Caption does not disclose clipping.")
    if "without relying on colour alone" not in accessibility:
        raise ValueError("Accessibility description lacks non-colour guidance.")
    if "ScholarOne file designation: Movie" not in submission:
        raise ValueError("Submission notes do not specify the Movie designation.")
    if JFM_PREPARING_URL not in submission or JFM_SUBMITTING_URL not in submission:
        raise ValueError("Submission notes omit official JFM links.")
    if "supplementary movie 2" not in reference.lower():
        raise ValueError("Suggested manuscript reference does not name movie 2.")
    combined = "\n".join((caption, accessibility, submission, reference, readme))
    if re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", combined):
        raise ValueError("A local absolute path leaked into a public-facing text file.")
    return {
        "caption_title": "Movie 2",
        "caption_tex_delimiter_count": caption.count("$$"),
        "official_jfm_links_present": True,
        "local_absolute_paths_absent": True,
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
        raise ValueError(f"Movie is not smaller than 50 MB: {size_bytes} bytes.")

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
    for segment in segments:
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
    if expected_start != media["frame_count"]:
        raise ValueError("Render segments do not cover every video frame.")
    if previous_scientific_time != {4: 50.0, 16: 50.0, 32: 50.0}:
        raise ValueError("All three movie chapters must end at 50 IP.")
    if manifest.get("vertical_modes") != [4, 16, 32]:
        raise ValueError("Render manifest vertical-mode order changed.")

    qc = manifest["representative_qc"]
    required_labels = {
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
    visual = check_render_outputs(output_directory, media)
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
            "under_50_mb": True,
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
