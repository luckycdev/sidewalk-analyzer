from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .types import ClipAsset, SegmentCandidate


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def get_video_duration_s(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = _run(cmd)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "ffprobe failed").strip())
    payload = json.loads(completed.stdout)
    duration = payload.get("format", {}).get("duration")
    try:
        return float(duration)
    except (TypeError, ValueError):
        raise RuntimeError(f"Could not parse duration from ffprobe output: {payload}")


def get_video_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    completed = _run(cmd)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "ffprobe failed").strip())
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    rate = None
    if streams and isinstance(streams, list) and isinstance(streams[0], dict):
        rate = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate")
    if not rate:
        return 30.0
    if isinstance(rate, str) and "/" in rate:
        num_s, den_s = rate.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
            if den == 0:
                return 30.0
            fps = num / den
            return fps if fps > 0 else 30.0
        except ValueError:
            return 30.0
    try:
        fps = float(rate)
        return fps if fps > 0 else 30.0
    except (TypeError, ValueError):
        return 30.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def extract_clips_and_stills(
    *,
    video_path: Path,
    candidates: list[SegmentCandidate],
    clips_dir: Path,
    clip_seconds: float,
    padding_seconds: float,
) -> tuple[list[ClipAsset], dict[str, Any]]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration_s(video_path)

    assets: list[ClipAsset] = []
    errors: list[str] = []

    for idx, cand in enumerate(candidates):
        mid = (cand.start_s + cand.end_s) / 2.0
        start = _clamp(mid - clip_seconds / 2.0 - padding_seconds, 0.0, max(0.0, duration - 0.1))
        end = _clamp(start + clip_seconds, 0.0, duration)
        clip_id = f"clip_{idx:03d}"
        clip_path = clips_dir / f"{clip_id}.mp4"
        still_path = clips_dir / f"{clip_id}.jpg"

        # Extract clip (re-encode for safety/compatibility)
        cmd_clip = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(video_path),
            "-an",
            "-vf",
            "scale='min(1280,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(clip_path),
        ]
        completed = _run(cmd_clip)
        if completed.returncode != 0:
            errors.append(f"{clip_id}: ffmpeg clip failed: {(completed.stderr or completed.stdout).strip()}")
            continue

        # Still-frame baseline: middle frame of the clip window
        still_t = _clamp((start + end) / 2.0, 0.0, duration)
        cmd_still = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{still_t:.3f}",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(still_path),
        ]
        completed2 = _run(cmd_still)
        if completed2.returncode != 0:
            errors.append(f"{clip_id}: ffmpeg still failed: {(completed2.stderr or completed2.stdout).strip()}")
            continue

        assets.append(
            ClipAsset(
                clip_id=clip_id,
                segment_id=cand.segment_id,
                start_s=start,
                end_s=end,
                path=clip_path,
                still_path=still_path,
            )
        )

    stats: dict[str, Any] = {
        "input_duration_s": round(duration, 3),
        "candidates": len(candidates),
        "clips_written": len(assets),
        "errors": errors[:10],
    }
    return assets, stats

