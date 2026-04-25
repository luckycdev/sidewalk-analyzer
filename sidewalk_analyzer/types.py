from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict


WidthClass = Literal["narrow", "standard", "wide"]
SurfaceCondition = Literal["good", "fair", "poor", "impassable"]
CurbRampStatus = Literal["compliant", "non_compliant", "missing"]


class PegasusResult(TypedDict, total=False):
    sidewalk_present: bool
    width_class: WidthClass
    surface_condition: SurfaceCondition
    curb_ramp_status: CurbRampStatus
    confidence: float
    raw_text: str


@dataclass(frozen=True)
class SegmentCandidate:
    segment_id: str
    start_s: float
    end_s: float
    score: float


@dataclass(frozen=True)
class ClipAsset:
    clip_id: str
    segment_id: str
    start_s: float
    end_s: float
    path: Path
    still_path: Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    clips_dir: Path
    exports_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_iso: str
    input_video: str
    s3_video_uri: str | None
    outputs: dict[str, str]
    stats: dict[str, Any]

