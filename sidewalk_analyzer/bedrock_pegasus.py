from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Literal

import boto3

from .config import Settings
from .s3_io import upload_video_to_s3
from .types import ClipAsset, PegasusResult


PEGASUS_MODEL_ID = "twelvelabs.pegasus-1-2-v1:0"


def _normalize_enum(value: str, *, allowed: set[str], default: str) -> str:
    v = (value or "").strip().lower()
    return v if v in allowed else default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True
        if v in {"false", "no", "n", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_result(payload: dict[str, Any], raw_text: str) -> PegasusResult:
    width_allowed = {"narrow", "standard", "wide"}
    surface_allowed = {"good", "fair", "poor", "impassable"}
    curb_allowed = {"compliant", "non_compliant", "missing"}

    sidewalk_present = _coerce_bool(payload.get("sidewalk_present"), False)
    width_class = _normalize_enum(str(payload.get("width_class", "")), allowed=width_allowed, default="standard")
    surface_condition = _normalize_enum(str(payload.get("surface_condition", "")), allowed=surface_allowed, default="fair")
    curb_ramp_status = _normalize_enum(str(payload.get("curb_ramp_status", "")), allowed=curb_allowed, default="missing")
    confidence = _coerce_float(payload.get("confidence"), 0.0)
    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    return PegasusResult(
        sidewalk_present=sidewalk_present,
        width_class=width_class,  # type: ignore[typeddict-item]
        surface_condition=surface_condition,  # type: ignore[typeddict-item]
        curb_ramp_status=curb_ramp_status,  # type: ignore[typeddict-item]
        confidence=confidence,
        raw_text=raw_text,
    )


def _invoke(runtime: Any, *, model_id: str, body: dict[str, Any]) -> str:
    resp = runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )
    b = resp.get("body")
    raw = b.read().decode("utf-8") if hasattr(b, "read") else (b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b))
    return raw


def _build_prompt() -> str:
    return (
        "You are labeling pedestrian infrastructure for ADA planning.\n"
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        '{\n'
        '  "sidewalk_present": true|false,\n'
        '  "width_class": "narrow"|"standard"|"wide",\n'
        '  "surface_condition": "good"|"fair"|"poor"|"impassable",\n'
        '  "curb_ramp_status": "compliant"|"non_compliant"|"missing",\n'
        '  "confidence": 0.0-1.0\n'
        "}\n"
        "If sidewalk is not present, set other fields to reasonable defaults and set confidence low.\n"
    )


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    # Direct JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Try to extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def run_pegasus_on_assets(
    *,
    assets: list[ClipAsset],
    settings: Settings,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    For each asset:
    - invoke Pegasus on the clip (video context)
    - invoke Pegasus on a still frame (baseline)
    Normalize both into a strict schema.
    """

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    runtime = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    prompt = _build_prompt()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for asset in assets:
        # Upload clip + still to S3 so the model can fetch them (avoids huge request payloads).
        # We reuse the same upload helper for mp4; for jpg we do simple put_object.
        clip_s3_uri = upload_video_to_s3(video_path=asset.path, settings=settings)
        still_s3_uri = _upload_image_to_s3(image_path=asset.still_path, settings=settings)

        clip_raw = ""
        still_raw = ""

        try:
            clip_raw = _invoke(
                runtime,
                model_id=PEGASUS_MODEL_ID,
                body={"input": {"s3Uri": clip_s3_uri}, "prompt": prompt},
            )
        except Exception as exc:
            errors.append(f"{asset.clip_id}: clip invoke failed: {exc}")

        try:
            still_raw = _invoke(
                runtime,
                model_id=PEGASUS_MODEL_ID,
                body={"input": {"s3Uri": still_s3_uri}, "prompt": prompt},
            )
        except Exception as exc:
            errors.append(f"{asset.clip_id}: still invoke failed: {exc}")

        clip_json = _parse_json_from_text(clip_raw) or {}
        still_json = _parse_json_from_text(still_raw) or {}
        clip_result = _validate_result(clip_json, clip_raw)
        still_result = _validate_result(still_json, still_raw)

        rows.append(
            {
                "clip_id": asset.clip_id,
                "segment_id": asset.segment_id,
                "segment_start_s": asset.start_s,
                "segment_end_s": asset.end_s,
                "clip_s3_uri": clip_s3_uri,
                "still_s3_uri": still_s3_uri,
                "pegasus_clip": dict(clip_result),
                "pegasus_still": dict(still_result),
            }
        )

        # Persist raw responses for audit
        (logs_dir / f"{asset.clip_id}_pegasus_clip.txt").write_text(clip_raw or "", encoding="utf-8")
        (logs_dir / f"{asset.clip_id}_pegasus_still.txt").write_text(still_raw or "", encoding="utf-8")
        time.sleep(0.2)  # polite pacing

    stats: dict[str, Any] = {
        "assets": len(assets),
        "errors": errors[:10],
    }
    (logs_dir / "pegasus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return rows, stats


def _upload_image_to_s3(*, image_path: Path, settings: Settings) -> str:
    if not settings.s3_bucket:
        raise RuntimeError("Missing S3_BUCKET in environment/.env")

    key = f"{settings.s3_prefix}images/{image_path.stem}{image_path.suffix.lower()}"
    s3 = boto3.client("s3", region_name=settings.aws_region)
    body = image_path.read_bytes()
    s3.put_object(Bucket=settings.s3_bucket, Key=key, Body=body, ContentType="image/jpeg")
    return f"s3://{settings.s3_bucket}/{key}"

