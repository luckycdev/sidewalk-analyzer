from __future__ import annotations

import json
import os
import re
from typing import Any


MODEL_ID = os.getenv("AWS_MODEL_ID", "twelvelabs.pegasus-1-2-v1:0")
REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_OWNER = os.getenv("AWS_BUCKET_OWNER", "564203970240")

_bedrock_client = None

# Pegasus tuning knobs (env-overridable)
MIN_SEGMENT_CONFIDENCE = float(os.getenv("MIN_SEGMENT_CONFIDENCE", "0.7"))
VERIFY_SEGMENTS = os.getenv("VERIFY_SEGMENTS", "1").strip().lower() not in {"0", "false", "no"}
VERIFY_MIN_HITS = int(os.getenv("VERIFY_MIN_HITS", "2"))  # out of 3 (start/mid/end)
VERIFY_MIN_AVG_CONF = float(os.getenv("VERIFY_MIN_AVG_CONF", "0.7"))


PEGASUS_PROMPT = """
You are analyzing a walking video where the camera points at the pedestrian path.

Task: find time ranges where SIDEWALK / PEDESTRIAN PATH SURFACE DAMAGE is clearly visible.

Only flag when the sidewalk surface is visible enough to judge condition. Do NOT flag:
- normal joints/expansion seams
- mild texture/grain
- shadows, lighting changes, wet patches
- camera blur or motion
- road/asphalt damage unless it is on the sidewalk/path surface
- grass/dirt edges unless it creates a clear tripping hazard on the walking surface

Damage types to consider (examples):
- crack (wide / alligator / long)
- spalling / missing chunks
- pothole / hole
- heave / uplift / vertical displacement (trip hazard)
- severe unevenness / buckling
- broken slab / collapsed edge
- obstruction hazard (if on path surface)

Output requirements:
- Return ONLY valid JSON (no markdown, no explanation).
- Output is an array of segments.
- Segments MUST be non-overlapping and sorted by start time.
- Use integer seconds for start/end (inclusive). If damage is visible for a single moment, set start=end.
- Keep segments coarse: if the same damage persists across adjacent seconds, produce ONE segment.

Severity rubric:
- low: visible defect but minor trip risk (hairline/wear, small cracks)
- medium: noticeable defect that could trip or impede mobility (wider cracks, small heave, moderate spalling)
- high: clear hazard likely to trip or block mobility (large holes, big heave, major breakup)

For each segment include:
{
  "start": int,
  "end": int,
  "severity": "low" | "medium" | "high",
  "damage_types": string[],
  "confidence": number,  // 0.0 to 1.0
  "description": string  // short, specific visual cues (e.g., "broken slab with 2-3cm heave")
}

Return [] if no clear sidewalk damage is visible.
""".strip()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, numeric))


def _pegasus_message_to_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, (dict, list)):
        try:
            return json.dumps(message, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(message)
    return str(message)


def _coerce_segment_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("segments", "detections", "results", "items", "output"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        if any(k in value for k in ("start", "end", "time")):
            return [value]
    return []


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return _coerce_segment_list(parsed)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return parsed
        return _coerce_segment_list(parsed)
    except json.JSONDecodeError:
        return []


def parse_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    message = result.get("message")
    segments = _extract_json_array(_pegasus_message_to_text(message))
    if segments:
        return [item for item in segments if isinstance(item, dict)]

    for key in ("segments", "detections", "results", "output"):
        nested = result.get(key)
        coerced = _coerce_segment_list(nested)
        if coerced:
            return coerced

    return _coerce_segment_list(result)


def normalize_and_merge_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        start = _as_int(det.get("start"), -1)
        end = _as_int(det.get("end"), -1)
        if start < 0 or end < 0:
            continue
        if end < start:
            start, end = end, start

        severity = str(det.get("severity") or "").strip().lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"

        damage_types = det.get("damage_types")
        if isinstance(damage_types, list):
            damage_types_clean = [str(item).strip().lower() for item in damage_types if str(item).strip()]
        else:
            damage_types_clean = []

        confidence = _clamp_float(det.get("confidence"), 0.5, 0.0, 1.0)
        description = str(det.get("description") or "").strip() or "Sidewalk damage visible"

        normalized.append(
            {
                "start": start,
                "end": end,
                "severity": severity,
                "damage_types": damage_types_clean,
                "confidence": confidence,
                "description": description,
            }
        )

    normalized.sort(key=lambda d: (d["start"], d["end"]))
    if not normalized:
        return []

    severity_rank = {"low": 0, "medium": 1, "high": 2}
    merged: list[dict[str, Any]] = [normalized[0]]
    for det in normalized[1:]:
        prev = merged[-1]
        if det["start"] <= prev["end"] + 1:
            prev["end"] = max(prev["end"], det["end"])
            if severity_rank[det["severity"]] > severity_rank[prev["severity"]]:
                prev["severity"] = det["severity"]
            prev["confidence"] = max(float(prev["confidence"]), float(det["confidence"]))
            prev_types = set(prev.get("damage_types") or [])
            det_types = set(det.get("damage_types") or [])
            prev["damage_types"] = sorted(prev_types | det_types)
            if len(det.get("description", "")) > len(prev.get("description", "")):
                prev["description"] = det["description"]
            continue
        merged.append(det)

    return merged


def invoke(prompt: str, *, s3_uri: str) -> dict[str, Any]:
    global _bedrock_client
    if _bedrock_client is None:
        import boto3

        _bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

    body = {
        "inputPrompt": prompt,
        "mediaSource": {
            "s3Location": {
                "uri": s3_uri,
                "bucketOwner": BUCKET_OWNER,
            }
        },
    }
    response = _bedrock_client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    raw_body = response["body"].read()
    if isinstance(raw_body, (bytes, bytearray)):
        raw_body = raw_body.decode("utf-8")
    return json.loads(raw_body)


def verify_segments(detections: list[dict[str, Any]], *, s3_uri: str) -> list[dict[str, Any]]:
    if not detections:
        return []

    checks: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        start = int(det["start"])
        end = int(det["end"])
        mid = (start + end) // 2
        checks.append({"segment_index": idx, "t": start})
        checks.append({"segment_index": idx, "t": mid})
        checks.append({"segment_index": idx, "t": end})

    verify_prompt = f"""
You are verifying whether sidewalk/path SURFACE DAMAGE is truly visible at specific timestamps.

Rules:
- Be conservative: if you cannot clearly see sidewalk damage, set damage=false.
- Ignore normal seams/joints, shadows, blur, wetness, and road-only damage.
- Only judge the pedestrian walking surface.

Return ONLY valid JSON array with one object per check:
{{
  "segment_index": int,
  "t": int,
  "damage": boolean,
  "severity": "low" | "medium" | "high" | null,
  "damage_types": string[],
  "confidence": number,
  "notes": string
}}

Checks:
{json.dumps(checks)}
""".strip()

    result = invoke(verify_prompt, s3_uri=s3_uri)
    raw_message = _pegasus_message_to_text(result.get("message"))
    rows = _extract_json_array(raw_message)

    by_segment: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        seg_idx = _as_int(row.get("segment_index"), -1)
        if seg_idx < 0:
            continue
        by_segment.setdefault(seg_idx, []).append(row)

    kept: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        segment_rows = by_segment.get(idx, [])
        hits = 0
        confs: list[float] = []
        merged_types: set[str] = set(det.get("damage_types") or [])

        for row in segment_rows:
            damage = bool(row.get("damage"))
            conf = _clamp_float(row.get("confidence"), 0.0, 0.0, 1.0)
            if damage:
                hits += 1
                confs.append(conf)
                types = row.get("damage_types")
                if isinstance(types, list):
                    merged_types |= {str(t).strip().lower() for t in types if str(t).strip()}

        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        if hits >= VERIFY_MIN_HITS and avg_conf >= VERIFY_MIN_AVG_CONF:
            det["damage_types"] = sorted(merged_types)
            det["confidence"] = max(float(det.get("confidence", 0.0)), avg_conf)
            kept.append(det)

    expected_rows = len(detections) * 3
    if detections and not kept and len(rows) < max(1, expected_rows // 2):
        return detections

    return kept

