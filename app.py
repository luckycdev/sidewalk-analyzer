from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_file

# Optional: only used when exporting Overture geometries as GeoJSON
try:
    from shapely.geometry import mapping as shapely_mapping  # type: ignore
except Exception:  # pragma: no cover
    shapely_mapping = None


USER_AGENT = "sidewalk-analyzer/1.0 (local flask app)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
MAPILLARY_IMAGES_URL = "https://graph.mapillary.com/images"
DEFAULT_OVERTURE_PADDING_M = 120.0
DEFAULT_OVERTURE_MAX_MATCH_M = 220.0
OVERTURE_ID_LOOKUP_URL = "https://geocoder.bradr.dev/id/{gers_id}"


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(Path(__file__).with_name(".env"))

APP_ROOT = Path(__file__).resolve().parent


def resolve_project_media_path(path_str: str) -> Path | None:
    """Resolve a relative or absolute path to an existing file under APP_ROOT."""
    raw = path_str.strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (APP_ROOT / raw).resolve()
    else:
        try:
            candidate = candidate.resolve()
        except OSError:
            return None
    try:
        candidate.relative_to(APP_ROOT)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def results_artifact_paths(sequence_id: str) -> tuple[Path, Path]:
    """Per-sequence results CSV and GERS GeoJSON paths (may not exist yet)."""
    safe = Path(sequence_id).name  # basename only
    return (
        APP_ROOT / f"results_{safe}.csv",
        APP_ROOT / f"results_{safe}.gers.geojson",
    )


def get_overture_explorer_link_via_api(
    gers_id: str, *, theme: str = "transportation", otype: str = "segment", zoom: float = 16.75
) -> str:
    """
    Given a GERS/Overture segment id, generate an Overture Explorer deep link by first
    resolving the feature bbox using a lightweight public lookup API.
    """
    gid = (gers_id or "").strip()
    if not gid:
        raise ValueError("gers_id is required")

    url = OVERTURE_ID_LOOKUP_URL.format(gers_id=gid)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    bbox = data.get("bbox") or {}
    try:
        lat = (float(bbox["ymin"]) + float(bbox["ymax"])) / 2.0
        lon = (float(bbox["xmin"]) + float(bbox["xmax"])) / 2.0
    except Exception as exc:
        raise ValueError("ID found but no valid bounding box returned") from exc

    return (
        "https://explore.overturemaps.org/"
        f"?mode=explore&feature={theme}.{otype}.{gid}#{zoom}/{lat:.6f}/{lon:.6f}"
    )


app = Flask(__name__, template_folder="templates", static_folder="static")

PIPELINE_RUNS: dict[str, dict[str, Any]] = {}
PIPELINE_LOCK = threading.Lock()

# AWS Configuration
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "sidewalk-analyzer-vincent")
REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("AWS_MODEL_ID", "twelvelabs.pegasus-1-2-v1:0")
BUCKET_OWNER = os.getenv("AWS_BUCKET_OWNER", "564203970240")

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# Pegasus tuning knobs (ported from `app_copyy.py` / `aws_test.py`)
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


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Best-effort extraction of the first JSON array in a model response."""
    if not text:
        return []
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        coerced = _coerce_segment_list(parsed)
        return coerced
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


def _pegasus_message_to_text(message: Any) -> str:
    """
    Bedrock/TwelveLabs responses sometimes return `message` as a string, but other times as
    structured JSON (dict/list). Normalize to a string suitable for `_extract_json_array`.
    """
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
    """Turn assorted Pegasus payload shapes into a list[dict] of segment-like objects."""
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("segments", "detections", "results", "items", "output"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        # Sometimes the model returns a single object instead of a list.
        if any(k in value for k in ("start", "end", "time")):
            return [value]
    return []


def _parse_pegasus_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parse segments from a Pegasus/Bedrock JSON wrapper.

    Primary path: `message` is JSON text containing an array of segments.
    Fallbacks: structured dict/list shapes.
    """
    message = result.get("message")
    segments = _extract_json_array(_pegasus_message_to_text(message))
    if segments:
        return [item for item in segments if isinstance(item, dict)]

    # Common alternate top-level keys
    for key in ("segments", "detections", "results", "output"):
        nested = result.get(key)
        coerced = _coerce_segment_list(nested)
        if coerced:
            return coerced

    return _coerce_segment_list(result)


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


def normalize_and_merge_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate + merge overlapping/adjacent segments to reduce duplicates."""
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
        description = str(det.get("description") or "").strip()
        if not description:
            description = "Sidewalk damage visible"

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


def _bedrock_pegasus_invoke(prompt: str, *, s3_uri: str) -> dict[str, Any]:
    body = {
        "inputPrompt": prompt,
        "mediaSource": {
            "s3Location": {
                "uri": s3_uri,
                "bucketOwner": BUCKET_OWNER,
            }
        },
    }
    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    raw_body = response["body"].read()
    if isinstance(raw_body, (bytes, bytearray)):
        raw_body = raw_body.decode("utf-8")
    return json.loads(raw_body)


def verify_segments(detections: list[dict[str, Any]], *, s3_uri: str) -> list[dict[str, Any]]:
    """
    Ask Pegasus to confirm damage at representative timestamps for each segment.
    Keeps segments where >= VERIFY_MIN_HITS timestamps are confirmed as damage and avg confidence is high enough.
    """
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

    result = _bedrock_pegasus_invoke(verify_prompt, s3_uri=s3_uri)
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

    # If verification didn't return enough structured rows to meaningfully evaluate checks,
    # don't "wipe" detections entirely. This commonly happens when the verify response isn't JSON.
    expected_rows = len(detections) * 3
    if detections and not kept and len(rows) < max(1, expected_rows // 2):
        return detections

    return kept


def fetch_json(url: str, *, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=64)
def geocode_street(query: str) -> dict[str, Any]:
    payload = fetch_json(
        NOMINATIM_URL,
        params={"q": query, "format": "jsonv2", "polygon_geojson": 1, "limit": 1},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    if not payload:
        raise ValueError(f"No street match for '{query}'.")

    item = payload[0]
    geometry = item.get("geojson")
    if not geometry:
        bbox = item.get("boundingbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"No usable geometry for '{query}'.")
        south, north, west, east = map(float, bbox)
        geometry = {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        }

    item["geometry"] = geometry
    return item


def geometry_lines(geometry: dict[str, Any]) -> list[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "LineString":
        return [coordinates]
    if geometry_type == "MultiLineString":
        return list(coordinates)
    if geometry_type == "Polygon":
        return [coordinates[0]]
    if geometry_type == "MultiPolygon":
        return [polygon[0] for polygon in coordinates if polygon]
    if geometry_type == "Point":
        lon, lat = coordinates
        delta = 0.00005
        return [[[lon - delta, lat - delta], [lon + delta, lat + delta]]]

    raise ValueError(f"Unsupported geometry type: {geometry_type}")


def project_point(lon: float, lat: float, lat0: float) -> tuple[float, float]:
    scale = math.cos(math.radians(lat0))
    return lon * 111_320.0 * scale, lat * 110_540.0


def unproject_point(x: float, y: float, lat0: float) -> tuple[float, float]:
    scale = math.cos(math.radians(lat0))
    if abs(scale) < 1e-9:
        scale = 1e-9
    return x / (111_320.0 * scale), y / 110_540.0


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def sample_line(coords: list[list[float]], spacing_m: float) -> list[tuple[float, float]]:
    if len(coords) < 2:
        lon, lat = coords[0]
        return [(lon, lat)]

    lat0 = sum(point[1] for point in coords) / len(coords)
    projected = [project_point(lon, lat, lat0) for lon, lat in coords]

    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total = 0.0
    for start, end in zip(projected[:-1], projected[1:]):
        segment_length = distance(start, end)
        if segment_length <= 0:
            continue
        segments.append((start, end, segment_length))
        total += segment_length

    if total == 0:
        lon, lat = coords[0]
        return [(lon, lat)]

    count = max(2, int(total / spacing_m) + 1)
    targets = [index * total / (count - 1) for index in range(count)]
    samples: list[tuple[float, float]] = []

    travelled = 0.0
    segment_index = 0
    for target in targets:
        while segment_index < len(segments) and travelled + segments[segment_index][2] < target:
            travelled += segments[segment_index][2]
            segment_index += 1

        if segment_index >= len(segments):
            lon, lat = coords[-1]
            samples.append((lon, lat))
            continue

        start, end, segment_length = segments[segment_index]
        if segment_length == 0:
            continue
        ratio = (target - travelled) / segment_length
        ratio = min(max(ratio, 0.0), 1.0)
        x = start[0] + (end[0] - start[0]) * ratio
        y = start[1] + (end[1] - start[1]) * ratio
        samples.append(unproject_point(x, y, sum(point[1] for point in coords) / len(coords)))

    return samples


def sample_street_points(geometry: dict[str, Any], spacing_m: float, max_points: int) -> list[tuple[float, float]]:
    sampled: list[tuple[float, float]] = []
    for line in geometry_lines(geometry):
        sampled.extend(sample_line(line, spacing_m))

    unique: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for lon, lat in sampled:
        key = (round(lon * 100000), round(lat * 100000))
        if key in seen:
            continue
        seen.add(key)
        unique.append((lon, lat))

    if len(unique) <= max_points:
        return unique

    if max_points <= 1:
        return unique[:1]

    step = (len(unique) - 1) / (max_points - 1)
    return [unique[round(index * step)] for index in range(max_points)]


def fetch_mapillary_images(token: str, *, lat: float, lng: float, radius: int, limit: int) -> list[dict[str, Any]]:
    payload = fetch_json(
        MAPILLARY_IMAGES_URL,
        params={
            "lat": lat,
            "lng": lng,
            "radius": radius,
            "limit": limit,
            "fields": "id,thumb_256_url,thumb_1024_url,captured_at,creator,geometry,is_pano",
        },
        headers={"Authorization": f"OAuth {token}", "Accept": "application/json"},
    )
    return payload.get("data", [])


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bbox_from_center_radius(lat: float, lng: float, radius_m: float) -> tuple[float, float, float, float]:
    delta_lat = radius_m / 110_540.0
    lon_scale = max(1e-9, 111_320.0 * math.cos(math.radians(lat)))
    delta_lon = radius_m / lon_scale
    return (lng - delta_lon, lat - delta_lat, lng + delta_lon, lat + delta_lat)


def fetch_mapillary_images_bbox(
    token: str,
    *,
    left: float,
    bottom: float,
    right: float,
    top: float,
    limit: int,
) -> list[dict[str, Any]]:
    bbox_value = f"{left:.8f},{bottom:.8f},{right:.8f},{top:.8f}"
    payload = fetch_json(
        MAPILLARY_IMAGES_URL,
        params={
            "bbox": bbox_value,
            "limit": min(max(limit, 1), 2000),
            "is_pano": "false",
            "fields": "id,thumb_256_url,thumb_1024_url,captured_at,creator,geometry,is_pano",
        },
        headers={"Authorization": f"OAuth {token}", "Accept": "application/json"},
    )
    return payload.get("data", [])


def collect_images_in_radius(
    token: str,
    *,
    center_lat: float,
    center_lng: float,
    search_radius_m: float,
    max_images: int,
) -> list[dict[str, Any]]:
    left, bottom, right, top = bbox_from_center_radius(center_lat, center_lng, search_radius_m)
    batch = fetch_mapillary_images_bbox(
        token,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        limit=max_images,
    )

    images: dict[str, dict[str, Any]] = {}
    for image in batch:
        if image.get("is_pano") is not False:
            continue
        image_id = image.get("id")
        if not image_id:
            continue
        coords = geometry_coords(image)
        if coords is None:
            continue
        lon, lat = coords
        distance_m = haversine_m(center_lat, center_lng, lat, lon)
        if distance_m > search_radius_m:
            continue

        image["distance_from_center_m"] = round(distance_m, 2)
        images[image_id] = image

    def sort_key(image: dict[str, Any]) -> tuple[float, int, str]:
        distance_m = float(image.get("distance_from_center_m") or 0.0)
        captured_at = int(image.get("captured_at") or 0)
        return (distance_m, -captured_at, str(image.get("id", "")))

    return sorted(images.values(), key=sort_key)


def collect_images(
    token: str,
    points: list[tuple[float, float]],
    *,
    radius: int,
    per_point_limit: int,
    max_images: int,
) -> list[dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    for lon, lat in points:
        batch = fetch_mapillary_images(token, lat=lat, lng=lon, radius=radius, limit=per_point_limit)
        for image in batch:
            if image.get("is_pano") is not False:
                continue
            image_id = image.get("id")
            if not image_id:
                continue
            images[image_id] = image
            if len(images) >= max_images:
                break
        if len(images) >= max_images:
            break

    def sort_key(image: dict[str, Any]) -> tuple[int, str]:
        captured_at = int(image.get("captured_at") or 0)
        return (-captured_at, str(image.get("id", "")))

    return sorted(images.values(), key=sort_key)


def format_timestamp(value: Any) -> str:
    if not value:
        return "Unknown"
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return "Unknown"


def image_url(image: dict[str, Any]) -> str | None:
    return image.get("thumb_1024_url") or image.get("thumb_256_url")


def mapillary_token() -> str:
    return os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def geometry_coords(image: dict[str, Any]) -> tuple[float, float] | None:
    geometry = image.get("geometry")
    if not isinstance(geometry, dict):
        return None
    coords = geometry.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    try:
        return float(coords[0]), float(coords[1])
    except (TypeError, ValueError):
        return None


def bbox_for_points(points: list[tuple[float, float]], padding_m: float) -> tuple[float, float, float, float]:
    lons = [lon for lon, _ in points]
    lats = [lat for _, lat in points]

    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lat_ref = sum(lats) / len(lats)

    pad_lat = padding_m / 110_540.0
    lon_scale = max(1e-9, 111_320.0 * math.cos(math.radians(lat_ref)))
    pad_lon = padding_m / lon_scale

    return min_lon - pad_lon, min_lat - pad_lat, max_lon + pad_lon, max_lat + pad_lat


def run_overture_download(bounds: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="overture_") as temp_dir:
        output_path = Path(temp_dir) / "segments.geojson"
        bbox_arg = ",".join(f"{value:.8f}" for value in bounds)
        command = [
            sys.executable,
            "-m",
            "overturemaps.cli",
            "download",
            f"--bbox={bbox_arg}",
            "-f",
            "geojson",
            "--type=segment",
            "-o",
            str(output_path),
        ]
        cli_env = os.environ.copy()
        cli_env["PYTHONUTF8"] = "1"
        cli_env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=cli_env)
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            message = stderr or stdout or "overturemaps download failed"
            raise RuntimeError(message)

        if not output_path.exists():
            return []

        data = json.loads(output_path.read_text(encoding="utf-8"))
        features = data.get("features", []) if isinstance(data, dict) else []
        return features if isinstance(features, list) else []


def is_sidewalk_segment(feature: dict[str, Any]) -> bool:
    properties = feature.get("properties")
    if not isinstance(properties, dict):
        return False

    if properties.get("subtype") != "road":
        return False

    class_value = str(properties.get("class", "")).lower()
    subclass_values: list[str] = []
    subclass_rules = properties.get("subclass_rules")
    if isinstance(subclass_rules, list):
        for rule in subclass_rules:
            if isinstance(rule, dict):
                subclass_values.append(str(rule.get("value", "")).lower())
    tags = " ".join([class_value, *subclass_values])

    sidewalk_keys = {
        "sidewalk",
        "footway",
        "pedestrian",
        "crossing",
        "steps",
        "path",
    }
    return any(keyword in tags for keyword in sidewalk_keys)


def is_road_segment(feature: dict[str, Any]) -> bool:
    properties = feature.get("properties")
    if not isinstance(properties, dict):
        return False
    return properties.get("subtype") == "road"


def geometry_linestrings(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "LineString" and isinstance(coordinates, list):
        return [[(float(pt[0]), float(pt[1])) for pt in coordinates if isinstance(pt, list) and len(pt) >= 2]]

    if geometry_type == "MultiLineString" and isinstance(coordinates, list):
        lines: list[list[tuple[float, float]]] = []
        for line in coordinates:
            if not isinstance(line, list):
                continue
            lines.append([(float(pt[0]), float(pt[1])) for pt in line if isinstance(pt, list) and len(pt) >= 2])
        return lines

    return []


def point_segment_distance_m(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    lon, lat = point
    ax, ay = a
    bx, by = b
    lat_ref = (lat + ay + by) / 3.0
    scale_x = 111_320.0 * max(1e-9, math.cos(math.radians(lat_ref)))
    scale_y = 110_540.0

    px, py = lon * scale_x, lat * scale_y
    x1, y1 = ax * scale_x, ay * scale_y
    x2, y2 = bx * scale_x, by * scale_y

    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy)


def point_linestring_distance_m(point: tuple[float, float], line: list[tuple[float, float]]) -> float:
    if len(line) < 2:
        return float("inf")
    return min(point_segment_distance_m(point, line[index], line[index + 1]) for index in range(len(line) - 1))


def nearest_sidewalk_match(
    point: tuple[float, float],
    sidewalk_features: list[dict[str, Any]],
    max_match_m: float,
) -> tuple[str | None, float | None]:
    best_id: str | None = None
    best_distance = float("inf")

    for feature in sidewalk_features:
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        feature_id = properties.get("id") or feature.get("id")
        geometry = feature.get("geometry")
        if not feature_id or not isinstance(geometry, dict):
            continue

        for line in geometry_linestrings(geometry):
            distance_m = point_linestring_distance_m(point, line)
            if distance_m < best_distance:
                best_distance = distance_m
                best_id = str(feature_id)

    if best_id is None or best_distance > max_match_m:
        return None, None

    return best_id, round(best_distance, 2)


def enrich_images_with_sidewalk_gers(images: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str | None]:
    image_points: list[tuple[float, float]] = []
    for image in images:
        coords = geometry_coords(image)
        if coords is not None:
            image_points.append(coords)

    if not image_points:
        return images, "skipped", "No valid image coordinates for sidewalk lookup."

    try:
        padding_m = as_float(os.getenv("OVERTURE_LOOKUP_PADDING_M"), DEFAULT_OVERTURE_PADDING_M)
        max_match_m = as_float(os.getenv("OVERTURE_LOOKUP_MAX_MATCH_M"), DEFAULT_OVERTURE_MAX_MATCH_M)
        bbox = bbox_for_points(image_points, padding_m)
        features = run_overture_download(bbox)
        sidewalk_features = [feature for feature in features if is_sidewalk_segment(feature)]
        road_features = [feature for feature in features if is_road_segment(feature)]

        for image in images:
            coords = geometry_coords(image)
            if coords is None:
                image["nearest_sidewalk_gers_id"] = None
                image["nearest_sidewalk_distance_m"] = None
                image["nearest_sidewalk_strategy"] = "none"
                continue

            sidewalk_id, sidewalk_distance = nearest_sidewalk_match(coords, sidewalk_features, max_match_m)
            strategy = "sidewalk"
            if sidewalk_id is None and road_features:
                sidewalk_id, sidewalk_distance = nearest_sidewalk_match(coords, road_features, max_match_m)
                strategy = "road_fallback" if sidewalk_id is not None else "none"
            elif sidewalk_id is None:
                strategy = "none"

            image["nearest_sidewalk_gers_id"] = sidewalk_id
            image["nearest_sidewalk_distance_m"] = sidewalk_distance
            image["nearest_sidewalk_strategy"] = strategy

        return images, "ok", None
    except Exception as exc:
        for image in images:
            image["nearest_sidewalk_gers_id"] = None
            image["nearest_sidewalk_distance_m"] = None
        return images, "error", str(exc)


def _run_analysis_job(run_id: str, video_path: str) -> None:
    try:
        sequence_id = os.path.splitext(os.path.basename(video_path))[0]

        # Step 1: Upload to S3
        s3_key = f"videos/{os.path.basename(video_path)}"
        s3.upload_file(video_path, BUCKET_NAME, s3_key)
        s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

        # Step 2: Detect segments (improved prompt + verification from `app_copyy.py`).
        result = _bedrock_pegasus_invoke(PEGASUS_PROMPT, s3_uri=s3_uri)
        segments = _parse_pegasus_segments(result)
        segments = normalize_and_merge_detections(segments)
        segments = [seg for seg in segments if float(seg.get("confidence") or 0.0) >= MIN_SEGMENT_CONFIDENCE]

        if VERIFY_SEGMENTS and segments:
            segments = verify_segments(segments, s3_uri=s3_uri)

        # Expand segments to per-frame issue strings (keep one best segment per frame).
        severity_rank = {"low": 0, "medium": 1, "high": 2}
        best_by_frame: dict[int, dict[str, Any]] = {}
        for seg in segments:
            start = int(seg["start"])
            end = int(seg["end"])
            for t in range(start, end + 1):
                if t < 0:
                    continue
                existing = best_by_frame.get(t)
                if existing is None:
                    best_by_frame[t] = seg
                    continue
                a = severity_rank.get(str(seg.get("severity")), 1)
                b = severity_rank.get(str(existing.get("severity")), 1)
                if a > b:
                    best_by_frame[t] = seg
                    continue
                if a == b and float(seg.get("confidence") or 0.0) > float(existing.get("confidence") or 0.0):
                    best_by_frame[t] = seg

        # Step 5: Load track CSV (GPS per frame)
        csv_path = APP_ROOT / "csvs" / f"{sequence_id}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Track CSV not found: {csv_path.relative_to(APP_ROOT)}")
        df = pd.read_csv(csv_path)

        # Step 6: Download Overture data
        import overturemaps
        import geopandas as gpd
        from shapely.geometry import Point

        bbox = (
            df['long'].min() - 0.001,
            df['lat'].min() - 0.001,
            df['long'].max() + 0.001,
            df['lat'].max() + 0.001
        )

        table = overturemaps.record_batch_reader("segment", bbox).read_all()
        # GeoPandas API mismatch: some versions don't ship GeoDataFrame.from_arrow()
        if hasattr(gpd.GeoDataFrame, "from_arrow"):
            gdf = gpd.GeoDataFrame.from_arrow(table)
        else:
            pdf = table.to_pandas()
            geometry = pdf.get("geometry")
            if geometry is not None:
                # If geometry is WKB bytes, decode to shapely geometries.
                if len(geometry) > 0 and isinstance(geometry.iloc[0], (bytes, bytearray, memoryview)):
                    try:
                        from shapely import from_wkb  # type: ignore[attr-defined]

                        geometry = geometry.apply(lambda v: None if v is None else from_wkb(bytes(v)))
                    except Exception:
                        import shapely.wkb  # type: ignore

                        geometry = geometry.apply(lambda v: None if v is None else shapely.wkb.loads(bytes(v)))
                gdf = gpd.GeoDataFrame(pdf, geometry=geometry)
            else:
                gdf = gpd.GeoDataFrame(pdf)

        sidewalk_keys = {'sidewalk', 'footway', 'pedestrian', 'path'}

        # Step 7: Build results (preserve existing output shape: `issues` string per frame).
        frame_issues = {i: [] for i in range(len(df))}
        for t, seg in best_by_frame.items():
            if 0 <= int(t) < len(df):
                desc = f"{seg.get('severity', '')}: {seg.get('description', '')}".strip(": ")
                if desc:
                    frame_issues[int(t)].append(desc)

        # Step 7a: Nearest Overture match per frame (compute once; reuse for rows + GeoJSON export).
        nearest_rows: list[dict[str, Any]] = []
        for i in range(len(df)):
            row = df.iloc[i]
            p = Point(row["long"], row["lat"])

            gdf["dist"] = gdf.geometry.distance(p)
            is_sidewalk = gdf["subclass"].isin(sidewalk_keys)
            sidewalks = gdf[is_sidewalk & (gdf["dist"] < 0.00015)]

            if not sidewalks.empty:
                best = sidewalks.loc[sidewalks["dist"].idxmin()]
            else:
                best = gdf.loc[gdf["dist"].idxmin()]

            nearest_rows.append({"i": i, "df_row": row, "best": best})

        gers_has_issue: dict[str, bool] = {}
        for item in nearest_rows:
            issues = " | ".join(frame_issues[item["i"]]) if frame_issues[item["i"]] else ""
            gers_id = str(item["best"].get("id") or "")
            if issues.strip():
                gers_has_issue[gers_id] = True

        rows = []
        gers_feature_map: dict[str, dict[str, Any]] = {}
        for item in nearest_rows:
            i = item["i"]
            row = item["df_row"]
            best = item["best"]

            issues = " | ".join(frame_issues[i]) if frame_issues[i] else ""
            gers_id = str(best.get("id") or "")

            if gers_id and gers_id not in gers_feature_map and shapely_mapping is not None:
                geom = getattr(best, "geometry", None)
                if geom is not None:
                    try:
                        gers_feature_map[gers_id] = {
                            "type": "Feature",
                            "geometry": shapely_mapping(geom),
                            "properties": {
                                "gers_id": gers_id,
                                "has_issue": bool(gers_has_issue.get(gers_id)),
                            },
                        }
                    except Exception:
                        pass

            rows.append(
                {
                    "frame": i,
                    "lat": row["lat"],
                    "lon": row["long"],
                    "gers_id": gers_id,
                    "segment_type": (best.get("subclass", "") or "").strip() or "footway",
                    "issues": issues,
                    "has_issue": 1 if issues.strip() else 0,
                }
            )

        for feature in gers_feature_map.values():
            props = feature.get("properties")
            if isinstance(props, dict):
                gid = str(props.get("gers_id") or "")
                props["has_issue"] = bool(gers_has_issue.get(gid))

        results = rows
        gers_geojson = {"type": "FeatureCollection", "features": list(gers_feature_map.values())}

        out_csv, out_gers = results_artifact_paths(sequence_id)
        pd.DataFrame(results).to_csv(out_csv, index=False)
        out_gers.write_text(json.dumps(gers_geojson), encoding="utf-8")

        with PIPELINE_LOCK:
            PIPELINE_RUNS[run_id]["status"] = "completed"
            PIPELINE_RUNS[run_id]["results"] = results
            PIPELINE_RUNS[run_id]["gers_geojson"] = gers_geojson
            PIPELINE_RUNS[run_id]["saved_csv"] = str(out_csv.relative_to(APP_ROOT))

    except Exception as exc:
        with PIPELINE_LOCK:
            PIPELINE_RUNS[run_id]["status"] = "error"
            PIPELINE_RUNS[run_id]["error"] = str(exc)


@app.get("/api/videos")
def api_videos():
    items: list[dict[str, Any]] = []
    for f in sorted(APP_ROOT.glob("*.mp4"), key=lambda p: p.name.lower()):
        out_csv, _ = results_artifact_paths(f.stem)
        rel = str(f.relative_to(APP_ROOT))
        items.append({"path": rel, "has_saved_results": out_csv.is_file()})
    return jsonify({"videos": items})


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}
    video_path = str(payload.get("video_path") or "").strip()
    resolved = resolve_project_media_path(video_path)
    if not resolved:
        return jsonify({"error": "Invalid video path"}), 400

    run_id = uuid.uuid4().hex[:12]
    with PIPELINE_LOCK:
        PIPELINE_RUNS[run_id] = {"status": "running", "created_at": datetime.now(tz=timezone.utc).isoformat()}

    thread = threading.Thread(
        target=_run_analysis_job,
        args=(run_id, str(resolved)),
        daemon=True,
    )
    thread.start()
    return jsonify({"run_id": run_id, "status": "running"})


@app.post("/api/load-saved-results")
def api_load_saved_results():
    """Load a prior run from results_<sequence_id>.csv (+ optional .gers.geojson) into memory."""
    payload = request.get_json(silent=True) or {}
    video_path = str(payload.get("video_path") or "").strip()
    resolved = resolve_project_media_path(video_path)
    if not resolved:
        return jsonify({"error": "Invalid video path"}), 400

    sequence_id = resolved.stem
    out_csv, out_gers = results_artifact_paths(sequence_id)
    if not out_csv.is_file():
        return jsonify({"error": f"No saved results for this sequence ({out_csv.name} missing)."}), 404

    try:
        df = pd.read_csv(out_csv).fillna("")
        # `to_json` ensures native Python scalars (CSV read often yields numpy types).
        results = json.loads(df.to_json(orient="records"))
        if not isinstance(results, list):
            raise ValueError("unexpected records shape")
    except Exception as exc:
        return jsonify({"error": f"Failed to read results CSV: {exc}"}), 500

    gers_geojson: dict[str, Any]
    if out_gers.is_file():
        try:
            gers_geojson = json.loads(out_gers.read_text(encoding="utf-8"))
            if not isinstance(gers_geojson, dict):
                raise ValueError("GeoJSON root must be an object")
        except Exception as exc:
            return jsonify({"error": f"Invalid GERS GeoJSON sidecar: {exc}"}), 500
    else:
        gers_geojson = {"type": "FeatureCollection", "features": []}

    run_id = uuid.uuid4().hex[:12]
    with PIPELINE_LOCK:
        PIPELINE_RUNS[run_id] = {
            "status": "completed",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "results": results,
            "gers_geojson": gers_geojson,
            "source": "saved_csv",
            "saved_csv": str(out_csv.relative_to(APP_ROOT)),
        }

    return jsonify({"run_id": run_id, "status": "completed"})


@app.get("/api/status/<run_id>")
def api_status(run_id: str):
    with PIPELINE_LOCK:
        run = PIPELINE_RUNS.get(run_id)
    if not run:
        return jsonify({"error": "run not found"}), 404
    return jsonify({"run_id": run_id, **run})


@app.get("/api/results/<run_id>")
def api_results(run_id: str):
    with PIPELINE_LOCK:
        run = PIPELINE_RUNS.get(run_id)
    if not run:
        return jsonify({"error": "run not found"}), 404
    if run.get("status") != "completed":
        return jsonify({"error": "run not completed", "status": run.get("status")}), 409
    return jsonify(run.get("results") or [])


@app.get("/api/overture-explorer-link/<path:gers_id>")
def api_overture_explorer_link(gers_id: str):
    theme = (request.args.get("theme") or "transportation").strip() or "transportation"
    otype = (request.args.get("otype") or "segment").strip() or "segment"
    try:
        zoom = float(request.args.get("zoom") or 16.75)
    except (TypeError, ValueError):
        zoom = 16.75

    try:
        link = get_overture_explorer_link_via_api(gers_id, theme=theme, otype=otype, zoom=zoom)
        return jsonify({"gers_id": (gers_id or "").strip(), "link": link})
    except requests.RequestException as exc:
        return jsonify({"error": f"API lookup failed: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/gers-geojson/<run_id>")
def api_gers_geojson(run_id: str):
    with PIPELINE_LOCK:
        run = PIPELINE_RUNS.get(run_id)
    if not run:
        return jsonify({"error": "run not found"}), 404
    if run.get("status") != "completed":
        return jsonify({"error": "run not completed", "status": run.get("status")}), 409
    return jsonify(run.get("gers_geojson") or {"type": "FeatureCollection", "features": []})


@app.get("/api/overture-sidewalks")
def api_overture_sidewalks():
    try:
        left = float(request.args.get("left"))
        bottom = float(request.args.get("bottom"))
        right = float(request.args.get("right"))
        top = float(request.args.get("top"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid bounds"}), 400

    try:
        features = run_overture_download((left, bottom, right, top))
        sidewalk_features = [f for f in features if is_sidewalk_segment(f)]
        geojson = {
            "type": "FeatureCollection",
            "features": sidewalk_features
        }
        return jsonify(geojson)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/csv-files")
def api_csv_files():
    csv_files = sorted(p.name for p in APP_ROOT.glob("results_*.csv"))
    return jsonify({"csv_files": csv_files})


@app.get("/api/load-csv/<filename>")
def api_load_csv(filename):
    if not filename.startswith('results_') or not filename.endswith('.csv'):
        return jsonify({"error": "Invalid filename"}), 400

    path = (APP_ROOT / filename).resolve()
    try:
        path.relative_to(APP_ROOT)
    except ValueError:
        return jsonify({"error": "Invalid filename"}), 400

    if not path.is_file():
        return jsonify({"error": "File not found"}), 404

    try:
        df = pd.read_csv(path)
        df = df.fillna("")
        data = df.to_dict("records")
        return jsonify({"data": data})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/")
def index() -> str:
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
    import json
    return render_template("index.html", mapbox_token_json=json.dumps(mapbox_token))


@app.get("/pipeline")
def pipeline_page() -> str:
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
    return render_template("pipeline.html", mapbox_token=mapbox_token)


def _run_pipeline_job(run_id: str, *, video_path: str, csv_path: str | None, out_dir: str) -> None:
    try:
        from sidewalk_analyzer.config import load_settings
        from sidewalk_analyzer.pipeline import run_pipeline

        settings = load_settings(env_path=Path(__file__).with_name(".env"))
        manifest = run_pipeline(
            video_path=Path(video_path),
            csv_path=Path(csv_path) if csv_path else None,
            out_dir=Path(out_dir),
            run_id=run_id,
            settings=settings,
            no_s3=False,
            skip_marengo=False,
            threshold_override=None,
        )
        with PIPELINE_LOCK:
            PIPELINE_RUNS[run_id]["status"] = "completed"
            PIPELINE_RUNS[run_id]["manifest"] = manifest
    except Exception as exc:
        with PIPELINE_LOCK:
            PIPELINE_RUNS[run_id]["status"] = "error"
            PIPELINE_RUNS[run_id]["error"] = str(exc)


@app.post("/api/pipeline/run")
def api_pipeline_run():
    payload = request.get_json(silent=True) or {}
    video_path = str(payload.get("video_path") or "").strip()
    csv_path = str(payload.get("csv_path") or "").strip() or None
    out_dir = str(payload.get("out_dir") or "outputs").strip()
    if not video_path:
        return jsonify({"error": "video_path required"}), 400

    run_id = payload.get("run_id") or uuid.uuid4().hex[:12]
    with PIPELINE_LOCK:
        PIPELINE_RUNS[run_id] = {"status": "running", "created_at": datetime.now(tz=timezone.utc).isoformat()}

    thread = threading.Thread(
        target=_run_pipeline_job,
        kwargs={"run_id": run_id, "video_path": video_path, "csv_path": csv_path, "out_dir": out_dir},
        daemon=True,
    )
    thread.start()
    return jsonify({"run_id": run_id, "status": "running"})


@app.get("/api/pipeline/status/<run_id>")
def api_pipeline_status(run_id: str):
    with PIPELINE_LOCK:
        run = PIPELINE_RUNS.get(run_id)
    if not run:
        return jsonify({"error": "run not found"}), 404
    return jsonify({"run_id": run_id, **run})


@app.get("/api/pipeline/result/<run_id>")
def api_pipeline_result(run_id: str):
    with PIPELINE_LOCK:
        run = PIPELINE_RUNS.get(run_id)
    if not run:
        return jsonify({"error": "run not found"}), 404
    if run.get("status") != "completed":
        return jsonify({"error": "run not completed", "status": run.get("status")}), 409
    return jsonify(run.get("manifest") or {})


@app.get("/api/pipeline/geojson/<run_id>")
def api_pipeline_geojson(run_id: str):
    with PIPELINE_LOCK:
        run = PIPELINE_RUNS.get(run_id)
    if not run or run.get("status") != "completed":
        return jsonify({"error": "run not completed"}), 404
    manifest = run.get("manifest") or {}
    outputs = manifest.get("outputs") or {}
    geojson_path = outputs.get("geojson")
    if not geojson_path:
        return jsonify({"error": "geojson not available"}), 404
    return send_file(geojson_path, mimetype="application/geo+json")

@app.post("/api/search")
def api_search():
    payload = request.get_json(silent=True) or {}

    try:
        center_lat = float(payload.get("center_lat"))
        center_lng = float(payload.get("center_lng"))
    except (TypeError, ValueError):
        return jsonify({"error": "center_lat and center_lng required."}), 400

    search_radius_m = as_float(payload.get("search_radius_m"), 100.0)
    if search_radius_m <= 0:
        return jsonify({"error": "search_radius_m must be > 0."}), 400

    token = mapillary_token()
    if not token:
        return jsonify({"error": "Mapillary token missing in .env."}), 400

    try:
        max_images = int(payload.get("max_images", 300))
        images = collect_images_in_radius(
            token,
            center_lat=center_lat,
            center_lng=center_lng,
            search_radius_m=search_radius_m,
            max_images=max_images,
        )
        images, overture_status, overture_error = enrich_images_with_sidewalk_gers(images)

        return jsonify(
            {
                "center_lat": center_lat,
                "center_lng": center_lng,
                "search_radius_m": search_radius_m,
                "images_count": len(images),
                "overture_status": overture_status,
                "overture_error": overture_error,
                "images": [
                    {
                        "id": image.get("id"),
                        "thumb_url": image_url(image),
                        "creator": image.get("creator", {}).get("username") if isinstance(image.get("creator"), dict) else "Unknown creator",
                        "captured_at_label": format_timestamp(image.get("captured_at")),
                        "longitude": geometry_coords(image)[0] if geometry_coords(image) is not None else None,
                        "latitude": geometry_coords(image)[1] if geometry_coords(image) is not None else None,
                        "distance_from_center_m": image.get("distance_from_center_m"),
                        "nearest_sidewalk_gers_id": image.get("nearest_sidewalk_gers_id"),
                        "nearest_sidewalk_distance_m": image.get("nearest_sidewalk_distance_m"),
                        "nearest_sidewalk_strategy": image.get("nearest_sidewalk_strategy"),
                    }
                    for image in images
                ],
                "status": "found" if images else "empty",
            }
        )
    except requests.HTTPError as exc:
        return jsonify({"error": f"API error: {exc}"}), 502
    except requests.RequestException as exc:
        return jsonify({"error": f"Network error: {exc}"}), 502
    except ValueError as exc:
        return jsonify({"error": str(exc), "status": "not_found"}), 404


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True)