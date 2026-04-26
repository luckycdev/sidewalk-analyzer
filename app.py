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
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


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
    return json.loads(response["body"].read())


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
    raw_message = result.get("message", "")
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
        raw_message = result.get("message", "")
        segments = _extract_json_array(raw_message)
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

        # Step 5: Load CSV
        csv_path = os.path.join("csvs", f"{sequence_id}.csv")
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

        rows = []
        # Export unique matched Overture geometries so the UI can draw the true GERS lines.
        gers_feature_map: dict[str, dict[str, Any]] = {}
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

            issues = " | ".join(frame_issues[i]) if frame_issues[i] else ""
            gers_id = str(best.get("id") or "")

            if gers_id and gers_id not in gers_feature_map and shapely_mapping is not None:
                geom = getattr(best, "geometry", None)
                if geom is not None:
                    try:
                        gers_feature_map[gers_id] = {
                            "type": "Feature",
                            "geometry": shapely_mapping(geom),
                            "properties": {"gers_id": gers_id},
                        }
                    except Exception:
                        # If mapping fails for any geometry, just skip exporting it.
                        pass

            rows.append({
                "frame": i,
                "lat": row["lat"],
                "lon": row["long"],
                "gers_id": gers_id,
                "segment_type": best.get("subclass", ""),
                "issues": issues
            })

        results = rows
        gers_geojson = {"type": "FeatureCollection", "features": list(gers_feature_map.values())}

        with PIPELINE_LOCK:
            PIPELINE_RUNS[run_id]["status"] = "completed"
            PIPELINE_RUNS[run_id]["results"] = results
            PIPELINE_RUNS[run_id]["gers_geojson"] = gers_geojson

    except Exception as exc:
        with PIPELINE_LOCK:
            PIPELINE_RUNS[run_id]["status"] = "error"
            PIPELINE_RUNS[run_id]["error"] = str(exc)


@app.get("/api/videos")
def api_videos():
    videos = [f for f in os.listdir('.') if f.endswith('.mp4')]
    return jsonify({"videos": videos})


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}
    video_path = str(payload.get("video_path") or "").strip()
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    run_id = uuid.uuid4().hex[:12]
    with PIPELINE_LOCK:
        PIPELINE_RUNS[run_id] = {"status": "running", "created_at": datetime.now(tz=timezone.utc).isoformat()}

    thread = threading.Thread(
        target=_run_analysis_job,
        args=(run_id, video_path),
        daemon=True,
    )
    thread.start()
    return jsonify({"run_id": run_id, "status": "running"})


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
    csv_files = [f for f in os.listdir('.') if f.startswith('results_') and f.endswith('.csv')]
    return jsonify({"csv_files": csv_files})


@app.get("/api/load-csv/<filename>")
def api_load_csv(filename):
    if not filename.startswith('results_') or not filename.endswith('.csv'):
        return jsonify({"error": "Invalid filename"}), 400
    
    try:
        df = pd.read_csv(filename)
        # Replace NaN with empty strings for JSON compatibility
        df = df.fillna('')
        data = df.to_dict('records')
        return jsonify({"data": data})
    except Exception as exc:
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