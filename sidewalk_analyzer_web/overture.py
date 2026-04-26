from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests

DEFAULT_OVERTURE_PADDING_M = 120.0
DEFAULT_OVERTURE_MAX_MATCH_M = 220.0

OVERTURE_ID_LOOKUP_URL = "https://geocoder.bradr.dev/id/{gers_id}"


def get_overture_explorer_link_via_api(
    gers_id: str, *, theme: str = "transportation", otype: str = "segment", zoom: float = 16.75
) -> str:
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


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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

    sidewalk_keys = {"sidewalk", "footway", "pedestrian", "crossing", "steps", "path"}
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
    from .mapillary import geometry_coords

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

