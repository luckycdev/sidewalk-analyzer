from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import requests


USER_AGENT = "sidewalk-analyzer/1.0 (local flask app)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
MAPILLARY_IMAGES_URL = "https://graph.mapillary.com/images"


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


def mapillary_token() -> str:
    return os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()


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


def format_timestamp(value: Any) -> str:
    if not value:
        return "Unknown"
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return "Unknown"


def image_url(image: dict[str, Any]) -> str | None:
    return image.get("thumb_1024_url") or image.get("thumb_256_url")

