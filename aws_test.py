import boto3
import json
import pandas as pd
import requests
import time
import os
import re
from typing import Any

# ---------------- CONFIG ----------------
BUCKET_NAME = "sidewalk-analyzer-dev"
CSV_DIR = "./csvs/"
REGION = "us-east-1"
MODEL_ID = "twelvelabs.pegasus-1-2-v1:0"
BUCKET_OWNER = "652992947084"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
FPS_ASSUMPTION = 1  # CSV row index == second timestamp when video is 1 FPS

# Tuning knobs (can also be set via env vars)
MIN_SEGMENT_CONFIDENCE = float(os.getenv("MIN_SEGMENT_CONFIDENCE", "0.7"))
VERIFY_SEGMENTS = os.getenv("VERIFY_SEGMENTS", "1").strip() not in {"0", "false", "no"}
VERIFY_MIN_HITS = int(os.getenv("VERIFY_MIN_HITS", "2"))  # out of 3 (start/mid/end)
VERIFY_MIN_AVG_CONF = float(os.getenv("VERIFY_MIN_AVG_CONF", "0.7"))

# Optional GERS enrichment (Overture segments)
USE_OVERTURE_GERS = os.getenv("USE_OVERTURE_GERS", "1").strip() not in {"0", "false", "no"}
OVERTURE_BBOX_PADDING_DEG = float(os.getenv("OVERTURE_BBOX_PADDING_DEG", "0.001"))
OVERTURE_MAX_MATCH_M = float(os.getenv("OVERTURE_MAX_MATCH_M", "25"))

# Output format
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "1"))  # 1 = every frame, 2 = every other frame, etc.
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "1").strip() not in {"0", "false", "no"}
OUTPUT_JSON = os.getenv("OUTPUT_JSON", "0").strip() in {"1", "true", "yes"}

# Optional (slow) OSM Overpass enrichment
USE_OSM_OVERPASS = os.getenv("USE_OSM_OVERPASS", "0").strip() in {"1", "true", "yes"}

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


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
        # merge if overlapping or directly adjacent in time
        if det["start"] <= prev["end"] + 1:
            prev["end"] = max(prev["end"], det["end"])
            if severity_rank[det["severity"]] > severity_rank[prev["severity"]]:
                prev["severity"] = det["severity"]
            prev["confidence"] = max(float(prev["confidence"]), float(det["confidence"]))
            prev_types = set(prev.get("damage_types") or [])
            det_types = set(det.get("damage_types") or [])
            prev["damage_types"] = sorted(prev_types | det_types)
            # keep the more specific description (simple heuristic: longer string)
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

    # Aggregate per segment
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


def _load_overture_segments_for_csv(df: pd.DataFrame):
    """
    Loads Overture 'segment' features for a bbox around the GPS trace.
    Returns (metric_gdf, sidewalk_mask) where geometries are projected to meters (EPSG:3857),
    or (None, None) if overture/geopandas not available.
    """
    try:
        import overturemaps  # type: ignore
        import geopandas as gpd  # type: ignore
    except Exception:
        return None, None

    if "lat" not in df.columns or "long" not in df.columns:
        return None, None

    min_lon = float(df["long"].min()) - OVERTURE_BBOX_PADDING_DEG
    min_lat = float(df["lat"].min()) - OVERTURE_BBOX_PADDING_DEG
    max_lon = float(df["long"].max()) + OVERTURE_BBOX_PADDING_DEG
    max_lat = float(df["lat"].max()) + OVERTURE_BBOX_PADDING_DEG
    bbox = (min_lon, min_lat, max_lon, max_lat)

    table = overturemaps.record_batch_reader("segment", bbox).read_all()
    # geopandas versions vary; from_arrow is not always available
    if hasattr(gpd.GeoDataFrame, "from_arrow"):
        gdf = gpd.GeoDataFrame.from_arrow(table)
    else:
        pdf = table.to_pandas()
        if "geometry" not in pdf.columns:
            return None, None

        geometry = pdf["geometry"]
        # If geometry is WKB bytes, decode to shapely
        try:
            from shapely import from_wkb  # type: ignore[attr-defined]

            if len(geometry) > 0 and isinstance(geometry.iloc[0], (bytes, bytearray, memoryview)):
                geometry = geometry.apply(lambda v: None if v is None else from_wkb(bytes(v)))
        except Exception:
            try:
                import shapely.wkb  # type: ignore

                if len(geometry) > 0 and isinstance(geometry.iloc[0], (bytes, bytearray, memoryview)):
                    geometry = geometry.apply(lambda v: None if v is None else shapely.wkb.loads(bytes(v)))
            except Exception:
                pass

        gdf = gpd.GeoDataFrame(pdf, geometry=geometry)
    if gdf.empty or "geometry" not in gdf.columns:
        return None, None

    # Overture geometries are lon/lat; ensure CRS then project to a meters CRS for accurate distances.
    try:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        metric_gdf = gdf.to_crs("EPSG:3857")
    except Exception:
        return None, None

    # Sidewalk-like classification: be resilient to schema variation by searching multiple fields.
    sidewalk_keywords = ("sidewalk", "footway", "pedestrian", "path", "crossing", "steps")
    text_fields = [col for col in ("subclass", "subtype", "class", "subclass_rules", "names") if col in metric_gdf.columns]
    if not text_fields:
        sidewalk_mask = pd.Series([False] * len(metric_gdf), index=metric_gdf.index)
    else:
        combined = metric_gdf[text_fields].astype(str).agg(" ".join, axis=1).str.lower()
        sidewalk_mask = combined.apply(lambda value: any(keyword in value for keyword in sidewalk_keywords))

    return metric_gdf, sidewalk_mask


def _nearest_overture_segment(point_lon: float, point_lat: float, gdf, sidewalk_mask):
    """
    Returns (gers_id, subclass_value, distance_m) for best match.
    Prefers sidewalk-like segments within OVERTURE_MAX_MATCH_M, else falls back to nearest segment overall.
    """
    try:
        from shapely.geometry import Point  # type: ignore
        from shapely.strtree import STRtree  # type: ignore
    except Exception:
        return None, None, None

    if gdf is None or sidewalk_mask is None or getattr(gdf, "empty", True):
        return None, None, None

    # Project point into the same meters CRS as the segment geometries (EPSG:3857).
    try:
        import geopandas as gpd  # type: ignore
    except Exception:
        return None, None, None

    point_gdf = gpd.GeoSeries([Point(float(point_lon), float(point_lat))], crs="EPSG:4326").to_crs("EPSG:3857")
    p = point_gdf.iloc[0]

    geoms = list(gdf.geometry.values)
    if not geoms:
        return None, None, None

    # Build a spatial index. For typical route bboxes this is fast enough and far more accurate than degree distances.
    tree = STRtree(geoms)
    geom_to_index = {id(geom): idx for idx, geom in zip(gdf.index, geoms)}

    def _as_geoms(items) -> list[Any]:
        """
        STRtree APIs vary by Shapely version:
        - some return geometries
        - some return integer indices (numpy ints)
        Normalize to a list of geometries.
        """
        normalized: list[Any] = []
        for item in items:
            if isinstance(item, (int,)) or str(type(item)).endswith("numpy.int64'>") or str(type(item)).endswith("numpy.int32'>"):
                try:
                    normalized.append(geoms[int(item)])
                except Exception:
                    continue
            else:
                normalized.append(item)
        return normalized

    def pick_best(geometry_list) -> tuple[Any, float] | tuple[None, float]:
        best_idx = None
        best_dist = float("inf")
        for geom in geometry_list:
            if not hasattr(geom, "distance"):
                continue
            dist = float(geom.distance(p))
            if dist < best_dist:
                best_dist = dist
                best_idx = geom_to_index.get(id(geom))
        return best_idx, best_dist

    max_m = float(OVERTURE_MAX_MATCH_M)

    # 1) Prefer sidewalk-like segments within max_m.
    sidewalk_candidates_geoms = []
    try:
        near_geoms = _as_geoms(list(tree.query(p.buffer(max_m))))
        if near_geoms:
            for geom in near_geoms:
                idx = geom_to_index.get(id(geom))
                if idx is None:
                    continue
                if bool(sidewalk_mask.loc[idx]):
                    sidewalk_candidates_geoms.append(geom)
    except Exception:
        sidewalk_candidates_geoms = []

    if sidewalk_candidates_geoms:
        idx, distance_m = pick_best(sidewalk_candidates_geoms)
    else:
        # 2) Otherwise pick nearest overall. Prefer nearest() if available; otherwise expand radius.
        idx = None
        distance_m = float("inf")
        if hasattr(tree, "nearest"):
            try:
                nearest_geom = tree.nearest(p)
                nearest_geom = _as_geoms([nearest_geom])[0] if nearest_geom is not None else None
                if nearest_geom is not None and hasattr(nearest_geom, "distance"):
                    idx = geom_to_index.get(id(nearest_geom))
                    if idx is not None:
                        distance_m = float(nearest_geom.distance(p))
            except Exception:
                idx = None

        if idx is None:
            # Fallback: expand search until we get candidates.
            for radius in (50.0, 100.0, 200.0, 500.0, 1000.0):
                try:
                    cand = _as_geoms(list(tree.query(p.buffer(radius))))
                except Exception:
                    cand = []
                if not cand:
                    continue
                idx, distance_m = pick_best(cand)
                if idx is not None:
                    break

    if idx is None:
        return None, None, None

    best = gdf.loc[idx]
    gers_id = None
    if "id" in getattr(best, "index", []) or (hasattr(best, "get") and best.get("id") is not None):
        try:
            gers_id = str(best.get("id"))
        except Exception:
            gers_id = None

    subclass_value = None
    for candidate in ("subclass", "subtype", "class"):
        if hasattr(best, "get") and best.get(candidate) is not None:
            subclass_value = str(best.get(candidate))
            break

    return gers_id, subclass_value, round(float(distance_m), 2)


def main():
    # 1. Select the video to analyze
    videos = [f for f in os.listdir('.') if f.endswith('.mp4')]
    
    if not videos:
        print("❌ No .mp4 files found in the current directory.")
        return

    print("\n--- Available Videos for Analysis ---")
    for idx, vid in enumerate(videos):
        print(f"[{idx}] {vid}")

    try:
        choice = int(input("\nSelect video to analyze: "))
        local_video = videos[choice]
        sequence_id = local_video.replace('.mp4', '')
    except (ValueError, IndexError):
        print("❌ Invalid selection.")
        return

    # 2. Check for matching CSV
    csv_path = os.path.join(CSV_DIR, f"{sequence_id}.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Required CSV not found at {csv_path}")
        return

    # 3. Upload to S3
    s3_key = f"videos/{local_video}"
    print(f"🚀 Uploading {local_video} to S3...")
    s3.upload_file(local_video, BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    # 4. Call Pegasus
    print(f"🧠 Pegasus is analyzing {sequence_id} (this may take a minute)...")
    result = _bedrock_pegasus_invoke(PEGASUS_PROMPT, s3_uri=s3_uri)
    
    # Pegasus returns the JSON string inside a 'message' field usually
    raw_message = result.get("message", "")
    detections = _extract_json_array(raw_message)
    detections = normalize_and_merge_detections(detections)
    detections = [det for det in detections if float(det.get("confidence") or 0.0) >= MIN_SEGMENT_CONFIDENCE]
    print(f"📍 Detected {len(detections)} segments after merge + confidence filter.")

    if VERIFY_SEGMENTS and detections:
        print("🔎 Verifying segments to reduce false positives...")
        detections = verify_segments(detections, s3_uri=s3_uri)
        print(f"✅ Kept {len(detections)} segments after verification.")

    # 5. Map Detections to GPS via CSV
    df = pd.read_csv(csv_path)
    flagged_points: list[dict[str, Any]] = []

    for det in detections:
        start, end = int(det["start"]), int(det["end"])
        # Map timestamps to CSV rows (assuming 1 FPS)
        for t in range(start, end + 1):
            if t < 0:
                continue
            if t < len(df):
                row = df.iloc[t]
                flagged_points.append({
                    "frame": t,
                    "lat": row["lat"],
                    "long": row["long"],
                    "severity": det["severity"],
                    "damage_types": det.get("damage_types", []),
                    "confidence": det.get("confidence", None),
                    "description": det["description"],
                })

    # De-duplicate per frame (keep highest severity, then highest confidence)
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    by_frame: dict[int, dict[str, Any]] = {}
    for pt in flagged_points:
        frame = int(pt["frame"])
        existing = by_frame.get(frame)
        if existing is None:
            by_frame[frame] = pt
            continue
        a = severity_rank.get(str(pt.get("severity")), 1)
        b = severity_rank.get(str(existing.get("severity")), 1)
        if a > b:
            by_frame[frame] = pt
            continue
        if a == b:
            ca = float(pt.get("confidence") or 0.0)
            cb = float(existing.get("confidence") or 0.0)
            if ca > cb:
                by_frame[frame] = pt

    flagged_points = [by_frame[k] for k in sorted(by_frame.keys())]

    overture_gdf = None
    overture_sidewalk_mask = None
    if USE_OVERTURE_GERS:
        print("🧭 Loading Overture segments for GERS matching...")
        overture_gdf, overture_sidewalk_mask = _load_overture_segments_for_csv(df)
        if overture_gdf is None:
            print("⚠️  Overture GERS enrichment unavailable (missing deps or no data).")

    # 6. Get Sidewalk IDs from OpenStreetMap (Overpass)
    def get_nearest_sidewalk(lat, lon):
        query = f"""
        [out:json];
        (
          way(around:25,{lat},{lon})["highway"="footway"]["footway"="sidewalk"];
          way(around:25,{lat},{lon})["highway"]["sidewalk"];
        );
        out center 1;
        """
        try:
            res = requests.post(OVERPASS_URL, data=query, timeout=10)
            data = res.json()
            if data["elements"]:
                el = data["elements"][0]
                return {"id": el["id"], "lat": el["center"]["lat"], "lon": el["center"]["lon"]}
        except:
            return None
        return None

    # Build per-frame issue lookup (empty if no detections)
    issues_by_frame: dict[int, dict[str, Any]] = {int(p["frame"]): p for p in flagged_points}

    if FRAME_STRIDE < 1:
        stride = 1
    else:
        stride = FRAME_STRIDE

    # Per-frame (or stride) table
    rows: list[dict[str, Any]] = []
    frames = list(range(0, len(df), stride))

    if USE_OSM_OVERPASS and frames:
        print(f"🗺️  Enriching {len(frames)} frames with OSM (slow)...")

    for frame in frames:
        lat = float(df.iloc[frame]["lat"])
        lon = float(df.iloc[frame]["long"])

        gers_id, gers_subclass, gers_distance_m = (None, None, None)
        if overture_gdf is not None:
            gers_id, gers_subclass, gers_distance_m = _nearest_overture_segment(
                point_lon=lon,
                point_lat=lat,
                gdf=overture_gdf,
                sidewalk_mask=overture_sidewalk_mask,
            )

        issue = issues_by_frame.get(frame)
        if issue is None:
            severity = ""
            confidence = ""
            damage_types = ""
            description = ""
            has_issue = 0
        else:
            severity = str(issue.get("severity") or "")
            conf_val = issue.get("confidence")
            confidence = "" if conf_val is None else str(conf_val)
            dt = issue.get("damage_types") or []
            if isinstance(dt, list):
                damage_types = "|".join(str(x) for x in dt)
            else:
                damage_types = str(dt)
            description = str(issue.get("description") or "")
            has_issue = 1

        osm_sidewalk = None
        if USE_OSM_OVERPASS:
            osm_sidewalk = get_nearest_sidewalk(lat, lon)
            time.sleep(0.3)  # Avoid hitting API limits

        rows.append(
            {
                "frame": frame,
                "lat": lat,
                "long": lon,
                "gers_id": gers_id or "",
                "gers_subclass": gers_subclass or "",
                "gers_distance_m": "" if gers_distance_m is None else gers_distance_m,
                "has_issue": has_issue,
                "severity": severity,
                "confidence": confidence,
                "damage_types": damage_types,
                "description": description,
                "osm_sidewalk_id": "" if not osm_sidewalk else osm_sidewalk.get("id", ""),
                "osm_sidewalk_lat": "" if not osm_sidewalk else osm_sidewalk.get("lat", ""),
                "osm_sidewalk_lon": "" if not osm_sidewalk else osm_sidewalk.get("lon", ""),
            }
        )

    # 7. Save results
    if OUTPUT_CSV:
        out_csv = f"results_{sequence_id}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n✅ DONE! Results saved to {out_csv}")

    if OUTPUT_JSON:
        out_json = f"results_{sequence_id}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"✅ Also wrote JSON: {out_json}")

if __name__ == "__main__":
    main()