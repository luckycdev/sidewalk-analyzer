from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import APP_ROOT, results_artifact_paths
from .pegasus import (
    MIN_SEGMENT_CONFIDENCE,
    PEGASUS_PROMPT,
    VERIFY_SEGMENTS,
    invoke as pegasus_invoke,
    normalize_and_merge_detections,
    parse_segments,
    verify_segments,
)


try:
    from shapely.geometry import mapping as shapely_mapping  # type: ignore
except Exception:  # pragma: no cover
    shapely_mapping = None


REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "sidewalk-analyzer-vincent")
_s3_client = None


def run_analysis(video_path: str) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Run upload → Pegasus → Overture join. Returns (results_rows, gers_geojson, saved_csv_relpath)."""
    import pandas as pd

    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3", region_name=REGION)

    sequence_id = os.path.splitext(os.path.basename(video_path))[0]

    s3_key = f"videos/{os.path.basename(video_path)}"
    _s3_client.upload_file(video_path, BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    result = pegasus_invoke(PEGASUS_PROMPT, s3_uri=s3_uri)
    segments = parse_segments(result)
    segments = normalize_and_merge_detections(segments)
    segments = [seg for seg in segments if float(seg.get("confidence") or 0.0) >= MIN_SEGMENT_CONFIDENCE]

    if VERIFY_SEGMENTS and segments:
        segments = verify_segments(segments, s3_uri=s3_uri)

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

    csv_path = APP_ROOT / "csvs" / f"{sequence_id}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Track CSV not found: {csv_path.relative_to(APP_ROOT)}")
    df = pd.read_csv(csv_path)

    import overturemaps
    import geopandas as gpd
    from shapely.geometry import Point

    bbox = (
        float(df["long"].min()) - 0.001,
        float(df["lat"].min()) - 0.001,
        float(df["long"].max()) + 0.001,
        float(df["lat"].max()) + 0.001,
    )

    table = overturemaps.record_batch_reader("segment", bbox).read_all()
    if hasattr(gpd.GeoDataFrame, "from_arrow"):
        gdf = gpd.GeoDataFrame.from_arrow(table)
    else:
        pdf = table.to_pandas()
        geometry = pdf.get("geometry")
        if geometry is not None and len(geometry) > 0 and isinstance(geometry.iloc[0], (bytes, bytearray, memoryview)):
            try:
                from shapely import from_wkb  # type: ignore[attr-defined]

                geometry = geometry.apply(lambda v: None if v is None else from_wkb(bytes(v)))
            except Exception:
                import shapely.wkb  # type: ignore

                geometry = geometry.apply(lambda v: None if v is None else shapely.wkb.loads(bytes(v)))
            gdf = gpd.GeoDataFrame(pdf, geometry=geometry)
        else:
            gdf = gpd.GeoDataFrame(pdf, geometry=geometry) if geometry is not None else gpd.GeoDataFrame(pdf)

    sidewalk_keys = {"sidewalk", "footway", "pedestrian", "path"}

    frame_issues: dict[int, list[str]] = {i: [] for i in range(len(df))}
    for t, seg in best_by_frame.items():
        if 0 <= int(t) < len(df):
            desc = f"{seg.get('severity', '')}: {seg.get('description', '')}".strip(": ")
            if desc:
                frame_issues[int(t)].append(desc)

    nearest_rows: list[dict[str, Any]] = []
    for i in range(len(df)):
        row = df.iloc[i]
        p = Point(row["long"], row["lat"])
        gdf["dist"] = gdf.geometry.distance(p)
        is_sidewalk = gdf["subclass"].isin(sidewalk_keys) if "subclass" in gdf.columns else False
        sidewalks = gdf[is_sidewalk & (gdf["dist"] < 0.00015)] if hasattr(is_sidewalk, "__len__") else gdf.iloc[0:0]
        best = sidewalks.loc[sidewalks["dist"].idxmin()] if not sidewalks.empty else gdf.loc[gdf["dist"].idxmin()]
        nearest_rows.append({"i": i, "df_row": row, "best": best})

    gers_has_issue: dict[str, bool] = {}
    for item in nearest_rows:
        issues = " | ".join(frame_issues[item["i"]]) if frame_issues[item["i"]] else ""
        gers_id = str(item["best"].get("id") or "")
        if issues.strip():
            gers_has_issue[gers_id] = True

    rows: list[dict[str, Any]] = []
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

    gers_geojson: dict[str, Any] = {"type": "FeatureCollection", "features": list(gers_feature_map.values())}

    out_csv, out_gers = results_artifact_paths(sequence_id)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_gers.write_text(json.dumps(gers_geojson), encoding="utf-8")

    return rows, gers_geojson, str(out_csv.relative_to(APP_ROOT))


def load_saved_results_for_sequence(sequence_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    import pandas as pd

    out_csv, out_gers = results_artifact_paths(sequence_id)
    if not out_csv.is_file():
        raise FileNotFoundError(f"No saved results for this sequence ({out_csv.name} missing).")

    df = pd.read_csv(out_csv).fillna("")
    results = json.loads(df.to_json(orient="records"))
    if not isinstance(results, list):
        raise ValueError("unexpected records shape")

    if out_gers.is_file():
        gers_geojson = json.loads(out_gers.read_text(encoding="utf-8"))
        if not isinstance(gers_geojson, dict):
            raise ValueError("GeoJSON root must be an object")
    else:
        gers_geojson = {"type": "FeatureCollection", "features": []}

    return results, gers_geojson, str(out_csv.relative_to(APP_ROOT))


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

