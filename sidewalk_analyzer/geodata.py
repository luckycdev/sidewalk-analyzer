from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from shapely.geometry import LineString


def build_geodataframe(
    *,
    pegasus_rows: list[dict[str, Any]],
    segment_geoms: dict[str, LineString | None],
    segment_meta: dict[str, dict[str, Any]] | None = None,
) -> gpd.GeoDataFrame:
    records: list[dict[str, Any]] = []
    for row in pegasus_rows:
        seg_id = str(row.get("segment_id", ""))
        clip_payload = row.get("pegasus_clip") or {}

        geom = segment_geoms.get(seg_id)
        meta = (segment_meta or {}).get(seg_id, {})

        records.append(
            {
                "segment_id": seg_id,
                "clip_id": row.get("clip_id"),
                "segment_start_s": row.get("segment_start_s"),
                "segment_end_s": row.get("segment_end_s"),
                "clip_s3_uri": row.get("clip_s3_uri"),
                "still_s3_uri": row.get("still_s3_uri"),
                "travel_bearing": meta.get("travel_bearing"),
                "left_side": meta.get("left_side"),
                "right_side": meta.get("right_side"),
                "image_id": meta.get("image_id"),
                "captured_at": meta.get("captured_at"),
                "sidewalk_present": bool(clip_payload.get("sidewalk_present")),
                "width_class": clip_payload.get("width_class"),
                "surface_condition": clip_payload.get("surface_condition"),
                "curb_ramp_status": clip_payload.get("curb_ramp_status"),
                "confidence": clip_payload.get("confidence"),
                "video_evidence_link": row.get("clip_s3_uri"),
                "geometry": geom,
                # keep baseline nested for evaluation
                "pegasus_still": row.get("pegasus_still"),
                "pegasus_clip_raw": clip_payload.get("raw_text"),
            }
        )

    df = pd.DataFrame.from_records(records)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf

