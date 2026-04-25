from __future__ import annotations

import os
from typing import Any

import duckdb
import geopandas as gpd


def attach_gers_id_duckdb(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """
    Attaches a `gers_id` by nearest match against an Overture transportation edges GeoParquet file.

    Configuration:
    - Set `OVERTURE_TRANSPORT_PARQUET=/path/or/s3_url_to_geoparquet` to enable.
    - Uses a 50m maximum distance, computed via ST_Distance_Sphere on centroids.

    If not configured, returns the input untouched with match-rate stats marked skipped.
    """

    transport_path = os.getenv("OVERTURE_TRANSPORT_PARQUET", "").strip()
    if not transport_path:
        gdf2 = gdf.copy()
        gdf2["gers_id"] = None
        return gdf2, {"skipped": True, "reason": "OVERTURE_TRANSPORT_PARQUET not set"}

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    # Write segments to an in-memory duckdb table
    seg_df = gdf.copy()
    seg_df["segment_wkt"] = seg_df.geometry.apply(lambda g: g.wkt if g is not None else None)
    seg_df["segment_centroid_wkt"] = seg_df.geometry.apply(lambda g: g.centroid.wkt if g is not None else None)
    con.register("segments_df", seg_df.drop(columns=["geometry"]))

    # Read Overture transport parquet.
    # Expect columns like: id (GERS), geometry.
    con.execute(
        """
        CREATE OR REPLACE VIEW overture_edges AS
        SELECT
          COALESCE(CAST(properties->>'id' AS VARCHAR), CAST(id AS VARCHAR)) AS gers_id,
          ST_GeomFromWKB(geometry) AS geom
        FROM read_parquet(?)
        """,
        [transport_path],
    )

    # Nearest join within 50m
    max_m = 50.0
    results = con.execute(
        """
        SELECT
          s.segment_id,
          e.gers_id,
          ST_Distance_Sphere(ST_GeomFromText(s.segment_centroid_wkt), e.geom) AS distance_m
        FROM segments_df s
        LEFT JOIN LATERAL (
          SELECT gers_id, geom
          FROM overture_edges
          ORDER BY ST_Distance_Sphere(ST_GeomFromText(s.segment_centroid_wkt), geom)
          LIMIT 1
        ) e ON TRUE
        """,
    ).df()

    # Apply threshold
    mapping: dict[str, str | None] = {}
    distances: dict[str, float | None] = {}
    for _, row in results.iterrows():
        seg_id = str(row["segment_id"])
        dist = float(row["distance_m"]) if row["distance_m"] is not None else None
        gers_id = str(row["gers_id"]) if row["gers_id"] is not None else None
        if dist is None or dist > max_m:
            mapping[seg_id] = None
            distances[seg_id] = dist
        else:
            mapping[seg_id] = gers_id
            distances[seg_id] = dist

    gdf2 = gdf.copy()
    gdf2["gers_id"] = gdf2["segment_id"].map(mapping)
    gdf2["gers_distance_m"] = gdf2["segment_id"].map(distances)

    matched = int(gdf2["gers_id"].notna().sum())
    total = int(len(gdf2))
    stats: dict[str, Any] = {
        "skipped": False,
        "overture_path": transport_path,
        "max_match_m": max_m,
        "total_segments": total,
        "matched_segments": matched,
        "match_rate": (matched / total) if total else 0.0,
    }
    return gdf2, stats

