from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd


def export_outputs(gdf: gpd.GeoDataFrame, *, exports_dir: Path) -> dict[str, Path]:
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Minimal required exports
    geojson_path = exports_dir / "sidewalk_segments.geojson"
    parquet_path = exports_dir / "sidewalk_segments.geoparquet"

    # GeoJSON
    gdf.to_file(geojson_path, driver="GeoJSON")

    # GeoParquet
    gdf.to_parquet(parquet_path, index=False)

    # Convenience CSV (attributes only)
    csv_path = exports_dir / "sidewalk_segments.csv"
    attrs = gdf.drop(columns=["geometry"], errors="ignore")
    attrs.to_csv(csv_path, index=False)

    return {
        "geojson": geojson_path,
        "geoparquet": parquet_path,
        "csv": csv_path,
    }

