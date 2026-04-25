from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import LineString


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    # fuzzy contains match
    for col in df.columns:
        cl = col.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return col
    return None


def load_csv_manifest(csv_path: str | Path) -> tuple[pd.DataFrame, str, str, str | None, str | None]:
    """
    Load the CSV manifest and resolve flexible column names.
    Returns: (df, lat_col, lon_col, ang_col, time_col)
    """

    path = Path(csv_path)
    df = pd.read_csv(path)

    lat_col = _find_col(df, ["latitude", "lat"])
    lon_col = _find_col(df, ["longitude", "lon", "lng"])
    ang_col = _find_col(df, ["angle", "bearing", "direction", "heading"])
    time_col = _find_col(df, ["captured_at", "timestamp", "time"])

    if not lat_col or not lon_col:
        raise ValueError("CSV manifest must include latitude/longitude columns (flex-matched).")

    return df, lat_col, lon_col, ang_col, time_col


def timestamp_to_row(timestamp_sec: float, fps: float, df: pd.DataFrame) -> int:
    """
    Map a timestamp (sec) to the nearest row index.
    If the CSV has no explicit frame/index column, row order acts as frame index.
    """

    if df.empty:
        return 0
    if fps <= 0:
        fps = 30.0
    idx = int(round(float(timestamp_sec) * float(fps)))
    idx = max(0, min(idx, len(df) - 1))
    return idx


def build_segment_geometry(
    start_sec: float,
    end_sec: float,
    fps: float,
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
) -> LineString | None:
    if df.empty:
        return None
    start_i = timestamp_to_row(start_sec, fps, df)
    end_i = timestamp_to_row(end_sec, fps, df)
    lo = min(start_i, end_i)
    hi = max(start_i, end_i)
    segment = df.iloc[lo : hi + 1]
    if len(segment) < 2:
        return None
    coords: list[tuple[float, float]] = []
    for _, row in segment.iterrows():
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
        except Exception:
            continue
        if math.isnan(lat) or math.isnan(lon):
            continue
        coords.append((lon, lat))
    # de-dup consecutive
    cleaned: list[tuple[float, float]] = []
    for c in coords:
        if not cleaned or cleaned[-1] != c:
            cleaned.append(c)
    return LineString(cleaned) if len(cleaned) >= 2 else None


def _wrap_angle_deg(a: float) -> float:
    v = float(a) % 360.0
    if v < 0:
        v += 360.0
    return v


def _angle_diff_deg(a: float, b: float) -> float:
    """
    Smallest absolute difference accounting for 0/360 wraparound.
    """

    a2 = _wrap_angle_deg(a)
    b2 = _wrap_angle_deg(b)
    d = abs(a2 - b2)
    return min(d, 360.0 - d)


def filter_stable_frames(df: pd.DataFrame, ang_col: str | None, max_angle_change: float = 30.0) -> pd.DataFrame:
    if df.empty or not ang_col or ang_col not in df.columns:
        return df

    angles = pd.to_numeric(df[ang_col], errors="coerce")
    keep = [True]
    prev = None
    for v in angles.iloc[1:]:
        try:
            cur = float(v)
        except Exception:
            keep.append(False)
            continue
        if math.isnan(cur):
            keep.append(False)
            continue
        if prev is None:
            prev = cur
            keep.append(True)
            continue
        diff = _angle_diff_deg(prev, cur)
        keep.append(diff <= max_angle_change)
        prev = cur
    return df.loc[df.index[keep]].reset_index(drop=True)


def _cardinal_8(angle_deg: float) -> str:
    a = _wrap_angle_deg(angle_deg)
    # 8-wind: N, NE, E, SE, S, SW, W, NW (45° bins)
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((a + 22.5) // 45) % 8
    return dirs[idx]


def get_street_side(angle_deg: float) -> dict[str, Any]:
    """
    Given travel bearing, return bearing and left/right side cardinals.
    """

    travel = _wrap_angle_deg(angle_deg)
    left = _wrap_angle_deg(travel - 90.0)
    right = _wrap_angle_deg(travel + 90.0)
    return {
        "travel_bearing": travel,
        "left_side": _cardinal_8(left),
        "right_side": _cardinal_8(right),
    }

