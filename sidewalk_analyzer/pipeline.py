from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .types import RunManifest, RunPaths


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _default_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_dirs(out_dir: Path, run_id: str) -> RunPaths:
    run_dir = out_dir / f"run_{run_id}"
    clips_dir = run_dir / "clips"
    exports_dir = run_dir / "exports"
    logs_dir = run_dir / "logs"
    for p in (run_dir, clips_dir, exports_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=run_dir, clips_dir=clips_dir, exports_dir=exports_dir, logs_dir=logs_dir)


def run_pipeline(
    *,
    video_path: Path,
    csv_path: Path | None = None,
    out_dir: Path,
    run_id: str | None,
    settings: Settings,
    no_s3: bool,
    skip_marengo: bool,
    threshold_override: float | None,
) -> dict[str, Any]:
    """
    Orchestrates the full pipeline. Individual steps live in other modules.
    This function is intentionally structured so you can run partial steps
    while iterating during the hackathon.
    """

    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    run_id_final = run_id or _default_run_id()
    paths = _ensure_dirs(out_dir, run_id_final)

    threshold = threshold_override if threshold_override is not None else settings.marengo_threshold

    timings: dict[str, float] = {}
    t_all = time.time()

    def mark(name: str, t0: float) -> None:
        timings[name] = round(time.time() - t0, 3)

    s3_video_uri: str | None = None
    if not no_s3:
        t0 = time.time()
        from .s3_io import upload_video_to_s3

        s3_video_uri = upload_video_to_s3(video_path=video_path, settings=settings)
        mark("s3_upload_video_s", t0)

    if skip_marengo:
        t0 = time.time()
        from .video_clips import get_video_duration_s
        from .types import SegmentCandidate

        duration = get_video_duration_s(video_path)
        candidates = [SegmentCandidate(segment_id="segment_000", start_s=0.0, end_s=duration, score=1.0)]
        marengo_stats: dict[str, Any] = {"skipped": True, "candidates": 1}
        mark("marengo_prefilter_s", t0)
    else:
        if not s3_video_uri:
            raise RuntimeError("Marengo requires S3 input video URI. Run without --no-s3 or use --skip-marengo.")
        t0 = time.time()
        from .bedrock_marengo import marengo_prefilter

        candidates, marengo_stats = marengo_prefilter(
            s3_video_uri=s3_video_uri,
            settings=settings,
            threshold=threshold,
            run_dir=paths.run_dir,
        )
        mark("marengo_prefilter_s", t0)

    from .video_clips import extract_clips_and_stills

    t0 = time.time()
    clip_assets, clip_stats = extract_clips_and_stills(
        video_path=video_path,
        candidates=candidates,
        clips_dir=paths.clips_dir,
        clip_seconds=settings.clip_seconds,
        padding_seconds=settings.clip_padding_seconds,
    )
    mark("clip_extract_s", t0)

    from .bedrock_pegasus import run_pegasus_on_assets

    t0 = time.time()
    pegasus_rows, pegasus_stats = run_pegasus_on_assets(
        assets=clip_assets,
        settings=settings,
        run_dir=paths.run_dir,
    )
    mark("pegasus_infer_s", t0)

    from .geodata import build_geodataframe
    from .video_clips import get_video_fps

    t0 = time.time()
    fps = get_video_fps(video_path)
    segment_geoms, segment_meta, gps_stats = _build_geoms_and_meta_from_csv(
        video_path=video_path,
        csv_path=csv_path,
        clip_assets=clip_assets,
        fps=fps,
    )
    gdf = build_geodataframe(pegasus_rows=pegasus_rows, segment_geoms=segment_geoms, segment_meta=segment_meta)
    mark("gps_and_geodata_s", t0)

    from .overture_join import attach_gers_id_duckdb

    t0 = time.time()
    gdf2, overture_stats = attach_gers_id_duckdb(gdf)
    mark("overture_join_s", t0)

    from .export import export_outputs

    t0 = time.time()
    outputs = export_outputs(gdf2, exports_dir=paths.exports_dir)
    mark("export_s", t0)

    from .eval import maybe_evaluate
    from .bench import summarize_benchmarks

    t0 = time.time()
    eval_stats = maybe_evaluate(paths.run_dir, clip_assets=clip_assets, pegasus_rows=pegasus_rows)
    mark("evaluation_s", t0)

    timings["total_s"] = round(time.time() - t_all, 3)
    (paths.logs_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    _write_cost_estimate(paths.logs_dir, clip_count=len(clip_assets))
    bench_stats = summarize_benchmarks(paths.run_dir)

    manifest: RunManifest = RunManifest(
        run_id=run_id_final,
        created_at_iso=_iso_now(),
        input_video=str(video_path),
        s3_video_uri=s3_video_uri,
        outputs={k: str(v) for k, v in outputs.items()},
        stats={
            "marengo": marengo_stats,
            "clips": clip_stats,
            "pegasus": pegasus_stats,
            "gps": gps_stats,
            "overture": overture_stats,
            "evaluation": eval_stats,
            "benchmarking": bench_stats,
        },
    )

    manifest_path = paths.run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")

    # Convenience pointer to last run
    try:
        (out_dir / "latest_run.txt").write_text(str(paths.run_dir), encoding="utf-8")
    except OSError:
        pass

    return manifest.__dict__


def _build_geoms_and_meta_from_csv(
    *,
    video_path: Path,
    csv_path: Path | None,
    clip_assets: list[Any],
    fps: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """
    If csv_path is provided, build:
    - segment geometry (LineString) using manifest lat/lon
    - segment metadata fields (bearing, sides, image_id, captured_at)
    Otherwise, return None geometries + empty metadata while preserving fallback behavior.
    """

    from shapely.geometry import LineString

    segment_geoms: dict[str, LineString | None] = {}
    segment_meta: dict[str, dict[str, Any]] = {}

    if not csv_path:
        for a in clip_assets:
            segment_geoms[a.segment_id] = None
            segment_meta[a.segment_id] = {
                "travel_bearing": None,
                "left_side": None,
                "right_side": None,
                "image_id": None,
                "captured_at": None,
            }
        return segment_geoms, segment_meta, {"csv_used": False, "missing_geometry": len(set(a.segment_id for a in clip_assets))}

    from .gps_extract import build_segment_geometry, filter_stable_frames, get_street_side, load_csv_manifest, timestamp_to_row

    df, lat_col, lon_col, ang_col, time_col = load_csv_manifest(csv_path)

    # Optional: filter out turning frames based on bearing stability
    df2 = filter_stable_frames(df, ang_col, max_angle_change=30.0)

    # Flexible IDs
    image_id_col = None
    for cand in ["image_id", "imageid", "id"]:
        if cand in [c.lower() for c in df2.columns]:
            # resolve actual case
            image_id_col = next(c for c in df2.columns if c.lower() == cand)
            break

    missing = 0
    for a in clip_assets:
        geom = build_segment_geometry(a.start_s, a.end_s, fps, df2, lat_col, lon_col)
        segment_geoms[a.segment_id] = geom
        if geom is None:
            missing += 1

        row_idx = timestamp_to_row(a.start_s, fps, df2)
        meta_row = df2.iloc[row_idx] if not df2.empty else None

        bearing = None
        if meta_row is not None and ang_col and ang_col in df2.columns:
            try:
                bearing = float(meta_row[ang_col])
            except Exception:
                bearing = None

        sides = get_street_side(bearing) if bearing is not None else {"travel_bearing": None, "left_side": None, "right_side": None}
        image_id = None
        if meta_row is not None:
            if "image_id" in df2.columns:
                image_id = meta_row.get("image_id")
            elif image_id_col:
                image_id = meta_row.get(image_id_col)
        captured_at = None
        if meta_row is not None and time_col and time_col in df2.columns:
            captured_at = meta_row.get(time_col)

        segment_meta[a.segment_id] = {
            "travel_bearing": sides.get("travel_bearing"),
            "left_side": sides.get("left_side"),
            "right_side": sides.get("right_side"),
            "image_id": image_id,
            "captured_at": captured_at,
        }

    stats: dict[str, Any] = {
        "csv_used": True,
        "csv_path": str(csv_path),
        "fps": fps,
        "rows": int(len(df2)),
        "lat_col": lat_col,
        "lon_col": lon_col,
        "ang_col": ang_col,
        "time_col": time_col,
        "missing_geometry": missing,
    }
    return segment_geoms, segment_meta, stats


def _write_cost_estimate(logs_dir: Path, *, clip_count: int) -> None:
    # Pricing varies; keep configurable placeholders in one place.
    # Users can overwrite with real hackathon budget numbers.
    estimate = {
        "clips": clip_count,
        "note": "Set real Bedrock pricing inputs for accurate cost estimates.",
        "estimated_usd": None,
    }
    (logs_dir / "cost_estimate.json").write_text(json.dumps(estimate, indent=2), encoding="utf-8")

