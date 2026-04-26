from __future__ import annotations

import json
import threading
import uuid
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request, send_file

from .analysis import load_saved_results_for_sequence, run_analysis, utc_now_iso
from .mapillary import (
    collect_images_in_radius,
    format_timestamp,
    geometry_coords,
    image_url,
    mapillary_token,
)
from .overture import enrich_images_with_sidewalk_gers, get_overture_explorer_link_via_api, is_sidewalk_segment, run_overture_download
from .paths import APP_ROOT, resolve_project_media_path, results_artifact_paths
from .state import PIPELINE_LOCK, PIPELINE_RUNS


def register_routes(app: Flask) -> None:
    @app.get("/api/videos")
    def api_videos():
        items: list[dict[str, Any]] = []
        for f in sorted(APP_ROOT.glob("*.mp4"), key=lambda p: p.name.lower()):
            out_csv, _ = results_artifact_paths(f.stem)
            rel = str(f.relative_to(APP_ROOT))
            items.append({"path": rel, "has_saved_results": out_csv.is_file()})
        return jsonify({"videos": items})

    @app.post("/api/analyze")
    def api_analyze():
        payload = request.get_json(silent=True) or {}
        video_path = str(payload.get("video_path") or "").strip()
        resolved = resolve_project_media_path(video_path)
        if not resolved:
            return jsonify({"error": "Invalid video path"}), 400

        run_id = uuid.uuid4().hex[:12]
        with PIPELINE_LOCK:
            PIPELINE_RUNS[run_id] = {"status": "running", "created_at": utc_now_iso()}

        def _job() -> None:
            try:
                results, gers_geojson, saved_csv = run_analysis(str(resolved))
                with PIPELINE_LOCK:
                    PIPELINE_RUNS[run_id].update(
                        {"status": "completed", "results": results, "gers_geojson": gers_geojson, "saved_csv": saved_csv}
                    )
            except Exception as exc:
                with PIPELINE_LOCK:
                    PIPELINE_RUNS[run_id]["status"] = "error"
                    PIPELINE_RUNS[run_id]["error"] = str(exc)

        threading.Thread(target=_job, daemon=True).start()
        return jsonify({"run_id": run_id, "status": "running"})

    @app.post("/api/load-saved-results")
    def api_load_saved_results():
        payload = request.get_json(silent=True) or {}
        video_path = str(payload.get("video_path") or "").strip()
        resolved = resolve_project_media_path(video_path)
        if not resolved:
            return jsonify({"error": "Invalid video path"}), 400

        sequence_id = resolved.stem
        try:
            results, gers_geojson, saved_csv = load_saved_results_for_sequence(sequence_id)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": f"Failed to read saved results: {exc}"}), 500

        run_id = uuid.uuid4().hex[:12]
        with PIPELINE_LOCK:
            PIPELINE_RUNS[run_id] = {
                "status": "completed",
                "created_at": utc_now_iso(),
                "results": results,
                "gers_geojson": gers_geojson,
                "source": "saved_csv",
                "saved_csv": saved_csv,
            }

        return jsonify({"run_id": run_id, "status": "completed"})

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

    @app.get("/api/overture-explorer-link/<path:gers_id>")
    def api_overture_explorer_link(gers_id: str):
        theme = (request.args.get("theme") or "transportation").strip() or "transportation"
        otype = (request.args.get("otype") or "segment").strip() or "segment"
        try:
            zoom = float(request.args.get("zoom") or 16.75)
        except (TypeError, ValueError):
            zoom = 16.75

        try:
            link = get_overture_explorer_link_via_api(gers_id, theme=theme, otype=otype, zoom=zoom)
            return jsonify({"gers_id": (gers_id or "").strip(), "link": link})
        except requests.RequestException as exc:
            return jsonify({"error": f"API lookup failed: {exc}"}), 502
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

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
            return jsonify({"type": "FeatureCollection", "features": sidewalk_features})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/csv-files")
    def api_csv_files():
        csv_files = sorted(p.name for p in APP_ROOT.glob("results_*.csv"))
        return jsonify({"csv_files": csv_files})

    @app.get("/api/load-csv/<filename>")
    def api_load_csv(filename: str):
        if not filename.startswith("results_") or not filename.endswith(".csv"):
            return jsonify({"error": "Invalid filename"}), 400

        path = (APP_ROOT / filename).resolve()
        try:
            path.relative_to(APP_ROOT)
        except ValueError:
            return jsonify({"error": "Invalid filename"}), 400

        if not path.is_file():
            return jsonify({"error": "File not found"}), 404

        try:
            import pandas as pd

            df = pd.read_csv(path).fillna("")
            data = json.loads(df.to_json(orient="records"))
            return jsonify({"data": data})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/search")
    def api_search():
        payload = request.get_json(silent=True) or {}

        try:
            center_lat = float(payload.get("center_lat"))
            center_lng = float(payload.get("center_lng"))
        except (TypeError, ValueError):
            return jsonify({"error": "center_lat and center_lng required."}), 400

        search_radius_m = float(payload.get("search_radius_m") or 100.0)
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
                            "creator": image.get("creator", {}).get("username")
                            if isinstance(image.get("creator"), dict)
                            else "Unknown creator",
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

    @app.get("/")
    def index() -> str:
        import os

        mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
        return render_template("index.html", mapbox_token_json=json.dumps(mapbox_token))

    @app.get("/pipeline")
    def pipeline_page() -> str:
        import os

        mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
        return render_template("pipeline.html", mapbox_token=mapbox_token)

    # Keep the existing pipeline endpoints in place; they depend on an external package.
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
            PIPELINE_RUNS[run_id] = {"status": "running", "created_at": utc_now_iso()}

        try:
            from pathlib import Path

            from sidewalk_analyzer.config import load_settings
            from sidewalk_analyzer.pipeline import run_pipeline

            settings = load_settings(env_path=Path(APP_ROOT) / ".env")
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

        return jsonify({"run_id": run_id, "status": PIPELINE_RUNS[run_id]["status"]})

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

