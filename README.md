# Sidewalk Analyzer

A small Flask app for **detecting sidewalk and path surface damage** in walking-style videos. It sends each video to **Amazon Bedrock** (TwelveLabs **Pegasus**), reads per-frame GPS from a companion CSV, matches locations to **Overture Maps** sidewalk segments, and shows results on a **Mapbox** map with optional GeoJSON export.

Supporting scripts help build CSV tracks from Mapillary sequences and batch-run analysis outside the web UI.

## What it does

1. **Video analysis** — You pick an `.mp4` in the project directory. The app uploads it to S3, invokes Pegasus with a structured prompt, optionally **verifies** candidate damage windows, then maps detections onto frame indices.
2. **Geospatial join** — For each frame, it uses `csvs/<basename>.csv` (same basename as the video) with at least `lat` and `long` columns, downloads Overture **segment** data for the track bounding box, and associates each point with the nearest sidewalk-related segment (and GERS-style segment ids where available).
3. **UI** — `/` lists local videos, runs analysis, polls status, shows a summary table, and draws the path plus segment styling on a Mapbox GL map.
4. **Mapillary search** (optional) — `POST /api/search` loads street-level images near a point and enriches them with nearest sidewalk segment metadata; requires a Mapillary token.

## Requirements

- **Python** 3.10+ recommended (GeoPandas, Overture, DuckDB stack).
- **AWS account** with credentials configured (e.g. `aws configure` or environment variables) and permission for:
  - **S3** — upload objects to your chosen bucket.
  - **Bedrock** — invoke the Pegasus model you configure.
- **Mapbox** [access token](https://account.mapbox.com/) for the map on `/` and `/pipeline`.
- **Per-sequence CSV** — For each `<id>.mp4`, a file `csvs/<id>.csv` with columns including `lat` and `long` (and any other columns you use downstream). The video basename must match the CSV basename.
- **FFmpeg** — Useful if you use `mp4_creator.py` or other MoviePy-based tooling to build videos locally.

## Setup

```bash
cd sidewalk-analyzer
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (the app loads it automatically). Example keys:

| Variable | Purpose |
|----------|---------|
| `MAPBOX_ACCESS_TOKEN` | Required for map tiles and the main UI. |
| `MAPILLARY_ACCESS_TOKEN` | Required for Mapillary-backed `/api/search`. |
| `AWS_BUCKET_NAME` | S3 bucket for uploaded videos (defaults exist in code; override for your account). |
| `AWS_REGION` | Region for S3 and Bedrock (default `us-east-1`). |
| `AWS_MODEL_ID` | Bedrock model id for Pegasus (default `twelvelabs.pegasus-1-2-v1:0`). |
| `AWS_BUCKET_OWNER` | Account id for `bucketOwner` in the Bedrock media payload. |
| `MIN_SEGMENT_CONFIDENCE` | Drop segments below this confidence (default `0.7`). |
| `VERIFY_SEGMENTS` | Set `0` / `false` to skip the second-pass verification calls. |
| `VERIFY_MIN_HITS`, `VERIFY_MIN_AVG_CONF` | Tuning for verification. |
| `OVERTURE_LOOKUP_PADDING_M`, `OVERTURE_LOOKUP_MAX_MATCH_M` | Used when enriching Mapillary results with Overture matches. |

Use standard AWS SDK environment variables or shared credentials for authentication (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, session token if applicable).

## Run the web app

```bash
source .venv/bin/activate
python app.py
```

Open `http://127.0.0.1:5000/` (Flask default). Use **Analyze** on a video whose matching CSV already exists under `csvs/`.

Health check: `GET http://127.0.0.1:5000/api/health`

## Other files in this repo

- **`aws_test.py`** — Standalone script oriented around batch CSV/video processing against S3 and Bedrock; edit the config block at the top for bucket, region, and owner, or align with your `.env` patterns.
- **`csv_creator.py`**, **`mp4_creator.py`** — Helpers for building track CSVs and MP4s from Mapillary-style inputs.
- **`pegasus_processor.py`** — Pegasus-related processing utilities.

## Code organization

The Flask entrypoint remains **`app.py`**, but most web-app logic now lives under **`sidewalk_analyzer_web/`**:

- **`sidewalk_analyzer_web/routes.py`**: Flask routes and API surface
- **`sidewalk_analyzer_web/analysis.py`**: upload → Pegasus → Overture join + saved results IO
- **`sidewalk_analyzer_web/pegasus.py`**, **`overture.py`**, **`mapillary.py`**: service-specific helpers

For downloading Mapillary assets outside this repo, the [mapillary_download](https://github.com/Stefal/mapillary_download) project is a common reference.

## Pipeline page (`/pipeline`)

The `/pipeline` route starts an optional **full pipeline** implemented in a Python package `sidewalk_analyzer` (`load_settings`, `run_pipeline`). That package is **not** included in this repository; if you do not have it installed, starting a pipeline run from the UI will fail on import. The main **`/`** flow (upload → Pegasus → Overture join → map) does not depend on that package.
