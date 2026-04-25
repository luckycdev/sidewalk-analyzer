# Street Lens

Small Flask app with separate backend API and frontend page. Backend geocodes street, samples along street geometry, then pulls nearby Mapillary images into one gallery.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` in project root:

   ```bash
   MAPILLARY_ACCESS_TOKEN=your_token_here
   ```

3. Run app:

   ```bash
   py app.py
   ```

## Use

- Enter street query like `Baker Street, London`.
- Tune spacing, radius, and max images.
- Click `Find imagery`.

Frontend no longer shows token. Backend reads `MAPILLARY_ACCESS_TOKEN` from `.env`.

If street has no coverage, frontend shows `No imagery found for this street` instead of blank page.

Frontend files live in [templates/index.html](templates/index.html), [static/app.js](static/app.js), and [static/styles.css](static/styles.css).

Each returned Mapillary image now includes nearest Overture sidewalk GERS ID and distance in meters.
If no explicit sidewalk-like segment is available, backend falls back to nearest road segment and marks strategy as `road_fallback`.

Optional lookup tuning in `.env`:

- `OVERTURE_LOOKUP_PADDING_M` (default `120`)
- `OVERTURE_LOOKUP_MAX_MATCH_M` (default `80`)

## Notes

- Mapillary API needs access token.
- Nominatim geocoding needs a friendly User-Agent and may rate limit heavy use.
- App uses street geometry sampling because Mapillary radius search is small.

## Pipeline CSV GPS manifest

If you run the video pipeline (`python -m sidewalk_analyzer run ...` or `/pipeline`), you can optionally provide a CSV manifest alongside the MP4 containing per-frame metadata.

Expected columns (names are matched flexibly; exact names don’t matter):

- `filename` (or frame number)
- `image_id`
- `latitude`
- `longitude`
- `captured_at`
- `angle` (compass bearing degrees 0–360)