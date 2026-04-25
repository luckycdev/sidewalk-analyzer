const form = document.getElementById('pipeline-form');
const message = document.getElementById('pipeline-message');
const runIdEl = document.getElementById('run-id');
const statusEl = document.getElementById('run-status');
const geojsonLinkEl = document.getElementById('geojson-link');
const featurePanel = document.getElementById('feature-panel');

function setMessage(text) {
  message.textContent = text || '';
}

function surfaceColor(value) {
  const v = (value || '').toLowerCase();
  if (v === 'good') return '#2dd4bf';
  if (v === 'fair') return '#fbbf24';
  if (v === 'poor') return '#fb7185';
  if (v === 'impassable') return '#f97316';
  return '#94a3b8';
}

let map = null;

function initMap() {
  const token = window.__MAPBOX_TOKEN__;
  if (!token) {
    setMessage('Missing MAPBOX_ACCESS_TOKEN in .env. Map will not render.');
    return;
  }

  mapboxgl.accessToken = token;
  map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/dark-v11',
    center: [-90.1994, 38.6270], // St. Louis default
    zoom: 12,
  });

  map.addControl(new mapboxgl.NavigationControl(), 'top-right');
}

function addOrUpdateGeoJson(runId, geojson) {
  if (!map) return;

  const sourceId = 'segments';
  const layerId = 'segments-line';

  const data = geojson;
  const hasFeatures = data && data.features && data.features.length;
  if (!hasFeatures) {
    setMessage('GeoJSON loaded but has no features.');
  }

  if (map.getSource(sourceId)) {
    map.getSource(sourceId).setData(data);
    return;
  }

  map.addSource(sourceId, { type: 'geojson', data });

  map.addLayer({
    id: layerId,
    type: 'line',
    source: sourceId,
    paint: {
      'line-width': 5,
      'line-color': [
        'match',
        ['get', 'surface_condition'],
        'good', surfaceColor('good'),
        'fair', surfaceColor('fair'),
        'poor', surfaceColor('poor'),
        'impassable', surfaceColor('impassable'),
        '#94a3b8',
      ],
      'line-opacity': 0.85,
    },
  });

  map.on('click', layerId, (e) => {
    const f = e.features && e.features[0];
    if (!f) return;
    const p = f.properties || {};
    const clipUri = p.clip_s3_uri || p.video_evidence_link || '';
    featurePanel.innerHTML = `
      <div><strong>segment_id</strong>: <span class="mono">${p.segment_id || '-'}</span></div>
      <div><strong>surface_condition</strong>: <span class="mono">${p.surface_condition || '-'}</span></div>
      <div><strong>width_class</strong>: <span class="mono">${p.width_class || '-'}</span></div>
      <div><strong>curb_ramp_status</strong>: <span class="mono">${p.curb_ramp_status || '-'}</span></div>
      <div><strong>confidence</strong>: <span class="mono">${p.confidence || '-'}</span></div>
      <div><strong>gers_id</strong>: <span class="mono">${p.gers_id || '-'}</span></div>
      <div class="small"><strong>clip_s3_uri</strong>: <span class="mono">${clipUri || '-'}</span></div>
      <div class="small">Tip: evidence playback uses presigned URLs (planned). For now, copy the S3 URI.</div>
    `;
  });

  map.on('mouseenter', layerId, () => (map.getCanvas().style.cursor = 'pointer'));
  map.on('mouseleave', layerId, () => (map.getCanvas().style.cursor = ''));

  // Fit bounds
  try {
    const bbox = turf.bbox(data);
    map.fitBounds(bbox, { padding: 40, duration: 800 });
  } catch (e) {
    // turf may not be present; ignore
  }
}

async function pollStatus(runId) {
  for (let i = 0; i < 240; i++) { // ~8 min at 2s
    const resp = await fetch(`/api/pipeline/status/${encodeURIComponent(runId)}`);
    const data = await resp.json();
    statusEl.textContent = data.status || 'unknown';
    if (data.status === 'completed') return true;
    if (data.status === 'error') {
      setMessage(`Pipeline error: ${data.error || 'unknown'}`);
      return false;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  setMessage('Timed out waiting for pipeline completion.');
  return false;
}

async function loadGeoJson(runId) {
  const resp = await fetch(`/api/pipeline/geojson/${encodeURIComponent(runId)}`);
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || 'Failed to fetch GeoJSON');
  }
  return await resp.json();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  setMessage('');
  featurePanel.textContent = '';

  const payload = {
    video_path: document.getElementById('video_path').value.trim(),
    csv_path: document.getElementById('csv_path').value.trim(),
    out_dir: document.getElementById('out_dir').value.trim() || 'outputs',
  };

  try {
    statusEl.textContent = 'starting';
    const resp = await fetch('/api/pipeline/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Failed to start pipeline');

    const runId = data.run_id;
    runIdEl.textContent = runId;
    statusEl.textContent = data.status;
    geojsonLinkEl.textContent = `/api/pipeline/geojson/${runId}`;

    const ok = await pollStatus(runId);
    if (!ok) return;

    setMessage('Pipeline complete. Loading GeoJSON...');
    const geojson = await loadGeoJson(runId);
    if (map && map.loaded()) {
      addOrUpdateGeoJson(runId, geojson);
    } else if (map) {
      map.once('load', () => addOrUpdateGeoJson(runId, geojson));
    }
    setMessage('Done.');
  } catch (err) {
    statusEl.textContent = 'error';
    setMessage(err.message || String(err));
  }
});

initMap();

