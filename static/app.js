const form = document.getElementById('search-form');
const gallery = document.getElementById('gallery');
const messageArea = document.getElementById('message-area');
const galleryNote = document.getElementById('gallery-note');
const streetName = document.getElementById('street-name');
const pointsCount = document.getElementById('points-count');
const imageCount = document.getElementById('image-count');
const status = document.getElementById('status');

function setMessage(kind, text) {
  if (!text) {
    messageArea.innerHTML = '';
    return;
  }

  messageArea.innerHTML = `<div class="${kind}">${text}</div>`;
}

function setLoading(isLoading) {
  form.classList.toggle('is-loading', isLoading);
  status.textContent = isLoading ? 'Loading' : status.textContent;
}

function renderGallery(images) {
  gallery.innerHTML = '';

  if (!images.length) {
    gallery.innerHTML = '<div class="empty">No images yet. Search street to populate gallery.</div>';
    return;
  }

  for (const image of images) {
    const card = document.createElement('article');
    card.className = 'photo-card';
    const sidewalkId = image.nearest_sidewalk_gers_id || 'Not found';
    const sidewalkDistance = image.nearest_sidewalk_distance_m != null ? `${image.nearest_sidewalk_distance_m} m` : '-';
    const centerDistance = image.distance_from_center_m != null ? `${image.distance_from_center_m} m` : '-';
    const strategy = image.nearest_sidewalk_strategy || 'none';
    const hasCoords = image.longitude != null && image.latitude != null;
    const coordinateLabel = hasCoords
      ? `${Number(image.latitude).toFixed(6)}, ${Number(image.longitude).toFixed(6)}`
      : 'Unknown';
    const mapLink = hasCoords
      ? `https://www.mapillary.com/app/?lat=${encodeURIComponent(image.latitude)}&lng=${encodeURIComponent(image.longitude)}&z=18`
      : null;
    card.innerHTML = `
      <div class="photo-frame">
        ${image.thumb_url ? `<img src="${image.thumb_url}" alt="Mapillary image ${image.id}">` : '<div class="photo-missing">No thumbnail</div>'}
      </div>
      <div class="photo-body">
        <div class="photo-meta">
          <strong>${image.creator || 'Unknown creator'}</strong>
          <span>${image.captured_at_label || 'Unknown time'}</span>
          <span>Coordinates: ${coordinateLabel}</span>
          <span>Distance from center: ${centerDistance}</span>
          <span>Nearest sidewalk GERS: ${sidewalkId}</span>
          <span>Sidewalk distance: ${sidewalkDistance}</span>
          <span>Match strategy: ${strategy}</span>
        </div>
        ${mapLink ? `<a href="${mapLink}" target="_blank" rel="noreferrer">Open Mapillary at coordinates</a>` : '<span>Map link unavailable (missing coordinates)</span>'}
      </div>
    `;
    gallery.appendChild(card);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  setMessage('', '');
  setLoading(true);
  galleryNote.textContent = 'Searching...';
  gallery.innerHTML = '';

  const payload = {
    center_lat: Number(document.getElementById('center_lat').value),
    center_lng: Number(document.getElementById('center_lng').value),
    search_radius_m: Number(document.getElementById('search_radius_m').value),
    max_images: Number(document.getElementById('max_images').value),
  };

  try {
    const response = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Search failed');
    }

    streetName.textContent = `${Number(payload.center_lat).toFixed(6)}, ${Number(payload.center_lng).toFixed(6)}`;
    pointsCount.textContent = String(data.search_radius_m ?? payload.search_radius_m);
    imageCount.textContent = String(data.images?.length ?? 0);
    status.textContent = data.status === 'empty' ? 'No imagery' : 'Found imagery';

    renderGallery(data.images || []);
    galleryNote.textContent = data.images && data.images.length ? 'Matched images from Mapillary.' : 'Nothing matched this street.';

    if (data.overture_status === 'error' && data.overture_error) {
      setMessage('notice', `Overture lookup failed: ${data.overture_error}`);
    }

    if (!data.images || data.images.length === 0) {
      setMessage('notice', 'No imagery found for this street. Try bigger radius, lower spacing, or different query near covered road.');
    }
  } catch (error) {
    status.textContent = 'Error';
    galleryNote.textContent = 'Search failed.';
    setMessage('error', error.message || String(error));
    gallery.innerHTML = '<div class="empty">Search failed. Fix error and try again.</div>';
  } finally {
    setLoading(false);
  }
});