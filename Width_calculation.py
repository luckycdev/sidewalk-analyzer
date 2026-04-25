import numpy as np

def estimate_width(mask, reference_pixels, reference_meters):
    """
    mask: binary sidewalk mask (1 = sidewalk)
    reference_pixels: width of known object in pixels
    reference_meters: real-world size of that object
    """

    # Step 1: meters per pixel
    meters_per_pixel = reference_meters / reference_pixels

    # Step 2: take bottom row of mask
    row = mask[-50]  # near bottom

    # Step 3: find sidewalk edges
    indices = np.where(row == 1)[0]

    if len(indices) == 0:
        return None

    width_pixels = indices[-1] - indices[0]

    # Step 4: convert to meters
    width_meters = width_pixels * meters_per_pixel

    return width_meters

rows = mask[-100:-50]

widths = []
for row in rows:
    indices = np.where(row == 1)[0]
    if len(indices) > 0:
        widths.append(indices[-1] - indices[0])

avg_width_pixels = np.mean(widths)
