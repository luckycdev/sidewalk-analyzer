import os
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_decimal_from_dms(dms, ref):
    """Converts Degrees, Minutes, Seconds to Decimal Degrees."""
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    
    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_exif_gps(image_path):
    """Extracts Lat/Long from an image's EXIF data."""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None, None

            gps_info = {}
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_info[sub_decoded] = value[t]

            if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                lat = get_decimal_from_dms(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
                long = get_decimal_from_dms(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))
                return lat, long
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    
    return None, None

# --- Main Logic ---
folder_path = './data/vPbqrpQm9BKEynlRtS8VN4' # Change this to your folder path
output_csv = 'image_coordinates.csv'
image_extensions = ('.jpg', '.jpeg', '.png', '.tiff')

data_rows = []
counter = 0

print("Processing images...")

for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        full_path = os.path.join(folder_path, filename)
        lat, long = get_exif_gps(full_path)
        
        if lat is not None:
            data_rows.append([counter, long, lat, full_path])
            counter += 1

# Write to CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['index', 'long', 'lat', 'file_path'])
    writer.writerows(data_rows)

print(f"Done! Saved {len(data_rows)} entries to {output_csv}")