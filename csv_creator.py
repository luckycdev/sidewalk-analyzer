import os
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_decimal_from_dms(dms, ref):
    """Converts Degrees, Minutes, Seconds to Decimal Degrees."""
    try:
        # Handle tuple/rational formats from different EXIF readers
        degrees = float(dms[0])
        minutes = float(dms[1]) / 60.0
        seconds = float(dms[2]) / 3600.0
        
        decimal = degrees + minutes + seconds
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except (TypeError, ZeroDivisionError, IndexError):
        return None

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
        print(f"⚠️ Error processing {image_path}: {e}")
    
    return None, None

def main():
    data_dir = './data/'
    csv_out_dir = './csvs/'
    image_extensions = ('.jpg', '.jpeg', '.png', '.tiff')

    # 1. Ensure directories exist
    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir} directory not found. Make sure you are in the project root.")
        return
    
    if not os.path.exists(csv_out_dir):
        os.makedirs(csv_out_dir)
        print(f"📁 Created folder: {csv_out_dir}")

    # 2. Get available sequences (subfolders in ./data/)
    sequences = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not sequences:
        print("❌ No sequence folders found in ./data/")
        return

    # 3. User Selection UI
    print("\n--- Available Data Sequences ---")
    for idx, seq in enumerate(sequences):
        print(f"[{idx}] {seq}")
    
    try:
        choice = int(input("\nSelect sequence to process: "))
        selected_seq = sequences[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection. Please enter a number from the list.")
        return

    folder_path = os.path.join(data_dir, selected_seq)
    output_csv = os.path.join(csv_out_dir, f"{selected_seq}.csv")

    # 4. Process Images
    data_rows = []
    print(f"\n🚀 Extracting GPS from: {selected_seq}...")
    
    # Sort files alphabetically to ensure index follows the walk order
    files = sorted(os.listdir(folder_path))
    counter = 0

    for filename in files:
        if filename.lower().endswith(image_extensions):
            full_path = os.path.join(folder_path, filename)
            
            # Normalize path to forward slashes for cross-platform stability
            clean_path = full_path.replace('\\', '/')
            
            lat, long = get_exif_gps(full_path)
            
            if lat is not None:
                data_rows.append([counter, long, lat, clean_path])
                counter += 1
            else:
                # Log skipped images so you know if data is missing
                print(f"  ⏭️  Skipped {filename} (No GPS data)")

    # 5. Write to CSV
    if data_rows:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'long', 'lat', 'file_path'])
            writer.writerows(data_rows)
        
        print(f"\n✅ SUCCESS!")
        print(f"📄 Saved {len(data_rows)} coordinates to: {output_csv}")
    else:
        print(f"\n❌ Failed: No images with GPS data were found in {selected_seq}")

if __name__ == "__main__":
    main()