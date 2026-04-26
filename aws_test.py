import boto3
import json
import pandas as pd
import os
import overturemaps
import math
from shapely.geometry import Point, LineString

# ---------------- CONFIG ----------------
BUCKET_NAME = "sidewalk-analyzer-vincent"
CSV_DIR = "./csvs/"
REGION = "us-east-1"
MODEL_ID = "twelvelabs.pegasus-1-2-v1:0"
BUCKET_OWNER = "564203970240"

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def get_point_to_line_dist(lon, lat, line_coords):
    """
    Calculates the minimum distance in meters between a point and a line segment.
    Uses the projection logic from your example for high accuracy.
    """
    p = Point(lon, lat)
    line = LineString(line_coords)
    # distance() in shapely on geographic coords is in degrees; 
    # we use a rough multiplier for meters or use the LineString logic
    return p.distance(line) * 111320.0 

def main():
    # 1. VIDEO SELECTION
    videos = sorted([f for f in os.listdir('.') if f.endswith('.mp4')])
    if not videos: return
    for idx, v in enumerate(videos): print(f"[{idx}] {v}")
    
    choice = int(input("\nSelect video index: "))
    local_video = videos[choice]
    sequence_id = os.path.splitext(local_video)[0]

    # 2. UPLOAD
    s3_key = f"videos/{local_video}"
    print(f"🚀 Uploading {local_video}...")
    s3.upload_file(local_video, BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    # 3. PEGASUS "EXPERT" PROMPT
    prompt = """
    You are a professional civil engineering inspector. Analyze this 1 FPS sidewalk video.
    Identify all structural hazards: vertical offsets > 1 inch, severe cracking (alligator cracking), 
    potholes, or missing sections of concrete.
    
    Return ONLY a JSON list:
    [{"start": sec, "end": sec, "severity": "low|medium|high", "description": "technical detail"}]
    """
    
    body = {
        "inputPrompt": prompt,
        "mediaSource": {"s3Location": {"uri": s3_uri, "bucketOwner": BUCKET_OWNER}}
    }

    print("🧠 Pegasus is scanning for damage...")
    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    
    raw_res = response["body"].read().decode('utf-8')
    res_json = json.loads(raw_res)
    msg = res_json.get("message", "[]")
    detections = json.loads(msg.replace('```json', '').replace('```', '').strip())
    
    print(f"📍 Found {len(detections)} hazardous areas.")

    # 4. DOWNLOAD OVERTURE BATCH
    csv_path = os.path.join(CSV_DIR, f"{sequence_id}.csv")
    df = pd.read_csv(csv_path)
    
    print("🌍 Downloading Overture segments for the route...")
    bbox = (df['long'].min()-0.001, df['lat'].min()-0.001, df['long'].max()+0.001, df['lat'].max()+0.001)
    
    import geopandas as gpd
    table = overturemaps.record_batch_reader("segment", bbox).read_all()
    local_gdf = gpd.GeoDataFrame.from_arrow(table)

    # Filter for sidewalks/footways specifically
    sidewalk_keys = {'sidewalk', 'footway', 'pedestrian', 'path'}
    
    # 5. SPATIAL MATCHING
    final_output = []
    for det in detections:
        for t in range(int(det["start"]), int(det["end"]) + 1):
            if t < len(df):
                row = df.iloc[t]
                p = Point(row["long"], row["lat"])
                
                # logic: Find nearest segment, prioritizing sidewalks
                # We calculate distances to all segments in our cached area
                local_gdf['dist'] = local_gdf.geometry.distance(p)
                
                # Check for sidewalks within a 15-meter threshold first
                is_sidewalk = local_gdf['subclass'].isin(sidewalk_keys)
                sidewalks_nearby = local_gdf[is_sidewalk & (local_gdf['dist'] < 0.00015)]
                
                if not sidewalks_nearby.empty:
                    best_seg = sidewalks_nearby.loc[sidewalks_nearby['dist'].idxmin()]
                else:
                    best_seg = local_gdf.loc[local_gdf['dist'].idxmin()]

                final_output.append({
                    "time": t,
                    "lat": row["lat"],
                    "lon": row["long"],
                    "hazard": det,
                    "gers_id": best_seg['id'],
                    "segment_type": best_seg.get('subclass', 'road')
                })

    # 6. SAVE
    out_file = f"results_{sequence_id}.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Report saved: {out_file}")

if __name__ == "__main__":
    main()