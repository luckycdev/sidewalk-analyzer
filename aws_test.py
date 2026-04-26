import boto3
import json
import pandas as pd
import os
import overturemaps
import geopandas as gpd
from shapely.geometry import Point

# ---------------- CONFIG ----------------
BUCKET_NAME = "sidewalk-analyzer-vincent"
CSV_DIR = "./csvs/"
REGION = "us-east-1"
MODEL_ID = "twelvelabs.pegasus-1-2-v1:0"
BUCKET_OWNER = "564203970240"

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def main():
    # ---------------- STEP 1: VIDEO SELECTION ----------------
    videos = sorted([f for f in os.listdir('.') if f.endswith('.mp4')])
    if not videos:
        print("No videos found.")
        return

    for idx, v in enumerate(videos):
        print(f"[{idx}] {v}")
    
    choice = int(input("\nSelect video index: "))
    local_video = videos[choice]
    sequence_id = os.path.splitext(local_video)[0]

    # ---------------- STEP 2: UPLOAD ----------------
    s3_key = f"videos/{local_video}"
    print(f"🚀 Uploading {local_video}...")
    s3.upload_file(local_video, BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    # ---------------- STEP 3: PROMPT ----------------
    prompt = """
You are a highly conservative civil engineering inspector analyzing a 1 FPS sidewalk video.

Your goal:
Identify individual frames (seconds) where the sidewalk is CLEARLY damaged.

STRICT RULES:
- Only report a frame if you are confident there is real structural damage
- If unsure, DO NOT include the frame
- It is OK to return an empty list

ONLY detect:
- large cracks
- potholes
- missing chunks of sidewalk
- major uneven slabs or trip hazards

IGNORE:
- shadows, lighting, blur
- minor wear or texture
- dirt or discoloration

IMPORTANT:
Treat EACH second independently. Do NOT group frames into ranges.

Return ONLY valid JSON:

[
  {
    "time": number,
    "severity": "medium | high",
    "description": "brief factual description"
  }
]
"""

    body = {
        "inputPrompt": prompt,
        "mediaSource": {
            "s3Location": {
                "uri": s3_uri,
                "bucketOwner": BUCKET_OWNER
            }
        }
    }

    print("🧠 Pegasus analyzing frames...")
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    raw_res = response["body"].read().decode("utf-8")
    res_json = json.loads(raw_res)

    msg = res_json.get("message", "[]")
    detections_raw = json.loads(
        msg.replace("```json", "").replace("```", "").strip()
    )

    print(f"📍 Raw detections: {len(detections_raw)}")

    # ---------------- STEP 4: NORMALIZE DETECTIONS ----------------
    detections = []

    for det in detections_raw:
        # Case 1: single frame
        if "time" in det:
            detections.append({
                "time": int(det["time"]),
                "severity": det.get("severity", ""),
                "description": det.get("description", "")
            })

        # Case 2: range → expand
        elif "start" in det and "end" in det:
            start = int(det["start"])
            end = int(det["end"])

            for t in range(start, end + 1):
                detections.append({
                    "time": t,
                    "severity": det.get("severity", ""),
                    "description": det.get("description", "")
                })

    print(f"✅ Normalized to {len(detections)} frame-level detections.")

    # ---------------- STEP 5: LOAD CSV ----------------
    csv_path = os.path.join(CSV_DIR, f"{sequence_id}.csv")
    df = pd.read_csv(csv_path)

    # ---------------- STEP 6: DOWNLOAD OVERTURE DATA ----------------
    print("🌍 Downloading Overture sidewalk data...")

    bbox = (
        df['long'].min() - 0.001,
        df['lat'].min() - 0.001,
        df['long'].max() + 0.001,
        df['lat'].max() + 0.001
    )

    table = overturemaps.record_batch_reader("segment", bbox).read_all()
    gdf = gpd.GeoDataFrame.from_arrow(table)

    sidewalk_keys = {'sidewalk', 'footway', 'pedestrian', 'path'}

    # ---------------- STEP 7: FRAME ISSUE STORAGE ----------------
    frame_issues = {i: [] for i in range(len(df))}

    for det in detections:
        t = det["time"]
        if t < len(df):
            desc = f"{det['severity']}: {det['description']}"
            frame_issues[t].append(desc)

    # ---------------- STEP 8: BUILD CSV ----------------
    rows = []

    for i in range(len(df)):
        row = df.iloc[i]
        p = Point(row["long"], row["lat"])

        gdf["dist"] = gdf.geometry.distance(p)

        is_sidewalk = gdf["subclass"].isin(sidewalk_keys)
        sidewalks = gdf[is_sidewalk & (gdf["dist"] < 0.00015)]

        if not sidewalks.empty:
            best = sidewalks.loc[sidewalks["dist"].idxmin()]
        else:
            best = gdf.loc[gdf["dist"].idxmin()]

        issues = " | ".join(frame_issues[i]) if frame_issues[i] else ""

        rows.append({
            "frame": i,
            "lat": row["lat"],
            "lon": row["long"],
            "gers_id": best["id"],
            "segment_type": best.get("subclass", ""),
            "issues": issues
        })

    out_df = pd.DataFrame(rows)

    out_file = f"results_{sequence_id}.csv"
    out_df.to_csv(out_file, index=False)

    print(f"✅ CSV saved: {out_file}")

if __name__ == "__main__":
    main()