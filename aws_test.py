import boto3
import json
import pandas as pd
import requests
import time
import os

# ---------------- CONFIG ----------------
BUCKET_NAME = "sidewalk-analyzer-vincent"
CSV_DIR = "./csvs/"
REGION = "us-east-1"
MODEL_ID = "twelvelabs.pegasus-1-2-v1:0"
BUCKET_OWNER = "564203970240"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def main():
    # 1. Select the video to analyze
    videos = [f for f in os.listdir('.') if f.endswith('.mp4')]
    
    if not videos:
        print("❌ No .mp4 files found in the current directory.")
        return

    print("\n--- Available Videos for Analysis ---")
    for idx, vid in enumerate(videos):
        print(f"[{idx}] {vid}")

    try:
        choice = int(input("\nSelect video to analyze: "))
        local_video = videos[choice]
        sequence_id = local_video.replace('.mp4', '')
    except (ValueError, IndexError):
        print("❌ Invalid selection.")
        return

    # 2. Check for matching CSV
    csv_path = os.path.join(CSV_DIR, f"{sequence_id}.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Required CSV not found at {csv_path}")
        return

    # 3. Upload to S3
    s3_key = f"videos/{local_video}"
    print(f"🚀 Uploading {local_video} to S3...")
    s3.upload_file(local_video, BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    # 4. Call Pegasus
    prompt = """
    You are analyzing a 1 FPS walking sidewalk video.
    Find ALL timestamps where sidewalk damage is clearly visible.
    Include: cracks, potholes, broken pavement, uneven surfaces, hazards.
    Return ONLY JSON:
    [{"start": number, "end": number, "severity": "low|medium|high", "description": string}]
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

    print(f"🧠 Pegasus is analyzing {sequence_id} (this may take a minute)...")
    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    result = json.loads(response["body"].read())
    
    # Pegasus returns the JSON string inside a 'message' field usually
    detections = json.loads(result.get("message", "[]"))
    print(f"📍 Detected {len(detections)} damaged segments.")

    # 5. Map Detections to GPS via CSV
    df = pd.read_csv(csv_path)
    flagged_points = []

    for det in detections:
        start, end = int(det["start"]), int(det["end"])
        # Map timestamps to CSV rows (assuming 1 FPS)
        for t in range(start, end + 1):
            if t < len(df):
                row = df.iloc[t]
                flagged_points.append({
                    "frame": t,
                    "lat": row["lat"],
                    "long": row["long"],
                    "severity": det["severity"],
                    "description": det["description"]
                })

    # 6. Get Sidewalk IDs from OpenStreetMap (Overpass)
    def get_nearest_sidewalk(lat, lon):
        query = f"""
        [out:json];
        way(around:15,{lat},{lon})["highway"="footway"]["footway"="sidewalk"];
        out center 1;
        """
        try:
            res = requests.post(OVERPASS_URL, data=query, timeout=10)
            data = res.json()
            if data["elements"]:
                el = data["elements"][0]
                return {"id": el["id"], "lat": el["center"]["lat"], "lon": el["center"]["lon"]}
        except:
            return None
        return None

    final_results = []
    print(f"🗺️  Mapping {len(flagged_points)} frames to OpenStreetMap...")
    
    for point in flagged_points:
        sidewalk = get_nearest_sidewalk(point["lat"], point["long"])
        final_results.append({**point, "osm_sidewalk": sidewalk})
        time.sleep(0.3) # Avoid hitting API limits

    # 7. Save results
    output_file = f"results_{sequence_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ DONE! Results saved to {output_file}")

if __name__ == "__main__":
    main()