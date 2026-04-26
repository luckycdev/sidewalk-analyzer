import boto3
import json
import pandas as pd
import requests
import time

# ---------------- CONFIG ----------------
BUCKET_NAME = "sidewalk-analyzer-vincent"
LOCAL_FILE = "./path_timelapse.mp4"
S3_KEY = "videos/path_timelapse.mp4"

CSV_FILE = "./image_coordinates.csv"
OUTPUT_FILE = "mapped_results.json"

REGION = "us-east-1"
MODEL_ID = "twelvelabs.pegasus-1-2-v1:0"
BUCKET_OWNER = "564203970240"

# Overture / Overpass API
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# ----------------------------------------

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# ---------------- STEP 1: Upload ----------------
print("Uploading video...")
s3.upload_file(LOCAL_FILE, BUCKET_NAME, S3_KEY)

s3_uri = f"s3://{BUCKET_NAME}/{S3_KEY}"
print("Uploaded:", s3_uri)

# ---------------- STEP 2: Call Pegasus ----------------
prompt = """
You are analyzing a 1 FPS walking sidewalk video.

Find ALL timestamps where sidewalk damage is clearly visible.

Include:
- cracks
- potholes
- broken pavement
- uneven surfaces
- hazards

Treat each second independently.

Return ONLY JSON:
[
  {
    "start": number,
    "end": number,
    "severity": "low | medium | high",
    "description": string
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

print("Calling Pegasus...")
response = bedrock.invoke_model(
    modelId=MODEL_ID,
    body=json.dumps(body)
)

result = json.loads(response["body"].read())

# Parse Pegasus output
raw_message = result.get("message", "[]")

try:
    detections = json.loads(raw_message)
except:
    detections = []

print("Detections:", detections)

# ---------------- STEP 3: Load CSV ----------------
df = pd.read_csv(CSV_FILE)

# ---------------- STEP 4: Expand timestamps → frames ----------------
flagged_points = []

for det in detections:
    start = int(det["start"])
    end = int(det["end"])

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

print(f"Total flagged frames: {len(flagged_points)}")

# ---------------- STEP 5: Find nearest sidewalk ----------------
def get_nearest_sidewalk(lat, lon):
    query = f"""
    [out:json];
    way(around:20,{lat},{lon})["highway"="footway"]["footway"="sidewalk"];
    out center 1;
    """

    try:
        res = requests.post(OVERPASS_URL, data=query)
        data = res.json()

        if data["elements"]:
            el = data["elements"][0]

            return {
                "id": el["id"],
                "lat": el["center"]["lat"],
                "lon": el["center"]["lon"]
            }

    except Exception as e:
        print("Overpass error:", e)

    return None

# ---------------- STEP 6: Attach sidewalk IDs ----------------
final_results = []

print("Querying sidewalks...")

for point in flagged_points:
    sidewalk = get_nearest_sidewalk(point["lat"], point["lon"])

    result_entry = {
        **point,
        "sidewalk": sidewalk
    }

    final_results.append(result_entry)

    time.sleep(0.5)  # avoid rate limiting

# ---------------- STEP 7: Save ----------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

print(f"Saved to {OUTPUT_FILE}")