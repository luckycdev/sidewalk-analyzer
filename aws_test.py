import boto3
import json

# ---------------- CONFIG ----------------
BUCKET_NAME = "sidewalk-analyzer-vincent"
LOCAL_FILE = "./path_timelapse.mp4"
S3_KEY = "videos/path_timelapse.mp4"
BUCKET_OWNER = "564203970240"

REGION = "us-east-1"

# IMPORTANT: keep YOUR working model id here
MODEL_ID = "twelvelabs.pegasus-1-2-v1:0"
# ----------------------------------------

# AWS clients
s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# ---------------- STEP 1: Upload video ----------------
print("Uploading video to S3...")

s3.upload_file(LOCAL_FILE, BUCKET_NAME, S3_KEY)

s3_uri = f"s3://{BUCKET_NAME}/{S3_KEY}"
print("Uploaded to:", s3_uri)

# ---------------- STEP 2: Build request ----------------
body = {
    "inputPrompt": """
You are analyzing a street-level video.

Find all timestamps where the sidewalk is in poor condition.

Include:
- cracks
- potholes
- uneven pavement
- broken surfaces
- hazards

Return ONLY valid JSON:

[
  {
    "start": number,
    "end": number,
    "description": string
  }
]
""",
    "mediaSource": {
        "s3Location": {
            "uri": s3_uri,
            "bucketOwner": BUCKET_OWNER
        }
    }
}

# ---------------- STEP 3: Call Bedrock ----------------
print("Calling Pegasus...")

response = bedrock.invoke_model(
    modelId=MODEL_ID,
    body=json.dumps(body)
)

result = json.loads(response["body"].read())

# ---------------- STEP 4: Output ----------------
print("\nRAW RESPONSE:\n")
print(json.dumps(result, indent=2))

# Try to extract usable text if present
try:
    # different models wrap output differently
    text = result.get("output", result)

    if isinstance(text, dict):
        text = json.dumps(text, indent=2)

    print("\nPROCESSED OUTPUT:\n")
    print(text)

except Exception as e:
    print("\nCould not parse output cleanly.")
    print("Error:", str(e))