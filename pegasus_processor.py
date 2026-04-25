import json
import re
import boto3

class PegasusProcessor:
    def __init__(self, region="us-east-1", model_id="twelvelabs.pegasus-1.2"):
        """
        Initialize AWS Bedrock client for Pegasus
        """
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    # -------------------------------
    # STEP 1: Send video clip to Pegasus
    # -------------------------------
    def analyze_clip(self, video_bytes, prompt=None):
        """
        Sends a video clip to Pegasus and returns raw text output
        """

        if prompt is None:
            prompt = (
                "Describe the sidewalk in this video. "
                "Include width (narrow/standard/wide), condition "
                "(good/fair/poor), and curb ramp presence."
            )

        body = {
            "input": {
                "video": video_bytes,  # binary video data
                "prompt": prompt
            }
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())

        # Adjust depending on actual response format
        text_output = result.get("output", {}).get("text", "")

        return text_output


    # -------------------------------
    # STEP 2: Parse Pegasus text output
    # -------------------------------
    def parse_description(self, text):
        """
        Convert Pegasus text into structured attributes
        """

        text = text.lower()

        # ---- WIDTH CLASS ----
        if "narrow" in text:
            width_class = "narrow"
        elif "wide" in text:
            width_class = "wide"
        elif "standard" in text or "moderate" in text:
            width_class = "standard"
        else:
            width_class = "unknown"

        # ---- CONDITION ----
        if any(word in text for word in ["crack", "broken", "damaged", "uneven"]):
            condition = "poor"
        elif any(word in text for word in ["good", "clean", "well-maintained"]):
            condition = "good"
        elif any(word in text for word in ["fair", "average"]):
            condition = "fair"
        else:
            condition = "unknown"

        # ---- CURB RAMP ----
        if "no curb ramp" in text or "missing ramp" in text:
            curb_ramp = "missing"
        elif "curb ramp" in text:
            curb_ramp = "present"
        else:
            curb_ramp = "unknown"

        return {
            "width_class_pegasus": width_class,
            "condition": condition,
            "curb_ramp": curb_ramp,
            "raw_description": text
        }


    # -------------------------------
    # STEP 3: Combine with width math
    # -------------------------------
    def combine_with_math(self, math_width, math_class, pegasus_data):
        """
        Combine mathematical measurement with Pegasus interpretation
        """

        confidence = 1.0

        # Adjust confidence if mismatch
        if pegasus_data["width_class_pegasus"] != "unknown":
            if pegasus_data["width_class_pegasus"] != math_class:
                confidence -= 0.2

        result = {
            "width_meters": math_width,
            "width_class_math": math_class,
            "width_class_pegasus": pegasus_data["width_class_pegasus"],
            "condition": pegasus_data["condition"],
            "curb_ramp": pegasus_data["curb_ramp"],
            "confidence": round(confidence, 2),
            "description": pegasus_data["raw_description"]
        }

        return result


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":

    # Initialize
    pegasus = PegasusProcessor()

    # Load a sample video clip (5–10 sec)
    with open("sidewalk_clip.mp4", "rb") as f:
        video_bytes = f.read()

    # Step 1: Analyze clip
    description = pegasus.analyze_clip(video_bytes)
    print("Pegasus Output:", description)

    # Step 2: Parse it
    parsed = pegasus.parse_description(description)
    print("Parsed:", parsed)

    # Step 3: Combine with your math result
    math_width = 1.6
    math_class = "standard"

    final = pegasus.combine_with_math(math_width, math_class, parsed)
    print("Final Output:", final)
