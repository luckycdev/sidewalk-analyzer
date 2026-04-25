from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import boto3

from .config import Settings
from .types import SegmentCandidate


MARENGO_MODEL_ID = "twelvelabs.marengo-embed-3-0-v1:0"


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expected s3://bucket/key")
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    return bucket, key


def marengo_prefilter(
    *,
    s3_video_uri: str,
    settings: Settings,
    threshold: float,
    run_dir: Path,
) -> tuple[list[SegmentCandidate], dict[str, Any]]:
    """
    Submits an async embedding job to Bedrock and uses the embedding output to score
    segments for the query 'sidewalk or pedestrian infrastructure'.

    Notes:
    - Bedrock async invoke returns results to S3 (outputLocation).
    - Exact response schema may differ slightly by model version; we persist raw outputs
      in run_dir/logs for audit/debug and keep parsing resilient.
    """

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Bedrock runtime clients:
    # - "bedrock-runtime" for InvokeModel / StartAsyncInvoke (runtime actions)
    runtime = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    in_bucket, in_key = _parse_s3_uri(s3_video_uri)
    out_key_prefix = f"{settings.s3_prefix}{settings.bedrock_output_s3_prefix}marengo/"
    out_s3_uri = f"s3://{settings.s3_bucket}/{out_key_prefix}"

    request_body = {
        # Model-specific payload; we persist raw outputs and keep parsing resilient.
        "input": {"s3Uri": s3_video_uri},
        "query": "sidewalk or pedestrian infrastructure",
    }

    # Start async job
    start = runtime.start_async_invoke(
        modelId=MARENGO_MODEL_ID,
        modelInput=request_body,
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": out_s3_uri}},
    )
    invocation_arn = start.get("invocationArn") or start.get("InvocationArn")
    if not invocation_arn:
        raise RuntimeError(f"Unexpected StartAsyncInvoke response: {start}")

    # Poll until complete
    t0 = time.time()
    status = "InProgress"
    output_s3_uri: str | None = None
    failure_message: str | None = None
    while True:
        info = runtime.get_async_invoke(invocationArn=invocation_arn)
        status = info.get("status") or info.get("Status") or status
        output_s3_uri = info.get("outputDataConfig", {}).get("s3Uri") or info.get("OutputDataConfig", {}).get("S3Uri") or output_s3_uri
        failure_message = info.get("failureMessage") or info.get("FailureMessage") or failure_message
        if status in {"Completed", "Failed", "Stopped", "Expired"}:
            (logs_dir / "marengo_async_status.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            break
        time.sleep(5)

    if status != "Completed":
        raise RuntimeError(f"Marengo async job did not complete (status={status}): {failure_message or 'unknown error'}")

    # Read results from S3 output prefix
    s3 = boto3.client("s3", region_name=settings.aws_region)
    out_bucket, out_prefix = _parse_s3_uri(output_s3_uri or out_s3_uri)
    # List objects and pick the newest JSON
    listed = s3.list_objects_v2(Bucket=out_bucket, Prefix=out_prefix)
    objects = listed.get("Contents") or []
    json_keys = [obj["Key"] for obj in objects if str(obj.get("Key", "")).lower().endswith(".json")]
    if not json_keys:
        raise RuntimeError(f"No JSON output found under {output_s3_uri or out_s3_uri}")

    # Heuristic: choose last modified newest
    newest = max(objects, key=lambda o: o.get("LastModified") or 0)
    output_key = newest.get("Key")
    if not output_key:
        output_key = json_keys[-1]

    payload = s3.get_object(Bucket=out_bucket, Key=output_key)["Body"].read().decode("utf-8")
    (logs_dir / "marengo_output.json").write_text(payload, encoding="utf-8")
    data = json.loads(payload)

    # Parsing strategy: be permissive; if we cannot find segment scores, fall back to one whole-video segment.
    candidates: list[SegmentCandidate] = []

    # Common patterns we handle:
    # - {"segments":[{"start":..,"end":..,"score":..}, ...]}
    # - {"results":[...]} / nested
    segments = None
    if isinstance(data, dict):
        for k in ("segments", "segmentScores", "results", "data"):
            if k in data and isinstance(data[k], list):
                segments = data[k]
                break

    if isinstance(segments, list):
        idx = 0
        for item in segments:
            if not isinstance(item, dict):
                continue
            start_s = item.get("start_s") or item.get("start") or item.get("startSec") or item.get("startSeconds")
            end_s = item.get("end_s") or item.get("end") or item.get("endSec") or item.get("endSeconds")
            score = item.get("score") or item.get("similarity") or item.get("confidence")
            try:
                start_f = float(start_s)
                end_f = float(end_s)
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            if score_f < threshold:
                continue
            candidates.append(
                SegmentCandidate(
                    segment_id=f"segment_{idx:03d}",
                    start_s=max(0.0, start_f),
                    end_s=max(start_f, end_f),
                    score=score_f,
                )
            )
            idx += 1

    stats: dict[str, Any] = {
        "invocation_arn": invocation_arn,
        "output_s3_uri": output_s3_uri or out_s3_uri,
        "output_key": output_key,
        "threshold": threshold,
        "candidates": len(candidates),
        "elapsed_s": round(time.time() - t0, 2),
    }

    if not candidates:
        from .video_clips import get_video_duration_s

        duration = get_video_duration_s_from_s3_uri(s3_video_uri, settings=settings)
        candidates = [SegmentCandidate(segment_id="segment_000", start_s=0.0, end_s=duration, score=1.0)]
        stats["fallback_whole_video"] = True

    (logs_dir / "marengo_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return candidates, stats


def get_video_duration_s_from_s3_uri(s3_uri: str, *, settings: Settings) -> float:
    """
    Duration estimation fallback when Marengo parsing can't produce segments.
    Downloads nothing; best effort: return 60s if unknown.
    """

    # Without downloading, we can't reliably know duration. Keep safe default.
    # Caller can override by skipping Marengo in early iteration.
    _ = (s3_uri, settings)
    return 60.0

