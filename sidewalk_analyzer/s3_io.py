from __future__ import annotations

import mimetypes
from pathlib import Path

import boto3

from .config import Settings


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def upload_video_to_s3(*, video_path: Path, settings: Settings) -> str:
    if not settings.s3_bucket:
        raise RuntimeError("Missing S3_BUCKET in environment/.env")

    video_id = video_path.stem
    key = f"{settings.s3_prefix}videos/{video_id}{video_path.suffix.lower()}"
    content_type, _ = mimetypes.guess_type(str(video_path))
    if not content_type:
        content_type = "video/mp4"

    s3 = boto3.client("s3", region_name=settings.aws_region)
    s3.upload_file(
        Filename=str(video_path),
        Bucket=settings.s3_bucket,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )
    return _s3_uri(settings.s3_bucket, key)


def presign_s3_uri(*, s3_uri: str, settings: Settings, expires_s: int = 3600) -> str:
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expected s3://bucket/key")
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    s3 = boto3.client("s3", region_name=settings.aws_region)
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(expires_s),
    )

