from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_file(path: Path) -> None:
    """
    Minimal .env loader (no dependencies).
    - Only sets keys not already in os.environ.
    - Supports simple KEY=VALUE lines with optional quotes.
    """

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    aws_region: str
    s3_bucket: str
    s3_prefix: str
    bedrock_output_s3_prefix: str

    marengo_threshold: float
    clip_seconds: float
    clip_padding_seconds: float

    mapbox_access_token: str


def _getenv(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is None:
        return default or ""
    return value.strip()


def _getenv_float(name: str, default: float) -> float:
    raw = _getenv(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_settings(*, env_path: Path | None = None) -> Settings:
    if env_path is None:
        env_path = Path.cwd() / ".env"
    load_dotenv_file(env_path)

    s3_prefix = _getenv("S3_PREFIX", "sidewalk-analyzer/").strip()
    if s3_prefix and not s3_prefix.endswith("/"):
        s3_prefix += "/"

    out_prefix = _getenv("BEDROCK_OUTPUT_S3_PREFIX", "bedrock/").strip()
    if out_prefix and not out_prefix.endswith("/"):
        out_prefix += "/"

    return Settings(
        aws_region=_getenv("AWS_REGION", "us-east-1") or "us-east-1",
        s3_bucket=_getenv("S3_BUCKET", ""),
        s3_prefix=s3_prefix,
        bedrock_output_s3_prefix=out_prefix,
        marengo_threshold=_getenv_float("MARENGO_THRESHOLD", 0.5),
        clip_seconds=_getenv_float("CLIP_SECONDS", 8.0),
        clip_padding_seconds=_getenv_float("CLIP_PADDING_SECONDS", 1.0),
        mapbox_access_token=_getenv("MAPBOX_ACCESS_TOKEN", ""),
    )

