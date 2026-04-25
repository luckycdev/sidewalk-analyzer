from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sidewalk_analyzer", description="MP4 → sidewalk inventory GeoParquet pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run full pipeline on a video")
    run.add_argument("--video", required=True, help="Path to input .mp4")
    run.add_argument("--csv", required=False, default="", help="Optional CSV manifest with lat/lon/bearing metadata")
    run.add_argument("--out", required=True, help="Output directory (created if missing)")
    run.add_argument("--run-id", default="", help="Optional run id (defaults to timestamp)")
    run.add_argument("--no-s3", action="store_true", help="Skip S3 upload (for local iteration)")
    run.add_argument("--skip-marengo", action="store_true", help="Skip Marengo prefilter and treat whole video as one segment")
    run.add_argument("--threshold", type=float, default=None, help="Override MARENGO_THRESHOLD")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings()

    from .pipeline import run_pipeline  # local import keeps CLI fast if deps missing

    result = run_pipeline(
        video_path=Path(args.video),
        csv_path=Path(args.csv) if str(args.csv or "").strip() else None,
        out_dir=Path(args.out),
        run_id=args.run_id or None,
        settings=settings,
        no_s3=bool(args.no_s3),
        skip_marengo=bool(args.skip_marengo),
        threshold_override=args.threshold,
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

