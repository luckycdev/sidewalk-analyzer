from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_benchmarks(run_dir: Path) -> dict[str, Any]:
    """
    Lightweight hook: other modules can write timing JSON into logs/.
    This consolidates them for the manifest.
    """

    logs = run_dir / "logs"
    files = [
        logs / "timings.json",
        logs / "cost_estimate.json",
    ]
    out: dict[str, Any] = {}
    for f in files:
        payload = _read_json_if_exists(f)
        if payload is not None:
            out[f.stem] = payload
    return out

