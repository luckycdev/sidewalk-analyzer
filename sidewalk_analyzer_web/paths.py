from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_media_path(path_str: str) -> Path | None:
    """Resolve a relative or absolute path to an existing file under APP_ROOT."""
    raw = path_str.strip()
    if not raw:
        return None

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (APP_ROOT / raw).resolve()
    else:
        try:
            candidate = candidate.resolve()
        except OSError:
            return None

    try:
        candidate.relative_to(APP_ROOT)
    except ValueError:
        return None

    return candidate if candidate.is_file() else None


def results_artifact_paths(sequence_id: str) -> tuple[Path, Path]:
    """Per-sequence results CSV and GERS GeoJSON paths (may not exist yet)."""
    safe = Path(sequence_id).name  # basename only
    return (
        APP_ROOT / f"results_{safe}.csv",
        APP_ROOT / f"results_{safe}.gers.geojson",
    )

