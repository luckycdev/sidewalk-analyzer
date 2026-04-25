from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


FEATURES = ["sidewalk_present", "width_class", "surface_condition", "curb_ramp_status"]


def _find_ground_truth(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "ground_truth.geojson",
        run_dir.parent / "ground_truth.geojson",
        Path.cwd() / "ground_truth.geojson",
        Path.cwd() / "ground_truth.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_ground_truth(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        gdf = gpd.read_file(path)
        df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    if "segment_id" not in df.columns:
        raise ValueError("Ground truth must include a segment_id column")
    return df


def _metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, float]:
    # For binary: use pos_label=True
    if all(isinstance(v, (bool, int)) for v in y_true + y_pred):
        yt = [bool(v) for v in y_true]
        yp = [bool(v) for v in y_pred]
        return {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }
    # Multiclass macro average
    return {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def maybe_evaluate(run_dir: Path, *, clip_assets: list[Any], pegasus_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_path = _find_ground_truth(run_dir)
    if gt_path is None:
        return {"skipped": True, "reason": "ground truth not found"}

    gt = _load_ground_truth(gt_path)

    pred_rows: list[dict[str, Any]] = []
    for row in pegasus_rows:
        seg_id = str(row.get("segment_id", ""))
        clip = row.get("pegasus_clip") or {}
        still = row.get("pegasus_still") or {}
        pred_rows.append(
            {
                "segment_id": seg_id,
                "pred_clip_sidewalk_present": clip.get("sidewalk_present"),
                "pred_clip_width_class": clip.get("width_class"),
                "pred_clip_surface_condition": clip.get("surface_condition"),
                "pred_clip_curb_ramp_status": clip.get("curb_ramp_status"),
                "pred_still_sidewalk_present": still.get("sidewalk_present"),
                "pred_still_width_class": still.get("width_class"),
                "pred_still_surface_condition": still.get("surface_condition"),
                "pred_still_curb_ramp_status": still.get("curb_ramp_status"),
            }
        )
    pred = pd.DataFrame.from_records(pred_rows)

    merged = gt.merge(pred, on="segment_id", how="inner", suffixes=("", ""))
    if merged.empty:
        return {"skipped": True, "reason": "no overlapping segment_id between ground truth and predictions"}

    out: dict[str, Any] = {
        "skipped": False,
        "ground_truth_path": str(gt_path),
        "segments_evaluated": int(len(merged)),
        "clip": {},
        "still": {},
    }

    # Expect ground truth columns named exactly like FEATURES
    for feat in FEATURES:
        if feat not in merged.columns:
            continue
        clip_col = f"pred_clip_{feat}"
        still_col = f"pred_still_{feat}"
        if clip_col in merged.columns:
            out["clip"][feat] = _metrics(merged[feat].tolist(), merged[clip_col].tolist())
        if still_col in merged.columns:
            out["still"][feat] = _metrics(merged[feat].tolist(), merged[still_col].tolist())

    (run_dir / "exports").mkdir(parents=True, exist_ok=True)
    (run_dir / "exports" / "evaluation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out

