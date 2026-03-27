from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .data import resolve_split_root


def get_default_data_root(config_path: str | Path = "configs/default.yaml") -> str:
    config = load_config(config_path)
    return config["data"]["root"]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def discover_training_runs(outputs_root: str | Path = "outputs") -> list[dict[str, Any]]:
    root = Path(outputs_root)
    if not root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True):
        history_path = run_dir / "history.json"
        if not history_path.is_file():
            continue

        history_payload = _read_json(history_path)
        history = history_payload.get("history", [])
        resolved_config = _read_yaml(run_dir / "resolved_config.yaml") if (run_dir / "resolved_config.yaml").is_file() else {}
        test_metrics = _read_json(run_dir / "test_metrics.json") if (run_dir / "test_metrics.json").is_file() else {}
        dataset_summary = _read_json(run_dir / "dataset_summary.json") if (run_dir / "dataset_summary.json").is_file() else {}

        best_val_accuracy = max((row.get("val_accuracy", 0.0) for row in history), default=0.0)
        latest_epoch = history[-1].get("epoch", 0) if history else 0
        runs.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "model_name": resolved_config.get("model", {}).get("name", run_dir.name),
                "epochs_completed": latest_epoch,
                "best_val_accuracy": best_val_accuracy,
                "test_accuracy": test_metrics.get("accuracy"),
                "class_count": dataset_summary.get("class_count"),
                "updated_at": run_dir.stat().st_mtime,
            }
        )
    return runs


def load_training_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir)
    return {
        "history": _read_json(target / "history.json").get("history", []),
        "test_metrics": _read_json(target / "test_metrics.json") if (target / "test_metrics.json").is_file() else {},
        "dataset_summary": _read_json(target / "dataset_summary.json") if (target / "dataset_summary.json").is_file() else {},
        "resolved_config": _read_yaml(target / "resolved_config.yaml") if (target / "resolved_config.yaml").is_file() else {},
    }


def scan_dataset_overview(data_root: str | Path) -> dict[str, Any]:
    split_root = resolve_split_root(data_root)
    split_counts: dict[str, int] = {}
    per_class_rows: list[dict[str, Any]] = []
    sample_images: list[dict[str, Any]] = []

    for split_name in ("train", "val", "test"):
        split_dir = split_root / split_name
        if not split_dir.is_dir():
            continue

        split_total = 0
        class_directories = sorted(path for path in split_dir.iterdir() if path.is_dir())
        for class_dir in class_directories:
            image_files = sorted(path for path in class_dir.iterdir() if path.is_file())
            image_count = len(image_files)
            split_total += image_count
            per_class_rows.append(
                {
                    "split": split_name,
                    "class_name": class_dir.name,
                    "image_count": image_count,
                }
            )

            if len(sample_images) < 9 and image_files:
                sample_images.append(
                    {
                        "split": split_name,
                        "class_name": class_dir.name,
                        "path": str(image_files[0]),
                    }
                )

        split_counts[split_name] = split_total

    class_names = sorted({row["class_name"] for row in per_class_rows})
    train_counts = Counter({row["class_name"]: row["image_count"] for row in per_class_rows if row["split"] == "train"})
    return {
        "split_root": str(split_root),
        "class_count": len(class_names),
        "class_names": class_names,
        "split_counts": split_counts,
        "per_class_rows": per_class_rows,
        "sample_images": sample_images,
        "top_train_classes": train_counts.most_common(10),
    }
