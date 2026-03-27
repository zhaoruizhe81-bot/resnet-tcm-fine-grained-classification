from __future__ import annotations

import copy
import platform
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_num_workers(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return 0 if platform.system().lower().startswith("win") else 4


def default_run_name(model_name: str) -> str:
    return f"{model_name}_tcm"


def resolve_runtime_config(
    raw_config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = merge_dict(raw_config, overrides or {})
    requested_run_name = ((overrides or {}).get("output") or {}).get("run_name")
    raw_default_run_name = default_run_name(raw_config["model"]["name"])
    current_run_name = config["output"].get("run_name")

    if not requested_run_name and (not current_run_name or current_run_name == raw_default_run_name):
        config["output"]["run_name"] = default_run_name(config["model"]["name"])

    config["data"]["root"] = str(Path(config["data"]["root"]))
    config["data"]["num_workers"] = default_num_workers(config["data"].get("num_workers"))
    config["output"]["root_dir"] = str(Path(config["output"]["root_dir"]))
    return config


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
