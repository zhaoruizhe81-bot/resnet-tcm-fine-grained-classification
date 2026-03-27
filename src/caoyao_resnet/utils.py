from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict,
    config: dict,
    class_names: list[str],
) -> dict:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": config,
        "class_names": class_names,
        "model_name": config["model"]["name"],
        "image_size": config["data"]["image_size"],
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload
