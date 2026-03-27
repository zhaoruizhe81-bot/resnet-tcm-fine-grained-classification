from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch
from torch import nn
from tqdm import tqdm


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / max(1, targets.size(0))


def _iter_batches(loader: Iterable, limit_batches: int | None):
    for batch_index, batch in enumerate(loader, start=1):
        yield batch_index, batch
        if limit_batches is not None and batch_index >= limit_batches:
            break


def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    use_amp: bool,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    loss_sum = 0.0
    correct = 0
    total = 0

    progress = tqdm(_iter_batches(loader, limit_batches), total=limit_batches or len(loader), desc="train", leave=False)
    for _, (images, labels) in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.cuda.amp.autocast if use_amp and device.type == "cuda" else nullcontext
        with autocast_context():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(1, total):.4f}")

    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
    }


@torch.inference_mode()
def evaluate(
    *,
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    split_name: str,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    progress = tqdm(_iter_batches(loader, limit_batches), total=limit_batches or len(loader), desc=split_name, leave=False)
    for _, (images, labels) in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(1, total):.4f}")

    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
    }


@torch.inference_mode()
def predict_topk(
    *,
    model: nn.Module,
    tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    top_k: int,
) -> list[dict[str, float | str]]:
    logits = model(tensor.to(device))
    probabilities = torch.softmax(logits, dim=1)
    values, indices = probabilities.topk(k=top_k, dim=1)

    results: list[dict[str, float | str]] = []
    for probability, class_index in zip(values[0].tolist(), indices[0].tolist()):
        results.append(
            {
                "class_name": class_names[class_index],
                "probability": round(probability, 6),
            }
        )
    return results
