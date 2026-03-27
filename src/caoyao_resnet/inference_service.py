from __future__ import annotations

import io
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD
from .logging_utils import safe_filename
from .models import build_model
from .utils import get_device


@dataclass
class CheckpointBundle:
    checkpoint_path: str
    model_name: str
    class_names: list[str]
    image_size: int
    config: dict[str, Any]
    device: str
    model: torch.nn.Module


def discover_checkpoints(outputs_root: str | Path = "outputs") -> list[Path]:
    root = Path(outputs_root)
    if not root.exists():
        return []
    candidates = [path for path in root.glob("*/best.pt") if path.is_file()]
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)


def read_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "checkpoint_path": str(Path(checkpoint_path)),
        "model_name": checkpoint["model_name"],
        "class_names": checkpoint["class_names"],
        "image_size": checkpoint["image_size"],
        "config": checkpoint["config"],
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
    }


def build_inference_transform(image_size: int) -> transforms.Compose:
    resize_size = max(256, int(image_size * 1.15))
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_checkpoint_bundle(checkpoint_path: str | Path, requested_device: str | None = None) -> CheckpointBundle:
    device = get_device(None if requested_device == "auto" else requested_device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        name=checkpoint["model_name"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
        dropout=checkpoint["config"]["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return CheckpointBundle(
        checkpoint_path=str(Path(checkpoint_path)),
        model_name=checkpoint["model_name"],
        class_names=checkpoint["class_names"],
        image_size=checkpoint["image_size"],
        config=checkpoint["config"],
        device=str(device),
        model=model,
    )


@torch.inference_mode()
def predict_pil_image(bundle: CheckpointBundle, image: Image.Image, top_k: int = 5) -> dict[str, Any]:
    preprocess = build_inference_transform(bundle.image_size)
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(bundle.device)

    started_at = time.perf_counter()
    logits = bundle.model(tensor)
    probabilities = torch.softmax(logits, dim=1)
    values, indices = probabilities.topk(k=min(top_k, len(bundle.class_names)), dim=1)
    duration_seconds = time.perf_counter() - started_at

    predictions = [
        {
            "class_name": bundle.class_names[class_index],
            "probability": round(probability, 6),
        }
        for probability, class_index in zip(values[0].tolist(), indices[0].tolist())
    ]
    top1 = predictions[0] if predictions else {"class_name": "N/A", "probability": 0.0}
    return {
        "predictions": predictions,
        "top1_class": top1["class_name"],
        "top1_probability": top1["probability"],
        "duration_seconds": duration_seconds,
    }


def predict_uploaded_images(
    bundle: CheckpointBundle,
    uploads: list[tuple[str, bytes]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for filename, payload in uploads:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        result = predict_pil_image(bundle, image, top_k=top_k)
        rows.append(
            {
                "filename": filename,
                "top1_class": result["top1_class"],
                "top1_probability": result["top1_probability"],
                "top_k_summary": " | ".join(
                    f"{item['class_name']}:{item['probability']:.4f}" for item in result["predictions"]
                ),
                "duration_seconds": round(result["duration_seconds"], 4),
            }
        )

    counts = Counter(row["top1_class"] for row in rows)
    dominant_class, dominant_count = counts.most_common(1)[0] if counts else ("N/A", 0)
    return {
        "rows": rows,
        "counts": dict(counts),
        "dominant_class": dominant_class,
        "dominant_count": dominant_count,
    }


def predict_video_frames(
    bundle: CheckpointBundle,
    video_path: str | Path,
    *,
    top_k: int = 5,
    sample_interval_seconds: float = 1.0,
    max_frames: int = 60,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0
    frame_interval = max(1, int(round(fps * sample_interval_seconds)))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration = total_frames / fps if total_frames > 0 else 0.0

    rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    frame_index = 0
    sampled_frames = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            result = predict_pil_image(bundle, image, top_k=top_k)
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 3),
                    "top1_class": result["top1_class"],
                    "top1_probability": result["top1_probability"],
                    "top_k_summary": " | ".join(
                        f"{item['class_name']}:{item['probability']:.4f}" for item in result["predictions"]
                    ),
                }
            )
            sampled_frames += 1
            if sampled_frames >= max_frames:
                break
        frame_index += 1

    capture.release()

    counts = Counter(row["top1_class"] for row in rows)
    dominant_class, dominant_count = counts.most_common(1)[0] if counts else ("N/A", 0)
    return {
        "rows": rows,
        "counts": dict(counts),
        "dominant_class": dominant_class,
        "dominant_count": dominant_count,
        "sampled_frames": sampled_frames,
        "fps": fps,
        "video_duration_seconds": video_duration,
        "processing_duration_seconds": time.perf_counter() - started_at,
        "frame_interval": frame_interval,
    }


def default_export_name(prefix: str, source_name: str, suffix: str = ".csv") -> str:
    return f"{safe_filename(prefix)}_{safe_filename(Path(source_name).stem)}{suffix}"
