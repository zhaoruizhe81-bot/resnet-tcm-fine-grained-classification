from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet 中药细粒度分类单图预测脚本")
    parser.add_argument("--checkpoint", required=True, help="训练得到的 checkpoint，例如 outputs/resnet50_tcm/best.pt")
    parser.add_argument("--image", required=True, help="待预测图片路径")
    parser.add_argument("--top-k", type=int, default=5, help="输出前 K 个候选类别")
    parser.add_argument("--device", default=None, help="指定设备，如 cuda 或 cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms

    from caoyao_resnet import build_model, get_device, predict_topk
    from caoyao_resnet.data import IMAGENET_MEAN, IMAGENET_STD

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    config = checkpoint["config"]

    model = build_model(
        name=checkpoint["model_name"],
        num_classes=len(class_names),
        pretrained=False,
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((max(256, int(image_size * 1.15)), max(256, int(image_size * 1.15)))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    predictions = predict_topk(
        model=model,
        tensor=tensor,
        class_names=class_names,
        device=device,
        top_k=min(args.top_k, len(class_names)),
    )
    for index, item in enumerate(predictions, start=1):
        print(f"{index}. {item['class_name']}: {item['probability']:.6f}")


if __name__ == "__main__":
    main()
