from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet 中药细粒度分类评估脚本")
    parser.add_argument("--checkpoint", required=True, help="训练得到的 checkpoint，例如 outputs/resnet50_tcm/best.pt")
    parser.add_argument("--data-root", default=None, help="覆盖数据根目录")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="评估的数据切分")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="覆盖 DataLoader worker 数")
    parser.add_argument("--device", default=None, help="指定设备，如 cuda 或 cpu")
    parser.add_argument("--limit-batches", type=int, default=None, help="仅评估前 N 个 batch，用于快速验证")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = None

    try:
        import torch
        from torch import nn

        from caoyao_resnet import build_model, create_dataloaders, evaluate, get_device, setup_logger

        checkpoint_path = Path(args.checkpoint)
        logger = setup_logger(
            f"evaluate.{checkpoint_path.parent.name}.{args.split}",
            checkpoint_path.parent / f"evaluate_{args.split}.log",
            level=logging.INFO,
        )
        logger.info("评估脚本启动")
        logger.info("Checkpoint: %s", checkpoint_path)

        device = get_device(args.device)
        logger.info("运行设备: %s", device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        if args.data_root:
            config["data"]["root"] = args.data_root
        if args.batch_size is not None:
            config["data"]["batch_size"] = args.batch_size
        if args.num_workers is not None:
            config["data"]["num_workers"] = args.num_workers

        dataloaders, class_names, dataset_sizes = create_dataloaders(
            data_root=config["data"]["root"],
            image_size=config["data"]["image_size"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
            device=device,
        )
        logger.info("数据目录: %s", config["data"]["root"])
        logger.info("数据集样本数: %s", dataset_sizes)
        if args.split not in dataloaders:
            raise ValueError(f"当前数据集中不存在 {args.split} 切分")

        model = build_model(
            name=checkpoint["model_name"],
            num_classes=len(class_names),
            pretrained=False,
            dropout=config["model"]["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(
            model=model,
            loader=dataloaders[args.split],
            criterion=criterion,
            device=device,
            split_name=args.split,
            limit_batches=args.limit_batches,
        )
        logger.info("%s | loss=%.4f acc=%.4f", args.split, metrics["loss"], metrics["accuracy"])
    except Exception:
        if logger is not None:
            logger.exception("评估脚本执行失败")
        raise


if __name__ == "__main__":
    main()
