from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet 中药细粒度分类训练脚本")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--data-root", default=None, help="覆盖数据根目录")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--run-name", default=None, help="覆盖实验名称")
    parser.add_argument("--model-name", default=None, help="覆盖模型名称，如 resnet18/resnet50")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--num-workers", type=int, default=None, help="覆盖 DataLoader worker 数")
    parser.add_argument("--device", default=None, help="指定设备，如 cuda 或 cpu")
    parser.add_argument("--resume", default=None, help="恢复训练的 checkpoint")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="仅跑前 N 个训练 batch，用于快速验证")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="仅跑前 N 个验证 batch，用于快速验证")
    parser.add_argument("--limit-test-batches", type=int, default=None, help="仅跑前 N 个测试 batch，用于快速验证")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {"data": {}, "model": {}, "train": {}, "output": {}}
    if args.data_root:
        overrides["data"]["root"] = args.data_root
    if args.batch_size is not None:
        overrides["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["data"]["num_workers"] = args.num_workers
    if args.model_name:
        overrides["model"]["name"] = args.model_name
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.lr is not None:
        overrides["train"]["learning_rate"] = args.lr
    if args.resume:
        overrides["train"]["checkpoint"] = args.resume
    if args.limit_train_batches is not None:
        overrides["train"]["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        overrides["train"]["limit_val_batches"] = args.limit_val_batches
    if args.limit_test_batches is not None:
        overrides["train"]["limit_test_batches"] = args.limit_test_batches
    if args.output_dir:
        overrides["output"]["root_dir"] = args.output_dir
    if args.run_name:
        overrides["output"]["run_name"] = args.run_name
    return {key: value for key, value in overrides.items() if value}


def main() -> None:
    args = parse_args()

    import torch
    from torch import nn

    from caoyao_resnet import build_model, create_dataloaders, evaluate, get_device, set_seed, train_one_epoch
    from caoyao_resnet.config import load_config, resolve_runtime_config, save_config
    from caoyao_resnet.utils import checkpoint_payload, ensure_dir, save_json

    config = resolve_runtime_config(load_config(args.config), build_overrides(args))
    set_seed(config["seed"])

    device = get_device(args.device)
    dataloaders, class_names, dataset_sizes = create_dataloaders(
        data_root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        device=device,
    )

    model = build_model(
        name=config["model"]["name"],
        num_classes=len(class_names),
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["train"]["label_smoothing"])

    start_epoch = 1
    best_accuracy = 0.0
    resume_path = config["train"]["checkpoint"]
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint.get("metrics", {}).get("val_accuracy", 0.0)

    run_dir = ensure_dir(Path(config["output"]["root_dir"]) / config["output"]["run_name"])
    save_config(run_dir / "resolved_config.yaml", config)
    save_json(
        run_dir / "dataset_summary.json",
        {
            "class_count": len(class_names),
            "class_names": class_names,
            "dataset_sizes": dataset_sizes,
        },
    )

    history: list[dict] = []
    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=config["train"]["mixed_precision"],
            limit_batches=config["train"]["limit_train_batches"],
        )
        val_metrics = evaluate(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
            split_name="val",
            limit_batches=config["train"]["limit_val_batches"],
        )
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        save_json(run_dir / "history.json", {"history": history})

        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_record,
                config=config,
                class_names=class_names,
            ),
            run_dir / "last.pt",
        )

        if val_metrics["accuracy"] >= best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=epoch_record,
                    config=config,
                    class_names=class_names,
                ),
                run_dir / "best.pt",
            )

        print(
            f"Epoch {epoch:02d}/{config['train']['epochs']} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

    if "test" in dataloaders:
        best_checkpoint = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = evaluate(
            model=model,
            loader=dataloaders["test"],
            criterion=criterion,
            device=device,
            split_name="test",
            limit_batches=config["train"]["limit_test_batches"],
        )
        save_json(run_dir / "test_metrics.json", test_metrics)
        print(f"Test loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f}")

    print(f"训练完成，结果保存在: {run_dir}")


if __name__ == "__main__":
    main()
