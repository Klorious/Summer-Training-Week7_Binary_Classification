import argparse
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import RoadImageDataset
from models import get_binary_model


TRAIN_CSV = "data/splits/train.csv"
VAL_CSV = "data/splits/val.csv"
REFERENCE_CSV = "data/splits/reference_100.csv"
IMG_DIR = "data/raw"
CHECKPOINT_DIR = Path("outputs/checkpoints")
RUN_MANIFEST_DIR = Path("outputs/training_runs")
DEFAULT_WANDB_ENTITY = "klorius-"
DEFAULT_WANDB_PROJECT = "training-Unit7-Binary-413410020"
REQUIRED_PYTHON = (3, 11)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Unit 7 binary Good/Bad image classifier."
    )
    parser.add_argument(
        "--config",
        default="configs/resnet18.yaml",
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 16 train/validation images and run only one epoch.",
    )
    parser.add_argument(
        "--run-name-suffix",
        default="",
        help="Optional suffix appended to the W&B run name.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B logging mode.",
    )
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--allow-non-python311",
        action="store_true",
        help="Permit diagnostics outside Python 3.11. Never use for formal runs.",
    )
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make the selected run reproducible on the same software/hardware stack.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def learning_rate_text(value):
    """Return a stable, human-readable learning-rate string for filenames."""
    return format(float(value), ".10g")


def build_checkpoint_path(config, smoke_test=False):
    lr_text = learning_rate_text(config["learning_rate"])
    prefix = "smoke_" if smoke_test else ""
    filename = f"{prefix}best_{config['model_name']}_lr{lr_text}.pth"
    return CHECKPOINT_DIR / filename


def build_manifest_path(config, smoke_test=False):
    lr_text = learning_rate_text(config["learning_rate"])
    prefix = "smoke_" if smoke_test else ""
    filename = f"{prefix}{config['model_name']}_lr{lr_text}.json"
    return RUN_MANIFEST_DIR / filename


def require_formal_python(args):
    actual = sys.version_info[:2]
    if actual != REQUIRED_PYTHON and not args.allow_non_python311:
        raise RuntimeError(
            "Formal runs require Python 3.11, but this interpreter is "
            f"{platform.python_version()}. Activate training-unit7-py311. "
            "Use --allow-non-python311 only for non-formal diagnostics."
        )


def validate_data_protocol():
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    reference = pd.read_csv(REFERENCE_CSV)
    tables = {
        "train": (train, 300),
        "validation": (val, 100),
        "reference": (reference, 100),
    }
    for name, (frame, expected_size) in tables.items():
        required = {"image_id", "image_path", "human_label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} split missing columns: {sorted(missing)}")
        if len(frame) != expected_size:
            raise ValueError(
                f"{name} split should contain {expected_size} rows, "
                f"found {len(frame)}"
            )
        if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
            raise ValueError(f"{name} split image IDs must be present and unique")
        if not frame["human_label"].isin({"Bad", "Good"}).all():
            raise ValueError(f"{name} split contains invalid labels")

    ids = {
        name: set(frame["image_id"].astype(str))
        for name, (frame, _) in tables.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "reference"),
        ("validation", "reference"),
    ):
        overlap = ids[left] & ids[right]
        if overlap:
            raise ValueError(
                f"{left}/{right} overlap detected: {sorted(overlap)[:10]}"
            )
    print(
        "Data protocol validation passed: "
        "train=300, validation=100, locked reference=100, no overlap"
    )


def main():
    args = parse_args()
    require_formal_python(args)
    validate_data_protocol()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required = {
        "model_name",
        "pretrained",
        "learning_rate",
        "batch_size",
        "epochs",
        "optimizer",
        "input_size",
        "weight_decay",
        "random_seed",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(
            f"{args.config} is missing required fields: {', '.join(sorted(missing))}"
        )
    if str(config["optimizer"]).lower() != "adamw":
        raise ValueError("This training script currently supports optimizer: AdamW")

    config = dict(config)
    config["config_path"] = args.config
    config["task"] = "binary_classification"
    config["label_encoding"] = "Bad=0, Good=1"
    config["positive_class"] = "Good"
    config["checkpoint_selection_metric"] = "validation F1"
    config["freeze_backbone"] = False
    config["augmentation"] = (
        "Resize(256), RandomResizedCrop(input_size), RandomHorizontalFlip, "
        "ImageNet normalization"
    )
    config["validation_transform"] = (
        "Resize(256), CenterCrop(input_size), ImageNet normalization"
    )
    config["human_reference_used_for_training"] = False
    config["human_reference_used_for_validation"] = False
    config["human_reference_used_for_checkpoint_selection"] = False
    config["human_reference_used_for_hyperparameter_selection"] = False
    config["human_reference_usage"] = "final comparison only"
    config["training_split_size"] = 300
    config["validation_split_size"] = 100
    config["locked_reference_size"] = 100
    config["python_version"] = platform.python_version()
    config["pytorch_version"] = torch.__version__
    config["torchvision_version"] = torchvision.__version__
    config["wandb_entity"] = args.wandb_entity
    config["wandb_project"] = args.wandb_project
    config["smoke_test"] = bool(args.smoke_test)
    if args.smoke_test:
        config["epochs"] = 1

    checkpoint_path = build_checkpoint_path(config, smoke_test=args.smoke_test)
    manifest_path = build_manifest_path(config, smoke_test=args.smoke_test)
    config["checkpoint_path"] = checkpoint_path.as_posix()
    config["run_manifest_path"] = manifest_path.as_posix()

    lr_text = learning_rate_text(config["learning_rate"])
    run_name = (
        f"{config['model_name']}_pretrained_lr{lr_text}_bs{config['batch_size']}"
    )
    if args.run_name_suffix:
        run_name = f"{run_name}_{args.run_name_suffix.strip()}"
    if args.smoke_test:
        run_name = f"SMOKE_TEST_{run_name}"

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config=config,
        mode=args.wandb_mode,
        job_type="training",
        tags=[
            "basic",
            "binary",
            "python311",
            "smoke-test" if args.smoke_test else "formal",
            config["model_name"],
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(int(config["random_seed"]))
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(config["input_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = RoadImageDataset(TRAIN_CSV, IMG_DIR, transform=train_transform)
    val_dataset = RoadImageDataset(VAL_CSV, IMG_DIR, transform=val_transform)
    if args.smoke_test:
        train_dataset = Subset(train_dataset, range(min(16, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(16, len(val_dataset))))
        print("Smoke test enabled: at most 16 train/validation images, one epoch")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config["random_seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = get_binary_model(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_epoch = 0
    best_labels = None
    best_predictions = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_precision = precision_score(
            all_labels, all_predictions, zero_division=0
        )
        val_recall = recall_score(all_labels, all_predictions, zero_division=0)
        val_f1 = f1_score(all_labels, all_predictions, zero_division=0)

        print(
            f"Epoch [{epoch}/{config['epochs']}] "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
            f"F1: {val_f1:.4f}"
        )

        wandb.log(
            {
                "train/loss": epoch_train_loss,
                "train/accuracy": epoch_train_acc,
                "val/loss": epoch_val_loss,
                "val/accuracy": epoch_val_acc,
                "val/precision": val_precision,
                "val/recall": val_recall,
                "val/f1": val_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_epoch = int(epoch)
            best_labels = list(all_labels)
            best_predictions = list(all_predictions)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config["model_name"],
                    "learning_rate": float(config["learning_rate"]),
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                    "config": config,
                    "wandb_run_id": run.id if run is not None else None,
                    "wandb_run_url": run.url if run is not None else None,
                    "wandb_entity": args.wandb_entity,
                    "wandb_project": args.wandb_project,
                },
                checkpoint_path,
            )
            print(
                f"Saved best checkpoint at epoch {best_epoch}: "
                f"val F1={best_val_f1:.4f}"
            )

    if best_labels is None or best_predictions is None:
        raise RuntimeError("Training completed without a valid checkpoint")

    wandb.log(
        {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "best/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=best_labels,
                preds=best_predictions,
                class_names=["Bad", "Good"],
            ),
        }
    )
    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_f1"] = best_val_f1
        run.summary["selected_checkpoint"] = checkpoint_path.as_posix()

    manifest = {
        "task": "binary_classification",
        "formal_run": not args.smoke_test,
        "model_name": config["model_name"],
        "display_name": {
            "vgg16": "VGG16",
            "resnet18": "ResNet18",
            "resnet50": "ResNet50",
        }[config["model_name"]],
        "learning_rate": float(config["learning_rate"]),
        "input_size": int(config["input_size"]),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "checkpoint_path": checkpoint_path.as_posix(),
        "wandb_run_id": run.id if run is not None else None,
        "wandb_run_url": run.url if run is not None else None,
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "random_seed": int(config["random_seed"]),
        "config_path": args.config,
        "checkpoint_selection_metric": "validation F1",
        "human_reference_usage": "final comparison only",
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    wandb.finish()
    print(
        f"Training complete. Best epoch={best_epoch}, "
        f"best validation F1={best_val_f1:.4f}"
    )
    print(f"Selected checkpoint: {checkpoint_path}")
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
