import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import get_binary_model


REFERENCE_CSV = "data/annotations/binary_reference_100.csv"
IMG_DIR = "data/raw"
OUTPUT_CSV = "outputs/predictions/reference_results.csv"
SELECTION_MANIFEST = "outputs/training_runs/selected_models.json"
DISPLAY_NAMES = {
    "vgg16": "VGG16",
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the three selected binary models on the locked reference set."
    )
    parser.add_argument("--reference-csv", default=REFERENCE_CSV)
    parser.add_argument("--image-dir", default=IMG_DIR)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--selection-manifest", default=SELECTION_MANIFEST)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    """Load both legacy raw state_dict files and new metadata checkpoints."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch releases that do not expose weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = {
            "model_name": checkpoint.get("model_name"),
            "learning_rate": checkpoint.get("learning_rate"),
            "source_wandb_run_id": checkpoint.get("wandb_run_id"),
            "source_wandb_run_url": checkpoint.get("wandb_run_url"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_f1": checkpoint.get("best_val_f1"),
            "config": checkpoint.get("config", {}),
        }
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    return metadata


def load_selected_model_specs(manifest_path):
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing selection manifest: {path}. Run "
            "python src/select_binary_checkpoints.py first."
        )
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("selection_metric") != "validation F1":
        raise ValueError("Checkpoints must be selected using validation F1")
    if manifest.get("human_reference_used_for_selection") is not False:
        raise ValueError("The human reference set must not select checkpoints")

    selected = manifest.get("selected_models", {})
    if set(selected) != set(DISPLAY_NAMES):
        raise ValueError(
            f"Expected selected models {sorted(DISPLAY_NAMES)}, "
            f"found {sorted(selected)}"
        )

    specs = {}
    for model_name, record in selected.items():
        display_name = DISPLAY_NAMES[model_name]
        specs[display_name] = {
            "model_name": model_name,
            "checkpoint": record["checkpoint_path"],
            "learning_rate": float(record["learning_rate"]),
            "source_wandb_run_id": record["wandb_run_id"],
            "best_epoch": int(record["best_epoch"]),
            "best_val_f1": float(record["best_val_f1"]),
            "input_size": int(record.get("input_size", 224)),
        }
    return specs, manifest


def build_inference_transform(input_size):
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def validate_checkpoint_metadata(display_name, spec, metadata):
    expected = {
        "model_name": spec["model_name"],
        "source_wandb_run_id": spec["source_wandb_run_id"],
        "best_epoch": spec["best_epoch"],
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"{display_name}: checkpoint {field}={metadata.get(field)!r}, "
                f"selection manifest expects {value!r}"
            )
    for field in ("learning_rate", "best_val_f1"):
        if not np.isclose(
            float(metadata.get(field)),
            float(spec[field]),
            atol=1e-12,
        ):
            raise ValueError(
                f"{display_name}: checkpoint {field} does not match manifest"
            )


def validate_reference_table(data_frame, csv_path):
    required = {"image_id", "image_path", "human_label"}
    missing = required.difference(data_frame.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {', '.join(sorted(missing))}"
        )
    if len(data_frame) != 100:
        raise ValueError(f"Expected 100 reference images, found {len(data_frame)}")
    if data_frame["image_id"].duplicated().any():
        duplicates = data_frame.loc[
            data_frame["image_id"].duplicated(), "image_id"
        ].tolist()
        raise ValueError(f"Duplicate reference image IDs: {duplicates[:10]}")
    invalid_labels = sorted(set(data_frame["human_label"]) - {"Good", "Bad"})
    if invalid_labels:
        raise ValueError(f"Invalid human labels: {invalid_labels}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    reference_df = pd.read_csv(args.reference_csv)
    validate_reference_table(reference_df, args.reference_csv)

    model_specs, _ = load_selected_model_specs(args.selection_manifest)

    all_results = []
    for display_name, spec in model_specs.items():
        checkpoint_path = Path(spec["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for {display_name}: {checkpoint_path}"
            )

        print(f"\nLoading {display_name}: {checkpoint_path}")
        model = get_binary_model(
            model_name=spec["model_name"],
            pretrained=False,
        )
        checkpoint_metadata = load_checkpoint(model, checkpoint_path, device)
        validate_checkpoint_metadata(display_name, spec, checkpoint_metadata)
        model = model.to(device)
        model.eval()
        val_transform = build_inference_transform(spec["input_size"])

        match_count = 0
        with torch.inference_mode():
            for _, row in reference_df.iterrows():
                image_name = os.path.basename(
                    str(row["image_path"]).replace("\\", "/")
                )
                image_path = Path(args.image_dir) / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                input_tensor = val_transform(image).unsqueeze(0).to(device)

                logits = model(input_tensor)
                probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()
                score_bad = float(probabilities[0])
                score_good = float(probabilities[1])
                predicted_class = int(score_good > score_bad)
                predicted_label = "Good" if predicted_class == 1 else "Bad"
                human_label = str(row["human_label"])
                is_match = int(human_label == predicted_label)
                match_count += is_match

                all_results.append(
                    {
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "human_label": human_label,
                        "model": display_name,
                        "model_name": spec["model_name"],
                        "predicted_label": predicted_label,
                        "score_good": round(score_good, 6),
                        "score_bad": round(score_bad, 6),
                        "match": is_match,
                        "checkpoint_path": checkpoint_path.as_posix(),
                        "learning_rate": checkpoint_metadata["learning_rate"],
                        "source_wandb_run_id": checkpoint_metadata[
                            "source_wandb_run_id"
                        ],
                        "best_epoch": checkpoint_metadata.get("best_epoch"),
                        "best_val_f1": checkpoint_metadata.get("best_val_f1"),
                    }
                )

        print(f"{display_name}: {match_count}/100 predictions match human labels")

    results_df = pd.DataFrame(all_results)
    if len(results_df) != 300:
        raise RuntimeError(f"Expected 300 result rows, found {len(results_df)}")
    counts = results_df.groupby("model").size().to_dict()
    if counts != {"ResNet18": 100, "ResNet50": 100, "VGG16": 100}:
        raise RuntimeError(f"Unexpected result counts: {counts}")
    if results_df.duplicated(["model", "image_id"]).any():
        raise RuntimeError("Duplicate model/image_id rows found in inference results")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} rows: {output_path}")
    print("Validation passed: three models x 100 unique reference images")


if __name__ == "__main__":
    main()
