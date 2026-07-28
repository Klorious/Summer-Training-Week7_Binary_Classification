import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from infer_reference import (
    SELECTION_MANIFEST,
    load_checkpoint,
    load_selected_model_specs,
    validate_checkpoint_metadata,
)
from models import get_binary_model


CANDIDATE_CSV = "data/splits/candidate_pool.csv"
IMAGE_DIR = "data/raw"
OUTPUT_CSV = "outputs/predictions/binary_candidate_pool_predictions.csv"
DISTRIBUTION_CSV = (
    "outputs/predictions/binary_candidate_pool_distribution.csv"
)

EXPECTED_CANDIDATE_COUNT = 500
EXPECTED_MODEL_COUNT = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the three validation-selected binary classifiers on the "
            "complete fixed candidate pool."
        )
    )
    parser.add_argument("--candidate-csv", default=CANDIDATE_CSV)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument(
        "--distribution-csv",
        default=DISTRIBUTION_CSV,
    )
    parser.add_argument("--selection-manifest", default=SELECTION_MANIFEST)
    return parser.parse_args()


def validate_candidate_pool(candidate_frame, csv_path):
    required_columns = {"image_id", "image_path"}
    missing_columns = required_columns.difference(candidate_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    if len(candidate_frame) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} candidate images, "
            f"but found {len(candidate_frame)}"
        )
    if candidate_frame["image_id"].isna().any():
        raise ValueError(f"{csv_path} contains missing image_id values")
    if candidate_frame["image_path"].isna().any():
        raise ValueError(f"{csv_path} contains missing image_path values")
    if candidate_frame["image_id"].duplicated().any():
        duplicates = candidate_frame.loc[
            candidate_frame["image_id"].duplicated(), "image_id"
        ].tolist()
        raise ValueError(f"Duplicate candidate image IDs: {duplicates[:10]}")


def build_inference_transform(input_size):
    # This is intentionally identical to the deterministic validation and
    # held-out reference transform used by src/infer.py.
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


def resolve_image_path(image_dir, stored_image_path):
    image_name = os.path.basename(str(stored_image_path).replace("\\", "/"))
    image_path = Path(image_dir) / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def validate_predictions(results_frame, model_specs):
    expected_rows = EXPECTED_CANDIDATE_COUNT * EXPECTED_MODEL_COUNT
    if len(results_frame) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} prediction rows, "
            f"but found {len(results_frame)}"
        )

    expected_counts = {
        model_name: EXPECTED_CANDIDATE_COUNT for model_name in model_specs
    }
    actual_counts = results_frame.groupby("model").size().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Unexpected prediction counts by model: "
            f"{actual_counts}; expected {expected_counts}"
        )

    if results_frame.duplicated(["model", "image_id"]).any():
        raise RuntimeError("Duplicate model/image_id prediction rows found")

    score_columns = ["score_good", "score_bad"]
    numeric_scores = results_frame[score_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric_scores.isna().any().any():
        raise RuntimeError("NaN or non-numeric probability values found")
    if not np.isfinite(numeric_scores.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Infinite probability values found")

    probability_sums = numeric_scores.sum(axis=1).to_numpy(dtype=np.float64)
    if not np.allclose(probability_sums, 1.0, atol=2e-6):
        raise RuntimeError("score_good and score_bad do not sum to one")

    valid_labels = {"Good", "Bad"}
    invalid_labels = set(results_frame["predicted_label"]) - valid_labels
    if invalid_labels:
        raise RuntimeError(
            f"Invalid predicted labels: {sorted(invalid_labels)}"
        )


def build_distribution_table(results_frame, model_specs):
    records = []
    for display_name in model_specs:
        model_rows = results_frame.loc[results_frame["model"] == display_name]
        label_counts = model_rows["predicted_label"].value_counts()
        predicted_bad = int(label_counts.get("Bad", 0))
        predicted_good = int(label_counts.get("Good", 0))
        total = int(len(model_rows))
        records.append(
            {
                "model": display_name,
                "predicted_bad": predicted_bad,
                "predicted_good": predicted_good,
                "total": total,
                "bad_ratio": round(predicted_bad / total, 4),
                "good_ratio": round(predicted_good / total, 4),
            }
        )
    return pd.DataFrame(records)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    candidate_frame = pd.read_csv(args.candidate_csv)
    validate_candidate_pool(candidate_frame, args.candidate_csv)
    print(
        f"Candidate pool validation passed: "
        f"{len(candidate_frame)} unique images"
    )

    model_specs, _ = load_selected_model_specs(args.selection_manifest)
    prediction_records = []

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
        checkpoint_metadata = load_checkpoint(
            model,
            checkpoint_path,
            device,
        )
        validate_checkpoint_metadata(display_name, spec, checkpoint_metadata)
        model = model.to(device)
        model.eval()
        inference_transform = build_inference_transform(spec["input_size"])

        with torch.inference_mode():
            for row_number, row in candidate_frame.iterrows():
                image_path = resolve_image_path(
                    args.image_dir,
                    row["image_path"],
                )
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                input_tensor = (
                    inference_transform(image).unsqueeze(0).to(device)
                )

                logits = model(input_tensor)
                probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()
                score_bad = float(probabilities[0])
                score_good = float(probabilities[1])
                predicted_label = (
                    "Good" if score_good > score_bad else "Bad"
                )

                prediction_records.append(
                    {
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "model": display_name,
                        "model_name": spec["model_name"],
                        "predicted_label": predicted_label,
                        "score_good": round(score_good, 6),
                        "score_bad": round(score_bad, 6),
                        "checkpoint_path": checkpoint_path.as_posix(),
                        "learning_rate": checkpoint_metadata["learning_rate"],
                        "source_wandb_run_id": checkpoint_metadata[
                            "source_wandb_run_id"
                        ],
                        "best_epoch": checkpoint_metadata.get("best_epoch"),
                        "best_val_f1": checkpoint_metadata.get("best_val_f1"),
                    }
                )

                completed = row_number + 1
                if completed % 100 == 0:
                    print(
                        f"{display_name}: "
                        f"{completed}/{EXPECTED_CANDIDATE_COUNT}"
                    )

    results_frame = pd.DataFrame(prediction_records)
    validate_predictions(results_frame, model_specs)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(output_path, index=False)

    distribution_frame = build_distribution_table(results_frame, model_specs)
    distribution_path = Path(args.distribution_csv)
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    distribution_frame.to_csv(distribution_path, index=False)

    print(f"\nSaved predictions: {output_path} ({len(results_frame)} rows)")
    print(f"Saved distribution: {distribution_path}")
    print("\nComplete candidate-pool prediction distribution:")
    print(distribution_frame.to_string(index=False))
    print(
        "\nValidation passed: three models x 500 unique candidate images; "
        "all probabilities are finite and sum to one"
    )


if __name__ == "__main__":
    main()
