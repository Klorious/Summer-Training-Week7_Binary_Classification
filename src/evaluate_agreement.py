import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


EXPECTED_MODELS = ("VGG16", "ResNet18", "ResNet50")
EXPECTED_REFERENCE_SIZE = 100
LABEL_MAP = {"Bad": 0, "Good": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate agreement between the locked human reference labels "
            "and the three binary classifiers."
        )
    )
    parser.add_argument(
        "--predictions",
        default="outputs/predictions/reference_results.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/binary_reference_metrics.csv",
    )
    return parser.parse_args()


def validate_predictions(frame):
    required = {
        "image_id",
        "human_label",
        "model",
        "predicted_label",
        "score_good",
        "score_bad",
        "match",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    if set(frame["model"]) != set(EXPECTED_MODELS):
        raise ValueError(
            f"Expected models {EXPECTED_MODELS}, found {sorted(frame['model'].unique())}"
        )

    for model in EXPECTED_MODELS:
        group = frame.loc[frame["model"] == model]
        if len(group) != EXPECTED_REFERENCE_SIZE:
            raise ValueError(
                f"{model}: expected {EXPECTED_REFERENCE_SIZE} rows, found {len(group)}"
            )
        if group["image_id"].isna().any() or group["image_id"].duplicated().any():
            raise ValueError(f"{model}: image IDs must be present and unique")
        if not group["human_label"].isin(LABEL_MAP).all():
            raise ValueError(f"{model}: invalid human label")
        if not group["predicted_label"].isin(LABEL_MAP).all():
            raise ValueError(f"{model}: invalid predicted label")

        scores = group[["score_good", "score_bad"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(scores.to_numpy()).all():
            raise ValueError(f"{model}: probabilities contain NaN or Inf")
        if not np.allclose(scores.sum(axis=1), 1.0, atol=2e-6):
            raise ValueError(f"{model}: Good/Bad probabilities do not sum to one")


def calculate_metrics(group):
    y_true = group["human_label"].map(LABEL_MAP)
    y_pred = group["predicted_label"].map(LABEL_MAP)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    agreement_count = int((y_true == y_pred).sum())
    agreement_rate = agreement_count / len(group)

    return {
        "agreement_count": agreement_count,
        "disagreement_count": int(len(group) - agreement_count),
        "agreement_rate": agreement_rate,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def main():
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    validate_predictions(predictions)

    rows = []
    for model in EXPECTED_MODELS:
        group = predictions.loc[predictions["model"] == model]
        rows.append({"model": model, **calculate_metrics(group)})

    metrics = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {output_path}")
    print(
        "Positive class: Good. Validation passed: "
        "three models x 100 unique reference images."
    )


if __name__ == "__main__":
    main()
