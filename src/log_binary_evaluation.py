import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import wandb
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PREDICTIONS_CSV = Path("outputs/predictions/reference_results.csv")
METRICS_CSV = Path("outputs/predictions/binary_reference_metrics.csv")
REFERENCE_CSV = Path("data/annotations/binary_reference_100.csv")
SELECTION_MANIFEST = Path("outputs/training_runs/selected_models.json")
CANDIDATE_PREDICTIONS_CSV = Path(
    "outputs/predictions/binary_candidate_pool_predictions.csv"
)
CANDIDATE_DISTRIBUTION_CSV = Path(
    "outputs/predictions/binary_candidate_pool_distribution.csv"
)

DEFAULT_WANDB_ENTITY = "klorius-"
DEFAULT_WANDB_PROJECT = "training-Unit7-Binary-413410020"
EXPECTED_REFERENCE_SIZE = 100
EXPECTED_MODELS = ("VGG16", "ResNet18", "ResNet50")
LABEL_MAP = {"Bad": 0, "Good": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log the finalized Python 3.11 binary evaluation to W&B."
    )
    parser.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--run-name",
        default="binary_final_evaluation_python311",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit W&B run ID. Omit to let W&B create one.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def calculate_metrics(group):
    y_true = group["human_label"].map(LABEL_MAP)
    y_pred = group["predicted_label"].map(LABEL_MAP)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    agreement_count = int((y_true == y_pred).sum())
    return {
        "agreement_count": agreement_count,
        "disagreement_count": int(len(group) - agreement_count),
        "agreement_rate": agreement_count / len(group),
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


def load_selection_manifest():
    if not SELECTION_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing {SELECTION_MANIFEST}. "
            "Run python src/select_binary_checkpoints.py first."
        )
    with SELECTION_MANIFEST.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("selection_metric") != "validation F1":
        raise ValueError("Selection manifest must use validation F1")
    if manifest.get("human_reference_used_for_selection") is not False:
        raise ValueError("Human reference set must not select checkpoints")

    selected = manifest.get("selected_models", {})
    if set(selected) != {"vgg16", "resnet18", "resnet50"}:
        raise ValueError("Selection manifest must contain three selected models")
    return manifest


def expected_provenance(manifest):
    return {
        record["display_name"]: {
            "model_name": record["model_name"],
            "checkpoint_path": Path(record["checkpoint_path"]).as_posix(),
            "learning_rate": float(record["learning_rate"]),
            "source_wandb_run_id": record["wandb_run_id"],
            "best_epoch": int(record["best_epoch"]),
            "best_val_f1": float(record["best_val_f1"]),
        }
        for record in manifest["selected_models"].values()
    }


def validate_reference_and_predictions(predictions, reference, provenance):
    required = {
        "image_id",
        "image_path",
        "human_label",
        "model",
        "model_name",
        "predicted_label",
        "score_good",
        "score_bad",
        "match",
        "checkpoint_path",
        "learning_rate",
        "source_wandb_run_id",
        "best_epoch",
        "best_val_f1",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")

    if len(reference) != EXPECTED_REFERENCE_SIZE:
        raise ValueError("Reference CSV must contain 100 rows")
    if reference["image_id"].isna().any() or reference["image_id"].duplicated().any():
        raise ValueError("Reference image IDs must be present and unique")
    if set(reference["human_label"]) != {"Bad", "Good"}:
        raise ValueError("Reference CSV must contain Bad and Good labels")

    if len(predictions) != 300 or set(predictions["model"]) != set(EXPECTED_MODELS):
        raise ValueError("Expected three models x 100 reference predictions")
    if predictions.duplicated(["model", "image_id"]).any():
        raise ValueError("Duplicate model/image_id prediction rows")
    if not predictions["human_label"].isin(LABEL_MAP).all():
        raise ValueError("Invalid human labels")
    if not predictions["predicted_label"].isin(LABEL_MAP).all():
        raise ValueError("Invalid predicted labels")

    scores = predictions[["score_good", "score_bad"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Probabilities contain NaN or Inf")
    if not np.allclose(scores.sum(axis=1), 1.0, atol=2e-6):
        raise ValueError("Good/Bad probabilities do not sum to one")

    reference_labels = reference.set_index("image_id")["human_label"].astype(str)
    reference_ids = set(reference_labels.index.astype(str))
    for model in EXPECTED_MODELS:
        group = predictions.loc[predictions["model"] == model]
        if len(group) != 100 or set(group["image_id"].astype(str)) != reference_ids:
            raise ValueError(f"{model} did not evaluate the exact reference set")
        labels = group.set_index("image_id")["human_label"].astype(str)
        if not labels.sort_index().equals(reference_labels.sort_index()):
            raise ValueError(f"{model} human labels do not match reference CSV")

        expected = provenance[model]
        for field in (
            "model_name",
            "checkpoint_path",
            "source_wandb_run_id",
            "best_epoch",
        ):
            values = group[field].drop_duplicates().tolist()
            if values != [expected[field]]:
                raise ValueError(
                    f"{model}: {field}={values}, expected {expected[field]!r}"
                )
        for field in ("learning_rate", "best_val_f1"):
            values = pd.to_numeric(group[field]).drop_duplicates().to_numpy()
            if len(values) != 1 or not np.isclose(
                float(values[0]), expected[field], atol=1e-12
            ):
                raise ValueError(f"{model}: {field} does not match manifest")


def recompute_metrics(predictions):
    rows = []
    confusion_data = {}
    for model in EXPECTED_MODELS:
        group = predictions.loc[predictions["model"] == model].copy()
        rows.append({"model": model, **calculate_metrics(group)})
        confusion_data[model] = {
            "y_true": group["human_label"].map(LABEL_MAP).tolist(),
            "y_pred": group["predicted_label"].map(LABEL_MAP).tolist(),
        }
    return pd.DataFrame(rows), confusion_data


def validate_stored_metrics(stored, recomputed):
    required = set(recomputed.columns)
    missing = required.difference(stored.columns)
    if missing:
        raise ValueError(f"Stored metrics missing columns: {sorted(missing)}")

    left = stored.set_index("model").sort_index()
    right = recomputed.set_index("model").sort_index()
    if set(left.index) != set(right.index):
        raise ValueError("Stored metric models do not match recomputation")

    numeric_columns = sorted(required - {"model"})
    if not np.allclose(
        left[numeric_columns].astype(float),
        right[numeric_columns].astype(float),
        atol=1e-12,
    ):
        raise ValueError("Stored metrics do not match independent recomputation")


def validation_run_table(manifest):
    selected_ids = {
        record["wandb_run_id"]
        for record in manifest["selected_models"].values()
    }
    rows = []
    for record in manifest["formal_validation_runs"]:
        rows.append(
            {
                "model": record["display_name"],
                "learning_rate": record["learning_rate"],
                "best_epoch": record["best_epoch"],
                "best_val_f1": record["best_val_f1"],
                "wandb_run_id": record["wandb_run_id"],
                "wandb_run_url": record["wandb_run_url"],
                "selected_for_reference_inference": (
                    record["wandb_run_id"] in selected_ids
                ),
            }
        )
    return pd.DataFrame(rows)


def load_optional_candidate_tables():
    if not CANDIDATE_PREDICTIONS_CSV.is_file():
        raise FileNotFoundError(f"Missing {CANDIDATE_PREDICTIONS_CSV}")
    if not CANDIDATE_DISTRIBUTION_CSV.is_file():
        raise FileNotFoundError(f"Missing {CANDIDATE_DISTRIBUTION_CSV}")

    predictions = pd.read_csv(CANDIDATE_PREDICTIONS_CSV)
    distribution = pd.read_csv(CANDIDATE_DISTRIBUTION_CSV)
    if len(predictions) != 1500:
        raise ValueError("Candidate predictions must contain 1500 rows")
    if predictions.duplicated(["model", "image_id"]).any():
        raise ValueError("Candidate predictions contain duplicates")
    if len(distribution) != 3:
        raise ValueError("Candidate distribution must contain three rows")
    return predictions, distribution


def main():
    args = parse_args()
    manifest = load_selection_manifest()
    provenance = expected_provenance(manifest)
    predictions = pd.read_csv(PREDICTIONS_CSV)
    reference = pd.read_csv(REFERENCE_CSV)
    stored_metrics = pd.read_csv(METRICS_CSV)

    validate_reference_and_predictions(predictions, reference, provenance)
    metrics, confusion_data = recompute_metrics(predictions)
    validate_stored_metrics(stored_metrics, metrics)
    validation_runs = validation_run_table(manifest)
    candidate_predictions, candidate_distribution = load_optional_candidate_tables()

    selected_records = list(manifest["selected_models"].values())
    best_validation = max(
        selected_records,
        key=lambda record: float(record["best_val_f1"]),
    )
    selected_resnet18 = manifest["selected_models"]["resnet18"]
    reference_counts = reference["human_label"].value_counts().to_dict()

    config = {
        "task": "binary_classification_final_evaluation",
        "evaluation_protocol": "locked held-out human reference set",
        "reference_size": EXPECTED_REFERENCE_SIZE,
        "reference_bad_count": int(reference_counts.get("Bad", 0)),
        "reference_good_count": int(reference_counts.get("Good", 0)),
        "candidate_pool_size": 500,
        "label_encoding": "Bad=0, Good=1",
        "positive_class": "Good",
        "checkpoint_selection_metric": "validation F1",
        "reported_metrics": [
            "agreement count",
            "agreement rate",
            "accuracy",
            "precision",
            "recall",
            "F1",
            "Cohen's Kappa",
            "confusion matrix",
        ],
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "human_reference_used_for_training": False,
        "human_reference_used_for_validation": False,
        "human_reference_used_for_checkpoint_selection": False,
        "human_reference_used_for_hyperparameter_selection": False,
        "human_reference_usage": "final comparison only",
        "formal_training_run_ids": validation_runs["wandb_run_id"].tolist(),
        "selected_resnet18_learning_rate": float(
            selected_resnet18["learning_rate"]
        ),
        "selected_resnet18_run_id": selected_resnet18["wandb_run_id"],
        "best_validation_architecture": best_validation["display_name"],
        "best_validation_f1": float(best_validation["best_val_f1"]),
    }

    with wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        id=args.run_id,
        resume="never" if args.run_id else None,
        job_type="evaluation",
        tags=["basic", "binary", "python311", "final-evaluation", "report"],
        notes=(
            "Final held-out reference evaluation including agreement rate "
            "and Cohen's Kappa. All checkpoint and hyperparameter selections "
            "were completed using validation F1 before reference inference."
        ),
        config=config,
        mode=args.wandb_mode,
    ) as run:
        run.log(
            {
                "reference_predictions": wandb.Table(dataframe=predictions),
                "binary_reference_metrics": wandb.Table(dataframe=metrics),
                "validation_run_comparison": wandb.Table(
                    dataframe=validation_runs
                ),
                "candidate_pool_predictions": wandb.Table(
                    dataframe=candidate_predictions
                ),
                "candidate_pool_distribution": wandb.Table(
                    dataframe=candidate_distribution
                ),
            }
        )

        for model in EXPECTED_MODELS:
            data = confusion_data[model]
            run.log(
                {
                    f"{model}/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=data["y_true"],
                        preds=data["y_pred"],
                        class_names=["Bad", "Good"],
                    )
                }
            )

        for _, row in metrics.iterrows():
            model = row["model"]
            for metric in (
                "agreement_rate",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "cohen_kappa",
            ):
                run.summary[f"{model}/{metric}"] = float(row[metric])
            for count in (
                "agreement_count",
                "disagreement_count",
                "tn",
                "fp",
                "fn",
                "tp",
            ):
                run.summary[f"{model}/{count}"] = int(row[count])

        best_reference = metrics.loc[metrics["f1"].idxmax()]
        run.summary["best_reference_f1_model"] = best_reference["model"]
        run.summary["best_reference_f1"] = float(best_reference["f1"])
        run.summary["best_validation_model"] = best_validation["display_name"]
        run.summary["best_validation_f1"] = float(best_validation["best_val_f1"])
        run.summary["selected_resnet18_learning_rate"] = float(
            selected_resnet18["learning_rate"]
        )
        run.summary["validation_status"] = (
            "300 reference rows and 1500 candidate rows validated; "
            "metrics and checkpoint provenance independently recomputed"
        )

        artifact = wandb.Artifact(
            name="unit7-binary-python311-final-results",
            type="evaluation",
            description="Final Python 3.11 binary classification results.",
        )
        for path in (
            PREDICTIONS_CSV,
            METRICS_CSV,
            SELECTION_MANIFEST,
            CANDIDATE_PREDICTIONS_CSV,
            CANDIDATE_DISTRIBUTION_CSV,
        ):
            artifact.add_file(str(path))
        run.log_artifact(artifact)

        print(f"W&B run URL: {run.url}")
        print(
            f"Best held-out reference F1: {best_reference['model']} "
            f"({float(best_reference['f1']):.4f})"
        )
        print(
            f"Best validation F1: {best_validation['display_name']} "
            f"({float(best_validation['best_val_f1']):.4f})"
        )
        print(
            "Validation passed: dynamic Python 3.11 provenance, "
            "three exact reference sets, and complete candidate pool."
        )


if __name__ == "__main__":
    main()
