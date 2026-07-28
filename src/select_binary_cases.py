import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


EXPECTED_MODELS = ("VGG16", "ResNet18", "ResNet50")
REFERENCE_SIZE = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select reproducible agreement and disagreement examples from "
            "the locked binary reference evaluation."
        )
    )
    parser.add_argument(
        "--predictions",
        default="outputs/predictions/reference_results.csv",
    )
    parser.add_argument(
        "--reference",
        default="data/annotations/binary_reference_100.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/report_data/binary_case_selection.csv",
    )
    parser.add_argument(
        "--image-output",
        default="report/figures/cases",
    )
    return parser.parse_args()


def series_name(image_id):
    match = re.match(r"^(p\d+_(?:img|seed))", str(image_id))
    return match.group(1) if match else str(image_id)


def predicted_confidence(frame):
    return frame[["score_good", "score_bad"]].max(axis=1)


def choose_diverse(frame, count, used_series=None):
    used = set() if used_series is None else set(used_series)
    chosen = []

    for index, row in frame.iterrows():
        series = row["series"]
        if series not in used:
            chosen.append(index)
            used.add(series)
        if len(chosen) == count:
            return chosen

    for index in frame.index:
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == count:
            return chosen

    raise ValueError(f"Only found {len(chosen)} examples; need {count}")


def choose_agreements(frame):
    selected = []
    used_series = set()
    for label, count in (("Good", 3), ("Bad", 2)):
        candidates = frame.loc[frame["human_label"] == label].sort_values(
            ["prediction_confidence", "image_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        indices = choose_diverse(candidates, count, used_series)
        selected.extend(indices)
        used_series.update(candidates.loc[indices, "series"])
    return frame.loc[selected]


def choose_disagreements(frame):
    candidates = frame.sort_values(
        ["prediction_confidence", "image_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    indices = choose_diverse(candidates, 5)
    selected = candidates.loc[indices].copy()
    selected["high_confidence_error"] = False
    top_two = candidates.head(2).index

    # Guarantee that the two highest-confidence errors are included.
    for index in reversed(top_two):
        if index not in selected.index:
            selected = pd.concat(
                [candidates.loc[[index]], selected.iloc[:-1]],
                axis=0,
            )
    selected["high_confidence_error"] = selected.index.isin(top_two)
    return selected


def validate_inputs(predictions, reference):
    required_prediction_columns = {
        "image_id",
        "image_path",
        "human_label",
        "model",
        "predicted_label",
        "score_good",
        "score_bad",
        "match",
    }
    missing = required_prediction_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    if set(predictions["model"]) != set(EXPECTED_MODELS):
        raise ValueError("Predictions do not contain exactly the three formal models")

    if len(reference) != REFERENCE_SIZE or reference["image_id"].duplicated().any():
        raise ValueError("Reference CSV must contain 100 unique images")

    for model in EXPECTED_MODELS:
        group = predictions.loc[predictions["model"] == model]
        if len(group) != REFERENCE_SIZE or group["image_id"].duplicated().any():
            raise ValueError(f"{model}: expected 100 unique reference predictions")


def copy_case_images(cases, output_root):
    output_root = Path(output_root)
    for model in EXPECTED_MODELS:
        for case_type in ("agreement", "disagreement"):
            managed_directory = output_root / model.lower() / case_type
            if managed_directory.is_dir():
                for path in managed_directory.iterdir():
                    if path.is_file():
                        path.unlink()

    for row in cases.itertuples(index=False):
        case_dir = output_root / row.model.lower() / row.case_type
        case_dir.mkdir(parents=True, exist_ok=True)
        source = Path(row.image_path)
        destination = case_dir / source.name
        shutil.copy2(source, destination)


def main():
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    reference = pd.read_csv(args.reference)
    validate_inputs(predictions, reference)

    human_columns = reference[
        ["image_id", "human_score", "reason"]
    ].rename(columns={"reason": "reference_reason"})
    frame = predictions.merge(
        human_columns,
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    frame["score_good"] = pd.to_numeric(frame["score_good"])
    frame["score_bad"] = pd.to_numeric(frame["score_bad"])
    frame["prediction_confidence"] = predicted_confidence(frame)
    frame["series"] = frame["image_id"].map(series_name)

    selections = []
    for model in EXPECTED_MODELS:
        model_rows = frame.loc[frame["model"] == model].copy()

        agreements = choose_agreements(model_rows.loc[model_rows["match"] == 1]).copy()
        agreements["case_type"] = "agreement"
        agreements["high_confidence_error"] = False

        disagreements = choose_disagreements(
            model_rows.loc[model_rows["match"] == 0]
        ).copy()
        disagreements["case_type"] = "disagreement"

        selections.extend([agreements, disagreements])

    cases = pd.concat(selections, ignore_index=True)
    cases["analysis_reason"] = ""
    cases["boundary_case"] = ""
    cases["possible_distribution_shift"] = ""

    columns = [
        "model",
        "case_type",
        "high_confidence_error",
        "image_id",
        "image_path",
        "series",
        "human_score",
        "human_label",
        "predicted_label",
        "score_good",
        "score_bad",
        "prediction_confidence",
        "reference_reason",
        "analysis_reason",
        "boundary_case",
        "possible_distribution_shift",
    ]
    cases = cases[columns]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_path, index=False)
    copy_case_images(cases, args.image_output)

    summary = cases.groupby(["model", "case_type"]).size()
    high_confidence = cases.groupby("model")["high_confidence_error"].sum()
    print(summary.to_string())
    print("\nHigh-confidence errors:")
    print(high_confidence.to_string())
    print(f"\nSaved: {output_path}")
    print(f"Copied case images under: {args.image_output}")


if __name__ == "__main__":
    main()
