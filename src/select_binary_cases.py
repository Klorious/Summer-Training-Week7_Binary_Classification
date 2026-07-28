import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


EXPECTED_MODELS = ("VGG16", "ResNet18", "ResNet50")
REFERENCE_SIZE = 100
CASE_REPLACEMENTS = {
    # Manual review replacement. Keep this explicit so rerunning the selector
    # cannot silently restore the rejected example.
    ("ResNet18", "disagreement", "p5_seed23"): "p1_seed10",
}
REVIEWED_CASE_NOTES = {
    ("ResNet18", "p1_seed10"): {
        "reference_reason": (
            "道路可見度 2 分、障礙物可辨識度 0 分、影像清晰度 1 分、"
            "遮擋程度 2 分、場景真實性 0 分，合計 5/10；"
            "圖片模糊、場景真實性低下且障礙物難以辨識。"
        ),
        "analysis_reason": (
            "模型可能較重視道路清楚可見且幾乎無遮擋，未充分考慮"
            "圖片模糊、障礙物難以辨識與場景真實性低下，因此將人工 "
            "Bad 錯判為 Good。"
        ),
        "boundary_case": "False",
        "possible_distribution_shift": "True",
    },
}


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


def apply_case_replacements(cases, source_frame):
    cases = cases.copy()

    for (model, case_type, old_image_id), new_image_id in CASE_REPLACEMENTS.items():
        old_mask = (
            (cases["model"] == model)
            & (cases["case_type"] == case_type)
            & (cases["image_id"] == old_image_id)
        )
        if old_mask.sum() != 1:
            raise ValueError(
                f"Expected exactly one case to replace: "
                f"{model}/{case_type}/{old_image_id}"
            )

        candidate = source_frame.loc[
            (source_frame["model"] == model)
            & (source_frame["image_id"] == new_image_id)
        ].copy()
        if len(candidate) != 1:
            raise ValueError(
                f"Replacement candidate must be unique: {model}/{new_image_id}"
            )

        candidate_row = candidate.iloc[0]
        expected_match = 1 if case_type == "agreement" else 0
        if int(candidate_row["match"]) != expected_match:
            raise ValueError(
                f"Replacement {model}/{new_image_id} is not a {case_type} case"
            )

        model_ids = set(cases.loc[cases["model"] == model, "image_id"])
        if new_image_id in model_ids:
            raise ValueError(
                f"Replacement {model}/{new_image_id} is already selected"
            )

        target_index = cases.index[old_mask][0]
        for column in source_frame.columns:
            if column in cases.columns:
                cases.at[target_index, column] = candidate_row[column]
        cases.at[target_index, "case_type"] = case_type
        cases.at[target_index, "high_confidence_error"] = False

        print(
            f"Applied reviewed replacement: "
            f"{model}/{old_image_id} -> {new_image_id}"
        )

    return cases


def apply_reviewed_case_notes(cases):
    cases = cases.copy()
    for (model, image_id), notes in REVIEWED_CASE_NOTES.items():
        mask = (cases["model"] == model) & (cases["image_id"] == image_id)
        if mask.sum() != 1:
            raise ValueError(
                f"Reviewed case note target must be unique: {model}/{image_id}"
            )
        for column, value in notes.items():
            cases.loc[mask, column] = value
    return cases


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
    cases = apply_case_replacements(cases, frame)
    cases["analysis_reason"] = ""
    cases["boundary_case"] = ""
    cases["possible_distribution_shift"] = ""
    cases = apply_reviewed_case_notes(cases)

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
