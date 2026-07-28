import argparse
import json
from pathlib import Path


DEFAULT_MANIFEST_DIR = Path("outputs/training_runs")
DEFAULT_OUTPUT = DEFAULT_MANIFEST_DIR / "selected_models.json"
EXPECTED_EXPERIMENTS = {
    ("vgg16", 0.0001),
    ("resnet18", 0.0001),
    ("resnet18", 0.001),
    ("resnet50", 0.0001),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select the three formal checkpoints using validation F1 only. "
            "The locked human reference set is never read."
        )
    )
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_formal_manifests(directory):
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == "selected_models.json":
            continue
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        if record.get("formal_run") is not True:
            continue
        record["_manifest_path"] = path.as_posix()
        records.append(record)
    return records


def validate_manifests(records):
    found = {
        (record.get("model_name"), float(record.get("learning_rate")))
        for record in records
    }
    if found != EXPECTED_EXPERIMENTS:
        raise ValueError(
            "Expected exactly four formal experiments "
            f"{sorted(EXPECTED_EXPERIMENTS)}, found {sorted(found)}"
        )

    run_ids = [record.get("wandb_run_id") for record in records]
    if any(not value for value in run_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError("Formal manifests must contain four unique W&B run IDs")

    for record in records:
        checkpoint = Path(record["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        if record.get("checkpoint_selection_metric") != "validation F1":
            raise ValueError("Every checkpoint must be selected by validation F1")
        if record.get("human_reference_usage") != "final comparison only":
            raise ValueError("Human reference provenance is invalid")


def select_models(records):
    by_model = {}
    for record in records:
        by_model.setdefault(record["model_name"], []).append(record)

    selected = {
        "vgg16": by_model["vgg16"][0],
        "resnet18": max(
            by_model["resnet18"],
            key=lambda item: (
                float(item["best_val_f1"]),
                -float(item["learning_rate"]),
            ),
        ),
        "resnet50": by_model["resnet50"][0],
    }
    return selected


def main():
    args = parse_args()
    records = load_formal_manifests(args.manifest_dir)
    validate_manifests(records)
    selected = select_models(records)

    output = {
        "selection_metric": "validation F1",
        "higher_is_better": True,
        "human_reference_used_for_selection": False,
        "human_reference_usage": "final comparison only",
        "formal_validation_runs": sorted(
            records,
            key=lambda item: (item["model_name"], item["learning_rate"]),
        ),
        "selected_models": selected,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("Validation-selected checkpoints:")
    for model_name, record in selected.items():
        print(
            f"  {model_name}: lr={record['learning_rate']}, "
            f"epoch={record['best_epoch']}, "
            f"val_f1={record['best_val_f1']:.6f}, "
            f"run={record['wandb_run_id']}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
