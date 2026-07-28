# Training Unit 7: Binary Image Classification

Student: 林慧心  
Student ID: 413410020

This repository contains only the Training Unit 7 binary-classification
assignment. It compares human Good/Bad labels with predictions from
ImageNet-pretrained VGG16, ResNet18, and ResNet50 models.

## Data protocol

- Candidate pool: 500 images.
- Training split: 300 images.
- Validation split: 100 images.
- Locked human reference set: 100 images (50 Good and 50 Bad).
- The reference set is excluded from training, validation, checkpoint
  selection, and hyperparameter selection.
- Label encoding: `Bad = 0`, `Good = 1`.
- Human-score threshold: scores 1--7 are Bad and scores 8--10 are Good.

The source images are intentionally excluded from Git. Place the 500 images
under `data/raw/` before running training or inference.

## Environment

The final formal reruns use Python 3.11, PyTorch 2.6, and an NVIDIA
Tesla V100-SXM2-32GB. See `ENVIRONMENT.md`, `requirements.txt`, and
`environment.yml`. Formal training exits immediately if Python is not 3.11.

## Train the models

```bash
python src/train_binary.py --config configs/vgg16.yaml
python src/train_binary.py --config configs/resnet18.yaml
python src/train_binary.py --config configs/resnet18_lr0.001.yaml
python src/train_binary.py --config configs/resnet50.yaml
```

The two ResNet18 configurations provide the required learning-rate
comparison.

After all four training runs finish, select checkpoints using validation F1
only:

```bash
python src/select_binary_checkpoints.py
```

This creates `outputs/training_runs/selected_models.json`. The locked human
reference set is not read during checkpoint or learning-rate selection.

## Run inference

Classify all 500 candidate images:

```bash
python src/infer_dataset.py
```

Evaluate the locked 100-image human reference set:

```bash
python src/infer_reference.py
python src/evaluate_agreement.py
```

Log the final binary evaluation to W&B:

```bash
python src/log_binary_evaluation.py
```

## Main outputs

```text
outputs/predictions/binary_candidate_pool_predictions.csv
outputs/predictions/binary_candidate_pool_distribution.csv
outputs/predictions/reference_results.csv
outputs/predictions/binary_reference_metrics.csv
```

The metrics file reports agreement count, agreement rate, Accuracy,
Precision, Recall, F1-score, Cohen's Kappa, and confusion-matrix counts.
`Good` is the positive class for all three models.

## Experiment tracking

- W&B project:
  https://wandb.ai/klorius-/training-Unit7-Binary-413410020

The final Overleaf source and compiled PDF belong under `report/`.
