# Training Unit 7：影像品質二元分類

本專案為 Training Unit 7 的 Binary Classification 作業，使用 ImageNet
預訓練的 VGG16、ResNet18 與 ResNet50，將道路影像分類為 Good 或 Bad，
並比較模型預測與人工標註結果。

- 姓名：林慧心
- 學號：413410020
- GitHub：https://github.com/Klorious/Summer-Training-Week7_Binary_Classification
- W&B：https://wandb.ai/klorius-/training-Unit7-Binary-413410020
- [完整技術報告](report/report.pdf)

## 1. 任務定義

本作業將影像分成兩類：

- `Bad = 0`
- `Good = 1`

人工品質分數與二元標籤的轉換規則：

- 1–7 分：Bad
- 8–10 分：Good

本專案所有 Accuracy、Precision、Recall 與 F1-score 均以 Good 作為正類。

## 2. 資料切分

完整資料集共 500 張圖片，切分如下：

| 資料集 | 數量 | 用途 |
|---|---:|---|
| Training set | 300 | 模型訓練 |
| Validation set | 100 | 選擇最佳 checkpoint 與超參數 |
| Human reference set | 100 | 最終人機比較 |

人工參考集包含 50 張 Good 與 50 張 Bad，且未用於：

- 模型訓練
- Validation
- Checkpoint 選擇
- 超參數選擇

因此，人工參考集的結果屬於 held-out evaluation。

原始影像未上傳至 GitHub。執行程式前，需自行將 500 張圖片放入：

```text
data/raw/
```

## 3. 模型與訓練設定

本作業使用以下三種模型：

- VGG16
- ResNet18
- ResNet50

共同設定：

| 項目 | 設定 |
|---|---|
| Pretrained weights | ImageNet |
| Backbone | 不凍結 |
| Classifier outputs | 2 |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Batch size | 32 |
| Epochs | 30 |
| Weight decay | 1e-4 |
| Input size | 224 × 224 |
| Random seed | 20260725 |
| Checkpoint rule | Validation F1 最高 |

ResNet18 另外比較兩組 learning rate：

- `1e-4`
- `1e-3`

每次實驗僅改變 learning rate，以維持比較公平性。模型 checkpoint 與
ResNet18 learning rate 的選擇皆只依據 validation F1，不使用人工參考集。

## 4. 資料前處理

訓練資料使用：

- Resize
- RandomResizedCrop
- RandomHorizontalFlip
- ImageNet normalization

Validation、人工參考集與完整候選池推論使用確定性前處理：

- Resize
- CenterCrop
- ImageNet normalization

人工參考集不使用隨機資料增強。

## 5. 執行環境

正式實驗環境：

- Python 3.11
- PyTorch 2.6
- CUDA 12.4
- NVIDIA Tesla V100-SXM2-32GB

完整套件與環境設定請參考：

- `ENVIRONMENT.md`
- `requirements.txt`
- `environment.yml`

建立並檢查環境：

```bash
conda env create -f environment.yml
conda activate training-unit7-py311
python check_environment.py
```

## 6. 訓練方式

執行四組正式訓練：

```bash
python src/train_binary.py --config configs/vgg16.yaml
python src/train_binary.py --config configs/resnet18.yaml
python src/train_binary.py --config configs/resnet18_lr0.001.yaml
python src/train_binary.py --config configs/resnet50.yaml
```

完成四組實驗後，依 validation F1 選擇正式 checkpoint：

```bash
python src/select_binary_checkpoints.py
```

選擇結果儲存於：

```text
outputs/training_runs/selected_models.json
```

## 7. 模型推論與評估

### 7.1 完整 500 張候選圖片推論

```bash
python src/infer_dataset.py
```

### 7.2 人工參考集推論

```bash
python src/infer_reference.py
python src/evaluate_agreement.py
```

### 7.3 記錄最終 W&B 評估

```bash
python src/log_binary_evaluation.py
```

## 8. 正式實驗結果

### 8.1 Validation 結果

| 模型 | Learning rate | Best epoch | Best validation F1 | W&B run ID |
|---|---:|---:|---:|---|
| VGG16 | 0.0001 | 29 | 0.8444 | `1xptjn7v` |
| ResNet18 | 0.0001 | 2 | 0.7869 | `g0osn4yx` |
| ResNet18 | 0.001 | 5 | 0.8321 | `dm8ovhqt` |
| ResNet50 | 0.0001 | 5 | **0.8531** | `p5r1yde1` |

ResNet18 的 learning rate 從 `1e-4` 調整為 `1e-3` 後，最佳 validation
F1 從 0.7869 提升至 0.8321，因此後續人工參考集推論採用
ResNet18 `1e-3` 的 checkpoint。三種模型中，ResNet50 取得最高的
validation F1。

### 8.2 人工參考集結果

| 模型 | Accuracy／一致率 | Precision | Recall | F1 | Cohen's Kappa |
|---|---:|---:|---:|---:|---:|
| VGG16 | 0.63 | 0.5823 | 0.92 | 0.7132 | 0.26 |
| ResNet18 | **0.66** | **0.6000** | 0.96 | **0.7385** | **0.32** |
| ResNet50 | 0.62 | 0.5682 | **1.00** | 0.7246 | 0.24 |

ResNet18 在人工參考集取得最高 F1-score 與 Cohen's Kappa。三個模型的
Recall 均較高，但 Precision 較低，顯示模型具有偏向預測 Good 的現象。

### 8.3 完整候選池分類分布

| 模型 | Predicted Bad | Predicted Good | Good ratio |
|---|---:|---:|---:|
| VGG16 | 154 | 346 | 0.692 |
| ResNet18 | 124 | 376 | 0.752 |
| ResNet50 | 110 | 390 | 0.780 |

## 9. 主要輸出檔案

```text
outputs/predictions/binary_candidate_pool_predictions.csv
outputs/predictions/binary_candidate_pool_distribution.csv
outputs/predictions/reference_results.csv
outputs/predictions/binary_reference_metrics.csv
outputs/training_runs/selected_models.json
```

`binary_reference_metrics.csv` 包含：

- Agreement count 與 agreement rate
- Accuracy
- Precision
- Recall
- F1-score
- Cohen's Kappa
- Confusion matrix counts

## 10. 實驗紀錄與報告

- [W&B 實驗專案](https://wandb.ai/klorius-/training-Unit7-Binary-413410020)
- [完整 Overleaf 技術報告](report/report.pdf)

報告包含：

- 人工標註規則
- 資料切分與隔離方式
- 模型架構與超參數
- 訓練與驗證曲線
- Learning-rate 比較
- 混淆矩陣
- 人機一致性指標
- 正確與錯誤預測案例分析
- 實驗限制與結論

## 11. GitHub 檔案政策

為避免儲存大型檔案與資料外洩，下列內容不納入 GitHub：

- `data/raw/` 原始圖片
- 模型 checkpoint
- W&B 本機執行資料
- 暫存與壓縮檔

CSV 標註、資料切分、程式碼、設定檔、評估結果與正式報告則保留於
repository，以支援結果檢查與實驗重現。
