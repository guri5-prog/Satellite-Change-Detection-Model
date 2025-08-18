```
# 🛰️ Satellite Image Change Detection  

This project implements a deep learning pipeline for satellite image change detection.  
It supports multiple datasets (DFC2020, LEVIR-CD, LEVIR-CD+, DSIFN, OSCD, Kaggle Change dataset), unifies them into a standard format, and trains a ChangeNet model to detect changes between before/after satellite images.  

---

## 📂 Repository Structure  

```
<img width="864" height="880" alt="image" src="https://github.com/user-attachments/assets/261472c9-8b55-44f0-a1e3-c08a34e38aad" />

````

---

## ⚙️ Installation  

```bash
git clone <repo_url>
cd <repo_name>
pip install -r requirements.txt
````

**Requirements**:

* Python 3.8+
* PyTorch + torchvision
* albumentations
* scikit-learn
* tqdm
* pyyaml
* opencv-python

---

## 📥 Step 1: Download Datasets

Download the following datasets and place them inside the `data/` folder using **exact folder names**:

| Dataset Name     | Folder Name        | Download Link                                                                                                                      |
| ---------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| DFC2020          | `DFC2020`          | [HuggingFace - DFC2020](https://huggingface.co/datasets/GFM-Bench/DFC2020/tree/main/data)                                          |
| DSIFN Train Test | `DSIFN Train Test` | [Kaggle - DSIFN](https://www.kaggle.com/datasets/agouazilynda/dsifn-train-test)                                                    |
| Kaggle Change    | `kaggle_change`    | [Kaggle - Satellite Change Detection](https://www.kaggle.com/datasets/ravi02516/satellite-change-detection?utm_source=chatgpt.com) |
| LEVIR CD         | `LEVIR CD`         | [Kaggle - LEVIR CD](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd)                                                    |
| LEVIR-CD+        | `LEVIR-CD+`        | [Kaggle - LEVIR-CD+](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection)                                  |
| OSCD             | `oscd`             | [OSCD Dataset](https://rcdaudt.github.io/oscd/)                                                                                    |

Your `data/` folder should look like this:

```
data/
 ├── DFC2020
 ├── DSIFN Train Test
 ├── kaggle_change
 ├── LEVIR CD
 ├── LEVIR-CD+
 └── oscd
```

---

## 🛠️ Step 2: Standardize Datasets

Each dataset has its **own standardization script** inside `scripts/` (e.g., `standardize_levircd.py`).
These scripts will:

* Convert dataset format into **unified format**
* Output three folders for each dataset:

  * `A/` → before images
  * `B/` → after images
  * `mask/` → binary change masks

Example usage:

```bash
python scripts/standardize_levircd.py
python scripts/standardize_dsifn.py
```

After running standardization, you will have:

```
data/standardized/{dataset_name}/
 ├── A/
 ├── B/
 └── mask/
```

---

## 🔀 Step 3: Split into Train/Val/Test

Once standardized, run the **split script**:

```bash
python scripts/split_standardized.py
```

This script does two things:

1. If dataset has **official splits** (e.g., DSIFN, DFC2020) → moves them into `data/standardized/{split}/{dataset_name}`.
2. If dataset has **no official splits** → automatically divides into Train (80%), Val (10%), Test (10%).

Final structure:

```
data/standardized/
 ├── train/{dataset}/A, B, mask
 ├── val/{dataset}/A, B, mask
 └── test/{dataset}/A, B, mask
```

---

## 🏗️ Model: ChangeNet

* Based on **ResNet backbone** (18/34/50, pretrained on ImageNet).
* Input: **6-channel tensor** (before + after image).
* Decoder upsamples features → outputs **binary change mask**.
* Activation: `sigmoid`.

📍 File: [`models/changenet.py`](models/changenet.py)

---

## 📚 Dataset Loader

Implemented in [`dataset/change_dataset.py`](dataset/change_dataset.py).

* Loads samples as `{A, B, mask}` triplets.
* Supports augmentation with **Albumentations**:

  * Resize
  * Flips
  * Brightness/contrast
  * Affine transforms
  * Normalization

---

## 📉 Loss & Metrics

* **Loss:** `BCEDiceLoss` = BCE + Dice loss
* **Metrics:**

  * `iou_score()` – Intersection over Union
  * `f1_score()` – F1-Score

📍 Files:

* [`utils/losses.py`](utils/losses.py)
* [`utils/metrics.py`](utils/metrics.py)

---

## 💾 Checkpoints

* Checkpoints saved in `checkpoints/`.
* **last\_checkpoint.pth** → always updated with latest epoch.
* **best\_model.pth** → saved when F1 score improves.
* Training automatically resumes from `last_checkpoint.pth` if available.

📍 File: [`utils/checkpoint.py`](utils/checkpoint.py)

---

## 🚀 Step 4: Train the Model

### Configuration (`configs/config.yaml`)

```yaml
dataset_path: data/standardized        # root path to datasets
image_size: 256                        # input image size
batch_size: 8                          # batch size for training
lr: 0.001                              # learning rate
epochs: 50                             # number of training epochs
save_every: 5                          # save checkpoint every N epochs
resume: true                           # resume from last checkpoint if available
checkpoint_path: checkpoints/latest.pth # path to latest checkpoint
best_model_path: checkpoints/best_model.pth # path to save best model
backbone: resnet34                     # resnet18, resnet34, resnet50 supported
augmentations: true                    # use data augmentations
```

### Run training

```bash
python train.py
```

During training, you will see:

* Epoch progress (loss, IoU, F1)
* Current LR
* Checkpoint saving logs

---

## 📊 Evaluation

After training:

* Best model is evaluated on validation set
* Best threshold for binary mask is found automatically
* Prints final metrics:

```
Accuracy:  0.9606
Precision: 0.9502
Recall:    0.8792
F1-Score:  0.9133
IoU:       0.8405
```

---

## 📝 Summary of Workflow

1. Download datasets → place inside `data/`.
2. Run **standardization script** for each dataset → `data/standardized/{dataset}/A,B,mask`.
3. Run **split script** → `data/standardized/train/val/test`.
4. Configure `configs/config.yaml`.
5. Run `train.py` to start training.

---

## 📌 Citation

If you use this repository, please cite:

* The original datasets (DFC2020, LEVIR-CD, LEVIR-CD+, DSIFN, OSCD, Kaggle Change)
* Pretrained ResNet models (ImageNet)
