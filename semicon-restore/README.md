# SEMICON-RESTORE

## AI-Based Noisy Low-Resolution Image Restoration

**SEMICON Hackathon 2026**

SEMICON-RESTORE is a deep learning-based image restoration system designed to reconstruct clean, high-resolution grayscale images from noisy and low-resolution inputs.

The project addresses the restoration problem:

**Noisy + Low Resolution → Clean + Full Resolution**

The system is designed with NVIDIA GPU acceleration in mind and uses a lightweight restoration network for efficient inference.

---

## Problem Statement

The input images contain:

* Noise degradation
* Reduced spatial resolution
* Grayscale image information

The objective is to reconstruct a clean, full-resolution image while preserving important image details.

### Input

```text
Noisy Low-Resolution Image
128 × 128
1 channel / grayscale
```

### Output

```text
Clean Full-Resolution Image
256 × 256
1 channel / grayscale
```

---

## Dataset

The project uses the official paired training dataset provided for the challenge.

Each training sample contains:

```text
NoisyLR → Ground Truth
```

Dataset verification performed during development:

* Total training pairs: **3,200**
* Training samples: **2,880**
* Validation samples: **320**
* Input resolution: **128 × 128**
* Ground-truth resolution: **256 × 256**
* Image channels: **1**
* Data type: **float32**

The original dataset is **not included in this repository**.

Place the official dataset locally in:

```text
data/official/
├── train/
└── Test_NoisyLR/
```

---

## Model

The project uses a lightweight convolutional restoration network implemented in PyTorch.

The network performs:

1. Feature extraction
2. Noise suppression
3. Image reconstruction
4. 2× spatial upscaling
5. Full-resolution output generation

### Model size

**776,705 trainable parameters**

The relatively small model size helps reduce computational requirements while maintaining good restoration quality.

---

## Project Structure

```text
SEMICON-RESTORE/
│
├── checkpoints/
│   └── best.pth
│
├── data/
│   └── official/
│       ├── train/
│       └── Test_NoisyLR/
│
├── reports/
│   ├── dataset_statistics.csv
│   └── dataset_visualization.png
│
├── results/
│
├── scripts/
│   ├── baseline.py
│   ├── evaluate.py
│   ├── folder_stats.py
│   ├── inspect_npy.py
│   ├── predict.py
│   ├── show_errors.py
│   ├── show_structure.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_training_step.py
│   ├── train.py
│   ├── verify_pairs.py
│   ├── visualize_dataset.py
│   └── visualize_results.py
│
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── kla_dataset.py
│   │
│   └── models/
│       ├── __init__.py
│       └── restoration_net.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Training

The dataset was divided into:

```text
Training:   2880 images
Validation:  320 images
```

The model was trained using paired noisy and ground-truth images.

Training can be started using:

```bash
python scripts/train.py
```

The training pipeline automatically selects CUDA when a compatible NVIDIA GPU and CUDA-enabled PyTorch installation are available.

---

## Evaluation

Evaluate the trained model using:

```bash
python scripts/evaluate.py
```

The project includes evaluation and visualization utilities for comparing restored images with the corresponding ground-truth images.

---

## Prediction

To generate restored images using the trained checkpoint:

```bash
python scripts/predict.py
```

The best trained model checkpoint is located at:

```text
checkpoints/best_28_4458.pth
```

---

## Testing

### Test dataset loading

```bash
python scripts/test_dataset.py
```

### Test model architecture

```bash
python scripts/test_model.py
```

The model produces:

```text
Input:  1 × 128 × 128
Output: 1 × 256 × 256
```

### Test training step

```bash
python scripts/test_training_step.py
```

---

## Baseline

A bicubic upsampling baseline is included for comparison.

Run:

```bash
python scripts/baseline.py
```

This provides a reference point for evaluating the improvement achieved by the learned restoration model.

---

## Results

The trained model achieved the following validation performance (via
`scripts/evaluate.py`, `skimage` PSNR/SSIM, with test-time augmentation),
re-measured 2026-08-16 against the checkpoint actually present in this repo.
Bicubic is measured by `scripts/baseline.py` on the identical 320-image
validation split (same seed-42 split as the model), so the two rows are a
fair comparison:

| Metric            |        Result |
| ----------------- | ------------: |
| Validation PSNR   | **27.6319 dB**|
| Validation SSIM   | **0.738996**  |
| Bicubic PSNR      |  23.0831 dB   |
| Bicubic SSIM      |  0.546892     |
| Improvement       |  **+4.5488 dB PSNR, +0.192104 SSIM** |
| Model Parameters  |   **776,705** |

Best checkpoint: `checkpoints/best.pth`. A prior checkpoint named
`best_28_4458.pth`, previously reported at 28.50 dB / 0.7624, is no longer
present in this repo and could not be reproduced — do not cite that number.

The repository also contains generated reports and visualization utilities for qualitative and quantitative analysis.

---

## GPU Acceleration

The project is designed to run with NVIDIA GPU acceleration using PyTorch and CUDA.

The training and inference pipeline detects the available device:

```text
CUDA GPU → GPU execution
No CUDA → CPU execution
```

For hackathon deployment, a CUDA-enabled PyTorch environment can be used to accelerate training and inference.

---

## Running on Google Colab

This repo's scripts already auto-detect CUDA and switch to GPU-tuned settings
(bigger batch, pinned memory, background workers, mixed precision) with no
code changes — running on Colab is just a matter of getting the code + data
there and using a GPU runtime.

### 1. Package the project locally

Zip the whole project folder (code + `data/` + `checkpoints/` — the good
checkpoint at `checkpoints/best_28_4458.pth` is what training/prediction
resume from and load by default, so make sure it's included):

```bash
# from the folder ABOVE semicon-restore
zip -r semicon-restore.zip semicon-restore -x "*/__pycache__/*"
```

(On Windows, right-click the `semicon-restore` folder → "Send to" →
"Compressed (zipped) folder" works just as well.)

### 2. Upload it to Google Drive

Upload `semicon-restore.zip` to your Google Drive (drive.google.com, or the
Drive desktop app) — this is far more reliable for a ~1GB file than
uploading directly through a Colab cell.

### 3. Open Colab and switch to a GPU runtime

Go to [colab.research.google.com](https://colab.research.google.com) →
New notebook → **Runtime → Change runtime type → T4 GPU** (free tier).

### 4. Mount Drive and unzip

```python
from google.colab import drive
drive.mount('/content/drive')

!unzip -q "/content/drive/MyDrive/semicon-restore.zip" -d /content/
%cd /content/semicon-restore
```

### 5. Install dependencies

```python
!pip install -q -r requirements.txt
```

Colab's Linux PyPI wheels for `torch` come CUDA-enabled by default (unlike
the Windows CPU-only wheel this project was developed against), so no
special `--index-url` is needed here.

### 6. Confirm the GPU is detected

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### 7. Train, evaluate, predict — same commands as local

```python
!python -m scripts.train      # auto-uses GPU settings; prints "Mixed precision (AMP): True"
!python -m scripts.evaluate   # PSNR / SSIM vs. bicubic baseline, with TTA
!python -m scripts.predict    # writes results/predictions.zip (with TTA)
```

`scripts/train.py` resumes from `checkpoints/best_28_4458.pth` by default and
protects `checkpoints/best.pth` from being overwritten by anything worse —
same safety behavior as running locally.

### 8. Get your results back out

```python
!cp checkpoints/best.pth /content/drive/MyDrive/
!cp results/predictions.zip /content/drive/MyDrive/
```

Copying to Drive is more reliable than `files.download()` for larger files
and survives the Colab session disconnecting.

---

## Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd SEMICON-RESTORE
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU execution, install a CUDA-compatible version of PyTorch appropriate for the target GPU and CUDA environment.

---

## Reproducing the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the official dataset

```text
data/official/
```

### 3. Verify the dataset

```bash
python scripts/verify_pairs.py
```

### 4. Test the model

```bash
python scripts/test_model.py
```

### 5. Train

```bash
python scripts/train.py
```

### 6. Evaluate

```bash
python scripts/evaluate.py
```

### 7. Generate predictions

```bash
python scripts/predict.py
```

---

## Key Features

* Paired noisy-to-clean image restoration
* 2× image super-resolution
* Grayscale image processing
* Lightweight neural network
* Approximately 776K parameters
* PyTorch implementation
* NVIDIA CUDA support
* Training and validation pipeline
* Evaluation scripts
* Dataset verification
* Visualization utilities
* Saved trained model checkpoint

---

## Hackathon Focus

The system focuses on balancing:

**Restoration Quality + Model Efficiency + GPU Compatibility**

The lightweight architecture is intended to provide a practical solution for image restoration while keeping the number of trainable parameters relatively small.

---

## Disclaimer

The official challenge dataset is not included in this repository. Users must obtain the dataset through the official hackathon distribution and place it under the expected `data/official/` directory.

---

## SEMICON Hackathon 2026

**Project:** SEMICON-RESTORE
**Task:** Noisy Low-Resolution → Clean Full-Resolution Image Restoration
**Framework:** PyTorch
**Target Hardware:** NVIDIA GPU
