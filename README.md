# 🧠 Explainable AI for Multi-Class Brain Tumor MRI Classification with EfficientNet-B0

Built a deep learning approach to recognize four types of brain MRI scans: **glioma, meningioma, pituitary tumor, and normal**. It compares multiple deep learning architectures and includes **cross-validation, performance analysis, and Grad-CAM visualizations** for a clearer understanding of model predictions.

---

## 🎯 Objectives

The main objectives of this project are to:

1. Build an automated multi-class brain tumor MRI classification pipeline.
2. Compare six modern deep-learning architectures under a unified experimental setup.
3. Apply an MRI-oriented preprocessing pipeline including resizing, z-score normalization, CLAHE contrast enhancement, and Non-Local Means denoising.
4. Expand the original dataset through controlled image augmentation and use a reproducible **80/10/10 stratified train/validation/test split**.
5. Evaluate model performance using accuracy, precision, recall, F1-score, Cohen's Kappa, ROC-AUC, PR-AUC, specificity, NPV, and 95% confidence intervals.
6. Analyze computational efficiency through training time, inference time, GPU memory usage, and RAM usage.
7. Improve model transparency using **Grad-CAM** to visualize tumor-relevant regions influencing predictions.
8. Validate the proposed EfficientNet-B0 model using **stratified 5-fold cross-validation**.

---

## 🗂️ Dataset

The project uses the following Kaggle dataset:

🔗 **PMRAM Bangladeshi Brain Cancer - MRI Dataset**  
https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset

The raw dataset contains **1,505 MRI images** distributed across four classes. The images are primarily grayscale JPG MRI scans.

### Classes

| Class | Raw Images | Description |
|---|---:|---|
| `512Glioma` | 373 | Glioma tumor MRI images |
| `512Meningioma` | 363 | Meningioma tumor MRI images |
| `512Normal` | 396 | Normal brain MRI images |
| `512Pituitary` | 373 | Pituitary tumor MRI images |
| **Total** | **1,505** | **Four-class MRI dataset** |

### Dataset Augmentation

Each original image is retained and **four augmentation operations** are randomly applied to generate additional images. The augmentation pipeline uses:

- Random flipping
- Random rotation in the range **−25° to +25°**
- Random zoom in the range **0.8× to 1.2×**
- Random brightness adjustment in the range **0.7× to 1.3×**

This produces **7,525 total images**:

| Class | Augmented Images |
|---|---:|
| Glioma | 1,865 |
| Meningioma | 1,815 |
| Normal | 1,980 |
| Pituitary | 1,865 |
| **Total** | **7,525** |

### Dataset Split

A stratified **80% / 10% / 10%** split is used after augmentation:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Glioma | 1,492 | 186 | 187 |
| Meningioma | 1,452 | 182 | 181 |
| Normal | 1,584 | 198 | 198 |
| Pituitary | 1,492 | 186 | 187 |
| **Total** | **6,020** | **752** | **753** |

---

## 🧠 Models Compared

Six architectures are evaluated under a unified training setup:

1. **ResNet50**
2. **Swin Tiny**
3. **DenseNet121**
4. **LeViT-128s**
5. **MobileViT-S**
6. **EfficientNet-B0**

### Test Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 99.34% | 99.34% | 99.33% | 99.33% | 0.9993 |
| Swin Tiny | 99.47% | 99.47% | 99.48% | 99.47% | 0.9996 |
| DenseNet121 | 98.94% | 98.93% | 98.93% | 98.93% | 0.9996 |
| LeViT-128s | 99.07% | 99.05% | 99.05% | 99.05% | 0.9990 |
| MobileViT-S | 98.94% | 98.93% | 98.94% | 98.93% | 0.9994 |
| **EfficientNet-B0** | **99.60%** | **99.60%** | **99.59%** | **99.60%** | **0.9999** |

**EfficientNet-B0 achieved the highest overall test performance and was selected as the proposed model.**

---

## 🔬 Methodology Overview

```text
Raw MRI Images (1,505, 4 classes)
        │
        ▼
MRI Preprocessing
  ├─ Resize → 224×224
  ├─ Grayscale conversion
  ├─ Z-score normalization
  ├─ Min-Max scaling → 0–1
  ├─ CLAHE contrast enhancement
  └─ Non-Local Means (NLM) denoising
        │
        ▼
Dataset Augmentation
  ├─ Random flip
  ├─ Random rotation (−25° to +25°)
  ├─ Random zoom (0.8× to 1.2×)
  └─ Random brightness (0.7× to 1.3×)
        │
        ▼
Augmented Dataset (7,525 images)
        │
        ▼
Stratified Split
  ├─ Train → 6,020 (80%)
  ├─ Validation → 752 (10%)
  └─ Test → 753 (10%)
        │
        ▼
Train & Compare 6 Architectures
  ├─ ResNet50
  ├─ Swin Tiny
  ├─ DenseNet121
  ├─ LeViT-128s
  ├─ MobileViT-S
  └─ EfficientNet-B0
        │
        ▼
Comprehensive Evaluation
  ├─ Accuracy · Precision · Recall · F1
  ├─ Cohen's Kappa · ROC-AUC · PR-AUC
  ├─ Specificity · NPV
  ├─ 95% Confidence Interval
  ├─ Confusion Matrix
  └─ Computational Profiling
        │
        ▼
Best Model → EfficientNet-B0
        │
        ├─► Learning Curves
        │
        ├─► Explainability: Grad-CAM
        │
        └─► Stratified 5-Fold Cross-Validation
```

### Preprocessing

All images are resized to **224×224** pixels. The preprocessing workflow converts MRI images to grayscale, applies z-score normalization with clipping, rescales intensities, enhances local contrast using **CLAHE**, and removes noise using **Non-Local Means (NLM)** denoising.

The notebook also evaluates preprocessing quality using **PSNR** and **SSIM**. In the sampled quality evaluation, CLAHE followed by denoising produced higher PSNR and SSIM values than the normalized image comparison across the four sampled classes.

### Training Configuration

All evaluated models use a common training configuration:

| Parameter | Value |
|---|---|
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Maximum Epochs | 40 |
| Early Stopping Patience | 5 |
| Optimizer | AdamW |
| Initial Learning Rate | 3 × 10⁻³ |
| Weight Decay | 1 × 10⁻³ |
| Dropout | 0.5 |
| Random Seed | 42 |
| Data Loader Workers | 2 |
| Pretrained Weights | Yes |
| Loss Function | Cross-Entropy Loss |
| LR Scheduler | ReduceLROnPlateau |

Training was performed in a Kaggle GPU environment using **dual T4 GPUs and 32 GB RAM**, with 2 data-loader workers.

### Explainability with Grad-CAM

After EfficientNet-B0 was selected as the best-performing model, **Grad-CAM** was used to generate class-specific activation maps from the model's final convolutional features.

The resulting heatmaps highlight regions of the MRI scans that contributed most strongly to the model's predictions, supporting interpretability and helping assess whether the model focuses on meaningful tumor-related areas.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- TIMM
- NumPy
- Pandas
- OpenCV
- Pillow
- Scikit-learn
- Scikit-image
- Matplotlib
- Seaborn
- Torchsummary
- PSNR / SSIM
- Grad-CAM
- Kaggle GPU environment

---

## ⚙️ Setup

The notebook was developed and executed in **Kaggle**. Download the dataset from Kaggle and add it to your Kaggle notebook environment:

```text
https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset
```

The notebook expects the original Kaggle dataset under the corresponding Kaggle input directory and generates the following working directories during execution:

```text
/kaggle/working/raw_augmented
/kaggle/working/raw_split
/kaggle/working/checkpoints
```

The raw dataset is read from its `Raw Data/Raw` directory.

---

## Requirements

```bash
pip install torch torchvision timm scikit-learn scikit-image opencv-python matplotlib seaborn pandas numpy pillow psutil torchsummary
```

For the exact experimental environment, using Kaggle with GPU acceleration is recommended because the original experiments were performed there.

---

## Run

1. Download the **PMRAM Bangladeshi Brain Cancer - MRI Dataset** from Kaggle and attach it to a Kaggle notebook.
2. Open `brain-tumor-classification-using-mri-images(1).ipynb`.
3. Run the notebook cells sequentially.
4. The notebook first inspects the raw dataset and class distribution.
5. Run the preprocessing and PSNR/SSIM evaluation stages.
6. Generate the augmented dataset and the stratified **80/10/10** train/validation/test split.
7. Train the six candidate architectures and evaluate them on the train, validation, and test sets.
8. Select **EfficientNet-B0** based on the comparative test results.
9. Generate learning curves, ROC/PR curves, confusion matrices, and Grad-CAM visualizations.
10. Run the stratified **5-fold cross-validation** experiment for EfficientNet-B0.
11. Model checkpoints are saved under `/kaggle/working/checkpoints`.

---

## 📈 Key Findings

1. **EfficientNet-B0 performed best** among the six evaluated architectures.

2. It achieved **99.60% test accuracy**, **99.60% precision**, **99.59% recall**, **99.60% F1-score**, and **0.9999 macro ROC-AUC**.

3. Stratified **5-fold cross-validation** produced a mean accuracy of **99.07% ± 0.50%**, with fold accuracies of 99.58%, 98.26%, 98.84%, 99.58%, and 99.09%.

4. EfficientNet-B0 achieved the highest Cohen's Kappa of **0.9947**, with a reported 95% confidence interval of **[0.9907, 1.0000]**.

5. The EfficientNet-B0 confusion matrix showed very limited misclassification: **185/187 Glioma**, **180/181 Meningioma**, **198/198 Normal**, and **187/187 Pituitary** test samples were correctly classified.

6. Grad-CAM visualizations showed that the model focused on class-specific regions associated with tumor-relevant structures, providing an interpretable view of its predictions.

7. Computational profiling showed that **LeViT-128s** was the fastest evaluated model in inference and required the lowest GPU memory, while **EfficientNet-B0** provided the strongest overall balance of predictive performance and computational cost.

8. The study combines model comparison, preprocessing evaluation, cross-validation, computational profiling, learning-curve analysis, and Grad-CAM explainability into a single reproducible evaluation pipeline.

---

## 📚 Research Paper

**Explainable AI for Multi-Class Brain Tumor MRI Classification: An EfficientNet-B0 Approach**

[Read the full paper on IEEE Xplore](https://ieeexplore.ieee.org/document/11502491)

### Authors

- **Md. Imam Hasan** — East West University
- **Amdadur Rahman** — East West University
- **Sami Al Zabid** — East West University

The paper presents the complete comparative evaluation of the six architectures and identifies EfficientNet-B0 as the proposed model for four-class brain tumor MRI classification.

---

## 🔐 License

Please make sure the license in this README matches the license file included in the GitHub repository. The provided project materials do not specify a repository license.

---

## ✉️ Contact

For questions or collaboration:

- **Md. Imam Hasan:** 2023-1-60-030@std.ewubd.edu
- **Amdadur Rahman:** 2023-1-60-037@std.ewubd.edu
- **Sami Al Zabid:** 2022-1-60-103@std.ewubd.edu
