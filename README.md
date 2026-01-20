![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)


# 🏠 Multi-Task CNN for Real-Estate Image Understanding

This repository contains an **end-to-end multi-task computer vision project** for real-estate interior images.  
A single CNN jointly predicts:

- **Room type** (Bathroom, Bedroom, Dining, Kitchen, Livingroom)
- **Photo quality** (GOOD / BAD, using proxy labels)

The project is built to reflect **real-world ML engineering practices**: data preprocessing, multi-task learning, proper evaluation, failure analysis, and an interactive demo.

---

## 🔍 Problem Motivation

Real-estate platforms rely heavily on images to:
- categorize listings by room type,
- filter or rank listings by photo quality,
- detect low-quality or misleading images.

This project demonstrates how a **shared CNN backbone** can efficiently solve both tasks while maintaining strong performance and interpretability.

---

## 📂 Project Structure

```text
.
├── src/
│   ├── train_multitask_tb.py      # Multi-task training with TensorBoard
│   ├── model_multitask.py         # Shared CNN + task-specific heads
│   ├── dataset_multitask.py       # Dataset loader
│   ├── make_quality_labels.py     # Proxy quality label generation
│   ├── export_room_failures.py    # Failure case extraction
│
├── app.py                         # Streamlit demo
├── requirements.txt
├── .gitignore
└── README.md
```
---
## 📊 Dataset

- [Source: Kaggle – House Rooms Image Dataset]{https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset?utm_source=chatgpt.com}
- Classes: Bathroom, Bedroom, Dining, Kitchen, Livingroom
- Total images: ~5,250

Photo Quality Labels

The dataset does not include photo quality annotations.
Therefore, proxy labels were generated using image heuristics:
- blur (Laplacian variance),
- brightness distribution,
- image resolution.

These labels approximate technical image quality, not subjective aesthetics.

---
## 🧠 Model Architecture

- Backbone: ResNet-18 (ImageNet pretrained)
- Shared feature extractor
- Two task heads:
      - Room classification (5-class softmax)
      - Photo quality classification (binary sigmoid)

---
## 🚀 Training Setup

- Framework: PyTorch
- Optimizer: AdamW
- Image size: 224 × 224
- Data split: 70% train / 15% validation / 15% test (stratified by room)
- Class imbalance handling: pos_weight for quality task
- Experiment tracking: TensorBoard

---

## 📈 Results (Test Set)
Room Classification

**Accuracy: 0.87** 

**Macro F1: 0.87**

| Class      | Precision | Recall | F1   |
| ---------- | --------- | ------ | ---- |
| Bathroom   | 0.94      | 0.88   | 0.91 |
| Bedroom    | 0.95      | 0.89   | 0.92 |
| Dining     | 0.86      | 0.82   | 0.84 |
| Kitchen    | 0.84      | 0.82   | 0.83 |
| Livingroom | 0.81      | 0.94   | 0.87 |



<img width="938" height="713" alt="image" src="https://github.com/user-attachments/assets/35108662-b627-4238-a51e-63e66919aabf" />

Photo Quality Classification (Proxy Labels)
- ROC-AUC: 0.964
- PR-AUC: 0.9997
- Best F1 threshold: 0.05

At the optimal threshold:
- F1: 0.997
- Precision: 0.994
- Recall: 1.000

The low optimal threshold reflects conservative probability calibration caused by class imbalance and loss weighting.

<img width="1916" height="870" alt="image" src="https://github.com/user-attachments/assets/3ef58523-089f-4c57-80d7-65bc1b903b59" />

<img width="1912" height="872" alt="image" src="https://github.com/user-attachments/assets/6d8cc1d4-ba81-4017-b6ef-6b42400a6e61" />

<img width="1911" height="885" alt="image" src="https://github.com/user-attachments/assets/8618705f-dbca-4ed9-a52b-279176d82397" />

---

## ❌ Failure Analysis

A systematic error analysis was performed by exporting misclassified samples from the test set.
- Total misclassifications: 266
- Common confusions:
  - Bedroom ↔ Livingroom
  - Kitchen ↔ Dining

- Observed causes:
  - Open-plan layouts
  - Wide-angle images covering multiple room types
  - Mixed furniture semantics
  - Ambiguous labels even for humans

<img width="1710" height="577" alt="image" src="https://github.com/user-attachments/assets/93425abc-fd7a-48db-b697-436af6320a76" />


These results suggest future improvements such as:
- multi-label room tagging,
- use of listing metadata,
- spatial layout or context-aware modeling.

---
## 🖥️ Interactive Demo

An interactive Streamlit application allows:
- uploading interior images,
- viewing room predictions and probabilities,
- adjusting the photo quality decision threshold.

Run locally
```sh
pip install -r requirements.txt
streamlit run app.py
```

<img width="1915" height="875" alt="image" src="https://github.com/user-attachments/assets/eb51e691-6b06-47ab-bf8d-44afd6dc168f" />

<img width="1911" height="871" alt="image" src="https://github.com/user-attachments/assets/e6b93672-5125-4856-ad2f-3cfcd55e8120" />

---

## 🔁 Reproducibility

To reproduce the full pipeline:

```bash
python src/make_quality_labels.py
python src/train_multitask_tb.py
tensorboard --logdir outputs/logs 
streamlit run app.py
```
Model weights, datasets, and logs are excluded from the repository for size.
---

## 🛠️ Tech Stack
1. Python
2. PyTorch / Torchvision
3. OpenCV, Pillow
4. Scikit-learn
5. TensorBoard
6. Streamlit

---
## 📌 Key Takeaways
1. Demonstrates multi-task learning with shared CNN representations
2. Applies real-world ML practices: imbalance handling, threshold tuning, and error analysis
3. Provides a production-style workflow from data preprocessing to an interactive demo
---

## 📎 Notes

Photo quality labels are proxy labels; future work includes human-rated calibration.

The project structure mirrors common industry ML pipelines.

---
